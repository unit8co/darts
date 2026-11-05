"""
Base Lightning Module
---------------------

Contains abstract classes for deterministic and probabilistic PyTorch Lightning Modules
"""

import copy
from abc import ABC, abstractmethod
from collections.abc import Sequence
from functools import wraps
from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics

from darts.logging import raise_log
from darts.models.components.layer_norm_variants import RINorm
from darts.utils.data.torch_datasets.utils import (
    ModuleStage,
    PLModuleInput,
    PLModuleOutput,
    TorchInferenceBatch,
    TorchTrainingBatch,
    _normalize_train_sample_shape,
)
from darts.utils.likelihood_models.torch import TorchLikelihood
from darts.utils.torch import MonteCarloDropout


def _rin_group_dim(group_config: bool | list[str], full_dim: int) -> int | None:
    """Number of components `RINorm` should normalize for one `use_reversible_instance_norm` group."""
    if group_config is False:
        return None
    return full_dim if group_config is True else len(group_config)


def io_processor(forward):
    """Applies some input / output processing to PLForecastingModule.forward.

    Note that this wrapper must be added to each of PLForecastingModule's subclasses forward methods.
    Here is an example how to add the decorator:

    .. highlight:: python
    .. code-block:: python

        @io_processor
        def forward(self, x_in: PLModuleInput) -> PLModuleOutput:
            pass
    ..

    Current applications include:

    - Reversible Instance Normalization: normalizes batch input target series, past covariates, and/or future
      covariates features (per `use_reversible_instance_norm`'s group config), and inverse transforms the forward
      output back to the original scale. Activated with `use_reversible_instance_norm` at model creation.
    """

    @wraps(forward)
    def forward_wrapper(self, x_in: PLModuleInput, *args, **kwargs):
        if self.rin is None:
            return forward(self, x_in, *args, **kwargs)

        # apply reversible instance normalization to the configured input groups
        x_in = self._rin_normalize_input(x_in)
        out = forward(self, x_in, *args, **kwargs)
        # inverse transform target output back to original scale
        return self._rin_denormalize_output(out)

    return forward_wrapper


class PLForecastingModule(pl.LightningModule, ABC):
    @abstractmethod
    def __init__(
        self,
        input_chunk_length: int,
        output_chunk_length: int,
        output_chunk_shift: int = 0,
        train_sample_shape: dict[str, tuple | None] | tuple | None = None,
        loss_fn: nn.modules.loss._Loss = nn.MSELoss(),
        torch_metrics: torchmetrics.Metric
        | torchmetrics.MetricCollection
        | Sequence[torchmetrics.Metric | torchmetrics.MetricCollection]
        | dict[str, torchmetrics.Metric | torchmetrics.MetricCollection]
        | None = None,
        likelihood: TorchLikelihood | None = None,
        optimizer_cls: type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: dict | None = None,
        lr_scheduler_cls: torch.optim.lr_scheduler._LRScheduler | None = None,
        lr_scheduler_kwargs: dict | None = None,
        use_reversible_instance_norm: bool | dict = False,
    ) -> None:
        """
        PyTorch Lightning-based Forecasting Module.

        This class is meant to be inherited to create a new PyTorch Lightning-based forecasting module.
        When subclassing this class, please make sure to add the following methods with the given signatures:

        - :func:`PLForecastingModule.__init__()`
        - :func:`PLForecastingModule.forward()`

        In subclass `MyModel`'s :func:`__init__` function call ``super(MyModel, self).__init__(**kwargs)`` where
        ``kwargs`` are the parameters of :class:`PLForecastingModule`.

        Parameters
        ----------
        input_chunk_length
            Number of time steps in the past to take as a model input (per chunk). Applies to the target
            series, and past and/or future covariates (if the model supports it).
        output_chunk_length
            Number of time steps predicted at once (per chunk) by the internal model. Also, the number of future values
            from future covariates to use as a model input (if the model supports future covariates). It is not the same
            as forecast horizon `n` used in `predict()`, which is the desired number of prediction points generated
            using either a one-shot- or autoregressive forecast. Setting `n <= output_chunk_length` prevents
            auto-regression. This is useful when the covariates don't extend far enough into the future, or to prohibit
            the model from using future values of past and / or future covariates for prediction (depending on the
            model's covariate support).
        train_sample_shape
            Shape of the model's input, used to instantiate model without calling ``fit_from_dataset`` and
            perform sanity check on new training/inference datasets used for re-training or prediction.
        loss_fn
            PyTorch loss function used for training.
            This parameter will be ignored for probabilistic models if the ``likelihood`` parameter is specified.
            Default: ``torch.nn.MSELoss()``.
        torch_metrics
            A ``torchmetric.Metric`` or a ``MetricCollection`` used for evaluation. A full list of available metrics
            can be found `here <https://torchmetrics.readthedocs.io/en/latest/>`__. Default: ``None``.
        likelihood
            One of Darts' :meth:`Likelihood <darts.utils.likelihood_models.torch.TorchLikelihood>` models to be used for
            probabilistic forecasts. Default: ``None``.
        optimizer_cls
            The PyTorch optimizer class to be used. Default: ``torch.optim.Adam``.
        optimizer_kwargs
            Optionally, some keyword arguments for the PyTorch optimizer (e.g., ``{'lr': 1e-3}``
            for specifying a learning rate). Otherwise the default values of the selected ``optimizer_cls``
            will be used. Default: ``None``.
        lr_scheduler_cls
            Optionally, the PyTorch learning rate scheduler class to be used. Specifying ``None`` corresponds
            to using a constant learning rate. Default: ``None``.
        lr_scheduler_kwargs
            Optionally, some keyword arguments for the PyTorch learning rate scheduler. Default: ``None``.
        use_reversible_instance_norm
            Whether to use reversible instance normalization `RINorm` against distribution shift as shown in [1]_.
            If ``True``, applies ``RINorm`` to the target `series` only, with default hyperparameters. If a
            dictionary, defines which component groups to normalize and the `RINorm` hyperparameters; see
            :meth:`RINorm.parse_config <darts.models.components.layer_norm_variants.RINorm.parse_config>` for the
            supported dict format (``"params"``, ``"series"``, ``"past_covariates"``, ``"future_covariates"`` keys).
            Default: ``False``. For example, to normalize all `series` components, two named
            `past_covariates` components, and no `future_covariates`:

            .. highlight:: python
            .. code-block:: python

                use_reversible_instance_norm={
                    "series": True,  # normalize all `series` components
                    "past_covariates": ["comp1", "compx"],  # normalize only these components, by name
                    "future_covariates": False,  # do not normalize `future_covariates` (also the default)
                }
            ..

        References
        ----------
        .. [1] T. Kim et al. "Reversible Instance Normalization for Accurate Time-Series Forecasting against
                Distribution Shift", https://openreview.net/forum?id=cGDAkQo1C0p
        """
        super().__init__()

        # save hyper parameters for saving/loading
        self.save_hyperparameters(ignore=["loss_fn", "torch_metrics"])

        self.input_chunk_length = input_chunk_length
        # output_chunk_length is a property
        self._output_chunk_length = output_chunk_length
        self.output_chunk_shift = output_chunk_shift

        # define the loss function
        self.criterion = loss_fn
        self.train_criterion = copy.deepcopy(loss_fn)
        self.val_criterion = copy.deepcopy(loss_fn)
        # reduction will be set to `None` when calling `TFM.fit()` with sample weights;
        # reset the actual criterion in method `on_fit_end()`
        self.train_criterion_reduction: str | None = None
        self.val_criterion_reduction: str | None = None

        # by default models are deterministic (i.e. not probabilistic)
        self.likelihood = likelihood

        # saved in checkpoint to be able to instantiate a model without calling fit_from_dataset
        self.train_sample_shape = _normalize_train_sample_shape(train_sample_shape)
        if self.train_sample_shape is not None:
            past_shape = self.train_sample_shape.get("past_target")
            self.n_targets = past_shape[1] if past_shape is not None else 1
        else:
            self.n_targets = 1

        # persist optimiser and LR scheduler parameters
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = dict() if optimizer_kwargs is None else optimizer_kwargs
        self.lr_scheduler_cls = lr_scheduler_cls
        self.lr_scheduler_kwargs = (
            dict() if lr_scheduler_kwargs is None else lr_scheduler_kwargs
        )

        # convert torch_metrics to torchmetrics.MetricCollection
        torch_metrics = self.configure_torch_metrics(torch_metrics)
        self.train_metrics = torch_metrics.clone(prefix="train_")
        self.val_metrics = torch_metrics.clone(prefix="val_")

        # reversible instance norm
        self.rin_config = RINorm.parse_config(use_reversible_instance_norm)
        # full (unselected) dims available in the batch, regardless of which components RIN targets
        shapes = self.train_sample_shape or {}
        past_cov_shape = shapes.get("past_covariates")
        future_cov_shape = shapes.get("future_covariates")
        self._past_cov_full_dim = past_cov_shape[1] if past_cov_shape else 0
        self._future_cov_full_dim = future_cov_shape[1] if future_cov_shape else 0
        if self.rin_config is None:
            self.rin: RINorm | None = None
        else:
            # Verify that if past_covariates or future_covariates is specified, the model actually
            # uses past_covariates or future_covariates
            for group_name, full_dim in (
                ("series", self.n_targets),
                ("past_covariates", self._past_cov_full_dim),
                ("future_covariates", self._future_cov_full_dim),
            ):
                group_cfg = self.rin_config[group_name]
                if full_dim > 0 or group_cfg is False:
                    continue
                condition = (
                    "selects components by name"
                    if isinstance(group_cfg, list)
                    else "is set to `True`"
                )
                raise_log(
                    ValueError(
                        f"`use_reversible_instance_norm['{group_name}']` {condition}, but this "
                        f"model does not use any `{group_name}`."
                    )
                )
            self.rin = RINorm(
                input_dim=_rin_group_dim(self.rin_config["series"], self.n_targets),
                past_cov_dim=_rin_group_dim(
                    self.rin_config["past_covariates"], self._past_cov_full_dim
                ),
                future_cov_dim=_rin_group_dim(
                    self.rin_config["future_covariates"], self._future_cov_full_dim
                ),
                **self.rin_config["params"],
            )
        # column indices selecting named components per group, resolved externally once actual
        # component order is known (e.g. by `TorchForecastingModel`); unused for `bool` group configs
        self.rin_component_indices: dict[str, Sequence[int] | None] = {
            "series": None,
            "past_covariates": None,
            "future_covariates": None,
        }

        # initialize prediction parameters
        self.pred_n: int | None = None
        self.pred_num_samples: int | None = None
        self.pred_roll_size: int | None = None
        self.pred_batch_size: int | None = None
        self.predict_likelihood_parameters: bool | None = None
        self.pred_mc_dropout: bool | None = None

    @abstractmethod
    def forward(self, x_in: PLModuleInput) -> PLModuleOutput:
        """Same as :meth:`torch.nn.Module.forward`.

        Parameters
        ----------
        x_in
            Named module input with independent past, future, and static tensors.
            Recurrent / cached values from a previous call are in ``x_in.state``.
            ``x_in.stage`` is the loop role (train / validate / predict); branch on
            it instead of ``self.trainer``. Keep ``self.training`` for dropout /
            BatchNorm.

        Returns
        -------
        PLModuleOutput
            Named module output. ``prediction`` is the forecast / likelihood-parameter
            tensor; ``state`` is carried to the next ``forward`` as ``PLModuleInput.state``.
        """

    def _onnx_wrapper(self, input_sample: PLModuleInput):
        """Build the generic ONNX export wrapper (features in, ``prediction`` out)."""
        from darts.utils.onnx.export import _prepare_onnx_export

        return _prepare_onnx_export(self, input_sample)

    def training_step(
        self, train_batch: TorchTrainingBatch, batch_idx: int
    ) -> torch.Tensor:
        """performs the training step"""
        return self._train_val_step(
            batch=train_batch,
            stage=ModuleStage.TRAIN,
            criterion=self.train_criterion,
            metrics=self.train_metrics,
        )

    def validation_step(
        self, val_batch: TorchTrainingBatch, batch_idx: int
    ) -> torch.Tensor:
        """performs the validation step"""
        return self._train_val_step(
            batch=val_batch,
            stage=ModuleStage.VALIDATE,
            criterion=self.val_criterion,
            metrics=self.val_metrics,
        )

    def _train_val_step(
        self,
        batch: TorchTrainingBatch,
        stage: ModuleStage,
        criterion,
        metrics,
    ) -> torch.Tensor:
        """performs a training or validation step"""
        output = self(batch.to_module_input(stage=stage))
        loss = self._compute_loss(
            output, batch.future_target, criterion, batch.sample_weight
        )
        self.log(
            f"{stage.value}_loss",
            loss,
            batch_size=batch.past_target.shape[0],
            prog_bar=True,
            sync_dist=True,
        )
        self._update_metrics(output, batch.future_target, metrics)
        return loss

    def on_fit_end(self) -> None:
        # revert the loss function reduction change when sample weights were used
        if self.train_criterion_reduction is not None:
            self.train_criterion.reduction = self.train_criterion_reduction
            self.train_criterion_reduction = None
        if self.val_criterion_reduction is not None:
            self.val_criterion.reduction = self.val_criterion_reduction
            self.val_criterion_reduction = None

    def on_train_epoch_end(self):
        self._compute_metrics(self.train_metrics)

    def on_validation_epoch_end(self):
        self._compute_metrics(self.val_metrics)

    def on_predict_start(self) -> None:
        # optionally, activate monte carlo dropout for prediction
        self.set_mc_dropout(active=self.pred_mc_dropout)

    def on_predict_end(self) -> None:
        # deactivate, monte carlo dropout for any downstream task
        self.set_mc_dropout(active=False)

    def predict_step(
        self,
        batch: TorchInferenceBatch,
        batch_idx: int,
        dataloader_idx: int | None = None,
    ) -> tuple[
        torch.Tensor,
        Sequence[dict[str, Any] | None],
        Sequence[Any],
    ]:
        """performs the prediction step

        batch
            Collated :class:`~darts.utils.data.torch_datasets.utils.TorchInferenceBatch` from
            Darts' :class:`TorchInferenceDataset`.
        batch_idx
            the batch index of the current batch
        dataloader_idx
            the dataloader index
        """
        # number of individual series to be predicted in current batch
        num_series = batch.past_target.shape[0]

        # number of times the input tensor should be tiled to produce predictions for multiple samples
        # this variable is larger than 1 only if the batch_size is at least twice as large as the number
        # of individual time series being predicted in current batch (`num_series`)
        batch_sample_size = min(
            max(self.pred_batch_size // num_series, 1), self.pred_num_samples
        )

        # counts number of produced prediction samples for every series to be predicted in current batch
        sample_count = 0

        # repeat prediction procedure for every needed sample
        batch_predictions = []
        while sample_count < self.pred_num_samples:
            # make sure we don't produce too many samples
            if sample_count + batch_sample_size > self.pred_num_samples:
                batch_sample_size = self.pred_num_samples - sample_count

            # stack multiple copies of the tensors to produce probabilistic forecasts
            input_batch_samples = batch.tile_tensors(batch_sample_size)

            # get predictions for 1 whole batch (can include predictions of multiple series
            # and for multiple samples if a probabilistic forecast is produced)
            batch_prediction = self._get_batch_prediction(
                self.pred_n, input_batch_samples, self.pred_roll_size
            )

            # reshape from 3d tensor (num_series x batch_sample_size, ...)
            # into 4d tensor (batch_sample_size, num_series, ...), where dim 0 represents the samples
            out_shape = batch_prediction.shape
            batch_prediction = batch_prediction.reshape(
                (
                    batch_sample_size,
                    num_series,
                )
                + out_shape[1:]
            )

            # save all predictions and update the `sample_count` variable
            batch_predictions.append(batch_prediction)
            sample_count += batch_sample_size

        # concatenate the batch of samples, to form self.pred_num_samples samples
        batch_predictions = torch.cat(batch_predictions, dim=0)
        return (
            batch_predictions,
            batch.series_schema,
            batch.pred_time,
        )

    def set_predict_parameters(
        self,
        n: int,
        num_samples: int,
        roll_size: int,
        batch_size: int,
        predict_likelihood_parameters: bool,
        mc_dropout: bool,
    ) -> None:
        """to be set from TorchForecastingModel before calling trainer.predict() and reset at self.on_predict_end()"""
        self.pred_n = n
        self.pred_num_samples = num_samples
        self.pred_roll_size = roll_size
        self.pred_batch_size = batch_size
        self.predict_likelihood_parameters = predict_likelihood_parameters
        self.pred_mc_dropout = mc_dropout

    def set_rin_component_indices(
        self,
        series: Sequence[int] | None = None,
        past_covariates: Sequence[int] | None = None,
        future_covariates: Sequence[int] | None = None,
    ) -> None:
        """Sets the column indices (within the corresponding batch tensor) that `RINorm` should select for
        groups configured with a list of component names in `use_reversible_instance_norm`. To be called
        once the actual component order is known (e.g. by `TorchForecastingModel`), before running the model.
        """
        self.rin_component_indices = {
            "series": series,
            "past_covariates": past_covariates,
            "future_covariates": future_covariates,
        }

    def _select_rin_group(
        self, tensor: torch.Tensor | None, group_name: str
    ) -> tuple[torch.Tensor | None, slice | Sequence[int] | None]:
        """Selects the `RINorm`-enabled components of `tensor` for `use_reversible_instance_norm`
        group `group_name`, returning the selection together with the index/slice needed to write
        the normalized values back into `tensor` (see :meth:`_merge_rin_group`).

        Returns ``(None, None)`` if `tensor` is `None` (the group has no data in this batch) or the
        group is disabled in `rin_config`.
        """
        cfg = self.rin_config[group_name]
        if tensor is None or cfg is False:
            return None, None
        if cfg is True:
            return tensor, slice(None)
        idx = self.rin_component_indices[group_name]
        if idx is None:
            raise_log(
                ValueError(
                    f"`use_reversible_instance_norm['{group_name}']` selects components by name, but the "
                    "corresponding indices have not been resolved yet. Call `set_rin_component_indices()` "
                    "before running the model."
                )
            )
        return tensor[:, :, idx], idx

    @staticmethod
    def _merge_rin_group(
        tensor: torch.Tensor | None,
        group_out: torch.Tensor | None,
        idx: slice | Sequence[int] | None,
    ) -> torch.Tensor | None:
        """Returns `tensor` with its columns `idx` (as returned by :meth:`_select_rin_group`)
        replaced by the normalized `group_out` values.

        Returns `tensor` unchanged if `group_out` is `None` (the group is not normalized). Never
        modifies `tensor` in-place, since it may share storage with the original batch (and
        modifying it would re-normalize it on a later call or corrupt the autograd graph).
        """
        if group_out is None:
            return tensor
        if isinstance(idx, slice):
            # all components were normalized; `RINorm` outputs are freshly allocated tensors
            return group_out
        tensor = tensor.clone()
        tensor[:, :, list(idx)] = group_out
        return tensor

    @staticmethod
    def _rin_as_tuple(
        result: torch.Tensor
        | tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None],
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """Normalizes an `RINorm` `forward`/`transform`/`inverse` call's result to always be a
        3-tuple ``(series, past_cov, future_cov)``, regardless of whether `RINorm` took its
        backward-compatible bare-tensor shortcut for that particular call.
        """
        if isinstance(result, tuple):
            return result
        return result, None, None

    def _rin_normalize_input(self, x_in: PLModuleInput) -> PLModuleInput:
        """Applies `RINorm` to the configured groups of `x_in` (past target, past covariates,
        historic future covariates, and future covariates), returning a new `PLModuleInput` with
        the normalized tensors to feed into the wrapped `forward()`.
        """
        series_in, series_idx = self._select_rin_group(x_in.past_target, "series")
        past_cov_in, past_cov_idx = self._select_rin_group(
            x_in.past_covariates, "past_covariates"
        )
        hist_future_cov_in, hist_future_cov_idx = self._select_rin_group(
            x_in.historic_future_covariates, "future_covariates"
        )

        series_out, past_cov_out, hist_future_cov_out = self._rin_as_tuple(
            self.rin(x=series_in, past_cov=past_cov_in, future_cov=hist_future_cov_in)
        )

        # the future window of future covariates reuses the historic window's normalization statistics
        future_cov_in, future_cov_idx = self._select_rin_group(
            x_in.future_covariates, "future_covariates"
        )
        _, _, future_cov_out = self._rin_as_tuple(
            self.rin.transform(future_cov=future_cov_in)
        )

        return x_in.replace(
            past_target=self._merge_rin_group(x_in.past_target, series_out, series_idx),
            past_covariates=self._merge_rin_group(
                x_in.past_covariates, past_cov_out, past_cov_idx
            ),
            historic_future_covariates=self._merge_rin_group(
                x_in.historic_future_covariates,
                hist_future_cov_out,
                hist_future_cov_idx,
            ),
            future_covariates=self._merge_rin_group(
                x_in.future_covariates, future_cov_out, future_cov_idx
            ),
        )

    def _rin_denormalize_output(self, out: PLModuleOutput) -> PLModuleOutput:
        """Inverse-transforms the wrapped `forward()`'s target output back to the original scale,
        if `series` is one of the active RIN groups (a no-op otherwise, since it was never
        normalized on the way in).
        """
        if not self.rin.has_series:
            return out

        # only the RIN-selected target columns were normalized on the way in (see
        # `_rin_normalize_input`); denormalize just those, leaving any non-selected target
        # columns (already in their original scale) untouched.
        prediction = out.prediction
        pred_sel, pred_idx = self._select_rin_group(prediction, "series")
        pred_inv, _, _ = self._rin_as_tuple(self.rin.inverse(x=pred_sel))
        return out.replace(
            prediction=self._merge_rin_group(prediction, pred_inv, pred_idx)
        )

    def _compute_loss(self, output: PLModuleOutput, target, criterion, sample_weight):
        # prediction is of shape (batch_size, n_timesteps, n_components, n_params)
        prediction = output.prediction
        if self.likelihood:
            loss = self.likelihood.compute_loss(prediction, target, sample_weight)
        else:
            # If there's no likelihood, nr_params=1, and we need to squeeze out the
            # last dimension of model prediction, for properly computing the loss.
            loss = criterion(prediction.squeeze(dim=-1), target)
            if sample_weight is not None:
                loss = (loss * sample_weight).mean()
        return loss

    def _update_metrics(self, output: PLModuleOutput, target: torch.Tensor, metrics):
        if not len(metrics):
            return

        prediction = output.prediction
        if self.likelihood:
            prediction = self.likelihood.sample(prediction)
        else:
            # If there's no likelihood, nr_params=1, and we need to squeeze out the
            # last dimension of model prediction, for properly computing the metric.
            prediction = prediction.squeeze(dim=-1)

        # torch metrics require 2D targets of shape (batch size * ocl, num targets)
        # contiguous() is needed because model outputs can be non-contiguous views
        # (e.g. NBEATS slices the last dimension), and some torchmetrics implementations
        # call .view() internally which requires a contiguous tensor.
        target = target.reshape(-1, self.n_targets).contiguous()
        prediction = prediction.reshape(-1, self.n_targets).contiguous()

        metrics.update(prediction, target)

    def _compute_metrics(self, metrics):
        if not len(metrics):
            return

        res = metrics.compute()
        self.log_dict(
            res,
            on_epoch=True,
            on_step=False,
            logger=True,
            prog_bar=True,
            sync_dist=True,
        )
        metrics.reset()

    def configure_optimizers(self):
        """configures optimizers and learning rate schedulers for model optimization."""

        # A utility function to create optimizer and lr scheduler from desired classes
        def _create_from_cls_and_kwargs(cls, kws):
            try:
                return cls(**kws)
            except (TypeError, ValueError) as e:
                raise_log(
                    ValueError(
                        "Error when building the optimizer or learning rate scheduler;"
                        "please check the provided class and arguments"
                        f"\nclass: {cls}"
                        f"\narguments (kwargs): {kws}"
                        f"\nerror:\n{e}"
                    ),
                )

        # Create the optimizer and (optionally) the learning rate scheduler
        # we have to create copies because we cannot save model.parameters into object state (not serializable)
        optimizer_kws = {k: v for k, v in self.optimizer_kwargs.items()}
        optimizer_kws["params"] = self.parameters()

        optimizer = _create_from_cls_and_kwargs(self.optimizer_cls, optimizer_kws)

        if self.lr_scheduler_cls is not None:
            lr_sched_kws = {k: v for k, v in self.lr_scheduler_kwargs.items()}
            lr_sched_kws["optimizer"] = optimizer

            # lr scheduler can be configured with lightning; defaults below
            lr_config_params = {
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
                "strict": True,
                "name": None,
            }
            # update config with user params
            lr_config_params = {
                k: (v if k not in lr_sched_kws else lr_sched_kws.pop(k))
                for k, v in lr_config_params.items()
            }

            lr_scheduler = _create_from_cls_and_kwargs(
                self.lr_scheduler_cls, lr_sched_kws
            )

            return [optimizer], dict({"scheduler": lr_scheduler}, **lr_config_params)
        else:
            return optimizer

    def _get_batch_prediction(
        self, n: int, input_batch: TorchInferenceBatch, roll_size: int
    ) -> torch.Tensor:
        """Generates batch predictions.

        Feeds `PLForecastingModule` with past, future, and static features to forecast the next ``n`` target values
        per target variable.

        Parameters
        ----------
        n
            prediction length
        input_batch
            Collated inference batch (tensors may already be tiled for multi-sample prediction).
        roll_size
            roll input arrays after every sequence by ``roll_size``. Initially, ``roll_size`` is equivalent to
            ``self.output_chunk_length``
        """

        past_target = input_batch.past_target
        past_covariates = input_batch.past_covariates
        future_past_covariates = input_batch.future_past_covariates
        historic_future_covariates = input_batch.historic_future_covariates
        future_covariates = input_batch.future_covariates
        static_covariates = input_batch.static_covariates

        def _build_pl_input(future_cov_slice, state=None):
            return PLModuleInput(
                past_target=past_target,
                past_covariates=past_covariates,
                historic_future_covariates=historic_future_covariates,
                future_covariates=future_cov_slice,
                static_covariates=static_covariates,
                future_target=None,
                state=state,
                stage=ModuleStage.PREDICT,
            )

        future_cov_slice = (
            future_covariates[:, : self.output_chunk_length, :]
            if future_covariates is not None
            else None
        )
        pl_input = _build_pl_input(future_cov_slice)
        module_out = self._produce_predict_output(x=pl_input)
        out = module_out.prediction
        state = module_out.state

        batch_prediction = [out[:, :roll_size, :]]
        prediction_length = roll_size

        # predict at least `output_chunk_length` points, so that we use the most recent target values
        min_n = n if n >= self.output_chunk_length else self.output_chunk_length
        while prediction_length < min_n:
            # we want the last prediction to end exactly at `min_n` into the future.
            # this means we may have to truncate the previous prediction and step
            # back the roll size for the last chunk
            if prediction_length + self.output_chunk_length > min_n:
                spillover_prediction_length = (
                    prediction_length + self.output_chunk_length - min_n
                )
                roll_size -= spillover_prediction_length
                prediction_length -= spillover_prediction_length
                batch_prediction[-1] = batch_prediction[-1][:, :roll_size, :]

            # roll over past input tensors to contain the latest target and covariates
            past_target = torch.roll(past_target, -roll_size, 1)
            if past_covariates is not None:
                past_covariates = torch.roll(past_covariates, -roll_size, 1)
            if historic_future_covariates is not None:
                historic_future_covariates = torch.roll(
                    historic_future_covariates, -roll_size, 1
                )

            # update target input to include next `roll_size` predictions
            if self.input_chunk_length >= roll_size:
                past_target[:, -roll_size:, :] = out[:, :roll_size, :]
            else:
                past_target[:, :, :] = out[:, -self.input_chunk_length :, :]

            # set left and right boundaries for extracting future elements
            if self.input_chunk_length >= roll_size:
                left_past, right_past = prediction_length - roll_size, prediction_length
            else:
                left_past, right_past = (
                    prediction_length - self.input_chunk_length,
                    prediction_length,
                )

            # update past covariates to include next `roll_size` future past covariates elements
            if past_covariates is not None and future_past_covariates is not None:
                if self.input_chunk_length >= roll_size:
                    past_covariates[:, -roll_size:, :] = future_past_covariates[
                        :, left_past:right_past, :
                    ]
                else:
                    past_covariates[:, :, :] = future_past_covariates[
                        :, left_past:right_past, :
                    ]

            # update historic future covariates to include next `roll_size` future covariates elements
            if historic_future_covariates is not None and future_covariates is not None:
                if self.input_chunk_length >= roll_size:
                    historic_future_covariates[:, -roll_size:, :] = future_covariates[
                        :, left_past:right_past, :
                    ]
                else:
                    historic_future_covariates[:, :, :] = future_covariates[
                        :, left_past:right_past, :
                    ]

            left_future, right_future = (
                right_past,
                right_past + self.output_chunk_length,
            )
            future_cov_slice = (
                future_covariates[:, left_future:right_future, :]
                if future_covariates is not None
                else None
            )

            pl_input = _build_pl_input(future_cov_slice, state=state)
            module_out = self._produce_predict_output(x=pl_input)
            out = module_out.prediction
            state = module_out.state

            batch_prediction.append(out)
            prediction_length += self.output_chunk_length

        # bring predictions into desired format and drop unnecessary values
        batch_prediction = torch.cat(batch_prediction, dim=1)
        batch_prediction = batch_prediction[:, :n, :]
        return batch_prediction

    def _get_mc_dropout_modules(self) -> set:
        def recurse_children(children, acc):
            for module in children:
                if isinstance(module, MonteCarloDropout):
                    acc.add(module)
                acc = recurse_children(module.children(), acc)
            return acc

        return recurse_children(self.children(), set())

    def set_mc_dropout(self, active: bool):
        # optionally, activate dropout in all MonteCarloDropout modules
        for module in self._get_mc_dropout_modules():
            module._mc_dropout_enabled = active

    @property
    def supports_probabilistic_prediction(self) -> bool:
        return self.likelihood is not None or len(self._get_mc_dropout_modules()) > 0

    def _produce_predict_output(self, x: PLModuleInput) -> PLModuleOutput:
        output = self(x)
        prediction = output.prediction
        if self.likelihood:
            if self.predict_likelihood_parameters:
                prediction = self.likelihood.predict_likelihood_parameters(prediction)
            else:
                prediction = self.likelihood.sample(prediction)
        else:
            prediction = prediction.squeeze(dim=-1)
        return output.replace(prediction=prediction)

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        # we must save the dtype for correct parameter precision at loading time
        checkpoint["model_dtype"] = self.dtype
        # we must save the shape of the input to be able to instantiate the model without calling fit_from_dataset
        checkpoint["train_sample_shape"] = self.train_sample_shape
        # we must save the loss to properly restore it when resuming training
        checkpoint["loss_fn"] = self.criterion
        # we must save the metrics to continue logging them when resuming training
        checkpoint["torch_metrics_train"] = self.train_metrics
        checkpoint["torch_metrics_val"] = self.val_metrics
        # column indices resolved from named `use_reversible_instance_norm` groups are not
        # recoverable from the checkpoint's tensors/hyperparameters alone (the original `TimeSeries`
        # component names are not saved), so they must be persisted directly
        checkpoint["rin_component_indices"] = self.rin_component_indices

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        # by default our models are initialized as float32. For other dtypes, we need to cast to the correct precision
        # before parameters are loaded by PyTorch-Lightning
        dtype = checkpoint["model_dtype"]
        self.to_dtype(dtype)

        # restoring attributes necessary to resume from training properly
        self.criterion = checkpoint["loss_fn"]
        self.train_metrics = checkpoint["torch_metrics_train"]
        self.val_metrics = checkpoint["torch_metrics_val"]
        # absent from checkpoints saved before named `use_reversible_instance_norm` groups existed;
        # `self.rin_component_indices` (all `None`, from `__init__`) is left as-is in that case
        if "rin_component_indices" in checkpoint:
            self.rin_component_indices = checkpoint["rin_component_indices"]

    def to_dtype(self, dtype):
        """Cast module precision (float32 by default) to another precision."""
        if dtype == torch.float16:
            self.half()
        elif dtype == torch.float32:
            self.float()
        elif dtype == torch.float64:
            self.double()
        else:
            raise_log(
                ValueError(
                    f"Trying to load dtype `{dtype}`. Loading for this type is not implemented yet. Please report this "
                    f"issue on https://github.com/unit8co/darts."
                ),
            )

    @property
    def epochs_trained(self):
        return self.current_epoch

    @property
    def output_chunk_length(self) -> int | None:
        """
        Number of time steps predicted at once by the model.
        """
        return self._output_chunk_length

    @staticmethod
    def configure_torch_metrics(
        torch_metrics: torchmetrics.Metric
        | torchmetrics.MetricCollection
        | Sequence[torchmetrics.Metric | torchmetrics.MetricCollection]
        | dict[str, torchmetrics.Metric | torchmetrics.MetricCollection],
    ) -> torchmetrics.MetricCollection:
        """process the torch_metrics parameter."""
        return torchmetrics.MetricCollection(
            torch_metrics if torch_metrics is not None else []
        )
