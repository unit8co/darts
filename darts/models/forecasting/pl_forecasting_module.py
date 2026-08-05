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
    PLModuleInput,
    TorchBatch,
    TorchInferenceBatch,
    TorchTrainingBatch,
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
        def forward(self, *args, **kwargs)
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

        x_past, x_future, x_static, future_target = x_in
        past_features, x_future_in = self._rin_normalize_input(x_past, x_future)
        out = forward(
            self, (past_features, x_future_in, x_static, future_target), *args, **kwargs
        )
        return self._rin_denormalize_output(out)

    return forward_wrapper


class PLForecastingModule(pl.LightningModule, ABC):
    @abstractmethod
    def __init__(
        self,
        input_chunk_length: int,
        output_chunk_length: int,
        output_chunk_shift: int = 0,
        train_sample_shape: tuple | None = None,
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
        - :func:`PLForecastingModule._process_input_batch()`
        - :func:`PLForecastingModule._produce_train_output()`
        - :func:`PLForecastingModule._get_batch_prediction()`

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
        self.train_sample_shape = train_sample_shape
        self.n_targets = (
            train_sample_shape[0][1] if train_sample_shape is not None else 1
        )

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
        self._past_cov_full_dim = (
            train_sample_shape[1][1]
            if train_sample_shape is not None
            and len(train_sample_shape) > 1
            and train_sample_shape[1] is not None
            else 0
        )
        self._future_cov_full_dim = (
            train_sample_shape[3][1]
            if train_sample_shape is not None
            and len(train_sample_shape) > 3
            and train_sample_shape[3] is not None
            else 0
        )
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

    @property
    def first_prediction_index(self) -> int:
        """
        Returns the index of the first predicted within the output of self.model.
        """
        return 0

    @abstractmethod
    def forward(self, x_in: PLModuleInput, *args, **kwargs) -> Any:
        """Same as :meth:`torch.nn.Module.forward`.

        Parameters
        ----------
        x_in
            ``(x_past, x_future, x_static, future_target)`` the past, future, and static features, as well as
            the future target.
        *args
            Whatever you decide to pass into the forward method.
        **kwargs
            Keyword arguments are also possible.

        Returns
        -------
        Any
            The module's output.
        """

    def training_step(
        self, train_batch: TorchTrainingBatch, batch_idx: int
    ) -> torch.Tensor:
        """performs the training step"""
        return self._train_val_step(
            batch=train_batch,
            name="train",
            criterion=self.train_criterion,
            metrics=self.train_metrics,
        )

    def validation_step(
        self, val_batch: TorchTrainingBatch, batch_idx: int
    ) -> torch.Tensor:
        """performs the validation step"""
        return self._train_val_step(
            batch=val_batch,
            name="val",
            criterion=self.val_criterion,
            metrics=self.val_metrics,
        )

    def _train_val_step(
        self,
        batch: TorchTrainingBatch,
        name: str,
        criterion,
        metrics,
    ) -> torch.Tensor:
        """performs a training or validation step"""
        (
            past_target,
            past_covariates,
            historic_future_covariates,
            future_covariates,
            static_covariates,
            sample_weight,
            future_target,
        ) = batch

        output = self._produce_train_output(
            (
                past_target,
                past_covariates,
                historic_future_covariates,
                future_covariates,
                static_covariates,
                future_target if name == "train" else None,
            ),
        )
        loss = self._compute_loss(output, future_target, criterion, sample_weight)
        self.log(
            f"{name}_loss",
            loss,
            batch_size=past_target.shape[0],
            prog_bar=True,
            sync_dist=True,
        )
        self._update_metrics(output, future_target, metrics)
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
            output of Darts' :class:`TorchInferenceDataset` - tuple of ``(past target, past cov,
            future past cov, historic future cov, future cov, static cov, target series schema,
            prediction start time step)``
        batch_idx
            the batch index of the current batch
        dataloader_idx
            the dataloader index
        """
        # batch has elements (past target, past cov, future past cov, historic future cov, future cov,
        # static cov, target series schema, pred start time)
        input_data_tuple, batch_series_schemas, batch_pred_starts = (
            batch[:-2],
            batch[-2],
            batch[-1],
        )

        # number of individual series to be predicted in current batch
        num_series = input_data_tuple[0].shape[0]

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
            input_data_tuple_samples = self._sample_tiling(
                input_data_tuple, batch_sample_size
            )

            # get predictions for 1 whole batch (can include predictions of multiple series
            # and for multiple samples if a probabilistic forecast is produced)
            batch_prediction = self._get_batch_prediction(
                self.pred_n, input_data_tuple_samples, self.pred_roll_size
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
            batch_series_schemas,
            batch_pred_starts,
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
        tensor: torch.Tensor,
        offset: int,
        group_out: torch.Tensor | None,
        idx: slice | Sequence[int] | None,
    ) -> None:
        """Writes the normalized `group_out` values into `tensor`'s columns `idx` (as returned by
        :meth:`_select_rin_group`), shifted by `offset` (this group's columns don't necessarily
        start at column `0` of `tensor`).

        Writes in-place: `tensor` must be a tensor freshly allocated for this purpose (e.g. via
        `.clone()`), not a view sharing storage with another RIN group's input, which would
        corrupt the autograd graph shared with that other group.
        """
        if group_out is None:
            return
        cols = (
            slice(offset, offset + group_out.shape[2])
            if idx == slice(None)
            else [offset + i for i in idx]
        )
        tensor[:, :, cols] = group_out

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

    def _rin_normalize_input(
        self, x_past: torch.Tensor, x_future: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Applies `RINorm` to the configured groups of `x_past` (target, past covariates, and
        historic future covariates, concatenated) and `x_future` (future covariates), returning the
        normalized replacements to feed into the wrapped `forward()`.
        """
        n_targets = self.n_targets
        n_past_cov = self._past_cov_full_dim
        n_future_cov = self._future_cov_full_dim

        series_slice = x_past[:, :, :n_targets]
        past_cov_slice = (
            x_past[:, :, n_targets : n_targets + n_past_cov] if n_past_cov else None
        )
        hist_future_cov_slice = (
            x_past[:, :, n_targets + n_past_cov : n_targets + n_past_cov + n_future_cov]
            if n_future_cov
            else None
        )

        series_in, series_idx = self._select_rin_group(series_slice, "series")
        past_cov_in, past_cov_idx = self._select_rin_group(
            past_cov_slice, "past_covariates"
        )
        hist_future_cov_in, future_cov_idx = self._select_rin_group(
            hist_future_cov_slice, "future_covariates"
        )

        series_out, past_cov_out, hist_future_cov_out = self._rin_as_tuple(
            self.rin(x=series_in, past_cov=past_cov_in, future_cov=hist_future_cov_in)
        )

        # write every active group's normalized values into a single freshly-allocated buffer,
        # instead of cloning per-group and then copying everything again via `torch.cat` (which
        # always allocates a new, full-width tensor anyway). At least one group is always active
        # here (`_rin_normalize_input` only runs when `self.rin is not None`, which guarantees it).
        past_features = x_past.clone()
        self._merge_rin_group(past_features, 0, series_out, series_idx)
        self._merge_rin_group(past_features, n_targets, past_cov_out, past_cov_idx)
        self._merge_rin_group(
            past_features,
            n_targets + n_past_cov,
            hist_future_cov_out,
            future_cov_idx,
        )

        # the future window of future covariates reuses the historic window's normalization statistics
        future_sel, future_idx = self._select_rin_group(x_future, "future_covariates")
        _, _, future_norm = self._rin_as_tuple(
            self.rin.transform(future_cov=future_sel)
        )
        if future_norm is None:
            x_future_in = x_future
        else:
            x_future_in = x_future.clone()
            self._merge_rin_group(x_future_in, 0, future_norm, future_idx)

        return past_features, x_future_in

    def _rin_denormalize_output(self, out: Any) -> Any:
        """Inverse-transforms the wrapped `forward()`'s target output back to the original scale,
        if `series` is one of the active RIN groups (a no-op otherwise, since it was never
        normalized on the way in).
        """
        if not self.rin.has_series:
            return out

        if isinstance(out, tuple):
            # RNNModel returns a tuple with hidden state
            out_series, out_extra = out[0], out[1:]
        else:
            out_series, out_extra = out, None

        # only the RIN-selected target columns were normalized on the way in (see
        # `_rin_normalize_input`); denormalize just those, leaving any non-selected target
        # columns (already in their original scale) untouched.
        series_sel, series_idx = self._select_rin_group(out_series, "series")
        inv_series, _, _ = self._rin_as_tuple(self.rin.inverse(x=series_sel))
        out_series = out_series.clone()
        self._merge_rin_group(out_series, 0, inv_series, series_idx)

        return (out_series, *out_extra) if out_extra is not None else out_series

    def _compute_loss(self, output, target, criterion, sample_weight):
        # output is of shape (batch_size, n_timesteps, n_components, n_params)
        if self.likelihood:
            loss = self.likelihood.compute_loss(output, target, sample_weight)
        else:
            # If there's no likelihood, nr_params=1, and we need to squeeze out the
            # last dimension of model output, for properly computing the loss.
            loss = criterion(output.squeeze(dim=-1), target)
            if sample_weight is not None:
                loss = (loss * sample_weight).mean()
        return loss

    def _update_metrics(self, output, target, metrics):
        if not len(metrics):
            return

        if self.likelihood:
            pred = self.likelihood.sample(output)
        else:
            # If there's no likelihood, nr_params=1, and we need to squeeze out the
            # last dimension of model output, for properly computing the metric.
            pred = output.squeeze(dim=-1)

        # torch metrics require 2D targets of shape (batch size * ocl, num targets)
        target = target.reshape(-1, self.n_targets)
        pred = pred.reshape(-1, self.n_targets)

        metrics.update(pred, target)

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

    def _produce_train_output(self, input_batch: TorchBatch):
        """Generates train output.

        Feeds `PLForecastingModule` with (past target + past cov + historic future cov (concatenated), future cov,
        static cov)

        Parameters
        ----------
        input_batch
            ``(past target, past cov, historic future cov, future cov, static cov, future target)``.
        """
        return self(self._process_input_batch(input_batch))

    def _process_input_batch(self, input_batch: TorchBatch) -> PLModuleInput:
        """Processes module input batch.

        Converts output of a dataset into a tuple of tensors (past target + past cov + historic future cov
        (concatenated), future cov, static cov)

        Parameters
        ----------
        input_batch
            ``(past target, past cov, historic future cov, future cov, static cov, future target)``.

        Returns
        -------
        tuple
            ``(x_past, x_future, x_static, future_target)`` the past, future, and static features, as well as
            the future target.
        """
        (
            past_target,
            past_covariates,
            historic_future_covariates,
            future_covariates,
            static_covariates,
            future_target,
        ) = input_batch
        dim_comp = 2

        x_past = torch.cat(
            [
                tensor
                for tensor in [
                    past_target,
                    past_covariates,
                    historic_future_covariates,
                ]
                if tensor is not None
            ],
            dim=dim_comp,
        )
        return x_past, future_covariates, static_covariates, future_target

    def _get_batch_prediction(
        self, n: int, input_batch: tuple[torch.Tensor | None, ...], roll_size: int
    ) -> torch.Tensor:
        """Generates batch predictions.

        Feeds `PLForecastingModule` with past, future, and static features to forecast the next ``n`` target values
        per target variable.

        Parameters
        ----------
        n
            prediction length
        input_batch
            (past target, past cov, future past cov, historic future cov, future cov, static cov)
        roll_size
            roll input arrays after every sequence by ``roll_size``. Initially, ``roll_size`` is equivalent to
            ``self.output_chunk_length``
        """

        dim_component = 2
        (
            past_target,
            past_covariates,
            future_past_covariates,
            historic_future_covariates,
            future_covariates,
            static_covariates,
        ) = input_batch

        n_targets = past_target.shape[dim_component]
        n_past_covs = (
            past_covariates.shape[dim_component] if past_covariates is not None else 0
        )
        n_future_covs = (
            future_covariates.shape[dim_component]
            if future_covariates is not None
            else 0
        )

        input_past, input_future, input_static, _ = self._process_input_batch((
            past_target,
            past_covariates,
            historic_future_covariates,
            (
                future_covariates[:, :roll_size, :]
                if future_covariates is not None
                else None
            ),
            static_covariates,
            None,  # future target
        ))

        out = self._produce_predict_output(
            x=(input_past, input_future, input_static, None)
        )[:, self.first_prediction_index :, :]

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

            # ==========> PAST INPUT <==========
            # roll over input series to contain the latest target and covariates
            input_past = torch.roll(input_past, -roll_size, 1)

            # update target input to include next `roll_size` predictions
            if self.input_chunk_length >= roll_size:
                input_past[:, -roll_size:, :n_targets] = out[:, :roll_size, :]
            else:
                input_past[:, :, :n_targets] = out[:, -self.input_chunk_length :, :]

            # set left and right boundaries for extracting future elements
            if self.input_chunk_length >= roll_size:
                left_past, right_past = prediction_length - roll_size, prediction_length
            else:
                left_past, right_past = (
                    prediction_length - self.input_chunk_length,
                    prediction_length,
                )

            # update past covariates to include next `roll_size` future past covariates elements
            if n_past_covs and self.input_chunk_length >= roll_size:
                input_past[:, -roll_size:, n_targets : n_targets + n_past_covs] = (
                    future_past_covariates[:, left_past:right_past, :]
                )
            elif n_past_covs:
                input_past[:, :, n_targets : n_targets + n_past_covs] = (
                    future_past_covariates[:, left_past:right_past, :]
                )

            # update historic future covariates to include next `roll_size` future covariates elements
            if n_future_covs and self.input_chunk_length >= roll_size:
                input_past[:, -roll_size:, n_targets + n_past_covs :] = (
                    future_covariates[:, left_past:right_past, :]
                )
            elif n_future_covs:
                input_past[:, :, n_targets + n_past_covs :] = future_covariates[
                    :, left_past:right_past, :
                ]

            # ==========> FUTURE INPUT <==========
            left_future, right_future = (
                right_past,
                right_past + self.output_chunk_length,
            )
            # update future covariates to include next `roll_size` future covariates elements
            if n_future_covs:
                input_future = future_covariates[:, left_future:right_future, :]

            # take only last part of the output sequence where needed
            out = self._produce_predict_output(
                x=(input_past, input_future, input_static, None)
            )[:, self.first_prediction_index :, :]

            batch_prediction.append(out)
            prediction_length += self.output_chunk_length

        # bring predictions into desired format and drop unnecessary values
        batch_prediction = torch.cat(batch_prediction, dim=1)
        batch_prediction = batch_prediction[:, :n, :]
        return batch_prediction

    @staticmethod
    def _sample_tiling(
        input_data_tuple: tuple[torch.Tensor | None, ...], batch_sample_size
    ) -> tuple[torch.Tensor | None, ...]:
        tiled_input_data = []
        for tensor in input_data_tuple:
            if tensor is not None:
                tiled_input_data.append(tensor.tile((batch_sample_size, 1, 1)))
            else:
                tiled_input_data.append(None)
        return tuple(tiled_input_data)

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

    def _produce_predict_output(self, x: PLModuleInput) -> torch.Tensor:
        if self.likelihood:
            output = self(x)
            if self.predict_likelihood_parameters:
                return self.likelihood.predict_likelihood_parameters(output)
            else:
                return self.likelihood.sample(output)
        else:
            return self(x).squeeze(dim=-1)

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
