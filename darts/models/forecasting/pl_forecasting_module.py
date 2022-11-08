"""
This file contains abstract classes for deterministic and probabilistic PyTorch Lightning Modules
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
from joblib import Parallel, delayed

from darts.logging import get_logger, raise_if, raise_log
from darts.timeseries import TimeSeries
from darts.utils.likelihood_models import Likelihood
from darts.utils.timeseries_generation import _build_forecast_series
from darts.utils.torch import MonteCarloDropout

logger = get_logger(__name__)

# Check whether we are running pytorch-lightning >= 1.6.0 or not:
tokens = pl.__version__.split(".")
pl_160_or_above = int(tokens[0]) >= 1 and int(tokens[1]) >= 6


class PLForecastingModule(pl.LightningModule, ABC):
    @abstractmethod
    def __init__(
        self,
        input_chunk_length: int,
        output_chunk_length: int,
        loss_fn: nn.modules.loss._Loss = nn.MSELoss(),
        torch_metrics: Optional[
            Union[torchmetrics.Metric, torchmetrics.MetricCollection]
        ] = None,
        likelihood: Optional[Likelihood] = None,
        optimizer_cls: torch.optim.Optimizer = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict] = None,
        lr_scheduler_cls: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        lr_scheduler_kwargs: Optional[Dict] = None,
    ) -> None:
        """
        PyTorch Lightning-based Forecasting Module.

        This class is meant to be inherited to create a new PyTorch Lightning-based forecasting module.
        When subclassing this class, please make sure to add the following methods with the given signatures:
            - :func:`PLTorchForecastingModel.__init__()`
            - :func:`PLTorchForecastingModel.forward()`
            - :func:`PLTorchForecastingModel._produce_train_output()`
            - :func:`PLTorchForecastingModel._get_batch_prediction()`

        In subclass `MyModel`'s :func:`__init__` function call ``super(MyModel, self).__init__(**kwargs)`` where
        ``kwargs`` are the parameters of :class:`PLTorchForecastingModel`.

        Parameters
        ----------
        input_chunk_length
            Number of input past time steps per chunk.
        output_chunk_length
            Number of output time steps per chunk.
        loss_fn
            PyTorch loss function used for training.
            This parameter will be ignored for probabilistic models if the ``likelihood`` parameter is specified.
            Default: ``torch.nn.MSELoss()``.
        torch_metrics
            A torch metric or a ``MetricCollection`` used for evaluation. A full list of available metrics can be found
            at https://torchmetrics.readthedocs.io/en/latest/. Default: ``None``.
        likelihood
            One of Darts' :meth:`Likelihood <darts.utils.likelihood_models.Likelihood>` models to be used for
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
        """
        super().__init__()

        # save hyper parameters for saving/loading
        # do not save type nn.Module params
        self.save_hyperparameters(ignore=["loss_fn", "torch_metrics"])

        raise_if(
            input_chunk_length is None or output_chunk_length is None,
            "Both `input_chunk_length` and `output_chunk_length` must be passed to `PLForecastingModule`",
            logger,
        )

        self.input_chunk_length = input_chunk_length
        self.output_chunk_length = output_chunk_length

        # define the loss function
        self.criterion = loss_fn
        # by default models are deterministic (i.e. not probabilistic)
        self.likelihood = likelihood

        # persist optimiser and LR scheduler parameters
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = dict() if optimizer_kwargs is None else optimizer_kwargs
        self.lr_scheduler_cls = lr_scheduler_cls
        self.lr_scheduler_kwargs = (
            dict() if lr_scheduler_kwargs is None else lr_scheduler_kwargs
        )

        if torch_metrics is None:
            torch_metrics = torchmetrics.MetricCollection([])
        elif isinstance(torch_metrics, torchmetrics.Metric):
            torch_metrics = torchmetrics.MetricCollection([torch_metrics])
        elif isinstance(torch_metrics, torchmetrics.MetricCollection):
            pass
        else:
            raise_log(
                AttributeError(
                    "`torch_metrics` only accepts type torchmetrics.Metric or torchmetrics.MetricCollection"
                ),
                logger,
            )
        self.train_metrics = torch_metrics.clone(prefix="train_")
        self.val_metrics = torch_metrics.clone(prefix="val_")

        # initialize prediction parameters
        self.pred_n: Optional[int] = None
        self.pred_num_samples: Optional[int] = None
        self.pred_roll_size: Optional[int] = None
        self.pred_batch_size: Optional[int] = None
        self.pred_n_jobs: Optional[int] = None

    @property
    def first_prediction_index(self) -> int:
        """
        Returns the index of the first predicted within the output of self.model.
        """
        return 0

    @abstractmethod
    def forward(self, *args, **kwargs) -> Any:
        super().forward(*args, **kwargs)

    def training_step(self, train_batch, batch_idx) -> torch.Tensor:
        """performs the training step"""
        output = self._produce_train_output(train_batch[:-1])
        target = train_batch[
            -1
        ]  # By convention target is always the last element returned by datasets
        loss = self._compute_loss(output, target)
        self.log("train_loss", loss, batch_size=train_batch[0].shape[0], prog_bar=True)
        self._calculate_metrics(output, target, self.train_metrics)
        return loss

    def validation_step(self, val_batch, batch_idx) -> torch.Tensor:
        """performs the validation step"""
        output = self._produce_train_output(val_batch[:-1])
        target = val_batch[-1]
        loss = self._compute_loss(output, target)
        self.log("val_loss", loss, batch_size=val_batch[0].shape[0], prog_bar=True)
        self._calculate_metrics(output, target, self.val_metrics)
        return loss

    def predict_step(
        self, batch: Tuple, batch_idx: int, dataloader_idx: Optional[int] = None
    ) -> Sequence[TimeSeries]:
        """performs the prediction step

        batch
            output of Darts' :class:`InferenceDataset` - tuple of ``(past_target, past_covariates,
            historic_future_covariates, future_covariates, future_past_covariates, input_timeseries)``
        batch_idx
            the batch index of the current batch
        dataloader_idx
            the dataloader index
        """
        input_data_tuple, batch_input_series = batch[:-1], batch[-1]

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
        batch_predictions = batch_predictions.cpu().detach().numpy()

        ts_forecasts = Parallel(n_jobs=self.pred_n_jobs)(
            delayed(_build_forecast_series)(
                [batch_prediction[batch_idx] for batch_prediction in batch_predictions],
                input_series,
            )
            for batch_idx, input_series in enumerate(batch_input_series)
        )
        return ts_forecasts

    def set_predict_parameters(
        self, n: int, num_samples: int, roll_size: int, batch_size: int, n_jobs: int
    ) -> None:
        """to be set from TorchForecastingModel before calling trainer.predict() and reset at self.on_predict_end()"""
        self.pred_n = n
        self.pred_num_samples = num_samples
        self.pred_roll_size = roll_size
        self.pred_batch_size = batch_size
        self.pred_n_jobs = n_jobs

    def _compute_loss(self, output, target):
        # output is of shape (batch_size, n_timesteps, n_components, n_params)
        if self.likelihood:
            return self.likelihood.compute_loss(output, target)
        else:
            # If there's no likelihood, nr_params=1, and we need to squeeze out the
            # last dimension of model output, for properly computing the loss.
            return self.criterion(output.squeeze(dim=-1), target)

    def _calculate_metrics(self, output, target, metrics):
        if not len(metrics):
            return

        if self.likelihood:
            _metric = metrics(self.likelihood.sample(output), target)
        else:
            # If there's no likelihood, nr_params=1, and we need to squeeze out the
            # last dimension of model output, for properly computing the metric.
            _metric = metrics(output.squeeze(dim=-1), target)

        self.log_dict(
            _metric,
            on_epoch=True,
            on_step=False,
            logger=True,
            prog_bar=True,
        )

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
                        "\nclass: {}"
                        "\narguments (kwargs): {}"
                        "\nerror:\n{}".format(cls, kws, e)
                    ),
                    logger,
                )

        # Create the optimizer and (optionally) the learning rate scheduler
        # we have to create copies because we cannot save model.parameters into object state (not serializable)
        optimizer_kws = {k: v for k, v in self.optimizer_kwargs.items()}
        optimizer_kws["params"] = self.parameters()

        optimizer = _create_from_cls_and_kwargs(self.optimizer_cls, optimizer_kws)

        if self.lr_scheduler_cls is not None:
            lr_sched_kws = {k: v for k, v in self.lr_scheduler_kwargs.items()}
            lr_sched_kws["optimizer"] = optimizer

            # ReduceLROnPlateau requires a metric to "monitor" which must be set separately, most others do not
            lr_monitor = lr_sched_kws.pop("monitor", None)

            lr_scheduler = _create_from_cls_and_kwargs(
                self.lr_scheduler_cls, lr_sched_kws
            )
            return [optimizer], {
                "scheduler": lr_scheduler,
                "monitor": lr_monitor if lr_monitor is not None else "val_loss",
            }
        else:
            return optimizer

    @abstractmethod
    def _produce_train_output(self, input_batch: Tuple) -> torch.Tensor:
        pass

    @abstractmethod
    def _get_batch_prediction(
        self, n: int, input_batch: Tuple, roll_size: int
    ) -> torch.Tensor:
        """
        In charge of applying the recurrent logic for non-recurrent models.
        Should be overwritten by recurrent models.
        """
        pass

    @staticmethod
    def _sample_tiling(input_data_tuple, batch_sample_size):
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
        for module in self._get_mc_dropout_modules():
            module.mc_dropout_enabled = active

    def _is_probabilistic(self) -> bool:
        return self.likelihood is not None or len(self._get_mc_dropout_modules()) > 0

    def _produce_predict_output(self, x: Tuple):
        if self.likelihood:
            output = self(x)
            return self.likelihood.sample(output)
        else:
            return self(x).squeeze(dim=-1)

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        # we must save the dtype for correct parameter precision at loading time
        checkpoint["model_dtype"] = self.dtype

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        # by default our models are initialized as float32. For other dtypes, we need to cast to the correct precision
        # before parameters are loaded by PyTorch-Lightning
        dtype = checkpoint["model_dtype"]
        if dtype == torch.float16:
            self.half()
        if dtype == torch.float32:
            self.float()
        elif dtype == torch.float64:
            self.double()
        else:
            raise_if(
                True,
                f"Trying to load dtype {dtype}. Loading for this type is not implemented yet. Please report this "
                f"issue on https://github.com/unit8co/darts",
                logger,
            )

    @property
    def epochs_trained(self):
        current_epoch = self.current_epoch

        # For PTL < 1.6.0 we have to adjust:
        if not pl_160_or_above and (self.current_epoch or self.global_step):
            current_epoch += 1

        return current_epoch


class PLPastCovariatesModule(PLForecastingModule, ABC):
    def _produce_train_output(self, input_batch: Tuple):
        """
        Feeds PastCovariatesTorchModel with input and output chunks of a PastCovariatesSequentialDataset for
        training.

        Parameters:
        ----------
        input_batch
            ``(past_target, past_covariates, static_covariates)``
        """
        past_target, past_covariates, static_covariates = input_batch
        # Currently all our PastCovariates models require past target and covariates concatenated
        inpt = (
            torch.cat([past_target, past_covariates], dim=2)
            if past_covariates is not None
            else past_target,
            static_covariates,
        )
        return self(inpt)

    def _get_batch_prediction(
        self, n: int, input_batch: Tuple, roll_size: int
    ) -> torch.Tensor:
        """
        Feeds PastCovariatesTorchModel with input and output chunks of a PastCovariatesSequentialDataset to forecast
        the next ``n`` target values per target variable.

        Parameters:
        ----------
        n
            prediction length
        input_batch
            ``(past_target, past_covariates, future_past_covariates, static_covariates)``
        roll_size
            roll input arrays after every sequence by ``roll_size``. Initially, ``roll_size`` is equivalent to
            ``self.output_chunk_length``
        """
        dim_component = 2
        (
            past_target,
            past_covariates,
            future_past_covariates,
            static_covariates,
        ) = input_batch

        n_targets = past_target.shape[dim_component]
        n_past_covs = (
            past_covariates.shape[dim_component] if past_covariates is not None else 0
        )

        input_past = torch.cat(
            [ds for ds in [past_target, past_covariates] if ds is not None],
            dim=dim_component,
        )

        out = self._produce_predict_output((input_past, static_covariates))[
            :, self.first_prediction_index :, :
        ]

        batch_prediction = [out[:, :roll_size, :]]
        prediction_length = roll_size

        while prediction_length < n:
            # we want the last prediction to end exactly at `n` into the future.
            # this means we may have to truncate the previous prediction and step
            # back the roll size for the last chunk
            if prediction_length + self.output_chunk_length > n:
                spillover_prediction_length = (
                    prediction_length + self.output_chunk_length - n
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
                input_past[
                    :, -roll_size:, n_targets : n_targets + n_past_covs
                ] = future_past_covariates[:, left_past:right_past, :]
            elif n_past_covs:
                input_past[
                    :, :, n_targets : n_targets + n_past_covs
                ] = future_past_covariates[:, left_past:right_past, :]

            # take only last part of the output sequence where needed
            out = self._produce_predict_output((input_past, static_covariates))[
                :, self.first_prediction_index :, :
            ]
            batch_prediction.append(out)
            prediction_length += self.output_chunk_length

        # bring predictions into desired format and drop unnecessary values
        batch_prediction = torch.cat(batch_prediction, dim=1)
        batch_prediction = batch_prediction[:, :n, :]
        return batch_prediction


class PLFutureCovariatesModule(PLForecastingModule, ABC):
    def _get_batch_prediction(
        self, n: int, input_batch: Tuple, roll_size: int
    ) -> torch.Tensor:
        raise NotImplementedError("TBD: Darts doesn't contain such a model yet.")


class PLDualCovariatesModule(PLForecastingModule, ABC):
    def _get_batch_prediction(
        self, n: int, input_batch: Tuple, roll_size: int
    ) -> torch.Tensor:
        raise NotImplementedError(
            "TBD: The only DualCovariatesModel is an RNN with a specific implementation."
        )


class PLMixedCovariatesModule(PLForecastingModule, ABC):
    def _produce_train_output(
        self, input_batch: Tuple
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Feeds MixedCovariatesTorchModel with input and output chunks of a MixedCovariatesSequentialDataset for
        training.

        Parameters:
        ----------
        input_batch
            ``(past_target, past_covariates, historic_future_covariates, future_covariates, static_covariates)``.
        """
        return self(self._process_input_batch(input_batch))

    def _process_input_batch(
        self, input_batch
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Converts output of MixedCovariatesDataset (training dataset) into an input/past- and
        output/future chunk.

        Parameters
        ----------
        input_batch
            ``(past_target, past_covariates, historic_future_covariates, future_covariates, static_covariates)``.

        Returns
        -------
        tuple
            ``(x_past, x_future, x_static)`` the input/past and output/future chunks.
        """
        (
            past_target,
            past_covariates,
            historic_future_covariates,
            future_covariates,
            static_covariates,
        ) = input_batch
        dim_variable = 2

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
            dim=dim_variable,
        )
        return x_past, future_covariates, static_covariates

    def _get_batch_prediction(
        self, n: int, input_batch: Tuple, roll_size: int
    ) -> torch.Tensor:
        """
        Feeds MixedCovariatesModel with input and output chunks of a MixedCovariatesSequentialDataset to forecast
        the next ``n`` target values per target variable.

        Parameters
        ----------
        n
            prediction length
        input_batch
            (past_target, past_covariates, historic_future_covariates, future_covariates, future_past_covariates)
        roll_size
            roll input arrays after every sequence by ``roll_size``. Initially, ``roll_size`` is equivalent to
            ``self.output_chunk_length``
        """

        dim_component = 2
        (
            past_target,
            past_covariates,
            historic_future_covariates,
            future_covariates,
            future_past_covariates,
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

        input_past, input_future, input_static = self._process_input_batch(
            (
                past_target,
                past_covariates,
                historic_future_covariates,
                future_covariates[:, :roll_size, :]
                if future_covariates is not None
                else None,
                static_covariates,
            )
        )

        out = self._produce_predict_output(x=(input_past, input_future, input_static))[
            :, self.first_prediction_index :, :
        ]

        batch_prediction = [out[:, :roll_size, :]]
        prediction_length = roll_size

        while prediction_length < n:
            # we want the last prediction to end exactly at `n` into the future.
            # this means we may have to truncate the previous prediction and step
            # back the roll size for the last chunk
            if prediction_length + self.output_chunk_length > n:
                spillover_prediction_length = (
                    prediction_length + self.output_chunk_length - n
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
                input_past[
                    :, -roll_size:, n_targets : n_targets + n_past_covs
                ] = future_past_covariates[:, left_past:right_past, :]
            elif n_past_covs:
                input_past[
                    :, :, n_targets : n_targets + n_past_covs
                ] = future_past_covariates[:, left_past:right_past, :]

            # update historic future covariates to include next `roll_size` future covariates elements
            if n_future_covs and self.input_chunk_length >= roll_size:
                input_past[
                    :, -roll_size:, n_targets + n_past_covs :
                ] = future_covariates[:, left_past:right_past, :]
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
                x=(input_past, input_future, input_static)
            )[:, self.first_prediction_index :, :]

            batch_prediction.append(out)
            prediction_length += self.output_chunk_length

        # bring predictions into desired format and drop unnecessary values
        batch_prediction = torch.cat(batch_prediction, dim=1)
        batch_prediction = batch_prediction[:, :n, :]
        return batch_prediction


class PLSplitCovariatesModule(PLForecastingModule, ABC):
    def _get_batch_prediction(
        self, n: int, input_batch: Tuple, roll_size: int
    ) -> torch.Tensor:
        raise NotImplementedError("TBD: Darts doesn't contain such a model yet.")
