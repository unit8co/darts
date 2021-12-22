"""
This file contains abstract classes for deterministic and probabilistic pytorch-lightning modules
"""

import numpy as np
import pandas as pd

from joblib import Parallel, delayed
from typing import Any, Optional, Dict, Tuple, Union, Sequence
from abc import ABC, abstractmethod
import torch
from torch import Tensor
import torch.nn as nn

from darts.timeseries import TimeSeries
from darts.utils.timeseries_generation import _generate_index

from darts.utils.likelihood_models import Likelihood
from darts.logging import get_logger, raise_log, raise_if

import pytorch_lightning as pl


logger = get_logger(__name__)


# TODO: better names
class PLTorchForecastingModel(pl.LightningModule, ABC):

    @abstractmethod
    def __init__(self,
                 loss_fn: nn.modules.loss._Loss = nn.MSELoss(),
                 optimizer_cls: torch.optim.Optimizer = torch.optim.Adam,
                 optimizer_kwargs: Optional[Dict] = None,
                 lr_scheduler_cls: torch.optim.lr_scheduler._LRScheduler = None,
                 lr_scheduler_kwargs: Optional[Dict] = None) -> None:

        super(PLTorchForecastingModel, self).__init__()

        # Define the loss function
        self.criterion = loss_fn

        # Persist optimiser and LR scheduler parameters
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = dict() if optimizer_kwargs is None else optimizer_kwargs
        self.lr_scheduler_cls = lr_scheduler_cls
        self.lr_scheduler_kwargs = dict() if lr_scheduler_kwargs is None else lr_scheduler_kwargs

        # by default models are deterministic (i.e. not probabilistic)
        self.likelihood = None

        # TODO: make better
        # initialize prediction settings
        self.pred_n: Optional[int] = None
        self.pred_num_samples: Optional[int] = None
        self.pred_n_jobs: Optional[int] = None
        self.pred_roll_size: Optional[int] = None
        self.pred_batch_size: Optional[int] = None

    @property
    def first_prediction_index(self) -> int:
        """
        Returns the index of the first predicted within the output of self.model.
        """
        return 0

    @abstractmethod
    def forward(self, *args, **kwargs) -> Any:
        super(PLTorchForecastingModel, self).forward(*args, **kwargs)

    def training_step(self, train_batch, batch_idx) -> Any:
        output = self._produce_train_output(train_batch[:-1])
        target = train_batch[-1]  # By convention target is always the last element returned by datasets
        loss = self._compute_loss(output, target)
        self.log('train_loss', loss, batch_size=train_batch[0].shape[0])
        return loss

    def validation_step(self, val_batch, batch_idx) -> Any:
        output = self._produce_train_output(val_batch[:-1])
        target = val_batch[-1]
        loss = self._compute_loss(output, target)
        self.log('val_loss', loss, batch_size=val_batch[0].shape[0])
        return loss

    def predict_step(self,
                     batch: Any,
                     batch_idx: int,
                     dataloader_idx: Optional[int] = None) -> Tuple[torch.Tensor, Sequence[TimeSeries]]:

        input_data_tuple, batch_input_series = batch[:-1], batch[-1]

        # number of individual series to be predicted in current batch
        num_series = input_data_tuple[0].shape[0]

        # number of of times the input tensor should be tiled to produce predictions for multiple samples
        # this variable is larger than 1 only if the batch_size is at least twice as large as the number
        # of individual time series being predicted in current batch (`num_series`)
        batch_sample_size = min(max(self.pred_batch_size // num_series, 1), self.pred_num_samples)

        # counts number of produced prediction samples for every series to be predicted in current batch
        sample_count = 0

        # repeat prediction procedure for every needed sample
        batch_predictions = []
        while sample_count < self.pred_num_samples:

            # make sure we don't produce too many samples
            if sample_count + batch_sample_size > self.pred_num_samples:
                batch_sample_size = self.pred_num_samples - sample_count

            # stack multiple copies of the tensors to produce probabilistic forecasts
            input_data_tuple_samples = self._sample_tiling(input_data_tuple, batch_sample_size)

            # get predictions for 1 whole batch (can include predictions of multiple series
            # and for multiple samples if a probabilistic forecast is produced)
            batch_prediction = self._get_batch_prediction(self.pred_n, input_data_tuple_samples, self.pred_roll_size)

            # reshape from 3d tensor (num_series x batch_sample_size, ...)
            # into 4d tensor (batch_sample_size, num_series, ...), where dim 0 represents the samples
            out_shape = batch_prediction.shape
            batch_prediction = batch_prediction.reshape((batch_sample_size, num_series,) + out_shape[1:])

            # save all predictions and update the `sample_count` variable
            batch_predictions.append(batch_prediction)
            sample_count += batch_sample_size

        # concatenate the batch of samples, to form self.pred_num_samples samples
        batch_predictions = torch.cat(batch_predictions, dim=0)
        batch_predictions = batch_predictions.cpu().detach().numpy()
        return batch_predictions, batch_input_series

    def on_predict_end(self) -> None:
        self.pred_n = None
        self.pred_num_samples = None
        self.pred_n_jobs = None
        self.pred_roll_size = None
        self.pred_batch_size = None

    def _compute_loss(self, output, target):
        return self.criterion(output, target)

    def configure_optimizers(self):
        """sets up optimizers"""

        # TODO: i think we can move this to to pl.Trainer(). and could probably be simplified

        # A utility function to create optimizer and lr scheduler from desired classes
        def _create_from_cls_and_kwargs(cls, kws):
            try:
                return cls(**kws)
            except (TypeError, ValueError) as e:
                raise_log(ValueError('Error when building the optimizer or learning rate scheduler;'
                                     'please check the provided class and arguments'
                                     '\nclass: {}'
                                     '\narguments (kwargs): {}'
                                     '\nerror:\n{}'.format(cls, kws, e)),
                          logger)

        # Create the optimizer and (optionally) the learning rate scheduler
        # we have to create copies because we cannot save model.parameters into object state (not serializable)
        optimizer_kws = {k: v for k, v in self.optimizer_kwargs.items()}
        optimizer_kws['params'] = self.parameters()

        optimizer = _create_from_cls_and_kwargs(self.optimizer_cls, optimizer_kws)

        if self.lr_scheduler_cls is not None:
            lr_sched_kws = {k: v for k, v in self.lr_scheduler_kwargs.items()}
            lr_sched_kws['optimizer'] = optimizer
            lr_scheduler = _create_from_cls_and_kwargs(self.lr_scheduler_cls, lr_sched_kws)
            return [optimizer], [lr_scheduler]
        else:
            return optimizer

    @abstractmethod
    def _produce_train_output(self, input_batch: Tuple) -> Tensor:
        pass

    @abstractmethod
    def _get_batch_prediction(self, n: int, input_batch: Tuple, roll_size: int) -> Tensor:
        """
        In charge of apply the recurrent logic for non-recurrent models.
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


class PLPastCovariatesTorchModel(PLTorchForecastingModel, ABC):
    def _produce_train_output(self, input_batch: Tuple):
        past_target, past_covariate = input_batch
        # Currently all our PastCovariates models require past target and covariates concatenated
        inpt = torch.cat([past_target, past_covariate], dim=2) if past_covariate is not None else past_target
        return self.model(inpt)

    def _get_batch_prediction(self, n: int, input_batch: Tuple, roll_size: int) -> torch.Tensor:
        """
        Feeds PastCovariatesTorchModel with input and output chunks of a PastCovariatesSequentialDataset to farecast
        the next `n` target values per target variable.

        Parameters:
        ----------
        n
            prediction length
        input_batch
            (past_target, past_covariates, future_past_covariates)
        roll_size
            roll input arrays after every sequence by `roll_size`. Initially, `roll_size` is equivalent to
            `self.output_chunk_length`
        """
        dim_component = 2
        past_target, past_covariates, future_past_covariates = input_batch

        n_targets = past_target.shape[dim_component]
        n_past_covs = past_covariates.shape[dim_component] if not past_covariates is None else 0

        input_past = torch.cat(
            [ds for ds in [past_target, past_covariates] if ds is not None],
            dim=dim_component
        )

        out = self._produce_predict_output(input_past)[:, self.first_prediction_index:, :]

        batch_prediction = [out[:, :roll_size, :]]
        prediction_length = roll_size

        while prediction_length < n:
            # we want the last prediction to end exactly at `n` into the future.
            # this means we may have to truncate the previous prediction and step
            # back the roll size for the last chunk
            if prediction_length + self.output_chunk_length > n:
                spillover_prediction_length = prediction_length + self.output_chunk_length - n
                roll_size -= spillover_prediction_length
                prediction_length -= spillover_prediction_length
                batch_prediction[-1] = batch_prediction[-1][:, :roll_size, :]

            # ==========> PAST INPUT <==========
            # roll over input series to contain latest target and covariate
            input_past = torch.roll(input_past, -roll_size, 1)

            # update target input to include next `roll_size` predictions
            if self.input_chunk_length >= roll_size:
                input_past[:, -roll_size:, :n_targets] = out[:, :roll_size, :]
            else:
                input_past[:, :, :n_targets] = out[:, -self.input_chunk_length:, :]

            # set left and right boundaries for extracting future elements
            if self.input_chunk_length >= roll_size:
                left_past, right_past = prediction_length - roll_size, prediction_length
            else:
                left_past, right_past = prediction_length - self.input_chunk_length, prediction_length

            # update past covariates to include next `roll_size` future past covariates elements
            if n_past_covs and self.input_chunk_length >= roll_size:
                input_past[:, -roll_size:, n_targets:n_targets + n_past_covs] = (
                    future_past_covariates[:, left_past:right_past, :]
                )
            elif n_past_covs:
                input_past[:, :, n_targets:n_targets + n_past_covs] = (
                    future_past_covariates[:, left_past:right_past, :]
                )

            # take only last part of the output sequence where needed
            out = self._produce_predict_output(input_past)[:, self.first_prediction_index:, :]
            batch_prediction.append(out)
            prediction_length += self.output_chunk_length

        # bring predictions into desired format and drop unnecessary values
        batch_prediction = torch.cat(batch_prediction, dim=1)
        batch_prediction = batch_prediction[:, :n, :]
        return batch_prediction

    def _produce_predict_output(self, x):
        return self.model(x)


class PLFutureCovariatesTorchModel(PLTorchForecastingModel, ABC):
    def _get_batch_prediction(self, n: int, input_batch: Tuple, roll_size: int) -> Tensor:
        raise NotImplementedError("TBD: Darts doesn't contain such a model yet.")


class PLDualCovariatesTorchModel(PLTorchForecastingModel, ABC):
    def _get_batch_prediction(self, n: int, input_batch: Tuple, roll_size: int) -> Tensor:
        raise NotImplementedError("TBD: The only DualCovariatesModel is an RNN with a specific implementation.")


class PLMixedCovariatesTorchModel(PLTorchForecastingModel, ABC):
    def _get_batch_prediction(self, n: int, input_batch: Tuple, roll_size: int) -> Tensor:
        raise NotImplementedError("TBD: Darts doesn't contain such a model yet.")


class PLSplitCovariatesTorchModel(PLTorchForecastingModel, ABC):
    def _get_batch_prediction(self, n: int, input_batch: Tuple, roll_size: int) -> Tensor:
        raise NotImplementedError("TBD: Darts doesn't contain such a model yet.")


# TODO: I think we could actually integrate probabilistic support already in the parent class and remove it from here?
class PLTorchParametricProbabilisticForecastingModel(PLTorchForecastingModel, ABC):
    def __init__(self, likelihood: Optional[Likelihood] = None, **kwargs):
        """ Pytorch Parametric Probabilistic Forecasting Model.

        This is a base class for pytroch parametric probabilistic models. "Parametric"
        means that these models are based on some predefined parametric distribution, say Gaussian.
        Make sure that subclasses contain the *likelihood* parameter in __init__ method
        and it is passed to the superclass via calling super().__init__. If the likelihood is not
        provided, the model is considered as deterministic.

        All TorchParametricProbabilisticForecastingModel's must produce outputs of shape
        (batch_size, n_timesteps, n_components, n_params). I.e., there's an extra dimension
        to store the distribution's parameters.

        Parameters
        ----------
        likelihood
            The likelihood model to be used for probabilistic forecasts.
        """
        super().__init__(**kwargs)
        self.likelihood = likelihood

    def _is_probabilistic(self):
        return self.likelihood is not None

    def _compute_loss(self, output, target):
        # output is of shape (batch_size, n_timesteps, n_components, n_params)
        if self.likelihood:
            return self.likelihood.compute_loss(output, target)
        else:
            # If there's no likelihood, nr_params=1 and we need to squeeze out the
            # last dimension of model output, for properly computing the loss.
            return super()._compute_loss(output.squeeze(dim=-1), target)

    @abstractmethod
    def _produce_predict_output(self, x):
        """
        This method has to be implemented by all children.
        """
        pass