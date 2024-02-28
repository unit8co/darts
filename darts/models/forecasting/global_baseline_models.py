"""
Global Baseline Models (Naive)
------------------------------

A collection of simple benchmark models working with univariate, mutlivariate, single, and multiple series.
"""

from abc import ABC, abstractmethod
from typing import Optional, Sequence, Tuple, Union

import pytorch_lightning as pl
import torch

from darts import TimeSeries
from darts.logging import get_logger
from darts.models.forecasting.forecasting_model import GlobalForecastingModel
from darts.models.forecasting.pl_forecasting_module import (
    PLMixedCovariatesModule,
    io_processor,
)
from darts.models.forecasting.torch_forecasting_model import (
    MixedCovariatesTorchModel,
    TorchForecastingModel,
)
from darts.utils.data.inference_dataset import InferenceDataset
from darts.utils.utils import seq2series

MixedCovariatesTrainTensorType = Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]


logger = get_logger(__name__)


def _extract_targets(batch: Tuple[torch.Tensor], n_target_components: int):
    """Extracts and returns the target components from an input batch

    Parameters
    ----------
    batch
        The input batch tuple for the forward method. Has elements `(x_past, x_future, x_static)`.
    n_target_components
        The number of target components to extract.
    """
    return batch[0][:, :, :n_target_components]


def _repeat_along_output_chunk(x: torch.Tensor, ocl: int) -> torch.Tensor:
    """Expands a tensor `x` of shape (batch size, n components) to a tensor of shape
    (batch size, `ocl`, n target components, 1 (n samples)), by repeating the values
    along the `output_chunk_length` axis.

    Parameters
    ----------
    x
        An input tensor of shape (batch size, n target components)
    ocl
        The output_chunk_length.
    """
    return x.view(-1, 1, x[0].shape[-1], 1).expand(-1, ocl, -1, -1)


class _GlobalNaiveModule(PLMixedCovariatesModule, ABC):
    def __init__(self, **kwargs):
        """Pytorch module for implementing naive models.

        Implement your own naive module by subclassing from `_GlobalNaiveModule`, and implement the
        logic for prediction in the private `_forward` method.
        """
        super().__init__(**kwargs)

        # will be set at inference time
        self.n_target_components = 0

    @io_processor
    def forward(
        self, x_in: Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]
    ) -> torch.Tensor:
        """Naive model forward pass.

        Parameters
        ----------
        x_in
            comes as tuple `(x_past, x_future, x_static)` where `x_past` is the input/past chunk and `x_future`
            is the output/future chunk. Input dimensions are `(batch_size, time_steps, components)`

        Returns
        -------
        torch.Tensor
            The output Tensor of shape `(batch_size, output_chunk_length, output_dim, nr_params)`
        """
        return self._forward(x_in)

    @abstractmethod
    def _forward(self, x_in) -> torch.Tensor:
        """Private method to implement the forward method in the subclasses."""
        pass


class _GlobalNaiveModel(MixedCovariatesTorchModel, ABC):
    def __init__(
        self,
        input_chunk_length: int,
        output_chunk_length: int,
        use_static_covariates: bool = True,
        **kwargs,
    ):
        """Base class for global naive models. The naive models inherit from `MixedCovariatesTorchModel` giving access
        to past, future, and static covariates in the model `forward()` method. This allows to create custom models
        naive models which can make use of the covariates. The built-in naive models will not use this information.

        To add a new naive model:
        - subclass from `_GlobalNaiveModel` with implementation of private method `_create_model` that creates an
            object of:
        - subclass from `_GlobalNaiveModule` with implemention of private method `_forward`

        Parameters
        ----------
        input_chunk_length
            The length of the input sequence fed to the model.
        output_chunk_length
            The length of the emitted forecast and output sequence fed to the model.
        use_static_covariates
            Whether the model should use static covariate information in case the input `series` passed to ``fit()``
            contain static covariates. If ``True``, and static covariates are available at fitting time, will enforce
            that all target `series` have the same static covariate dimensionality in ``fit()`` and ``predict()``.
        **kwargs
            Optional arguments to initialize the pytorch_lightning.Module, pytorch_lightning.Trainer, and
            Darts' :class:`TorchForecastingModel`.
            Since naive models are not trained, the following parameters will have no effect:
            `loss_fn`, `likelihood`, `optimizer_cls`, `optimizer_kwargs`, `lr_scheduler_cls`, `lr_scheduler_kwargs`,
            `n_epochs`, `save_checkpoints`, and some of the `pl_trainer_kwargs`.
        """
        super().__init__(**self._extract_torch_model_params(**self.model_params))

        # extract pytorch lightning module kwargs
        self.pl_module_params = self._extract_pl_module_params(**self.model_params)

        self._considers_static_covariates = use_static_covariates

        # naive models do not have to be trained
        self.model = self._create_model(tuple())
        self._module_name = self.model.__class__.__name__
        self._fit_called = True

    def fit(
        self,
        series: Union[TimeSeries, Sequence[TimeSeries]],
        past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        *args,
        **kwargs,
    ) -> TorchForecastingModel:
        """Fit/train the model on a (or potentially multiple) series.
        This method is only implemented for naive baseline models to provide a unified fit/predict API with other
        forecasting models.

        The models are not really trained on the input, but they store the training `series` in case only a single
        `TimeSeries` was passed. This allows to call `predict()` without having to pass the single `series`.

        All baseline models compute the forecasts for each series directly when calling `predict()`.

        Parameters
        ----------
        series
            A series or sequence of series serving as target (i.e. what the model will be trained to forecast)
        past_covariates
            Optionally, a series or sequence of series specifying past-observed covariates
        future_covariates
            Optionally, a series or sequence of series specifying future-known covariates
        **kwargs
            Optionally, some keyword arguments.

        Returns
        -------
        self
            Fitted model.
        """
        GlobalForecastingModel.fit(
            self,
            series=seq2series(series),
            past_covariates=seq2series(past_covariates),
            future_covariates=seq2series(future_covariates),
        )
        return self

    def predict_from_dataset(
        self,
        n: int,
        input_series_dataset: InferenceDataset,
        trainer: Optional[pl.Trainer] = None,
        batch_size: Optional[int] = None,
        verbose: Optional[bool] = None,
        n_jobs: int = 1,
        roll_size: Optional[int] = None,
        num_samples: int = 1,
        num_loader_workers: int = 0,
        mc_dropout: bool = False,
        predict_likelihood_parameters: bool = False,
    ) -> Sequence[TimeSeries]:
        # we retrieve the number of target components
        self.model.n_target_components = input_series_dataset[0][0].shape[1]
        return super().predict_from_dataset(
            n=n,
            input_series_dataset=input_series_dataset,
            trainer=trainer,
            batch_size=batch_size,
            verbose=verbose,
            n_jobs=n_jobs,
            roll_size=roll_size,
            num_samples=num_samples,
            num_loader_workers=num_loader_workers,
            mc_dropout=mc_dropout,
            predict_likelihood_parameters=predict_likelihood_parameters,
        )

    @abstractmethod
    def _create_model(
        self, train_sample: MixedCovariatesTrainTensorType
    ) -> _GlobalNaiveModule:
        pass

    def _verify_predict_sample(self, predict_sample: Tuple):
        # naive models do not have to be trained, predict sample does not
        # have to match the training sample
        pass

    def supports_likelihood_parameter_prediction(self) -> bool:
        return False

    def _is_probabilistic(self) -> bool:
        return False

    @property
    def supports_static_covariates(self) -> bool:
        return False

    @property
    def supports_multivariate(self) -> bool:
        return True


class _GlobalNaiveMeanModule(_GlobalNaiveModule):
    def _forward(self, x_in) -> torch.Tensor:
        y_target = _extract_targets(x_in, self.n_target_components)
        mean = torch.mean(y_target, dim=1)
        return _repeat_along_output_chunk(mean, self.output_chunk_length)


class GlobalNaiveMean(_GlobalNaiveModel):
    def _create_model(
        self, train_sample: MixedCovariatesTrainTensorType
    ) -> _GlobalNaiveModule:
        return _GlobalNaiveMeanModule(**self.pl_module_params)


class _GlobalNaiveSeasonalModule(_GlobalNaiveModule):
    def _forward(self, x_in) -> torch.Tensor:
        y_target = _extract_targets(x_in, self.n_target_components)
        season = y_target[:, 0, :]
        return _repeat_along_output_chunk(season, self.output_chunk_length)


class GlobalNaiveSeasonal(_GlobalNaiveModel):
    def _create_model(
        self, train_sample: MixedCovariatesTrainTensorType
    ) -> _GlobalNaiveModule:
        return _GlobalNaiveSeasonalModule(**self.pl_module_params)


class _GlobalNaiveDrift(_GlobalNaiveModule):
    def _forward(self, x_in) -> torch.Tensor:
        y_target = _extract_targets(x_in, self.n_target_components)
        slope = _repeat_along_output_chunk(
            (y_target[:, -1, :] - y_target[:, 0, :]) / self.input_chunk_length,
            self.output_chunk_length,
        )

        x = torch.arange(1, self.output_chunk_length + 1, device=self.device).view(
            1, self.output_chunk_length, 1, 1
        )

        y_0 = y_target[:, -1, :].view(-1, 1, y_target.shape[-1], 1)
        return slope * x + y_0


class GlobalNaiveDrift(_GlobalNaiveModel):
    def _create_model(
        self, train_sample: MixedCovariatesTrainTensorType
    ) -> _GlobalNaiveModule:
        return _GlobalNaiveDrift(**self.pl_module_params)
