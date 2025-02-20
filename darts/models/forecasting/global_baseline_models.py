"""
Global Baseline Models (Naive)
------------------------------

A collection of simple benchmark models working with univariate, multivariate, single, and multiple series.

- :class:`GlobalNaiveAggregate`
- :class:`GlobalNaiveDrift`
- :class:`GlobalNaiveSeasonal`
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Callable, Optional, Union

import torch

from darts import TimeSeries
from darts.logging import get_logger, raise_log
from darts.models.forecasting.pl_forecasting_module import (
    PLMixedCovariatesModule,
    io_processor,
)
from darts.models.forecasting.torch_forecasting_model import (
    MixedCovariatesTorchModel,
    TorchForecastingModel,
)
from darts.utils.data.sequential_dataset import MixedCovariatesSequentialDataset
from darts.utils.data.training_dataset import MixedCovariatesTrainingDataset

MixedCovariatesTrainTensorType = tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]


logger = get_logger(__name__)


def _extract_targets(batch: tuple[torch.Tensor], n_targets: int):
    """Extracts and returns the target components from an input batch

    Parameters
    ----------
    batch
        The input batch tuple for the forward method. Has elements `(x_past, x_future, x_static)`.
    n_targets
        The number of target components to extract.
    """
    return batch[0][:, :, :n_targets]


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
    def __init__(self, *args, **kwargs):
        """Pytorch module for implementing naive models.

        Implement your own naive module by subclassing from `_GlobalNaiveModule`, and implement the
        logic for prediction in the private `_forward` method.
        """
        super().__init__(*args, **kwargs)

    @io_processor
    def forward(
        self, x_in: tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]
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
        output_chunk_shift: int = 0,
        use_static_covariates: bool = True,
        **kwargs,
    ):
        """Base class for global naive models. The naive models inherit from `MixedCovariatesTorchModel` giving access
        to past, future, and static covariates in the model `forward()` method. This allows to create custom models
        naive models which can make use of the covariates. The built-in naive models will not use this information.

        The naive models do not have to be trained before generating predictions.

        To add a new naive model:
        - subclass from `_GlobalNaiveModel` with implementation of private method `_create_model` that creates an
            object of:
        - subclass from `_GlobalNaiveModule` with implementation of private method `_forward`

        .. note::
            - Model checkpointing with `save_checkpoints=True`, and checkpoint loading with `load_from_checkpoint()`
              and `load_weights_from_checkpoint()` are not supported for global naive models.

        Parameters
        ----------
        input_chunk_length
            The length of the input sequence fed to the model.
        output_chunk_length
            The length of the emitted forecast and output sequence fed to the model.
        output_chunk_shift
            Optionally, the number of steps to shift the start of the output chunk into the future (relative to the
            input chunk end). This will create a gap between the input and output. If the model supports
            `future_covariates`, the future values are extracted from the shifted output chunk. Predictions will start
            `output_chunk_shift` steps after the end of the target `series`. If `output_chunk_shift` is set, the model
            cannot generate autoregressive predictions (`n > output_chunk_length`).
        use_static_covariates
            Whether the model should use static covariate information in case the input `series` passed to ``fit()``
            contain static covariates. If ``True``, and static covariates are available at fitting time, will enforce
            that all target `series` have the same static covariate dimensionality in ``fit()`` and ``predict()``.
        **kwargs
            Optional arguments to initialize the pytorch_lightning.Module, pytorch_lightning.Trainer, and
            Darts' :class:`TorchForecastingModel`.
            Since naive models are not trained, the following parameters will have no effect:
            `loss_fn`, `likelihood`, `optimizer_cls`, `optimizer_kwargs`, `lr_scheduler_cls`, `lr_scheduler_kwargs`,
            `n_epochs`, `save_checkpoints`, and some of `pl_trainer_kwargs`.
        """
        super().__init__(**self._extract_torch_model_params(**self.model_params))

        # extract pytorch lightning module kwargs
        self.pl_module_params = self._extract_pl_module_params(**self.model_params)

        self._considers_static_covariates = use_static_covariates

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

        The model is not really trained on the input, but `fit()` is used to setup the model based on the input series.
        Also, it stores the training `series` in case only a single `TimeSeries` was passed. This allows to call
        `predict()` without having to pass the single `series`.

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
        return super().fit(series, past_covariates, future_covariates, *args, **kwargs)

    @staticmethod
    def load_from_checkpoint(
        model_name: str,
        work_dir: str = None,
        file_name: str = None,
        best: bool = True,
        **kwargs,
    ) -> "TorchForecastingModel":
        raise_log(
            NotImplementedError(
                "GlobalNaiveModels do not support loading from checkpoint since they are never trained."
            ),
            logger=logger,
        )

    def load_weights_from_checkpoint(
        self,
        model_name: str = None,
        work_dir: str = None,
        file_name: str = None,
        best: bool = True,
        strict: bool = True,
        load_encoders: bool = True,
        skip_checks: bool = False,
        **kwargs,
    ):
        raise_log(
            NotImplementedError(
                "GlobalNaiveModels do not support weights loading since they do not have any weights/parameters."
            ),
            logger=logger,
        )

    @abstractmethod
    def _create_model(
        self, train_sample: MixedCovariatesTrainTensorType
    ) -> _GlobalNaiveModule:
        pass

    def _verify_predict_sample(self, predict_sample: tuple):
        # naive models do not have to be trained, predict sample does not
        # have to match the training sample
        pass

    @property
    def supports_likelihood_parameter_prediction(self) -> bool:
        return False

    @property
    def supports_probabilistic_prediction(self) -> bool:
        return False

    @property
    def supports_static_covariates(self) -> bool:
        return True

    @property
    def supports_multivariate(self) -> bool:
        return True

    @property
    def _requires_training(self) -> bool:
        # naive models do not have to be trained.
        return False

    def _build_train_dataset(
        self,
        target: Sequence[TimeSeries],
        past_covariates: Optional[Sequence[TimeSeries]],
        future_covariates: Optional[Sequence[TimeSeries]],
        sample_weight: Optional[Sequence[TimeSeries]],
        max_samples_per_ts: Optional[int],
    ) -> MixedCovariatesTrainingDataset:
        return MixedCovariatesSequentialDataset(
            target_series=target,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            input_chunk_length=self.input_chunk_length,
            output_chunk_length=0,
            output_chunk_shift=self.output_chunk_shift,
            max_samples_per_ts=max_samples_per_ts,
            use_static_covariates=self.uses_static_covariates,
            sample_weight=sample_weight,
        )


class _NoCovariatesMixin:
    @property
    def supports_static_covariates(self) -> bool:
        return False

    @property
    def supports_future_covariates(self) -> bool:
        return False

    @property
    def supports_past_covariates(self) -> bool:
        return False


class _GlobalNaiveAggregateModule(_GlobalNaiveModule):
    def __init__(
        self, agg_fn: Callable[[torch.Tensor, int], torch.Tensor], *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.agg_fn = agg_fn

    def _forward(self, x_in) -> torch.Tensor:
        y_target = _extract_targets(x_in, self.n_targets)
        aggregate = self.agg_fn(y_target, dim=1)
        return _repeat_along_output_chunk(aggregate, self.output_chunk_length)


class GlobalNaiveAggregate(_NoCovariatesMixin, _GlobalNaiveModel):
    def __init__(
        self,
        input_chunk_length: int,
        output_chunk_length: int,
        output_chunk_shift: int = 0,
        agg_fn: Union[str, Callable[[torch.Tensor, int], torch.Tensor]] = "mean",
        **kwargs,
    ):
        """Global Naive Aggregate Model.

        The model generates forecasts for each `series` as described below:

        - take an aggregate (computed with `agg_fn`, default: mean) from each target component over the last
          `input_chunk_length` points
        - the forecast is the component aggregate repeated `output_chunk_length` times

        Depending on the horizon `n` used when calling `model.predict()`, the forecasts are either:

        - a constant aggregate value (default: mean) if `n <= output_chunk_length`, or
        - a moving aggregate if `n > output_chunk_length`, as a result of the autoregressive prediction.

        This model is equivalent to:

        - :class:`~darts.models.forecasting.baselines.NaiveMean`, when `input_chunk_length` is equal to the length of
          the input target `series`, and `agg_fn='mean'`.
        - :class:`~darts.models.forecasting.baselines.NaiveMovingAverage`, with identical `input_chunk_length`
          and `output_chunk_length=1`, and `agg_fn='mean'`.

        .. note::
            - Model checkpointing with `save_checkpoints=True`, and checkpoint loading with `load_from_checkpoint()`
              and `load_weights_from_checkpoint()` are not supported for global naive models.

        Parameters
        ----------
        input_chunk_length
            The length of the input sequence fed to the model.
        output_chunk_length
            The length of the emitted forecast and output sequence fed to the model.
        output_chunk_shift
            Optionally, the number of steps to shift the start of the output chunk into the future (relative to the
            input chunk end). This will create a gap between the input and output. If the model supports
            `future_covariates`, the future values are extracted from the shifted output chunk. Predictions will start
            `output_chunk_shift` steps after the end of the target `series`. If `output_chunk_shift` is set, the model
            cannot generate autoregressive predictions (`n > output_chunk_length`).
        agg_fn
            The aggregation function to use. If a string, must be the name of `torch` function that can be imported
            directly from `torch` (e.g. `"mean"` for `torch.mean`, `"sum"` for `torch.sum`).
            The function must have the signature below. If a `Callable`, it must also have the signature below.

            .. highlight:: python
            .. code-block:: python

                def agg_fn(x: torch.Tensor, dim: int, *args, **kwargs) -> torch.Tensor:
                    # x has shape `(batch size, input_chunk_length, n targets)`, `dim` is always `1`.
                    # function must return a tensor of shape `(batch size, n targets)`
                    return torch.mean(x, dim=dim)
            ..
        **kwargs
            Optional arguments to initialize the pytorch_lightning.Module, pytorch_lightning.Trainer, and
            Darts' :class:`TorchForecastingModel`.
            Since naive models are not trained, the following parameters will have no effect:
            `loss_fn`, `likelihood`, `optimizer_cls`, `optimizer_kwargs`, `lr_scheduler_cls`, `lr_scheduler_kwargs`,
            `n_epochs`, `save_checkpoints`, and some of `pl_trainer_kwargs`.

        Examples
        --------
        >>> from darts.datasets import IceCreamHeaterDataset
        >>> from darts.models import GlobalNaiveAggregate
        >>> # create list of multivariate series
        >>> series_1 = IceCreamHeaterDataset().load()
        >>> series_2 = series_1 + 100.
        >>> series = [series_1, series_2]
        >>> # predict 3 months, take mean over last 60 months
        >>> horizon, icl = 3, 60
        >>> # naive mean over last 60 months (with `output_chunk_length = horizon`)
        >>> model = GlobalNaiveAggregate(input_chunk_length=icl, output_chunk_length=horizon)
        >>> # predict after end of each multivariate series
        >>> pred = model.fit(series).predict(n=horizon, series=series)
        >>> [p.values() for p in pred]
        [array([[29.666668, 50.983337],
               [29.666668, 50.983337],
               [29.666668, 50.983337]]), array([[129.66667, 150.98334],
               [129.66667, 150.98334],
               [129.66667, 150.98334]])]
        >>> # naive moving mean (with `output_chunk_length < horizon`)
        >>> model = GlobalNaiveAggregate(input_chunk_length=icl, output_chunk_length=1, agg_fn="mean")
        >>> pred = model.fit(series).predict(n=horizon, series=series)
        >>> [p.values() for p in pred]
        [array([[29.666668, 50.983337],
               [29.894447, 50.88306 ],
               [30.109352, 50.98111 ]]), array([[129.66667, 150.98334],
               [129.89445, 150.88307],
               [130.10936, 150.98111]])]
        >>> # naive moving sum (with `output_chunk_length < horizon`)
        >>> model = GlobalNaiveAggregate(input_chunk_length=icl, output_chunk_length=1, agg_fn="sum")
        >>> pred = model.fit(series).predict(n=horizon, series=series)
        >>> [p.values() for p in pred]
        [array([[ 1780.,  3059.],
               [ 3544.,  6061.],
               [ 7071., 12077.]]), array([[ 7780.,  9059.],
               [15444., 17961.],
               [30771., 35777.]])]
        """
        super().__init__(
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            output_chunk_shift=output_chunk_shift,
            use_static_covariates=False,
            **kwargs,
        )
        if isinstance(agg_fn, str):
            agg_fn = getattr(torch, agg_fn, None)
            if agg_fn is None:
                raise_log(
                    ValueError(
                        "When `agg_fn` is a string, must be the name of a PyTorch function that "
                        "can be imported directly from `torch`. E.g., `'mean'` for `torch.mean`"
                    ),
                    logger=logger,
                )
        if not isinstance(agg_fn, Callable):
            raise_log(
                ValueError("`agg_fn` must be a string or callable."),
                logger=logger,
            )

        # check that `agg_fn` returns the expected output
        batch_size, n_targets = 5, 3
        x = torch.ones((batch_size, 4, n_targets))
        try:
            agg = agg_fn(x, dim=1)
            assert isinstance(agg, torch.Tensor), (
                "`agg_fn` output must be a torch Tensor."
            )
            assert agg.shape == (
                batch_size,
                n_targets,
            ), "Unexpected `agg_fn` output shape."
        except Exception as err:
            raise_log(
                ValueError(
                    f"`agg_fn` sanity check raised the following error: ({err}) Read the parameter "
                    f"description to properly define the aggregation function."
                ),
                logger=logger,
            )
        self.agg_fn = agg_fn

    def _create_model(
        self, train_sample: MixedCovariatesTrainTensorType
    ) -> _GlobalNaiveModule:
        return _GlobalNaiveAggregateModule(agg_fn=self.agg_fn, **self.pl_module_params)


class _GlobalNaiveSeasonalModule(_GlobalNaiveModule):
    def _forward(self, x_in) -> torch.Tensor:
        y_target = _extract_targets(x_in, self.n_targets)
        season = y_target[:, 0, :]
        return _repeat_along_output_chunk(season, self.output_chunk_length)


class GlobalNaiveSeasonal(_NoCovariatesMixin, _GlobalNaiveModel):
    def __init__(
        self,
        input_chunk_length: int,
        output_chunk_length: int,
        output_chunk_shift: int = 0,
        **kwargs,
    ):
        """Global Naive Seasonal Model.

        The model generates forecasts for each `series` as described below:

        - take the value from each target component at the `input_chunk_length`th point before the end of the
          target `series`.
        - the forecast is the component value repeated `output_chunk_length` times.

        Depending on the horizon `n` used when calling `model.predict()`, the forecasts are either:

        - a constant value if `n <= output_chunk_length`, or
        - a moving (seasonal) value if `n > output_chunk_length`, as a result of the autoregressive prediction.

        This model is equivalent to:

        - :class:`~darts.models.forecasting.baselines.NaiveSeasonal`, when `input_chunk_length` is equal to the length
          of  the input target `series` and `output_chunk_length=1`.

        .. note::
            - Model checkpointing with `save_checkpoints=True`, and checkpoint loading with `load_from_checkpoint()`
              and `load_weights_from_checkpoint()` are not supported for global naive models.

        Parameters
        ----------
        input_chunk_length
            The length of the input sequence fed to the model.
        output_chunk_length
            The length of the emitted forecast and output sequence fed to the model.
        output_chunk_shift
            Optionally, the number of steps to shift the start of the output chunk into the future (relative to the
            input chunk end). This will create a gap between the input and output. If the model supports
            `future_covariates`, the future values are extracted from the shifted output chunk. Predictions will start
            `output_chunk_shift` steps after the end of the target `series`. If `output_chunk_shift` is set, the model
            cannot generate autoregressive predictions (`n > output_chunk_length`).
        **kwargs
            Optional arguments to initialize the pytorch_lightning.Module, pytorch_lightning.Trainer, and
            Darts' :class:`TorchForecastingModel`.
            Since naive models are not trained, the following parameters will have no effect:
            `loss_fn`, `likelihood`, `optimizer_cls`, `optimizer_kwargs`, `lr_scheduler_cls`, `lr_scheduler_kwargs`,
            `n_epochs`, `save_checkpoints`, and some of `pl_trainer_kwargs`.

        Examples
        --------
        >>> from darts.datasets import IceCreamHeaterDataset
        >>> from darts.models import GlobalNaiveSeasonal
        >>> # create list of multivariate series
        >>> series_1 = IceCreamHeaterDataset().load()
        >>> series_2 = series_1 + 100.
        >>> series = [series_1, series_2]
        >>> # predict 3 months, use value from 12 months ago
        >>> horizon, icl = 3, 12
        >>> # repeated seasonal value (with `output_chunk_length = horizon`)
        >>> model = GlobalNaiveSeasonal(input_chunk_length=icl, output_chunk_length=horizon)
        >>> # predict after end of each multivariate series
        >>> pred = model.fit(series).predict(n=horizon, series=series)
        >>> [p.values() for p in pred]
        [array([[ 21., 100.],
               [ 21., 100.],
               [ 21., 100.]]), array([[121., 200.],
               [121., 200.],
               [121., 200.]])]
        >>> # moving seasonal value (with `output_chunk_length < horizon`)
        >>> model = GlobalNaiveSeasonal(input_chunk_length=icl, output_chunk_length=1)
        >>> pred = model.fit(series).predict(n=horizon, series=series)
        >>> [p.values() for p in pred]
        [array([[ 21., 100.],
               [ 21.,  68.],
               [ 24.,  51.]]), array([[121., 200.],
               [121., 168.],
               [124., 151.]])]
        """
        super().__init__(
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            output_chunk_shift=output_chunk_shift,
            use_static_covariates=False,
            **kwargs,
        )

    def _create_model(
        self, train_sample: MixedCovariatesTrainTensorType
    ) -> _GlobalNaiveModule:
        return _GlobalNaiveSeasonalModule(**self.pl_module_params)


class _GlobalNaiveDrift(_GlobalNaiveModule):
    def _forward(self, x_in) -> torch.Tensor:
        y_target = _extract_targets(x_in, self.n_targets)
        slope = _repeat_along_output_chunk(
            (y_target[:, -1, :] - y_target[:, 0, :]) / (self.input_chunk_length - 1),
            self.output_chunk_length,
        )

        x = torch.arange(
            start=self.output_chunk_shift + 1,
            end=self.output_chunk_length + self.output_chunk_shift + 1,
            device=self.device,
        ).view(1, self.output_chunk_length, 1, 1)

        y_0 = y_target[:, -1, :].view(-1, 1, y_target.shape[-1], 1)
        return slope * x + y_0


class GlobalNaiveDrift(_NoCovariatesMixin, _GlobalNaiveModel):
    def __init__(
        self,
        input_chunk_length: int,
        output_chunk_length: int,
        output_chunk_shift: int = 0,
        **kwargs,
    ):
        """Global Naive Drift Model.

        The model generates forecasts for each `series` as described below:

        - take the slope `m` from each target component between the `input_chunk_length`th and last point before the
          end of the `series`.
        - the forecast is `m * x + c` per component where `x` are the values
          `range(1 + output_chunk_shift, 1 + output_chunk_length + output_chunk_shift)`, and `c` are the last values
          from each target component.

        Depending on the horizon `n` used when calling `model.predict()`, the forecasts are either:

        - a linear drift if `n <= output_chunk_length`, or
        - a moving drift if `n > output_chunk_length`, as a result of the autoregressive prediction.

        This model is equivalent to:

        - :class:`~darts.models.forecasting.baselines.NaiveDrift`, when `input_chunk_length` is equal to the length
          of  the input target `series` and `output_chunk_length=n`.

        .. note::
            - Model checkpointing with `save_checkpoints=True`, and checkpoint loading with `load_from_checkpoint()`
              and `load_weights_from_checkpoint()` are not supported for global naive models.

        Parameters
        ----------
        input_chunk_length
            The length of the input sequence fed to the model.
        output_chunk_length
            The length of the emitted forecast and output sequence fed to the model.
        output_chunk_shift
            Optionally, the number of steps to shift the start of the output chunk into the future (relative to the
            input chunk end). This will create a gap between the input and output. If the model supports
            `future_covariates`, the future values are extracted from the shifted output chunk. Predictions will start
            `output_chunk_shift` steps after the end of the target `series`. If `output_chunk_shift` is set, the model
            cannot generate autoregressive predictions (`n > output_chunk_length`).
        **kwargs
            Optional arguments to initialize the pytorch_lightning.Module, pytorch_lightning.Trainer, and
            Darts' :class:`TorchForecastingModel`.
            Since naive models are not trained, the following parameters will have no effect:
            `loss_fn`, `likelihood`, `optimizer_cls`, `optimizer_kwargs`, `lr_scheduler_cls`, `lr_scheduler_kwargs`,
            `n_epochs`, `save_checkpoints`, and some of `pl_trainer_kwargs`.

        Examples
        --------
        >>> from darts.datasets import IceCreamHeaterDataset
        >>> from darts.models import GlobalNaiveDrift
        >>> # create list of multivariate series
        >>> series_1 = IceCreamHeaterDataset().load()
        >>> series_2 = series_1 + 100.
        >>> series = [series_1, series_2]
        >>> # predict 3 months, use drift over the last 60 months
        >>> horizon, icl = 3, 60
        >>> # linear drift (with `output_chunk_length = horizon`)
        >>> model = GlobalNaiveDrift(input_chunk_length=icl, output_chunk_length=horizon)
        >>> # predict after end of each multivariate series
        >>> pred = model.fit(series).predict(n=horizon, series=series)
        >>> [p.values() for p in pred]
        [array([[24.135593, 74.28814 ],
               [24.271187, 74.57627 ],
               [24.40678 , 74.86441 ]]), array([[124.13559, 174.28813],
               [124.27119, 174.57628],
               [124.40678, 174.86441]])]
        >>> # moving drift (with `output_chunk_length < horizon`)
        >>> model = GlobalNaiveDrift(input_chunk_length=icl, output_chunk_length=1)
        >>> pred = model.fit(series).predict(n=horizon, series=series)
        >>> [p.values() for p in pred]
        [array([[24.135593, 74.28814 ],
               [24.256536, 74.784546],
               [24.34563 , 75.45886 ]]), array([[124.13559, 174.28813],
               [124.25653, 174.78455],
               [124.34563, 175.45886]])]
        """
        super().__init__(
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            output_chunk_shift=output_chunk_shift,
            use_static_covariates=False,
            **kwargs,
        )

    def _create_model(
        self, train_sample: MixedCovariatesTrainTensorType
    ) -> _GlobalNaiveModule:
        return _GlobalNaiveDrift(**self.pl_module_params)
