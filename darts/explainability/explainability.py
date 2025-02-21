"""
Forecasting Model Explainer Base Class

A `_ForecastingModelExplainer` takes a fitted forecasting model as input and generates explanations for it.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Optional, Union

from darts import TimeSeries
from darts.explainability.explainability_result import _ExplainabilityResult
from darts.explainability.utils import process_horizons_and_targets, process_input
from darts.logging import get_logger, raise_log
from darts.models.forecasting.forecasting_model import ForecastingModel

logger = get_logger(__name__)

MIN_BACKGROUND_SAMPLE = 10


class _ForecastingModelExplainer(ABC):
    @abstractmethod
    def __init__(
        self,
        model: ForecastingModel,
        background_series: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        background_past_covariates: Optional[
            Union[TimeSeries, Sequence[TimeSeries]]
        ] = None,
        background_future_covariates: Optional[
            Union[TimeSeries, Sequence[TimeSeries]]
        ] = None,
        requires_background: bool = False,
        requires_covariates_encoding: bool = False,
        check_component_names: bool = False,
        test_stationarity: bool = False,
    ):
        """
        The base class for forecasting model explainers. It defines the *minimal* behavior that all
        forecasting model explainers support.

        Naming:

        - A background series is a `TimeSeries` with which to 'train' the `Explainer` model.
        - A foreground series is the `TimeSeries` to explain using the fitted `Explainer` model.

        Parameters
        ----------
        model
            A `ForecastingModel` to be explained. It must be fitted first.
        background_series
            A series or list of series to *train* the `_ForecastingModelExplainer` along with any foreground series.
            Consider using a reduced well-chosen background to reduce computation time.
            Optional if `model` was fit on a single target series. By default, it is the `series` used at fitting time.
            Mandatory if `model` was fit on multiple (sequence of) target series.
        background_past_covariates
            A past covariates series or list of series that the model needs once fitted.
        background_future_covariates
            A future covariates series or list of series that the model needs once fitted.
        requires_background
            Whether the explainer requires background series as an input. If `True`, raises an error if no background
            series were provided and `model` was fit using multiple series.
        requires_covariates_encoding
            Whether to apply the model's encoders to the input covariates. This should only be `True` if the
            Explainer will not call model methods `fit()` or `predict()` directly.
        check_component_names
            Whether to enforce that, in the case of multiple time series, all series of the same type (target and/or
            *_covariates) must have the same component names.
        test_stationarity
            Whether to raise a warning if not all `background_series` are stationary.
        """
        if not model._fit_called:
            raise_log(
                ValueError(
                    f"The model must be fitted before instantiating a {self.__class__.__name__}."
                ),
                logger,
            )
        self.model = model
        # default forecasting horizon
        self.n: Optional[int] = getattr(self.model, "output_chunk_length", None)

        # check background input validity and process it
        (
            self.background_series,
            self.background_past_covariates,
            self.background_future_covariates,
            self.target_components,
            self.static_covariates_components,
            self.past_covariates_components,
            self.future_covariates_components,
        ) = process_input(
            model=model,
            input_type="background",
            series=background_series,
            past_covariates=background_past_covariates,
            future_covariates=background_future_covariates,
            fallback_series=model.training_series,
            fallback_past_covariates=model.past_covariate_series,
            fallback_future_covariates=model.future_covariate_series,
            check_component_names=check_component_names,
            requires_input=requires_background,
            requires_covariates_encoding=requires_covariates_encoding,
            test_stationarity=test_stationarity,
        )
        self.requires_foreground = self.background_series is None
        self.requires_covariates_encoding = requires_covariates_encoding
        self.check_component_names = check_component_names
        self.test_stationarity = test_stationarity

    @abstractmethod
    def explain(
        self,
        foreground_series: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        foreground_past_covariates: Optional[
            Union[TimeSeries, Sequence[TimeSeries]]
        ] = None,
        foreground_future_covariates: Optional[
            Union[TimeSeries, Sequence[TimeSeries]]
        ] = None,
        horizons: Optional[Sequence[int]] = None,
        target_components: Optional[Sequence[str]] = None,
    ) -> _ExplainabilityResult:
        """
        Explains a foreground time series, and returns a :class:`_ExplainabilityResult
        <darts.explainability.explainability_result._ExplainabilityResult>` that can be used for downstream tasks.

        Parameters
        ----------
        foreground_series
            Optionally, one or a sequence of target `TimeSeries` to be explained. Can be multivariate.
            If not provided, the background `TimeSeries` will be explained instead.
        foreground_past_covariates
            Optionally, one or a sequence of past covariates `TimeSeries` if required by the forecasting model.
        foreground_future_covariates
            Optionally, one or a sequence of future covariates `TimeSeries` if required by the forecasting model.
        horizons
            Optionally, an integer or sequence of integers representing the future time steps to be explained.
            `1` corresponds to the first timestamp being forecasted.
            All values must be `<=output_chunk_length` of the explained forecasting model.
        target_components
            Optionally, a string or sequence of strings with the target components to explain.

        Returns
        -------
        _ExplainabilityResult
            The explainability result.
        """
        pass

    def _process_foreground(
        self,
        foreground_series: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        foreground_past_covariates: Optional[
            Union[TimeSeries, Sequence[TimeSeries]]
        ] = None,
        foreground_future_covariates: Optional[
            Union[TimeSeries, Sequence[TimeSeries]]
        ] = None,
    ):
        return process_input(
            model=self.model,
            input_type="foreground",
            series=foreground_series,
            past_covariates=foreground_past_covariates,
            future_covariates=foreground_future_covariates,
            fallback_series=self.background_series,
            fallback_past_covariates=self.background_past_covariates,
            fallback_future_covariates=self.background_future_covariates,
            check_component_names=self.check_component_names,
            requires_input=self.requires_foreground,
            requires_covariates_encoding=self.requires_covariates_encoding,
            test_stationarity=self.test_stationarity,
        )

    def _process_horizons_and_targets(
        self,
        horizons: Optional[Union[int, Sequence[int]]],
        target_components: Optional[Union[str, Sequence[str]]],
    ) -> tuple[Sequence[int], Sequence[str]]:
        return process_horizons_and_targets(
            horizons=horizons,
            fallback_horizon=self.n,
            target_components=target_components,
            fallback_target_components=self.target_components,
            check_component_names=self.check_component_names,
        )
