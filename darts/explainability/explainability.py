"""
Forecasting Model Explainer Base Class

A forecasting model explainer takes a fitted forecasting model as input and applies an Explainability model
to it. Its purpose is to explain each past input contribution to a given model forecast. This 'explanation'
depends on the characteristics of the XAI model chosen (shap, lime etc...).

"""
from abc import ABC, abstractmethod
from typing import Optional, Sequence, Tuple, Union

from darts import TimeSeries
from darts.explainability.explainability_result import ExplainabilityResult
from darts.explainability.utils import process_horizons_and_targets, process_input
from darts.logging import get_logger, raise_log
from darts.models.forecasting.forecasting_model import ForecastingModel

logger = get_logger(__name__)

MIN_BACKGROUND_SAMPLE = 10


class ForecastingModelExplainer(ABC):
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
            A series or list of series to *train* the `ForecastingModelExplainer` along with any foreground series.
            Consider using a reduced well-chosen background to reduce computation time.

                - optional if `model` was fit on a single target series. By default, it is the `series` used
                at fitting time.
                - mandatory if `model` was fit on multiple (sequence of) target series.
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
                    "The model must be fitted before instantiating a ForecastingModelExplainer."
                ),
                logger,
            )
        self.model = model
        # default forecasting horizon
        self.n: Optional[int] = getattr(self.model, "output_chunk_length")

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
    ) -> ExplainabilityResult:
        """
        Explains a foreground time series, returns an :class:`ExplainabilityResult
        <darts.explainability.explainability_result.ExplainabilityResult>`.
        Results can be retrieved via the method
        :func:`ExplainabilityResult.get_explanation(horizon, target_component)
        <darts.explainability.explainability_result.ExplainabilityResult.get_explanation>`.
        The result is a multivariate `TimeSeries` instance containing the 'explanation'
        for the (horizon, target_component) forecast at any timestamp forecastable corresponding to
        the foreground `TimeSeries` input.

        The component name convention of this multivariate `TimeSeries` is:
        ``"{name}_{type_of_cov}_lag_{idx}"``, where:

        - ``{name}`` is the component name from the original foreground series (target, past, or future).
        - ``{type_of_cov}`` is the covariates type. It can take 3 different values:
          ``"target"``, ``"past_cov"`` or ``"future_cov"``.
        - ``{idx}`` is the lag index.

        **Example:**

        Say we have a model with 2 target components named ``"T_0"`` and ``"T_1"``,
        3 past covariates with default component names ``"0"``, ``"1"``, and ``"2"``,
        and one future covariate with default component name ``"0"``.
        Also, ``horizons = [1, 2]``.
        The model is a regression model, with ``lags = 3``, ``lags_past_covariates=[-1, -3]``,
        ``lags_future_covariates = [0]``.

        We provide `foreground_series`, `foreground_past_covariates`, `foreground_future_covariates` each of length 5.


        >>> explain_results = explainer.explain(
        >>>     foreground_series=foreground_series,
        >>>     foreground_past_covariates=foreground_past_covariates,
        >>>     foreground_future_covariates=foreground_future_covariates,
        >>>     horizons=[1, 2],
        >>>     target_names=["T_0", "T_1"])
        >>> output = explain_results.get_explanation(horizon=1, target="T_1")

        Then the method returns a multivariate TimeSeries containing the *explanations* of
        the corresponding `ForecastingModelExplainer`, with the following component names:

             - T_0_target_lag-1
             - T_0_target_lag-2
             - T_0_target_lag-3
             - T_1_target_lag-1
             - T_1_target_lag-2
             - T_1_target_lag-3
             - 0_past_cov_lag-1
             - 0_past_cov_lag-3
             - 1_past_cov_lag-1
             - 1_past_cov_lag-3
             - 2_past_cov_lag-1
             - 2_past_cov_lag-3
             - 0_fut_cov_lag_0

        This series has length 3, as the model can explain 5-3+1 forecasts
        (timestamp indexes 4, 5, and 6)

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
            Optionally, an integer or sequence of integers representing the future time step/s to be explained.
            `1` corresponds to the first timestamp being forecasted.
            All values must be `<=output_chunk_length` of the explained forecasting model.
        target_components
            Optionally, a string or sequence of strings with the target components to explain.

        Returns
        -------
        ExplainabilityResult
            The forecast explanations
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
    ) -> Tuple[Sequence[int], Sequence[str]]:

        return process_horizons_and_targets(
            horizons=horizons,
            fallback_horizon=self.n,
            target_components=target_components,
            fallback_target_components=self.target_components,
            check_component_names=self.check_component_names,
        )
