"""
Forecasting Model Explainer Base Class

A forecasting model explainer takes a fitted forecasting model as input and applies an Explainability model
to it. Its purpose is to explain each past input contribution to a given model forecast. This 'explanation'
depends on the characteristics of the XAI model chosen (shap, lime etc...).

"""
from abc import ABC, abstractmethod
from typing import Collection, List, Optional, Sequence, Tuple, Union

from darts import TimeSeries
from darts.explainability.explainability_result import ExplainabilityResult
from darts.logging import get_logger, raise_if, raise_if_not, raise_log
from darts.models.forecasting.forecasting_model import ForecastingModel
from darts.utils.statistics import stationarity_tests
from darts.utils.utils import series2seq

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
            Whether to apply the model's encoders to the input covariates. This is should only be `True` if the
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
            self.past_covariates_components,
            self.future_covariates_components,
        ) = self._process_background(
            model=model,
            background_series=background_series,
            background_past_covariates=background_past_covariates,
            background_future_covariates=background_future_covariates,
            check_component_names=check_component_names,
            requires_background=requires_background,
            requires_covariates_encoding=requires_covariates_encoding,
            test_stationarity=test_stationarity,
        )
        self.requires_foreground = self.background_series is None
        self.requires_covariates_encoding = requires_covariates_encoding
        self.check_component_names = check_component_names

    @classmethod
    def _process_background(
        cls,
        model,
        background_series,
        background_past_covariates,
        background_future_covariates,
        check_component_names: bool = True,
        requires_background: bool = True,
        requires_covariates_encoding: bool = False,
        test_stationarity: bool = False,
    ):
        if (
            background_series is not None
            or background_past_covariates is not None
            or background_future_covariates is not None
        ):
            requires_background = True

        # if `background_series` was not passed, use `training_series` saved in fitted forecasting model.
        if background_series is None:
            raise_if(
                (background_past_covariates is not None)
                or (background_future_covariates is not None),
                "Supplied background past or future covariates but no background series. Please also provide "
                "`background_series`.",
                logger,
            )
            raise_if(
                requires_background and model.training_series is None,
                "`background_series` must be provided if `model` was fit on multiple time series.",
                logger,
            )
            background_series = model.training_series
            background_past_covariates = model.past_covariate_series
            background_future_covariates = model.future_covariate_series
        # otherwise use the passed background, and optionally generate the covariate encodings
        else:
            if model.encoders.encoding_available and requires_covariates_encoding:
                (
                    background_past_covariates,
                    background_future_covariates,
                ) = model.generate_fit_encodings(
                    series=background_series,
                    past_covariates=background_past_covariates,
                    future_covariates=background_future_covariates,
                )

        background_series = series2seq(background_series)
        background_past_covariates = series2seq(background_past_covariates)
        background_future_covariates = series2seq(background_future_covariates)

        target_components = None
        if background_series is not None:
            target_components = background_series[0].columns.to_list()
        past_covariates_components = None
        if background_past_covariates is not None:
            past_covariates_components = background_past_covariates[0].columns.to_list()
        future_covariates_components = None
        if background_future_covariates is not None:
            future_covariates_components = background_future_covariates[
                0
            ].columns.to_list()

        cls._check_input(
            model,
            "background",
            background_series,
            background_past_covariates,
            background_future_covariates,
            target_components,
            past_covariates_components,
            future_covariates_components,
            check_component_names=check_component_names,
            requires_input=requires_background,
        )

        if test_stationarity and background_series is not None:
            if not cls._test_stationarity(background_series):
                logger.warning(
                    "At least one time series component of the background time series is not stationary."
                    " Beware of wrong interpretation with chosen explainability."
                )
        return (
            background_series,
            background_past_covariates,
            background_future_covariates,
            target_components,
            past_covariates_components,
            future_covariates_components,
        )

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
        requires_foreground = self.requires_foreground
        if (
            foreground_series is not None
            or foreground_past_covariates is not None
            or foreground_future_covariates is not None
        ):
            requires_foreground = True

        # if `foreground_series` was not passed, use `background_series` (if not available, raise error)
        if foreground_series is None:
            if requires_foreground:
                raise_log(ValueError("Must provide a `foreground_series`."), logger)
            if (foreground_past_covariates is not None) or (
                foreground_future_covariates is not None
            ):
                raise_log(
                    ValueError(
                        "Supplied foreground past or future covariates but no foreground series. Please also provide "
                        "`foreground_series`."
                    ),
                    logger,
                )
            foreground_series = self.background_series
            foreground_past_covariates = self.background_past_covariates
            foreground_future_covariates = self.background_future_covariates
        # otherwise use the passed foreground, and optionally generate the covariate encodings
        else:
            if (
                self.model.encoders.encoding_available
                and self.requires_covariates_encoding
            ):
                (
                    foreground_past_covariates,
                    foreground_future_covariates,
                ) = self.model.generate_fit_encodings(
                    series=foreground_series,
                    past_covariates=foreground_past_covariates,
                    future_covariates=foreground_future_covariates,
                )

        foreground_series = series2seq(foreground_series)
        foreground_past_covariates = series2seq(foreground_past_covariates)
        foreground_future_covariates = series2seq(foreground_future_covariates)

        target_components = None
        if foreground_series is not None:
            target_components = foreground_series[0].columns.to_list()
        past_covariates_components = None
        if foreground_past_covariates is not None:
            past_covariates_components = foreground_past_covariates[0].columns.to_list()
        future_covariates_components = None
        if foreground_future_covariates is not None:
            future_covariates_components = foreground_future_covariates[
                0
            ].columns.to_list()

        self._check_input(
            self.model,
            "foreground",
            foreground_series,
            foreground_past_covariates,
            foreground_future_covariates,
            target_components,
            past_covariates_components,
            future_covariates_components,
            check_component_names=self.check_component_names,
            requires_input=requires_foreground,
        )

        return (
            foreground_series,
            foreground_past_covariates,
            foreground_future_covariates,
            target_components,
            past_covariates_components,
            future_covariates_components,
        )

    @staticmethod
    def _check_input(
        model,
        input_type: str,
        series: Optional[Sequence[TimeSeries]],
        past_covariates: Optional[Sequence[TimeSeries]],
        future_covariates: Optional[Sequence[TimeSeries]],
        target_components: Optional[List[str]],
        past_covariates_components: Optional[List[str]],
        future_covariates_components: Optional[List[str]],
        check_component_names: bool = True,
        requires_input: bool = True,
    ) -> None:
        if input_type not in ["background", "foreground"]:
            raise_log(
                ValueError(
                    f"Unknown `input_type='{input_type}'`. Must be one of ['background', 'foreground']."
                ),
                logger,
            )
        if past_covariates is not None:
            raise_if_not(
                len(series) == len(past_covariates),
                f"The number of {input_type} series and past covariates must be the same.",
                logger,
            )

        if future_covariates is not None:
            raise_if_not(
                len(series) == len(future_covariates),
                f"The number of {input_type} series and future covariates must be the same.",
                logger,
            )

        if requires_input:
            raise_if(
                model.uses_past_covariates and past_covariates is None,
                f"A {input_type} past covariates is not provided, but the model requires past covariates.",
                logger,
            )
            raise_if(
                model.uses_future_covariates and future_covariates is None,
                f"A {input_type} future covariates is not provided, but the model requires future covariates.",
                logger,
            )

        if not check_component_names:
            return

        # ensure we have the same names between TimeSeries (if list of). Important to ensure homogeneity
        # for explained features.
        for idx in range(len(series)):
            raise_if_not(
                all(
                    [
                        series[idx].columns.to_list() == target_components,
                        past_covariates[idx].columns.to_list()
                        == past_covariates_components
                        if past_covariates is not None
                        else True,
                        future_covariates[idx].columns.to_list()
                        == future_covariates_components
                        if future_covariates is not None
                        else True,
                    ]
                ),
                "Columns names must be identical between TimeSeries list components (multi-TimeSeries).",
            )

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
        horizons: Optional[Collection[int]] = None,
        target_components: Optional[Collection[str]] = None,
    ) -> ExplainabilityResult:
        """
        Explains a foreground time series, returns an :class:`ExplainabilityResult`.

        Results can be retrieved via the method
        :func:`ExplainabilityResult.get_explanation(horizon, target_component)`.
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
            Optionally, the target `TimeSeries` to be explained. Can be multivariate.
            If not provided, the background `TimeSeries` will be explained instead.
        foreground_past_covariates
            Optionally, past covariate timeseries if needed by the ForecastingModel.
        foreground_future_covariates
            Optionally, future covariate timeseries if needed by the ForecastingModel.
        horizons
            Optionally, a collection of integers representing the future lags to be explained.
            Horizon 1 corresponds to the first timestamp being forecasted.
            All values must be no larger than `output_chunk_length` of the explained model.
        target_components
            Optionally, A list of string naming the target components to be explained.

         Returns
         -------
         ExplainabilityResult
             The forecast explanations.

        """
        pass

    @staticmethod
    def _test_stationarity(background_series: Union[TimeSeries, Sequence[TimeSeries]]):
        return all(
            [
                (stationarity_tests(bs[c]) for c in bs.components)
                for bs in background_series
            ]
        )

    def _process_horizons_and_targets(
        self,
        horizons: Optional[Union[int, Sequence[int]]],
        target_components: Optional[Union[str, Sequence[str]]],
    ) -> Tuple[Sequence[int], Sequence[str]]:

        if target_components is not None:
            if isinstance(target_components, str):
                target_components = [target_components]
            if self.check_component_names:
                raise_if(
                    any(
                        [
                            target_name not in self.target_components
                            for target_name in target_components
                        ]
                    ),
                    "One of the target names doesn't exist. Please review your target_names input",
                )
        else:
            target_components = self.target_components

        if horizons is not None:
            if isinstance(horizons, int):
                horizons = [horizons]

            if self.n is not None:
                raise_if(
                    max(horizons) > self.n,
                    "One of the horizons is than `output_chunk_length`.",
                )
            raise_if(min(horizons) < 1, "One of the horizons is too small.")
        else:
            horizons = range(1, self.n + 1)

        return horizons, target_components
