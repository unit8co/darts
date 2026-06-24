"""
Explainability Result
---------------------

Contains the explainability results obtained from :func:`_ForecastingModelExplainer.explain()
<darts.explainability.explainability._ForecastingModelExplainer.explain>`.

- :class:`ShapExplainabilityResult <ShapExplainabilityResult>` for :class:`ShapExplainer
  <darts.explainability.shap_explainer.ShapExplainer>`. Contains general forecasting model explainability result
  based on SHAP values.
- :class:`ShapSingleExplainabilityResult <ShapSingleExplainabilityResult>` for :class:`ShapExplainer
  <darts.explainability.shap_explainer.ShapExplainer>`. Contains the explainability result for a single model forecast.
- :class:`TFTExplainabilityResult <TFTExplainabilityResult>` for :class:`TFTExplainer
  <darts.explainability.tft_explainer.TFTExplainer>`.
- :class:`ComponentBasedExplainabilityResult <ComponentBasedExplainabilityResult>` for generic component-based
  explainability result.
- :class:`HorizonBasedExplainabilityResult <HorizonBasedExplainabilityResult>` for generic horizon-based
  explainability results.
"""

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd
import shap

from darts import TimeSeries
from darts.logging import get_logger, raise_if, raise_if_not, raise_log
from darts.typing import TimeSeriesLike

logger = get_logger(__name__)


class _ExplainabilityResult(ABC):
    """
    Abstract class for explainability results of a :class:`_ForecastingModelExplainer`.
    The subclasses should implement convenient access to explanations.
    """

    @abstractmethod
    def get_explanation(self, *args, **kwargs):
        """Returns one or multiple explanations based on some input parameters."""


class ComponentBasedExplainabilityResult(_ExplainabilityResult):
    """
    Stores the explainability results of a :class:`_ForecastingModelExplainer
    <darts.explainability.explainability._ForecastingModelExplainer>` with convenient access to component-based
    results.

    Parameters
    ----------
    explained_components
        The component-based explainability results.

    Examples
    --------
    >>> explainer = SomeComponentBasedExplainer(model)
    >>> result = explainer.explain()
    >>> explanation = result.get_explanation(component="some_component")
    """

    def __init__(
        self,
        explained_components: dict[str, Any] | list[dict[str, Any]],
    ):
        if isinstance(explained_components, list):
            comps_available = explained_components[0].keys()
            if not all(comp.keys() == comps_available for comp in explained_components):
                raise_log(
                    ValueError(
                        "When giving a list of explained component dicts, the dict keys must match."
                    ),
                    logger=logger,
                )
        else:
            comps_available = explained_components.keys()
        self.explained_components = explained_components
        self.available_components = list(comps_available)

    def get_explanation(self, component: str | None = None) -> Any | list[Any]:
        """
        Returns one or several explanations for a given component.

        Parameters
        ----------
        component
            Optionally, the target series component for which to return the explanation. Must be supplied for
            multivariate forecasting models.
        """
        return self._query_explainability_result(self.explained_components, component)

    def _query_explainability_result(
        self,
        attr: dict[str, Any] | list[dict[str, Any]],
        component: str | None,
    ) -> Any:
        """
        Helper that extracts and returns the explainability result attribute for a given component.

        Parameters
        ----------
        attr
            An explainability result attribute from which to extract the component.
        component
            Optionally, the target series component for which to return the explanation. Must be supplied for
            multivariate forecasting models.
        """
        component = self._validate_input_for_querying_explainability_result(component)
        if isinstance(attr, list):
            return [attr_[component] for attr_ in attr]
        else:
            return attr[component]

    def _validate_input_for_querying_explainability_result(
        self, component: str | None
    ) -> str:
        """
        Helper that validates the input parameters of a method that queries the `ComponentBasedExplainabilityResult`.

        Parameters
        ----------
        component
            Optionally, the target series component for which to return the explanation. Must be supplied for
            multivariate forecasting models.
        """
        # validate component argument
        if component is None and len(self.available_components) > 1:
            raise_log(
                ValueError(
                    "The component parameter is required when the model has more than one component."
                ),
                logger,
            )

        component_out = self.available_components[0] if component is None else component

        if component_out not in self.available_components:
            raise_log(
                ValueError(
                    f'Component "{component_out}" is not available. '
                    f"Available components are: {self.available_components}"
                ),
                logger,
            )
        return component_out


class HorizonBasedExplainabilityResult(_ExplainabilityResult):
    """
    Stores the explainability results of a :class:`_ForecastingModelExplainer
    <darts.explainability.explainability._ForecastingModelExplainer>` with convenient access to horizon-based
    results.

    The result is a multivariate ``TimeSeries`` instance containing the "explanation" for the
    ``(horizon, target_component)`` forecast at any timestamp forecastable in the foreground series.

    The components of the ``TimeSeries`` correspond to the input features used by the model to produce the
    forecasts. They are named according to the convention: ``"{name}_{type_of_cov}_lag{idx}"``, where:

    - ``{name}`` is the component name from the original foreground series (target, past covariates, or future
      covariates).
    - ``{type_of_cov}`` is the covariates type. It can take 3 different values:
      ``"target"``, ``"pastcov"``,  ``"futcov"``.
    - ``{idx}`` is the lag index, where ``0`` represents the position of the first predicted step.

    Static covariates are named according to the convention: ``"{name}_statcov_target_{comp}"``, where:

    - ``{name}`` is the variable name of the static covariate.
    - ``{comp}`` is the component name of the target series if static covariates are component-specific, or
      ``"global_components"`` if they are global.

    Examples
    --------
    Say we have a ``SKLearnModel`` instance with:

        - 1 target component named ``"Y"``,
        - 1 future covariate named ``"month"``,
        - ``lags = 2``, and ``lags_future_covariates = [-1, 0]``.

    Let's explain the background series that the model was trained on:

    >>> from darts.datasets import AusBeerDataset
    >>> from darts.explainability import ShapExplainer
    >>> from darts.models import LinearRegressionModel
    >>> from darts.utils.timeseries_generation import datetime_attribute_timeseries as dta
    >>>
    >>> # load a target series and create future covariates holding the calendar month values
    >>> series = AusBeerDataset().load()
    >>> fc = dta(series, attribute="month", add_length=12)
    >>>
    >>> # create and fit a model
    >>> model = LinearRegressionModel(lags=2, lags_future_covariates=[-1, 0])
    >>> model.fit(series, future_covariates=fc)
    >>>
    >>> # create an explainer; requires background series if the model was trained on multiple series
    >>> explainer = ShapExplainer(model)
    >>> # explain the background series (or foreground if passed to `explain()`)
    >>> result = explainer.explain()
    >>> result.get_explanation(horizon=1)
                Y_target_lag-2  Y_target_lag-1  month_futcov_lag-1  month_futcov_lag0
    1956-07-01      -56.332566     -106.927156          -24.253184          33.064478
    1956-10-01      -88.545937      -99.569541           13.642416          84.727725
    1957-01-01      -82.194005      -57.000488           51.538016         -70.262016
    1957-04-01      -45.443539      -81.175506          -62.148784         -18.598769
    1957-07-01      -66.314174      -99.043998          -24.253184          33.064478
    ...                    ...             ...                 ...                ...
    2007-10-01      -11.415330      -11.803715           13.642416          84.727725
    2008-01-01       -6.424526       29.714250           51.538016         -70.262016
    2008-04-01       29.418521        1.860425          -62.148784         -18.598769
    2008-07-01        5.371920      -13.905891          -24.253184          33.064478
    2008-10-01       -8.239364       -3.395013           13.642416          84.727725

    shape: (210, 4, 1), freq: QS-OCT, size: 6.56 KB

    The explanation has length 210, containing the feature SHAP values for all possible forecast start points
    over the background series.
    """

    def __init__(
        self,
        explained_forecasts: dict[int, dict[str, TimeSeries]]
        | list[dict[int, dict[str, TimeSeries]]],
    ):
        self.explained_forecasts = explained_forecasts
        if isinstance(self.explained_forecasts, list):
            if not isinstance(self.explained_forecasts[0], dict):
                raise_log(
                    ValueError(
                        "The `explained_forecasts` list must consist of dictionaries."
                    ),
                    logger,
                )
            if not all(
                isinstance(key, int) for key in self.explained_forecasts[0].keys()
            ):
                raise_log(
                    ValueError(
                        "The `explained_forecasts` dictionary list must have all integer keys."
                    ),
                    logger,
                )
            self.available_horizons = list(self.explained_forecasts[0].keys())
            h_0 = self.available_horizons[0]
            self.available_components = list(self.explained_forecasts[0][h_0].keys())
        elif isinstance(self.explained_forecasts, dict):
            if all(isinstance(key, int) for key in self.explained_forecasts.keys()):
                self.available_horizons = list(self.explained_forecasts.keys())
                h_0 = self.available_horizons[0]
                self.available_components = list(self.explained_forecasts[h_0].keys())
            else:
                raise_log(
                    ValueError(
                        "The `explained_forecasts` dictionary must have all integer keys."
                    ),
                    logger,
                )
        else:
            raise_log(
                ValueError(
                    "The `explained_forecasts` must be a dictionary or a list of dictionaries."
                ),
                logger,
            )

    # TODO(oswald): TSS migration — wrap `list[TimeSeries]` return as TimeSeriesSequence once TSS lands.
    def get_explanation(
        self, horizon: int, component: str | None = None
    ) -> TimeSeriesLike:
        """
        Returns one or several ``TimeSeries`` representing the explanations
        for a given horizon and component.

        Parameters
        ----------
        horizon
            The horizon for which to return the explanation.
        component
            Optionally, the target series component for which to return the explanation. Must be supplied for
            multivariate forecasting models.
        """
        return self._query_explainability_result(
            self.explained_forecasts, horizon, component
        )

    def _query_explainability_result(
        self,
        attr: dict[int, dict[str, Any]] | list[dict[int, dict[str, Any]]],
        horizon: int,
        component: str | None = None,
    ) -> Any:
        """
        Helper that extracts and returns the explainability result attribute for a specified horizon and component from
        the input attribute.

        Parameters
        ----------
        attr
            An explainability result attribute from which to extract the content for a certain horizon and component.
        horizon
            The horizon for which to return the content of the attribute.
        component
            Optionally, the target series component for which to return the explanation. Must be supplied for
            multivariate forecasting models.
        """
        component = self._validate_input_for_querying_explainability_result(
            horizon, component
        )

        if isinstance(attr, list):
            return [attr[i][horizon][component] for i in range(len(attr))]
        elif all(isinstance(key, int) for key in attr.keys()):
            return attr[horizon][component]
        else:
            raise_log(
                ValueError(
                    f"Something went wrong. {self.__class__.__name__} got instantiated with an unexpected type."
                ),
                logger,
            )

    def _validate_input_for_querying_explainability_result(
        self, horizon: int, component: str | None = None
    ) -> str:
        """
        Helper that validates the input parameters of a method that queries the `HorizonBasedExplainabilityResult`.

        Parameters
        ----------
        horizon
            The horizon for which to return the explanation.
        component
            Optionally, the target series component for which to return the explanation. Must be supplied for
            multivariate forecasting models.
        """
        # validate component argument
        if component is None and len(self.available_components) > 1:
            raise_log(
                ValueError(
                    "The component parameter is required when the model has more than one component."
                ),
                logger,
            )

        component_out = self.available_components[0] if component is None else component

        if component_out not in self.available_components:
            raise_log(
                ValueError(
                    f'Component "{component_out}" is not available. '
                    f"Available components are: {self.available_components}"
                ),
                logger,
            )

        if horizon not in self.available_horizons:
            raise_log(
                ValueError(
                    f"Horizon {horizon} is not available. Available horizons are: {self.available_horizons}"
                ),
                logger,
            )
        return component_out


class ShapExplainabilityResult(HorizonBasedExplainabilityResult):
    """
    Stores the explainability results of a :class:`ShapExplainer <darts.explainability.shap_explainer.ShapExplainer>`
    with convenient access to the results.

    It extends the :class:`HorizonBasedExplainabilityResult
    <HorizonBasedExplainabilityResult>` and carries additional information specific to the SHAP explainers.

    - :func:`get_explanation() <get_explanation>`: SHAP values for a given horizon and component in
      multivariate ``TimeSeries`` format.
    - :func:`get_feature_values() <get_feature_values>`: input feature values for a given horizon and component in
      multivariate ``TimeSeries`` format.
    - :func:`get_shap_explanation_object() <get_shap_explanation_object>`: ``shap.Explanation`` object for a given
      horizon and component.

    Examples
    --------
    >>> from darts.datasets import AusBeerDataset
    >>> from darts.explainability import ShapExplainer
    >>> from darts.models import LinearRegressionModel
    >>> from darts.utils.timeseries_generation import datetime_attribute_timeseries as dta
    >>>
    >>> # load a target series and create future covariates holding the calendar month values
    >>> series = AusBeerDataset().load()
    >>> fc = dta(series, attribute="month", add_length=12)
    >>>
    >>> # create and fit a model
    >>> model = LinearRegressionModel(lags=2, lags_future_covariates=[-1, 0])
    >>> model.fit(series, future_covariates=fc)
    >>>
    >>> # create an explainer; requires background series if the model was trained on multiple series
    >>> explainer = ShapExplainer(model)
    >>> # explain the background series (or foreground if passed to `explain()`)
    >>> result = explainer.explain()
    >>>
    >>> # get explanations for a specific horizon (and optional component for multivariate models)
    >>> # the feature SHAP values for all possible forecast start points
    >>> explanation = result.get_explanation(horizon=1)
    >>> # the feature values used as model inputs for all possible forecast start points
    >>> feature_values = result.get_feature_values(horizon=1)
    >>> # the raw shap objects for further processing
    >>> shap_object = result.get_shap_explanation_object(horizon=1)
    """

    def __init__(
        self,
        explained_forecasts: dict[int, dict[str, TimeSeries]]
        | list[dict[int, dict[str, TimeSeries]]],
        feature_values: dict[int, dict[str, TimeSeries]]
        | list[dict[int, dict[str, TimeSeries]]],
        shap_explanation_object: dict[int, dict[str, shap.Explanation]]
        | list[dict[int, dict[str, shap.Explanation]]],
    ):
        super().__init__(explained_forecasts)
        self.feature_values = feature_values
        self.shap_explanation_object = shap_explanation_object

    # TODO(oswald): TSS migration — wrap `list[TimeSeries]` return as TimeSeriesSequence once TSS lands.
    def get_feature_values(
        self, horizon: int, component: str | None = None
    ) -> TimeSeriesLike:
        """
        Returns one or several ``TimeSeries`` representing the feature values
        for a given horizon and component.

        Parameters
        ----------
        horizon
            The horizon for which to return the feature values.
        component
            Optionally, the target series component for which to return the feature values. Must be supplied for
            multivariate forecasting models.
        """
        return self._query_explainability_result(
            self.feature_values, horizon, component
        )

    def get_shap_explanation_object(
        self, horizon: int, component: str | None = None
    ) -> shap.Explanation | list[shap.Explanation]:
        """
        Returns the underlying ``shap.Explanation`` object for a given horizon and component.

        Parameters
        ----------
        horizon
            The horizon for which to return the ``shap.Explanation`` object.
        component
            Optionally, the target series component for which to return the ``shap.Explanation object``. Must be
            supplied for multivariate forecasting models.
        """
        return self._query_explainability_result(
            self.shap_explanation_object, horizon, component
        )


class ShapSingleExplainabilityResult(ComponentBasedExplainabilityResult):
    """
    Stores the explainability results of a :class:`ShapExplainer <darts.explainability.shap_explainer.ShapExplainer>`
    for a single model forecast with convenient access to the results.

    It extends the :class:`ComponentBasedExplainabilityResult <ComponentBasedExplainabilityResult>` and
    carries additional information specific to the SHAP explainers.

    - :func:`get_explanation() <get_explanation>`: SHAP values for a given component in multivariate ``TimeSeries``
      format.
    - :func:`get_feature_values() <get_feature_values>`: input feature values for a given component in
      single-timestamp multivariate ``TimeSeries`` format.
    - :func:`get_shap_explanation_object() <get_shap_explanation_object>`: ``shap.Explanation`` object for a given
      component.

    Examples
    --------
    >>> from darts.datasets import AusBeerDataset
    >>> from darts.explainability import ShapExplainer
    >>> from darts.models import LinearRegressionModel
    >>> from darts.utils.timeseries_generation import datetime_attribute_timeseries as dta
    >>>
    >>> # load a target series and create future covariates holding the calendar month values
    >>> series = AusBeerDataset().load()
    >>> fc = dta(series, attribute="month", add_length=12)
    >>>
    >>> # create and fit a model
    >>> model = LinearRegressionModel(lags=2, lags_future_covariates=[-1, 0])
    >>> model.fit(series, future_covariates=fc)
    >>>
    >>> # create an explainer; requires background series if the model was trained on multiple series
    >>> explainer = ShapExplainer(model)
    >>> # explain the background forecast (or foreground forecast if passed to `explain_single()`)
    >>> result = explainer.explain_single()
    >>> # get explanations for that forecast (and optional component for multivariate models)
    >>> # the feature SHAP values for that forecast
    >>> explanation = result.get_explanation()
    >>> # the feature values used as model inputs for that forecast
    >>> feature_values = result.get_feature_values()
    >>> # the raw shap objects for further processing
    >>> shap_object = result.get_shap_explanation_object()
    """

    def __init__(
        self,
        explained_components: dict[str, TimeSeries],
        feature_values: dict[str, TimeSeries],
        shap_explanation_object: dict[str, shap.Explanation],
    ):
        super().__init__(explained_components)
        self.feature_values = feature_values
        self.shap_explanation_object = shap_explanation_object

    def get_explanation(self, component: str | None = None) -> TimeSeries:
        """
        Returns the ``TimeSeries`` representing the explanation for a given component.

        The components of the ``TimeSeries`` correspond to the input features used by the model to produce the
        forecasts. The time index contains the forecasted timestamps in the future. Therefore, the values of
        ``TimeSeries`` are the SHAP values of the features for the forecast at each forecasted timestamp.

        Parameters
        ----------
        component
            Optionally, the target series component for which to return the explanation. Must be supplied for
            multivariate forecasting models.
        """
        return self._query_explainability_result(self.explained_components, component)

    def get_feature_values(self, component: str | None = None) -> TimeSeries:
        """
        Returns the ``TimeSeries`` representing the feature values for a given component.

        The components of the ``TimeSeries`` correspond to the input features used by the model to produce the
        forecasts. The time index contains only one timestamp, which is the first forecasted timestamp in the future.
        The values of the ``TimeSeries`` are the feature values used by the model to produce the forecast starting
        at that timestamp.

        Parameters
        ----------
        component
            The component for which to return the feature values. Must be supplied for multivariate forecasting models.
        """
        return self._query_explainability_result(self.feature_values, component)

    def get_shap_explanation_object(
        self, component: str | None = None
    ) -> shap.Explanation:
        """
        Returns the underlying ``shap.Explanation`` object for a given component.

        Parameters
        ----------
        component
            The component for which to return the ``shap.Explanation`` object. Must be supplied for multivariate
            forecasting models.
        """
        return self._query_explainability_result(
            self.shap_explanation_object, component
        )


class TFTExplainabilityResult(ComponentBasedExplainabilityResult):
    """
    Stores the explainability results of a :class:`TFTExplainer <darts.explainability.tft_explainer.TFTExplainer>`
    with convenient access to the results. It extends the :class:`ComponentBasedExplainabilityResult` and carries
    information specific to the TFT explainer.

    - :func:`get_attention() <TFTExplainabilityResult.get_attention>`: self attention over the encoder and decoder
    - :func:`get_encoder_importance() <TFTExplainabilityResult.get_encoder_importance>`: encoder feature importances
      including past target, past covariates, and historic part of future covariates.
    - :func:`get_decoder_importance() <TFTExplainabilityResult.get_decoder_importance>`: decoder feature importances
      including future part of future covariates.
    - :func:`get_static_covariates_importance() <TFTExplainabilityResult.get_static_covariates_importance>`: static
      covariates importances.
    - :func:`get_feature_importances() <TFTExplainabilityResult.get_feature_importances>`: get all feature importances
      at once.

    Examples
    --------
    >>> from darts.datasets import AusBeerDataset
    >>> from darts.explainability import TFTExplainer
    >>> from darts.models import TFTModel
    >>> from darts.utils.timeseries_generation import datetime_attribute_timeseries as dta
    >>>
    >>> # load a target series and create future covariates holding the calendar month values
    >>> series = AusBeerDataset().load().astype("float32")
    >>> fc = dta(series, attribute="month", add_length=12, dtype=series.dtype)
    >>>
    >>> # create and fit a model
    >>> model = TFTModel(
    >>>     input_chunk_length=12,
    >>>     output_chunk_length=12,
    >>>     use_reversible_instance_norm=True
    >>> )
    >>> model.fit(series, future_covariates=fc)
    >>>
    >>> # create an explainer
    >>> explainer = TFTExplainer(model)
    >>> # explain a single forecast:
    >>> # - by default, if foreground is not provided, it is the forecast of the background
    >>> # - otherwise, it is the forecast of the foreground
    >>> result = explainer.explain()
    >>> attention = result.get_attention()
    >>> feature_importances = result.get_feature_importances()
    >>> encoder_importance = result.get_encoder_importance()
    >>> decoder_importance = result.get_decoder_importance()
    >>> static_cov_importance = result.get_static_covariates_importance()
    """

    def __init__(
        self,
        explanations: dict[str, Any] | list[dict[str, Any]],
    ):
        super().__init__(explanations)
        self.feature_importances = [
            "encoder_importance",
            "decoder_importance",
            "static_covariates_importance",
        ]

    # TODO(oswald): TSS migration — wrap `list[TimeSeries]` return as TimeSeriesSequence once TSS lands.
    def get_attention(self) -> TimeSeries | list[TimeSeries]:
        """
        Returns the time-dependent attention on the encoder and decoder for each `horizon` in (1,
        `output_chunk_length`). The time index ranges from the prediction series' start time - input_chunk_length and
        ends at the prediction series' end time. If multiple series were used when calling
        :func:`TFTExplainer.explain() <darts.explainability.tft_explainer.TFTExplainer.explain>`, returns a list of
        TimeSeries.
        """
        attention = self.get_explanation("attention")
        return attention

    def get_feature_importances(
        self,
    ) -> dict[str, pd.DataFrame | list[pd.DataFrame]]:
        """
        Returns the feature importances for the encoder, decoder and static covariates as pd.DataFrames.
        If multiple series were used in :func:`TFTExplainer.explain()
        <darts.explainability.tft_explainer.TFTExplainer.explain>`, returns a list of pd.DataFrames per importance.
        """
        return {comp: self.get_explanation(comp) for comp in self.feature_importances}

    def get_encoder_importance(self) -> pd.DataFrame | list[pd.DataFrame]:
        """
        Returns the time-dependent encoder importances as a pd.DataFrames.
        If multiple series were used in :func:`TFTExplainer.explain()
        <darts.explainability.tft_explainer.TFTExplainer.explain>`, returns a list of pd.DataFrames.
        """
        return self.get_explanation("encoder_importance")

    def get_decoder_importance(self) -> pd.DataFrame | list[pd.DataFrame]:
        """
        Returns the time-dependent decoder importances as a pd.DataFrames.
        If multiple series were used in :func:`TFTExplainer.explain()
        <darts.explainability.tft_explainer.TFTExplainer.explain>`, returns a list of pd.DataFrames.
        """
        return self.get_explanation("decoder_importance")

    def get_static_covariates_importance(
        self,
    ) -> pd.DataFrame | list[pd.DataFrame]:
        """
        Returns the numeric and categorical static covariates importances as a pd.DataFrames.
        If multiple series were used in :func:`TFTExplainer.explain()
        <darts.explainability.tft_explainer.TFTExplainer.explain>`, returns a list of pd.DataFrames.
        """
        return self.get_explanation("static_covariates_importance")
