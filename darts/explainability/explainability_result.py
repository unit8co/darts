"""
Explainability Result
--------------------

Contains the explainability results obtained from :func:`_ForecastingModelExplainer.explain()
<darts.explainability.explainability._ForecastingModelExplainer.explain>`.

- :class:`ShapExplainabilityResult <ShapExplainabilityResult>` for :class:`ShapExplainer
  <darts.explainability.shap_explainer.ShapExplainer>`
- :class:`TFTExplainabilityResult <TFTExplainabilityResult>` for :class:`TFTExplainer
  <darts.explainability.tft_explainer.TFTExplainer>`
- :class:`ComponentBasedExplainabilityResult <ComponentBasedExplainabilityResult>` for component based explainability
  results
- :class:`HorizonBasedExplainabilityResult <HorizonBasedExplainabilityResult>` for horizon based explainability results
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Union

import pandas as pd
import shap

from darts import TimeSeries
from darts.logging import get_logger, raise_if, raise_if_not, raise_log

logger = get_logger(__name__)


class _ExplainabilityResult(ABC):
    """
    Abstract class for explainability results of a :class:`_ForecastingModelExplainer`.
    The subclasses should implement convenient access to explanations.
    """

    @abstractmethod
    def get_explanation(self, *args, **kwargs):
        """Returns one or multiple explanations based on some input parameters."""
        pass


class ComponentBasedExplainabilityResult(_ExplainabilityResult):
    """Explainability result for general component objects.
    The explained components can describe anything.

    Example
    -------
    >>> explainer = SomeComponentBasedExplainer(model)
    >>> explain_results = explainer.explain()
    >>> output = explain_results.get_explanation(component="some_component")
    """

    def __init__(
        self,
        explained_components: Union[dict[str, Any], list[dict[str, Any]]],
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
        self.available_components = comps_available

    def get_explanation(self, component) -> Union[Any, list[Any]]:
        """
        Returns one or several explanations for a given component.

        Parameters
        ----------
        component
            The component for which to return the explanation.
        """
        return self._query_explainability_result(self.explained_components, component)

    def _query_explainability_result(
        self,
        attr: Union[dict[str, Any], list[dict[str, Any]]],
        component: str,
    ) -> Any:
        """
        Helper that extracts and returns the explainability result attribute for a given component.

        Parameters
        ----------
        attr
            An explainability result attribute from which to extract the component.
        component
            The component for which to return the content of the attribute.
        """
        component = self._validate_input_for_querying_explainability_result(component)
        if isinstance(attr, list):
            return [attr_[component] for attr_ in attr]
        else:
            return attr[component]

    def _validate_input_for_querying_explainability_result(self, component) -> str:
        """
        Helper that validates the input parameters of a method that queries the `ComponentBasedExplainabilityResult`.

        Parameters
        ----------
        component
            The component for which to return the explanation. Does not
            need to be specified for univariate series.
        """
        # validate component argument
        raise_if(
            component is None and len(self.explained_components) > 1,
            f"The component parameter is required when the `{self.__class__.__name__}` has more than one component.",
            logger,
        )

        if component is None:
            component = self.available_components[0]

        raise_if_not(
            component in self.available_components,
            f"Component {component} is not available. Available components are: {self.available_components}",
            logger,
        )
        return component


class HorizonBasedExplainabilityResult(_ExplainabilityResult):
    """
    Stores the explainability results of a :class:`_ForecastingModelExplainer
    <darts.explainability.explainability._ForecastingModelExplainer>` with convenient access to the horizon
    based results.

    The result is a multivariate `TimeSeries` instance containing the 'explanation' for the (horizon, target_component)
    forecast at any timestamp forecastable corresponding to the foreground `TimeSeries` input.

    The component name convention of this multivariate `TimeSeries` is:
    ``"{name}_{type_of_cov}_lag_{idx}"``, where:

    - ``{name}`` is the component name from the original foreground series (target, past, or future).
    - ``{type_of_cov}`` is the covariates type. It can take 3 different values:
      ``"target"``, ``"past_cov"`` or ``"future_cov"``.
    - ``{idx}`` is the lag index.

    Example
    -------

    Say we have a model with 2 target components named ``"T_0"`` and ``"T_1"``,
    3 past covariates with default component names ``"0"``, ``"1"``, and ``"2"``,
    and one future covariate with default component name ``"0"``.
    Also, ``horizons = [1, 2]``.
    The model is a regression model, with ``lags = 3``, ``lags_past_covariates=[-1, -3]``,
    ``lags_future_covariates = [0]``.

    We provide `foreground_series`, `foreground_past_covariates`, `foreground_future_covariates` each of length 5.

    >>> explainer = SomeHorizonBasedExplainer(model)
    >>> explain_results = explainer.explain(
    >>>     foreground_series=foreground_series,
    >>>     foreground_past_covariates=foreground_past_covariates,
    >>>     foreground_future_covariates=foreground_future_covariates,
    >>>     horizons=[1, 2],
    >>>     target_names=["T_0", "T_1"]
    >>> )
    >>> output = explain_results.get_explanation(horizon=1, target="T_1")

    Then the method returns a multivariate TimeSeries containing the *explanations* of
    the corresponding `_ForecastingModelExplainer`, with the following component names:

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
    """

    def __init__(
        self,
        explained_forecasts: Union[
            dict[int, dict[str, TimeSeries]],
            list[dict[int, dict[str, TimeSeries]]],
        ],
    ):
        self.explained_forecasts = explained_forecasts
        if isinstance(self.explained_forecasts, list):
            raise_if_not(
                isinstance(self.explained_forecasts[0], dict),
                "The explained_forecasts list must consist of dicts.",
                logger,
            )
            raise_if_not(
                all(isinstance(key, int) for key in self.explained_forecasts[0].keys()),
                "The explained_forecasts dict list must have all integer keys.",
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
                        "The explained_forecasts dictionary must have all integer keys."
                    ),
                    logger,
                )
        else:
            raise_log(
                ValueError(
                    "The explained_forecasts must be a dictionary or a list of dictionaries."
                ),
                logger,
            )

    def get_explanation(
        self, horizon: int, component: Optional[str] = None
    ) -> Union[TimeSeries, list[TimeSeries]]:
        """
        Returns one or several `TimeSeries` representing the explanations
        for a given horizon and component.

        Parameters
        ----------
        horizon
            The horizon for which to return the explanation.
        component
            The component for which to return the explanation. Does not
            need to be specified for univariate series.
        """
        return self._query_explainability_result(
            self.explained_forecasts, horizon, component
        )

    def _query_explainability_result(
        self,
        attr: Union[dict[int, dict[str, Any]], list[dict[int, dict[str, Any]]]],
        horizon: int,
        component: Optional[str] = None,
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
            The component for which to return the content of the attribute. Does not
            need to be specified for univariate series.
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
        self, horizon: int, component: Optional[str] = None
    ) -> str:
        """
        Helper that validates the input parameters of a method that queries the `HorizonBasedExplainabilityResult`.

        Parameters
        ----------
        horizon
            The horizon for which to return the explanation.
        component
            The component for which to return the explanation. Does not
            need to be specified for univariate series.
        """
        # validate component argument
        raise_if(
            component is None and len(self.available_components) > 1,
            "The component parameter is required when the model has more than one component.",
            logger,
        )

        if component is None:
            component = self.available_components[0]

        raise_if_not(
            component in self.available_components,
            f"Component {component} is not available. Available components are: {self.available_components}",
            logger,
        )

        raise_if_not(
            horizon in self.available_horizons,
            f"Horizon {horizon} is not available. Available horizons are: {self.available_horizons}",
            logger,
        )
        return component


class ShapExplainabilityResult(HorizonBasedExplainabilityResult):
    """
    Stores the explainability results of a :class:`ShapExplainer <darts.explainability.shap_explainer.ShapExplainer>`
    with convenient access to the results. It extends the :class:`HorizonBasedExplainabilityResult
    <HorizonBasedExplainabilityResult>` and carries additional information specific to the Shap explainers.
    In particular, in addition to the `explained_forecasts` (which in the case of the `ShapExplainer` are the
    shap values), it also provides access to the corresponding `feature_values` and the underlying `shap.Explanation`
    object.

    - :func:`get_explanation() <ShapExplainabilityResult.get_explanation>`: explained forecast for a given horizon
      (and target component)
    - :func:`get_feature_values() <ShapExplainabilityResult.get_feature_values>`: feature values for a given horizon
      (and target component).
    - :func:`get_shap_explanation_object() <ShapExplainabilityResult.get_shap_explanation_object>`: `shap.Explanation`
      object for a given horizon (and target component).

    Examples
    --------
    >>> explainer = ShapExplainer(model)  # requires `background` if model was trained on multiple series
    >>> explain_results = explainer.explain()
    >>> exlained_fc = explain_results.get_explanation(horizon=1)
    >>> feature_values = explain_results.get_feature_values(horizon=1)
    >>> shap_objects = explain_results.get_shap_explanation_objects(horizon=1)
    """

    def __init__(
        self,
        explained_forecasts: Union[
            dict[int, dict[str, TimeSeries]],
            list[dict[int, dict[str, TimeSeries]]],
        ],
        feature_values: Union[
            dict[int, dict[str, TimeSeries]],
            list[dict[int, dict[str, TimeSeries]]],
        ],
        shap_explanation_object: Union[
            dict[int, dict[str, shap.Explanation]],
            list[dict[int, dict[str, shap.Explanation]]],
        ],
    ):
        super().__init__(explained_forecasts)
        self.feature_values = feature_values
        self.shap_explanation_object = shap_explanation_object

    def get_feature_values(
        self, horizon: int, component: Optional[str] = None
    ) -> Union[TimeSeries, list[TimeSeries]]:
        """
        Returns one or several `TimeSeries` representing the feature values
        for a given horizon and component.

        Parameters
        ----------
        horizon
            The horizon for which to return the feature values.
        component
            The component for which to return the feature values. Does not
            need to be specified for univariate series.
        """
        return self._query_explainability_result(
            self.feature_values, horizon, component
        )

    def get_shap_explanation_object(
        self, horizon: int, component: Optional[str] = None
    ) -> Union[shap.Explanation, list[shap.Explanation]]:
        """
        Returns the underlying `shap.Explanation` object for a given horizon and component.

        Parameters
        ----------
        horizon
            The horizon for which to return the `shap.Explanation` object.
        component
            The component for which to return the `shap.Explanation` object. Does not
            need to be specified for univariate series.
        """
        return self._query_explainability_result(
            self.shap_explanation_object, horizon, component
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
    >>> explainer = TFTExplainer(model)  # requires `background` if model was trained on multiple series
    >>> explain_results = explainer.explain()
    >>> attention = explain_results.get_attention()
    >>> importances = explain_results.get_feature_importances()
    >>> encoder_importance = explain_results.get_encoder_importance()
    >>> decoder_importance = explain_results.get_decoder_importance()
    >>> static_covariates_importance = explain_results.get_static_covariates_importance()
    """

    def __init__(
        self,
        explanations: Union[
            dict[str, Any],
            list[dict[str, Any]],
        ],
    ):
        super().__init__(explanations)
        self.feature_importances = [
            "encoder_importance",
            "decoder_importance",
            "static_covariates_importance",
        ]

    def get_attention(self) -> Union[TimeSeries, list[TimeSeries]]:
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
    ) -> dict[str, Union[pd.DataFrame, list[pd.DataFrame]]]:
        """
        Returns the feature importances for the encoder, decoder and static covariates as pd.DataFrames.
        If multiple series were used in :func:`TFTExplainer.explain()
        <darts.explainability.tft_explainer.TFTExplainer.explain>`, returns a list of pd.DataFrames per importance.
        """
        return {comp: self.get_explanation(comp) for comp in self.feature_importances}

    def get_encoder_importance(self) -> Union[pd.DataFrame, list[pd.DataFrame]]:
        """
        Returns the time-dependent encoder importances as a pd.DataFrames.
        If multiple series were used in :func:`TFTExplainer.explain()
        <darts.explainability.tft_explainer.TFTExplainer.explain>`, returns a list of pd.DataFrames.
        """
        return self.get_explanation("encoder_importance")

    def get_decoder_importance(self) -> Union[pd.DataFrame, list[pd.DataFrame]]:
        """
        Returns the time-dependent decoder importances as a pd.DataFrames.
        If multiple series were used in :func:`TFTExplainer.explain()
        <darts.explainability.tft_explainer.TFTExplainer.explain>`, returns a list of pd.DataFrames.
        """
        return self.get_explanation("decoder_importance")

    def get_static_covariates_importance(
        self,
    ) -> Union[pd.DataFrame, list[pd.DataFrame]]:
        """
        Returns the numeric and categorical static covariates importances as a pd.DataFrames.
        If multiple series were used in :func:`TFTExplainer.explain()
        <darts.explainability.tft_explainer.TFTExplainer.explain>`, returns a list of pd.DataFrames.
        """
        return self.get_explanation("static_covariates_importance")
