"""
ExplainabilityResult
--------------------

Contains the explainability results obtained from :func:`ForecastingModelExplainer.explain()`.
"""

from abc import ABC
from typing import Any, Dict, Optional, Sequence, Union

import shap

from darts import TimeSeries
from darts.logging import get_logger, raise_if, raise_if_not, raise_log

logger = get_logger(__name__)


class ExplainabilityResult(ABC):
    """
    Stores the explainability results of a :class:`ForecastingModelExplainer`
    with convenient access to the results.
    """

    def __init__(
        self,
        explained_forecasts: Union[
            Dict[str, TimeSeries],
            Dict[int, Dict[str, TimeSeries]],
            Sequence[Dict[int, Dict[str, TimeSeries]]],
        ],
    ):
        self.explained_forecasts = explained_forecasts
        if isinstance(self.explained_forecasts, list):
            raise_if_not(
                isinstance(self.explained_forecasts[0], dict),
                "The explained_forecasts Sequence must consist of dicts.",
                logger,
            )
            raise_if_not(
                all(isinstance(key, int) for key in self.explained_forecasts[0].keys()),
                "The explained_forecasts dict Sequence must have all integer keys.",
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
            elif all(isinstance(key, str) for key in self.explained_forecasts.keys()):
                self.available_horizons = []
                self.available_components = list(self.explained_forecasts.keys())
            else:
                raise_log(
                    ValueError(
                        "The explained_forecasts dictionary must have all integer or all string keys."
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
        self, horizon: Optional[int] = None, component: Optional[str] = None
    ) -> Union[TimeSeries, Sequence[TimeSeries]]:
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
        attr: Union[Dict[int, Dict[str, Any]], Sequence[Dict[int, Dict[str, Any]]]],
        horizon: Optional[int] = None,
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
        self._validate_input_for_querying_explainability_result(horizon, component)
        if component is None:
            component = self.available_components[0]

        if isinstance(attr, list):
            return [attr[i][horizon][component] for i in range(len(attr))]
        elif all(isinstance(key, int) for key in attr.keys()):
            return attr[horizon][component]
        elif all(isinstance(key, str) for key in attr.keys()):
            return attr[component]
        else:
            raise_log(
                ValueError(
                    "Something went wrong. ExplainabilityResult got instantiated with an unexpected type."
                ),
                logger,
            )

    def _validate_input_for_querying_explainability_result(
        self, horizon: Optional[int] = None, component: Optional[str] = None
    ) -> None:
        """
        Helper that validates the input parameters of a method that queries the ExplainabilityResult.

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

        # validate horizon argument
        if horizon is not None:
            raise_if(
                len(self.available_horizons) == 0,
                "The horizon parameter can not be used for a model where all time horizons are saved in the component.",
                logger,
            )

            raise_if_not(
                horizon in self.available_horizons,
                f"Horizon {horizon} is not available. Available horizons are: {self.available_horizons}",
                logger,
            )


class ShapExplainabilityResult(ExplainabilityResult):
    """
    Stores the explainability results of a :class:`ShapExplainer`
    with convenient access to the results. It extends the :class:`ExplainabilityResult` and carries additional
    information specific to the Shap explainers. In particular, in addition to the `explained_forecasts` (which in
    the case of the :class:`ShapExplainer` are the shap values), it also provides access to the corresponding
    `feature_values` and the underlying `shap.Explanation` object.
    """

    def __init__(
        self,
        explained_forecasts: Union[
            Dict[int, Dict[str, TimeSeries]],
            Sequence[Dict[int, Dict[str, TimeSeries]]],
        ],
        feature_values: Union[
            Dict[int, Dict[str, TimeSeries]],
            Sequence[Dict[int, Dict[str, TimeSeries]]],
        ],
        shap_explanation_object: Union[
            Dict[int, Dict[str, shap.Explanation]],
            Sequence[Dict[int, Dict[str, shap.Explanation]]],
        ],
    ):
        super().__init__(explained_forecasts)
        self.feature_values = feature_values
        self.shap_explanation_object = shap_explanation_object

    def get_feature_values(
        self, horizon: int, component: Optional[str] = None
    ) -> Union[TimeSeries, Sequence[TimeSeries]]:
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
    ) -> Union[shap.Explanation, Sequence[shap.Explanation]]:
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


# class TFTExplainabilityResult(ExplainabilityResult):
#     def __init__(
#         self,
#         attention_series: Union[
#             Dict[int, Dict[str, TimeSeries]],
#             Sequence[Dict[int, Dict[str, TimeSeries]]],
#         ],
#         encoder_importance: Union[
#             Dict[int, Dict[str, TimeSeries]],
#             Sequence[Dict[int, Dict[str, TimeSeries]]],
#         ],
#         decoder_importance: Union[
#             Dict[int, Dict[str, TimeSeries]],
#             Sequence[Dict[int, Dict[str, TimeSeries]]],
#         ],
#         static_importance: Union[
#             Dict[int, Dict[str, TimeSeries]],
#             Sequence[Dict[int, Dict[str, TimeSeries]]],
#         ],
#     ):
#         super().__init__(explained_forecasts)
#         self.feature_values = feature_values
#         self.shap_explanation_object = shap_explanation_object
