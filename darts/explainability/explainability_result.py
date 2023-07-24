"""
ExplainabilityResult
--------------------

Contains the explainability results obtained from :func:`ForecastingModelExplainer.explain()`.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import shap

from darts import TimeSeries
from darts.logging import get_logger, raise_if, raise_if_not, raise_log

logger = get_logger(__name__)


class ExplainabilityResult(ABC):
    """
    Abstract class for explainability results of a :class:`ForecastingModelExplainer`.
    The subclasses should implement convenient access to explanations.
    """

    @abstractmethod
    def get_explanation(self, *args, **kwargs):
        """Returns one or multiple explanations based on some input parameters."""
        pass


class ComponentBasedExplainabilityResult:
    """Explainability result for general component objects.
    The explained components can describe anything.
    """

    def __init__(
        self,
        explained_components: Union[Dict[str, Any], List[Dict[str, Any]]],
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

    def get_explanation(self, component: Optional[str] = None) -> Union[Any, List[Any]]:
        """
        Returns one or several explanations for a given component.

        Parameters
        ----------
        component
            The component for which to return the explanation. Does not
            need to be specified for univariate series.
        """
        return self._query_explainability_result(self.explained_components, component)

    def _query_explainability_result(
        self,
        attr: Union[Dict[str, Any], List[Dict[str, Any]]],
        component: Optional[str] = None,
    ) -> Any:
        """
        Helper that extracts and returns the explainability result attribute for a given component.

        Parameters
        ----------
        attr
            An explainability result attribute from which to extract the component.
        component
            The component for which to return the content of the attribute. Does not
            need to be specified when attribute is a single dict.
        """
        component = self._validate_input_for_querying_explainability_result(component)

        if isinstance(attr, list):
            return [attr_[component] for attr_ in attr]
        else:
            return attr[component]

    def _validate_input_for_querying_explainability_result(
        self, component: Optional[str] = None
    ) -> str:
        """
        Helper that validates the input parameters of a method that queries the ComponentBasedExplainabilityResult.

        Parameters
        ----------
        component
            The component for which to return the explanation. Does not
            need to be specified for univariate series.
        """
        # validate component argument
        raise_if(
            component is None and len(self.explained_components) > 1,
            "The component parameter is required when the `ExplainabilityResult` has more than one component.",
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


class HorizonBasedExplainabilityResult(ExplainabilityResult):
    """
    Stores the explainability results of a :class:`ForecastingModelExplainer`
    with convenient access to the horizon based results.
    """

    def __init__(
        self,
        explained_forecasts: Union[
            Dict[str, Union[TimeSeries, pd.DataFrame, np.ndarray]],
            List[Dict[str, Union[TimeSeries, pd.DataFrame, np.ndarray]]],
            Dict[int, Dict[str, TimeSeries]],
            List[Dict[int, Dict[str, TimeSeries]]],
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
    ) -> Union[TimeSeries, List[TimeSeries]]:
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
        attr: Union[Dict[int, Dict[str, Any]], List[Dict[int, Dict[str, Any]]]],
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
        component = self._validate_input_for_querying_explainability_result(
            horizon, component
        )

        if isinstance(attr, list):
            return [attr[i][horizon][component] for i in range(len(attr))]
        elif all(isinstance(key, int) for key in attr.keys()):
            return attr[horizon][component]
        elif all(isinstance(key, str) for key in attr.keys()):
            return attr[component]
        else:
            raise_log(
                ValueError(
                    "Something went wrong. HorizonBasedExplainabilityResult got instantiated with an unexpected type."
                ),
                logger,
            )

    def _validate_input_for_querying_explainability_result(
        self, horizon: Optional[int] = None, component: Optional[str] = None
    ) -> str:
        """
        Helper that validates the input parameters of a method that queries the HorizonBasedExplainabilityResult.

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
        return component


class ShapExplainabilityResult(HorizonBasedExplainabilityResult):
    """
    Stores the explainability results of a :class:`ShapExplainer`
    with convenient access to the results. It extends the :class:`HorizonBasedExplainabilityResult` and carries
    additional information specific to the Shap explainers. In particular, in addition to the `explained_forecasts`
    (which in the case of the :class:`ShapExplainer` are the shap values), it also provides access to the corresponding
    `feature_values` and the underlying `shap.Explanation` object.
    """

    def __init__(
        self,
        explained_forecasts: Union[
            Dict[int, Dict[str, TimeSeries]],
            List[Dict[int, Dict[str, TimeSeries]]],
        ],
        feature_values: Union[
            Dict[int, Dict[str, TimeSeries]],
            List[Dict[int, Dict[str, TimeSeries]]],
        ],
        shap_explanation_object: Union[
            Dict[int, Dict[str, shap.Explanation]],
            List[Dict[int, Dict[str, shap.Explanation]]],
        ],
    ):
        super().__init__(explained_forecasts)
        self.feature_values = feature_values
        self.shap_explanation_object = shap_explanation_object

    def get_feature_values(
        self, horizon: int, component: Optional[str] = None
    ) -> Union[TimeSeries, List[TimeSeries]]:
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
    ) -> Union[shap.Explanation, List[shap.Explanation]]:
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
    def __init__(
        self,
        explanations: Union[
            Dict[str, Any],
            List[Dict[str, Any]],
        ],
    ):
        super().__init__(explanations)
        self.feature_importances = [
            "encoder_importance",
            "decoder_importance",
            "static_covariates_importance",
        ]

    def get_attention(self) -> Union[TimeSeries, List[TimeSeries]]:
        """
        Returns the time-dependent attention on the encoder and decoder for a given `horizon`.
        The time index ranges from the prediction series' start time - input_chunk-length and ends
        at the prediction series' end time. If multiple series were used when calling
        `TFTExplainer.explain()`, returns a list of TimeSeries.

        Parameters
        ----------
        horizon
            Optionally, an integer or list of integers for which to extract the attention.
            Maximum value corresponds to either `output_chunk_length` of the trained `TFTModel` or
            the maximum used when calling `TFTExplainer.explain()`
        """
        attention = self.get_explanation("attention")
        return attention

    def get_feature_importances(
        self,
    ) -> Dict[str, Union[pd.DataFrame, List[pd.DataFrame]]]:
        """
        Returns the feature importances for the encoder, decoder and static covariates as pd.DataFrames.
        If multiple series were used in `TFTExplainer.explain()`, returns a list of pd.DataFrames per
        importance.
        """
        return {comp: self.get_explanation(comp) for comp in self.feature_importances}

    def get_encoder_importance(self) -> Union[pd.DataFrame, List[pd.DataFrame]]:
        """
        Returns the time-dependent encoder importances as a pd.DataFrames.
        If multiple series were used in `TFTExplainer.explain()`, returns a list of pd.DataFrames.
        """
        return self.get_explanation("encoder_importance")

    def get_decoder_importance(self) -> Union[pd.DataFrame, List[pd.DataFrame]]:
        """
        Returns the time-dependent decoder importances as a pd.DataFrames.
        If multiple series were used in `TFTExplainer.explain()`, returns a list of pd.DataFrames.
        """
        return self.get_explanation("decoder_importance")

    def get_static_covariates_importance(
        self,
    ) -> Union[pd.DataFrame, List[pd.DataFrame]]:
        """
        Returns the numeric and categorical static covariates importances as a pd.DataFrames.
        If multiple series were used in `TFTExplainer.explain()`, returns a list of pd.DataFrames.
        """
        return self.get_explanation("static_covariates_importance")
