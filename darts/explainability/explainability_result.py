"""
ExplainabilityResult
--------------------

Contains the explainability results obtained from :func:`ForecastingModelExplainer.explain()`.
"""

from abc import ABC
from typing import Dict, Optional, Sequence, Union

from numpy import integer

from darts import TimeSeries
from darts.logging import get_logger, raise_if, raise_if_not

logger = get_logger(__name__)


class ExplainabilityResult(ABC):
    """
    Stores the explainability results of a :class:`ForecastingModelExplainer`
    with convenient access to the results.
    """

    def __init__(
        self,
        explained_forecasts: Union[
            Dict[integer, Dict[str, TimeSeries]],
            Sequence[Dict[integer, Dict[str, TimeSeries]]],
        ],
    ):

        self.explained_forecasts = explained_forecasts
        if isinstance(self.explained_forecasts, list):
            self.available_horizons = list(self.explained_forecasts[0].keys())
            h_0 = self.available_horizons[0]
            self.available_components = list(self.explained_forecasts[0][h_0].keys())
        else:
            self.available_horizons = list(self.explained_forecasts.keys())
            h_0 = self.available_horizons[0]
            self.available_components = list(self.explained_forecasts[h_0].keys())

    def get_explanation(
        self, horizon: int, component: Optional[str] = None
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

        raise_if(
            component is None and len(self.available_components) > 1,
            ValueError(
                "The component parameter is required when the model has more than one component."
            ),
            logger,
        )

        if component is None:
            component = self.available_components[0]

        raise_if_not(
            horizon in self.available_horizons,
            "Horizon {} is not available. Available horizons are: {}".format(
                horizon, self.available_horizons
            ),
        )

        raise_if_not(
            component in self.available_components,
            "Component {} is not available. Available components are: {}".format(
                component, self.available_components
            ),
        )

        if isinstance(self.explained_forecasts, list):
            return [
                self.explained_forecasts[i][horizon][component]
                for i in range(len(self.explained_forecasts))
            ]
        else:
            return self.explained_forecasts[horizon][component]
