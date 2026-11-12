"""
Tabularization API
------------------

Helpers that convert time series data into tabular format with lagged features. The public helpers
consume a :class:`ForecastConfig` and resolve it against the components of the series they receive.
"""

from darts.utils.data.tabularization.features import LaggedFeatures
from darts.utils.data.tabularization.forecast_config import ForecastConfig
from darts.utils.data.tabularization.tabularization import (
    add_static_covariates_to_lagged_data,
    create_lagged_component_names,
    create_lagged_data,
    create_lagged_prediction_data,
    create_lagged_training_data,
    get_shared_times,
    get_shared_times_bounds,
    strided_moving_window,
)

__all__ = [
    "ForecastConfig",
    "LaggedFeatures",
    "add_static_covariates_to_lagged_data",
    "create_lagged_component_names",
    "create_lagged_data",
    "create_lagged_prediction_data",
    "create_lagged_training_data",
    "get_shared_times",
    "get_shared_times_bounds",
    "strided_moving_window",
]
