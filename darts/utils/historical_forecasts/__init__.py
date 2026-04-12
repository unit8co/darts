"""
Utils for Historical Forecasting
--------------------------------

Utilities for generating and optimizing historical forecasts.
"""

from darts.utils._lazy import setup_lazy_imports

_LAZY_IMPORTS: dict[str, str] = {
    "_optimized_historical_forecasts_regression": (
        "darts.utils.historical_forecasts.optimized_historical_forecasts_regression"
    ),
    "_check_optimizable_historical_forecasts_global_models": "darts.utils.historical_forecasts.utils",
    "_get_historical_forecast_boundaries": "darts.utils.historical_forecasts.utils",
    "_historical_forecasts_general_checks": "darts.utils.historical_forecasts.utils",
    "_process_historical_forecast_input": "darts.utils.historical_forecasts.utils",
}

__all__, __getattr__, __dir__ = setup_lazy_imports(_LAZY_IMPORTS, __name__, globals())
