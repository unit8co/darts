"""
Utils for Historical Forecasting
--------------------------------

Utilities for generating and optimizing historical forecasts.
"""

import importlib

_LAZY_IMPORTS: dict[str, str] = {
    "_optimized_historical_forecasts_regression": (
        "darts.utils.historical_forecasts.optimized_historical_forecasts_regression"
    ),
    "_check_optimizable_historical_forecasts_global_models": "darts.utils.historical_forecasts.utils",
    "_get_historical_forecast_boundaries": "darts.utils.historical_forecasts.utils",
    "_historical_forecasts_general_checks": "darts.utils.historical_forecasts.utils",
    "_process_historical_forecast_input": "darts.utils.historical_forecasts.utils",
}

__all__ = list(_LAZY_IMPORTS.keys())


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        mod = importlib.import_module(_LAZY_IMPORTS[name])
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return __all__
