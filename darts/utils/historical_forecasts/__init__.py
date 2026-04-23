"""
Utils for Historical Forecasting
--------------------------------

Utilities for generating and optimizing historical forecasts.
"""

from typing import TYPE_CHECKING

from darts.utils._lazy import setup_lazy_imports

if TYPE_CHECKING:
    from darts.utils.historical_forecasts.optimized_historical_forecasts_regression import (
        _optimized_historical_forecasts_regression as _optimized_historical_forecasts_regression,
    )
    from darts.utils.historical_forecasts.utils import (
        _check_optimizable_historical_forecasts_global_models as _check_optimizable_historical_forecasts_global_models,
    )
    from darts.utils.historical_forecasts.utils import (
        _get_historical_forecast_boundaries as _get_historical_forecast_boundaries,
    )
    from darts.utils.historical_forecasts.utils import (
        _historical_forecasts_general_checks as _historical_forecasts_general_checks,
    )
    from darts.utils.historical_forecasts.utils import (
        _process_historical_forecast_input as _process_historical_forecast_input,
    )

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
