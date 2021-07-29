"""
Auto-ARIMA
----------
"""

from pmdarima import AutoARIMA as PmdAutoARIMA
from typing import Optional

from .forecasting_model import DualCovariatesForecastingModel
from ..timeseries import TimeSeries
from ..logging import get_logger, raise_if

logger = get_logger(__name__)


class AutoARIMA(DualCovariatesForecastingModel):
    def __init__(self, *autoarima_args, **autoarima_kwargs):
        """ Auto-ARIMA

        This implementation is a thin wrapper around `pmdarima.txt AutoARIMA model
        <https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.AutoARIMA.html>`_,
        which provides functionality similar to R's `auto.arima
        <https://www.rdocumentation.org/packages/forecast/versions/7.3/topics/auto.arima>`_.

        This model supports the same parameters as the pmdarima.txt AutoARIMA model.
        See `pmdarima.txt documentation
        <https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.AutoARIMA.html>`_
        for an extensive documentation and a list of supported parameters.

        Parameters
        ----------
        autoarima_args
            Positional arguments for the pmdarima.txt AutoARIMA model
        autoarima_kwargs
            Keyword arguments for the pmdarima.txt AutoARIMA model
        """
        super().__init__()
        self.model = PmdAutoARIMA(*autoarima_args, **autoarima_kwargs)
        self.trend = self.model.trend

    def __str__(self):
        return 'Auto-ARIMA'

    def fit(self, series: TimeSeries, future_covariates: Optional[TimeSeries] = None):
        super().fit(series, future_covariates)
        series = self.training_series
        self.model.fit(series.values(),
                       X=future_covariates.values() if future_covariates else None)

    def predict(self, n: int,
                future_covariates: Optional[TimeSeries] = None,
                num_samples: int = 1):
        super().predict(n, future_covariates, num_samples)
        forecast = self.model.predict(n_periods=n,
                                      X=future_covariates.values() if future_covariates else None)
        return self._build_forecast_series(forecast)

    @property
    def min_train_series_length(self) -> int:
        return 30

    def _supports_range_index(self) -> bool:
        raise_if(self.trend and self.trend != "c",
            "'trend' is not None. Range indexing is not supported in that case.",
            logger
        )
        return True
