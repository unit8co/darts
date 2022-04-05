"""
Auto-ARIMA
----------
"""

from typing import Optional

import numpy as np
from pmdarima import AutoARIMA as PmdAutoARIMA
from statsforecast.arima import AutoARIMA as SFAutoARIMA

from darts.logging import get_logger, raise_if
from darts.models.forecasting.forecasting_model import DualCovariatesForecastingModel
from darts.timeseries import TimeSeries

logger = get_logger(__name__)


class AutoARIMA(DualCovariatesForecastingModel):
    def __init__(self, *autoarima_args, **autoarima_kwargs):
        """Auto-ARIMA

        This implementation is a thin wrapper around `pmdarima AutoARIMA model
        <https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.AutoARIMA.html>`_,
        which provides functionality similar to R's `auto.arima
        <https://www.rdocumentation.org/packages/forecast/versions/7.3/topics/auto.arima>`_.

        This model supports the same parameters as the pmdarima AutoARIMA model.
        See `pmdarima documentation
        <https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.AutoARIMA.html>`_
        for an extensive documentation and a list of supported parameters.

        Parameters
        ----------
        autoarima_args
            Positional arguments for the pmdarima.AutoARIMA model
        autoarima_kwargs
            Keyword arguments for the pmdarima.AutoARIMA model
        """
        super().__init__()
        self.model = PmdAutoARIMA(*autoarima_args, **autoarima_kwargs)
        self.trend = self.model.trend

    def __str__(self):
        return "Auto-ARIMA"

    def _fit(self, series: TimeSeries, future_covariates: Optional[TimeSeries] = None):
        super()._fit(series, future_covariates)
        series = self.training_series
        self.model.fit(
            series.values(), X=future_covariates.values() if future_covariates else None
        )
        return self

    def _predict(
        self,
        n: int,
        future_covariates: Optional[TimeSeries] = None,
        num_samples: int = 1,
    ):
        super()._predict(n, future_covariates, num_samples)
        forecast = self.model.predict(
            n_periods=n, X=future_covariates.values() if future_covariates else None
        )
        return self._build_forecast_series(forecast)

    @property
    def min_train_series_length(self) -> int:
        return 10

    def _supports_range_index(self) -> bool:
        raise_if(
            self.trend and self.trend != "c",
            "'trend' is not None. Range indexing is not supported in that case.",
            logger,
        )
        return True


class AutoARIMASF(DualCovariatesForecastingModel):
    def __init__(self, *autoarima_args, **autoarima_kwargs):
        """Auto-ARIMA based on `Statsforecasts package
        <https://github.com/Nixtla/statsforecast>`_.

        This implementation can perform faster than the ``AutoARIMA`` model,
        but typically requires more time on the first call, because it relies
        on Numba and jit compilation.

        It is probabilistic, whereas ``AutoARIMA`` is not.

        TODO: auto-seasonality

        Parameters
        ----------
        autoarima_args
            Positional arguments for ``statsforecasts.arima.AutoARIMA``.
        autoarima_kwargs
            Keyword arguments for ``statsforecasts.arima.AutoARIMA``.
        """
        super().__init__()
        self.model = SFAutoARIMA(*autoarima_args, **autoarima_kwargs)

    def __str__(self):
        return "Auto-ARIMA-Statsforecasts"

    def _fit(self, series: TimeSeries, future_covariates: Optional[TimeSeries] = None):
        super()._fit(series, future_covariates)
        series._assert_univariate()
        series = self.training_series
        self.model.fit(
            series.values(copy=False).flatten(),
            X=future_covariates.values(copy=False) if future_covariates else None,
        )
        return self

    def _predict(
        self,
        n: int,
        future_covariates: Optional[TimeSeries] = None,
        num_samples: int = 1,
    ):
        super()._predict(n, future_covariates, num_samples)
        forecast_df = self.model.predict(
            h=n,
            X=future_covariates.values(copy=False) if future_covariates else None,
            level=68,  # ask one std for the confidence interval. Note, we're limited to int...
        )

        mu = forecast_df["mean"].values
        if num_samples > 1:
            std = 2 * (forecast_df["hi_68%"].values - mu)
            samples = np.random.normal(loc=mu, scale=std, size=(num_samples, n)).T
            samples = np.expand_dims(samples, axis=1)
        else:
            samples = mu

        return self._build_forecast_series(samples)

    @property
    def min_train_series_length(self) -> int:
        return 10

    def _supports_range_index(self) -> bool:
        return True

    def _is_probabilistic(self) -> bool:
        return True
