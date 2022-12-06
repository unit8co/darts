"""
StatsForecastAutoARIMA
-----------
"""

from typing import Optional

import numpy as np
from statsforecast.models import AutoARIMA as SFAutoARIMA

from darts import TimeSeries
from darts.models.forecasting.forecasting_model import (
    FutureCovariatesLocalForecastingModel,
)


class StatsForecastAutoARIMA(FutureCovariatesLocalForecastingModel):
    def __init__(self, *autoarima_args, **autoarima_kwargs):
        """Auto-ARIMA based on `Statsforecasts package
        <https://github.com/Nixtla/statsforecast>`_.

        This implementation can perform faster than the :class:`AutoARIMA` model,
        but typically requires more time on the first call, because it relies
        on Numba and jit compilation.

        It is probabilistic, whereas :class:`AutoARIMA` is not.

        We refer to the `statsforecast AutoARIMA documentation
        <https://nixtla.github.io/statsforecast/models.html#arima-methods>`_
        for the documentation of the arguments.

        Parameters
        ----------
        autoarima_args
            Positional arguments for ``statsforecasts.models.AutoARIMA``.
        autoarima_kwargs
            Keyword arguments for ``statsforecasts.models.AutoARIMA``.

        Examples
        --------
        >>> from darts.models import StatsForecastAutoARIMA
        >>> from darts.datasets import AirPassengersDataset
        >>> series = AirPassengersDataset().load()
        >>> model = StatsForecastAutoARIMA(season_length=12)
        >>> model.fit(series[:-36])
        >>> pred = model.predict(36, num_samples=100)
        """
        super().__init__()
        self.model = SFAutoARIMA(*autoarima_args, **autoarima_kwargs)

    def __str__(self):
        return "Auto-ARIMA-Statsforecasts"

    def _fit(self, series: TimeSeries, future_covariates: Optional[TimeSeries] = None):
        super()._fit(series, future_covariates)
        self._assert_univariate(series)
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
        verbose: bool = False,
    ):
        super()._predict(n, future_covariates, num_samples)
        forecast_df = self.model.predict(
            h=n,
            X=future_covariates.values(copy=False) if future_covariates else None,
            level=(68.27,),  # ask one std for the confidence interval.
        )

        mu = forecast_df["mean"]
        if num_samples > 1:
            std = forecast_df["hi-68.27"] - mu
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
