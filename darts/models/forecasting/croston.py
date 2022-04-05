import numpy as np
from statsforecast.models import croston_classic, croston_optimized

from darts.models.forecasting.forecasting_model import ForecastingModel
from darts.timeseries import TimeSeries


class Croston(ForecastingModel):
    def __init__(self, optimized=False):
        """ """
        super().__init__()
        self.method = croston_optimized if optimized else croston_classic

    def __str__(self):
        return "Croston"

    def fit(self, series: TimeSeries):
        super().fit(series)
        series._assert_univariate()
        series = self.training_series
        self.forecast_val = self.method(
            series.values(copy=False), h=1, future_xreg=None
        )
        return self

    def predict(
        self,
        n: int,
        num_samples: int = 1,
    ):
        super().predict(n, num_samples)
        values = np.tile(self.forecast_val, n)
        return self._build_forecast_series(values)

    @property
    def min_train_series_length(self) -> int:
        return 10

    def _supports_range_index(self) -> bool:
        return True

    def _is_probabilistic(self) -> bool:
        return False
