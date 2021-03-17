"""
LightGBM
----------------------
"""

from lightgbm import LGBMRegressor

from darts.models.forecasting_model import ForecastingModel
from darts.timeseries import TimeSeries
from darts.logging import get_logger

logger = get_logger(__name__)


class LightGBM(ForecastingModel):
    def __init__(self, *lightgbm_args, **lightgbm_kwargs):
        """ LightGBM

        This implementation is a thin wrapper around lightgbm Python library

        Parameters
        ----------
        lightgbm_args
            Positional arguments for the lightgbm model
        lightgbm_kwargs
            Keyword arguments for the lightgbm model
        """

        super().__init__()
        self.model = LGBMRegressor(*lightgbm_args, **lightgbm_kwargs)

    def __str__(self):
        return 'LightGBM'

    def fit(self, series: TimeSeries):
        super().fit(series)
        series = self.training_series

        X = series.time_index()
        y = series.univariate_values()

        self.model.fit(X, y)
        self.last_tick = X[-1]

    def predict(self, n):
        super().predict(n)
        forecast = self.model.predict(list(range(self.last_tick+1, self.last_tick + n+1)))
        return self._build_forecast_series(forecast)

    @property
    def min_train_series_length(self) -> int:
        return 30


if __name__ == "__main__":
    pass