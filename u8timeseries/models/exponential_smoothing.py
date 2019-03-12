from .autoregressive_model import AutoRegressiveModel
from ..timeseries import TimeSeries
import statsmodels.tsa.holtwinters as hw


class ExponentialSmoothing(AutoRegressiveModel):

    def __init__(self, trend='additive', seasonal='additive', seasonal_periods=12):
        super().__init__()
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.model = None

    def __str__(self):
        return 'Exponential smoothing'

    def fit(self, series: TimeSeries):
        super().fit(series)
        self.model = hw.ExponentialSmoothing(series.values(),
                                             trend=self.trend,
                                             seasonal=self.trend,
                                             seasonal_periods=self.seasonal_periods).fit()

    def predict(self, n):
        super().predict(n)
        forecast = self.model.forecast(n)
        return self._build_forecast_series(forecast)
