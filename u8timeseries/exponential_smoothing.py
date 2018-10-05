from .timeseries_model import TimeseriesModel
import statsmodels.tsa.holtwinters as hw


class ExponentialSmoothing(TimeseriesModel):

    def __init__(self, trend='additive', seasonal='additive', seasonal_periods=12):
        super(ExponentialSmoothing, self).__init__()
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.model = None

    def __str__(self):
        return 'Exponential smoothing'

    def fit(self, df, target_column, time_column=None, stepduration_str=None):
        super(ExponentialSmoothing, self).fit(df, target_column, time_column, stepduration_str)
        values = df[target_column].values
        self.model = hw.ExponentialSmoothing(values,
                                             trend=self.trend,
                                             seasonal=self.seasonal,
                                             seasonal_periods=self.seasonal_periods).fit()

    def predict(self, n):
        forecast = self.model.forecast(n)
        return self._build_forecast_df(forecast)
