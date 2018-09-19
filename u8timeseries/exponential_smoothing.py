from .timeseries_model import TimeseriesModel
import statsmodels.tsa.holtwinters as hw


class ExponentialSmoothing(TimeseriesModel):

    def __init__(self, trend='additive', seasonal='additive', seasonal_periods=12):
        super(ExponentialSmoothing, self).__init__()
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.model = None

    def fit(self, df, target_column, time_column=None, periodicity_str=None):
        super(ExponentialSmoothing, self).fit(df, target_column, time_column, periodicity_str)
        values = df[target_column].values
        self.model = hw.ExponentialSmoothing(values,
                                             trend=self.trend,
                                             seasonal=self.trend,
                                             seasonal_periods=self.seasonal_periods).fit()

    def predict(self, n):
        forecast = self.model.forecast(n)
        return self._build_forecast_df(forecast)
