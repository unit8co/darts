from .timeseries_model import TimeseriesModel


class KthValueAgoBaseline(TimeseriesModel):
    """
    A baseline model that always predict value of K time steps ago
    E.g., always predict last observed value when K=1, or last year value with K=12 (and monthly data)
    """

    def __init__(self, K=1):
        super(KthValueAgoBaseline, self).__init__()
        self.kth_value_ago = None
        self.K = K

    def fit(self, df, target_column, time_column=None, stepduration_str=None):
        super(KthValueAgoBaseline, self).fit(df, target_column, time_column, stepduration_str)
        values = df[target_column].values
        assert len(values) >= self.K, 'The time series has to contain at least K={} points'.format(self.K)
        self.kth_value_ago = values[-self.K]

    def predict(self, n):
        forecast = [self.kth_value_ago] * n
        return self._build_forecast_df(forecast)
