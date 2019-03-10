from u8timeseries.models.autoregressive_model import AutoRegressiveModel
import pandas as pd

import fbprophet


class Prophet(AutoRegressiveModel):

    def __init__(self, weekly_seasonality=False, daily_seasonality=False):
        super(Prophet, self).__init__()

        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.model = None

    def __str__(self):
        return 'Prophet'

    def fit(self, series):
        super(Prophet, self).fit(series)

        in_df = pd.DataFrame(data={
            'ds': series.time_index(),
            'y': series.values()
        })

        # TODO: user-provided seasonalities, or "auto" based on stepduration
        self.model = fbprophet.Prophet(weekly_seasonality=self.weekly_seasonality,
                                       daily_seasonality=self.daily_seasonality)
        self.model.fit(in_df)

    def predict(self, n):
        new_dates = self._generate_new_dates(n)
        new_dates_df = pd.DataFrame(data={'ds': new_dates})

        predictions = self.model.predict(new_dates_df)

        forecast = predictions['yhat'][-n:].values
        conf_lo = predictions['yhat_lower'][-n:].values
        conf_hi = predictions['yhat_upper'][-n:].values
        return self._build_forecast_series(forecast, conf_lo, conf_hi)
