from u8timeseries.models.autoregressive_model import AutoRegressiveModel
import pandas as pd

import fbprophet


class Prophet(AutoRegressiveModel):

    def __init__(self):
        super(Prophet, self).__init__()
        self.model = None

    def __str__(self):
        return 'Prophet'

    def fit(self, series):
        super(Prophet, self).fit(series)

        in_df = pd.DataFrame(data={
            'ds': series.get_time_index(),
            'y': series.get_values()
        })

        # TODO: user-provided seasonalities, or "auto" based on stepduration
        self.model = fbprophet.Prophet(weekly_seasonality=False, daily_seasonality=False)
        self.model.fit(in_df)

    def predict(self, n):
        # First we have to find which dates the next n points correspond to
        new_dates = self.training_series.get_time_index().append(self._generate_new_dates(n))

        new_dates_df = pd.DataFrame(data={'ds': new_dates})
        predictions = self.model.predict(new_dates_df)

        forecast = predictions['yhat'][-n:].values
        return self._build_forecast_series(forecast)
