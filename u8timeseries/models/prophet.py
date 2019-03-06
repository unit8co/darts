from u8timeseries.models.timeseries_model import TimeseriesModel
import pandas as pd

import fbprophet


class Prophet(TimeseriesModel):

    def __init__(self):
        super(Prophet, self).__init__()
        self.model = None

    def __str__(self):
        return 'Prophet'

    def fit(self, df, target_column, time_column, stepduration_str):
        assert time_column is not None, 'Prophet model requires a time column'
        super(Prophet, self).fit(df, target_column, time_column, stepduration_str)

        values = df[target_column].values

        in_df = pd.DataFrame(data={
            'ds': self.training_dates,
            'y': values
        })

        # TODO: user-provided seasonalities, or "auto" based on stepduration
        self.model = fbprophet.Prophet(weekly_seasonality=False, daily_seasonality=False)
        self.model.fit(in_df)

    def predict(self, n):
        # First we have to find which dates the next n points correspond to
        new_dates = [d for d in self.training_dates] + self._get_new_dates(n)

        new_dates_df = pd.DataFrame(data={'ds': new_dates})
        predictions = self.model.predict(new_dates_df)

        forecast = predictions['yhat'][-n:].values
        return self._build_forecast_df(forecast)
