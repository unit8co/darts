from u8timeseries.models.autoregressive_model import AutoRegressiveModel
import pandas as pd

import fbprophet


class Prophet(AutoRegressiveModel):
    """
    Implementation of the Prophet model.

    Currently just a wrapper around the fbprophet implementation.

    :param country_holidays: An optional country code, for which holidays can be taken into account by Prophet.

                             See: https://github.com/dr-prodigy/python-holidays

                             In addition to those countries, Prophet includes holidays for these
                             countries: Brazil (BR), Indonesia (ID), India (IN), Malaysia (MY), Vietnam (VN),
                             Thailand (TH), Philippines (PH), Turkey (TU), Pakistan (PK), Bangladesh (BD),
                             Egypt (EG), China (CN), and Russian (RU).
    :param weekly_seasonality:
    :param daily_seasonality:
    """

    def __init__(self, weekly_seasonality=False, daily_seasonality=False, country_holidays: str = None):

        super().__init__()

        self.country_holidays = country_holidays
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.model = None

    def __str__(self):
        return 'Prophet'

    def fit(self, series):
        super().fit(series)

        in_df = pd.DataFrame(data={
            'ds': series.time_index(),
            'y': series.values()
        })

        # TODO: user-provided seasonalities, or "auto" based on stepduration
        self.model = fbprophet.Prophet(weekly_seasonality=self.weekly_seasonality,
                                       daily_seasonality=self.daily_seasonality)

        # Input built-in country holidays
        if self.country_holidays is not None:
            self.model.add_country_holidays(self.country_holidays)

        self.model.fit(in_df)

    def predict(self, n):
        super().predict(n)
        new_dates = self._generate_new_dates(n)
        new_dates_df = pd.DataFrame(data={'ds': new_dates})

        predictions = self.model.predict(new_dates_df)

        forecast = predictions['yhat'][-n:].values
        conf_lo = predictions['yhat_lower'][-n:].values
        conf_hi = predictions['yhat_upper'][-n:].values
        return self._build_forecast_series(forecast, conf_lo, conf_hi)
