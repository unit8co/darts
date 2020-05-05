"""
Implementation of an Prophet model.
-------------------------------------------------------
"""

import fbprophet
import pandas as pd

from u8timeseries.models.autoregressive_model import AutoRegressiveModel
from ..custom_logging import time_log, get_logger

logger = get_logger(__name__)


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
    :param yearly_seasonality:
    :param weekly_seasonality:
    :param daily_seasonality:
    :param mode: The seasonality mode, either `additive` or `multiplicative`.
    """

    def __init__(self, frequency: int = None, yearly_seasonality=False, weekly_seasonality=False,
                 daily_seasonality=False, country_holidays: str = None, mode: str = "additive"):

        super().__init__()

        self.country_holidays = country_holidays
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.mode = mode
        self.freq = frequency
        self.model = None

    def __str__(self):
        return 'Prophet'

    @time_log(logger=logger)
    def fit(self, series):
        super().fit(series)

        in_df = pd.DataFrame(data={
            'ds': series.time_index(),
            'y': series.values()
        })

        # TODO: user-provided seasonalities, or "auto" based on stepduration
        self.model = fbprophet.Prophet(yearly_seasonality=self.yearly_seasonality,
                                       weekly_seasonality=self.weekly_seasonality,
                                       daily_seasonality=self.daily_seasonality,
                                       seasonality_mode=self.mode)
        if self.freq is not None:
            if series.freq_str() in ['MS', 'M', 'ME']:
                interval_length = 30.4375
            elif series.freq_str() == 'Y':
                interval_length = 365.25
            else:
                interval_length = pd.to_timedelta(series.freq_str()).days
            self.model.add_seasonality(name='custom', period=self.freq * interval_length,
                                       fourier_order=5)

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
