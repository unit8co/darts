"""
Facebook Prophet
----------------
"""

from typing import Optional
import logging
import numpy as np

from darts.timeseries import TimeSeries
from darts.models.forecasting.forecasting_model import DualCovariatesForecastingModel
import pandas as pd
from darts.logging import get_logger, execute_and_suppress_output
import prophet


logger = get_logger(__name__)
logger.level = logging.WARNING  # set to warning to suppress prophet logs


class Prophet(DualCovariatesForecastingModel):
    def __init__(self,
                 frequency: Optional[int] = None,
                 country_holidays: Optional[str] = None,
                 **prophet_kwargs):
        """ Facebook Prophet

        This class provides a basic wrapper around `Facebook Prophet <https://github.com/facebook/prophet>`_.
        It also supports country holidays.

        Parameters
        ----------
        frequency
            Optionally, some frequency, specifying a known seasonality, which will be added to prophet.
        country_holidays
            An optional country code, for which holidays can be taken into account by Prophet.

            See: https://github.com/dr-prodigy/python-holidays

            In addition to those countries, Prophet includes holidays for these
            countries: Brazil (BR), Indonesia (ID), India (IN), Malaysia (MY), Vietnam (VN),
            Thailand (TH), Philippines (PH), Turkey (TU), Pakistan (PK), Bangladesh (BD),
            Egypt (EG), China (CN), and Russia (RU).
        prophet_kwargs
            Some optional keyword arguments for Prophet.
            For information about the parameters see:
            `The Prophet source code <https://github.com/facebook/prophet/blob/master/python/prophet/forecaster.py>`_.

        """

        super().__init__()

        self.country_holidays = country_holidays
        self.freq = frequency
        self.prophet_kwargs = prophet_kwargs
        self.model = None

    def __str__(self):
        return 'Prophet'

    def fit(self, series: TimeSeries, future_covariates: Optional[TimeSeries] = None):
        super().fit(series, future_covariates)
        series = self.training_series

        fit_df = pd.DataFrame(data={
            'ds': series.time_index,
            'y': series.univariate_values()
        })

        # TODO: user-provided seasonalities, or "auto" based on stepduration
        self.model = prophet.Prophet(**self.prophet_kwargs)
        if self.freq is not None:
            if series.freq_str in ['MS', 'M', 'ME']:
                interval_length = 30.4375
            elif series.freq_str == 'Y':
                interval_length = 365.25
            else:
                interval_length = pd.to_timedelta(series.freq_str).days
            self.model.add_seasonality(name='custom', period=self.freq * interval_length,
                                       fourier_order=5)

        if future_covariates is not None:
            fit_df = fit_df.merge(future_covariates.pd_dataframe(), left_on='ds', right_index=True, how='left')
            for covariate in future_covariates.columns:
                self.model.add_regressor(covariate)

        # Input built-in country holidays
        if self.country_holidays is not None:
            self.model.add_country_holidays(self.country_holidays)

        execute_and_suppress_output(self.model.fit, logger, logging.WARNING, fit_df)

    def predict(self,
                n: int,
                future_covariates: Optional[TimeSeries] = None,
                num_samples: int = 1) -> TimeSeries:
        super().predict(n, future_covariates, num_samples)

        predict_df = pd.DataFrame(data={'ds': self._generate_new_dates(n)})
        if future_covariates is not None:
            predict_df = predict_df.merge(future_covariates.pd_dataframe(), left_on='ds', right_index=True, how='left')

        if num_samples == 1:
            forecast = self.model.predict(predict_df)['yhat'].values
        else:
            forecast = np.expand_dims(self.stochastic_samples(predict_df, n_samples=num_samples), axis=1)
        return self._build_forecast_series(forecast)

    def stochastic_samples(self, predict_df, n_samples) -> np.ndarray:
        """Returns stochastic forecast of `n_samples` samples.
        This method is a replicate of Prophet.predict() which suspends simplification of stochastic samples to
        deterministic target values."""

        n_samples_default = self.model.uncertainty_samples
        self.model.uncertainty_samples = n_samples

        if self.model.history is None:
            raise Exception('Model has not been fit.')

        if predict_df is None:
            predict_df = self.model.history.copy()
        else:
            if predict_df.shape[0] == 0:
                raise ValueError('Dataframe has no rows.')
            predict_df = self.model.setup_dataframe(predict_df.copy())

        predict_df['trend'] = self.model.predict_trend(predict_df)
        seasonal_components = self.model.predict_seasonal_components(predict_df)

        forecast = self.model.sample_posterior_predictive(predict_df)['yhat']

        self.model.uncertainty_samples = n_samples_default
        return forecast

    def _is_probabilistic(self) -> bool:
        return True
