"""
Facebook Prophet
----------------
"""

from typing import Optional, Union, List
import logging
import numpy as np

from darts.timeseries import TimeSeries
from darts.models.forecasting.forecasting_model import DualCovariatesForecastingModel
import pandas as pd
from darts.logging import get_logger, execute_and_suppress_output, raise_if
import prophet


logger = get_logger(__name__)
logger.level = logging.WARNING  # set to warning to suppress prophet logs


class Prophet(DualCovariatesForecastingModel):
    def __init__(self,
                 add_seasonalities: Optional[Union[dict, List[dict]]] = None,
                 country_holidays: Optional[str] = None,
                 **prophet_kwargs):
        """ Facebook Prophet

        This class provides a basic wrapper around `Facebook Prophet <https://github.com/facebook/prophet>`_.
        It supports country holidays as well as custom seasonalities and adds support for stochastic forecasting and
        future covariates.

        Parameters
        ----------
        add_seasonalities
            Optionally, a dict or list of dicts with custom seasonality/ies to add to the model.
            Each dict takes the following mandatory and optional data:

            dict({
                `'name': str` (name of the seasonality component),

                `'seasonal_periods': int` (number of timesteps after which the custom seasonal cycle repeats),

                `'fourier_order': int` (number of Fourier components to use),

                `'prior_scale': Optional[float]` (optionally, a prior scale for this component),

                `'mode': Optional[str]` (optionally, 'additive' or 'multiplicative')
                })

            An example for `seasonal_periods`: If you have hourly data (frequency='H') and your seasonal cycle repeats
            after 48 hours -> `seasonal_periods=48`
            Apart from `seasonal_periods`, this is very similar to how you would call Facebook Prophet's
            `add_seasonality()` method.
            Alternatively, you can add seasonalities after model creation and before fitting with
            `self.add_seasonality()`
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

        self.auto_seasonalities = self._extract_auto_seasonality(prophet_kwargs)
        self.add_seasonalities = dict()
        if not isinstance(add_seasonalities, list):
            add_seasonalities = [add_seasonalities]
        for add_seasonality in add_seasonalities:
            self._process_seasonality_call(seasonality_call=add_seasonality)
        self.country_holidays = country_holidays
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

        self.model = prophet.Prophet(**self.prophet_kwargs)

        # add user defined seasonalities (from model creation and/or pre-fit self.add_seasonalities())
        if self.add_seasonalities:
            interval_length = self.freq_to_days(series.freq_str)
            for seasonality_name, attributes in self.add_seasonalities.items():
                self.model.add_seasonality(name=seasonality_name,
                                           period=attributes['seasonal_periods'] * interval_length,
                                           fourier_order=attributes['fourier_order'])

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

        predict_df = self.generate_predict_df(n=n, future_covariates=future_covariates)

        if num_samples == 1:
            forecast = self.model.predict(predict_df)['yhat'].values
        else:
            forecast = np.expand_dims(self.stochastic_samples(predict_df, n_samples=num_samples), axis=1)

        return self._build_forecast_series(forecast)

    def generate_predict_df(self,
                            n: int,
                            future_covariates: Optional[TimeSeries] = None) -> pd.DataFrame:
        """Returns a pandas DataFrame in the format required for Prophet.predict() with `n` dates after the end of
        the fitted TimeSeries"""

        predict_df = pd.DataFrame(data={'ds': self._generate_new_dates(n)})
        if future_covariates is not None:
            predict_df = predict_df.merge(future_covariates.pd_dataframe(), left_on='ds', right_index=True, how='left')
        return predict_df

    def _is_probabilistic(self) -> bool:
        return True

    def stochastic_samples(self,
                           predict_df,
                           n_samples) -> np.ndarray:
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

        forecast = self.model.sample_posterior_predictive(predict_df)

        self.model.uncertainty_samples = n_samples_default
        return forecast['yhat']

    def predict_raw(self,
                    n: int,
                    future_covariates: Optional[TimeSeries] = None) -> pd.DataFrame:
        """Returns the output of the base Prophet model in form of a pandas DataFrame. Note however, that the outpu of
        method is not supported for further processing with the Darts API.

        Methods of the base Prophet model can be accessed with self.model.method() (i.e. self.model.plot_components())
        """
        super().predict(n, future_covariates, num_samples=1)

        predict_df = self.generate_predict_df(n=n, future_covariates=future_covariates)

        return self.model.predict(predict_df)

    def add_seasonality(self, name, seasonal_periods, fourier_order, **kwargs) -> None:
        """Adds a custom seasonality to the model that reapeats after every n=`seasonal_periods` timesteps.
        An example for `seasonal_periods`: If you have hourly data (frequency='H') and your seasonal cycle repeats
        after 48 hours -> `seasonal_periods=48`.

        Apart from `seasonal_periods`, this is very similar to how you would call Facebook Prophet's
        `add_seasonality()` method. For information about the parameters see:
        `The Prophet source code <https://github.com/facebook/prophet/blob/master/python/prophet/forecaster.py>`_.

        Parameters
        ----------
        name
            string name of the seasonality component.
        seasonal_periods
            name of the seasonality component
        fourier_order
            number of timesteps after which the custom seasonal cycle repeats
        prior_scale
            optionally, number of Fourier components to use
        mode
            optionally, 'additive' or 'multiplicative'
        """
        args = {'name': name, 'seasonal_periods': seasonal_periods, 'fourier_order': fourier_order}
        function_call = dict(args, **kwargs)
        self._process_seasonality_call(seasonality_call=function_call)

    def _process_seasonality_call(self,
                                 seasonality_call: Optional[dict] = None) -> None:
        """Checks the validity of a add_seasonality() call and stores valid calls.

        Raises
        ----------
        ValueError
            if `seasonality_call` has missing or empty mandatory keys/arguments

            if `seasonality_call` with `name` already exists.

            if `seasonality_call` has invalid keys/arguments
        """

        if seasonality_call is None:
            return

        seasonality_default = {
            'name': None,
            'seasonal_periods': None,
            'fourier_order': None,
            'prior_scale': None,
            'mode': None
        }
        mandatory_args = ['name', 'seasonal_periods', 'fourier_order']

        add_seasonality_call = dict(seasonality_default, **seasonality_call)

        missing_args = [arg for arg in mandatory_args if add_seasonality_call[arg] is None]
        raise_if(len(missing_args) > 0,
                 f'add_seasonality has missing or empty mandatory add_seasonality keys/arguments: {missing_args}.',
                 logger)

        seasonality_name = add_seasonality_call['name']
        raise_if(seasonality_name in self.auto_seasonalities or seasonality_name in self.add_seasonalities,
                 f'Adding seasonality with `name={seasonality_name}` failed. A seasonality with this name already '
                 f'exists.')

        invalid_args = [arg for arg in add_seasonality_call.keys() if arg not in seasonality_default]
        raise_if(len(invalid_args) > 0,
                 f'invalid add_seasonality keys/arguments: {invalid_args}. Only the following arguments are supported: '
                 f'{list(seasonality_default)}',
                 logger)

        self.add_seasonalities[seasonality_name] = add_seasonality_call

    @staticmethod
    def _extract_auto_seasonality(prophet_kwargs: dict) -> list:
        """Returns the automatically added seasonalities by Prophet's base model based on kwargs of model creation"""
        auto_seasonalities = []
        for auto_seasonality in ['daily', 'weekly', 'yearly']:
            s_name = auto_seasonality + '_seasonality'
            if not (s_name in prophet_kwargs and not prophet_kwargs[s_name]):
                auto_seasonalities.append(auto_seasonality)
        return auto_seasonalities

    @staticmethod
    def freq_to_days(freq: str) -> float:
        """Converts a frequency to number of days required by Prophet

        Parameters
        ----------
        freq
            frequency string of the underlying TimeSeries's time index (pd.DateTimeIndex.freq_str)
        """
        seconds_per_day = 86400
        if freq == ['A', 'BYS', 'BA', 'RE'] or freq.startswith(('A', 'BYS', 'BA', 'RE-')):  # year
            days = 365.25
        elif freq == ['Q', 'BQ', 'REQ'] or freq.startswith(('Q', 'BQ', 'REQ')):  # quarter
            days = 3 * 30.4375
        elif freq in ['M', 'BM', 'CBM', 'SM'] or freq.startswith(('M', 'BM', 'BS', 'CBM', 'SM')):  # month
            days = 30.4375
        elif freq in 'W' or freq.startswith('W-'):  # week
            days = 7.
        elif freq in ['D', 'B', 'C'] or freq.startswith(('D-', 'B-', 'C-')):  # day
            days = 1.
        elif freq in ['H', 'BH', 'CBH'] or freq.startswith(('H', 'BH', 'CBH')):  # hour
            days = 1/24
        elif freq in ['T', 'min'] or freq.startswith(('H', 'BH', 'CBH')):  # minute
            days = 1 / seconds_per_day
        elif freq == 'S' or freq.startswith('S'):  # second
            days = 1 / (seconds_per_day * 10**3)
        elif freq in ['L', 'ms'] or freq.startswith(('L', 'ms')):  # millisecond
            days = 1 / (seconds_per_day * 10**6)
        elif freq == ['U', 'us'] or freq.startswith(('U', 'us')):  # microsecond
            days = 1 / (seconds_per_day * 10**9)
        elif freq == 'N' or freq.startswith('N'):  # nanosecond
            days = 1 / (seconds_per_day * 10**12)
        else:
            raise ValueError("freq {} not understood. Please report if you think this is in error.".format(freq))
        return days
