"""
Facebook Prophet
----------------
"""

from typing import Optional, Union, List

import logging
import prophet
import numpy as np
import pandas as pd

from darts.timeseries import TimeSeries
from darts.models.forecasting.forecasting_model import DualCovariatesForecastingModel
from darts.logging import get_logger, execute_and_suppress_output, raise_if


logger = get_logger(__name__)
logger.level = logging.WARNING  # set to warning to suppress prophet logs


class Prophet(DualCovariatesForecastingModel):
    def __init__(self,
                 add_seasonalities: Optional[Union[dict, List[dict]]] = None,
                 country_holidays: Optional[str] = None,
                 **prophet_kwargs):
        """ Facebook Prophet

        This class provides a basic wrapper around `Facebook Prophet <https://github.com/facebook/prophet>`_.
        It supports adding country holidays as well as custom seasonalities and adds support for stochastic
        forecasting and future covariates.

        Parameters
        ----------
        add_seasonalities
            Optionally, a dict or list of dicts with custom seasonality/ies to add to the model.
            Each dict takes the following mandatory and optional data:

            dict({
                `'name': str` (name of the seasonality component),

                `'seasonal_periods': int` (number of timesteps after which the seasonal cycle repeats),

                `'fourier_order': int` (number of Fourier components to use),

                `'prior_scale': Optional[float]` (optionally, a prior scale for this component),

                `'mode': Optional[str]` (optionally, 'additive' or 'multiplicative')
                })

            An example for `seasonal_periods`: If you have hourly data (frequency='H') and your seasonal cycle repeats
            after 48 hours then set `seasonal_periods=48`.

            Apart from `seasonal_periods`, this is very similar to how you would call Facebook Prophet's
            `add_seasonality()` method.
            Alternatively, you can add seasonalities after model creation and before fitting with
            :meth:`add_seasonality() <Prophet.add_seasonality()>`.
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

        self._auto_seasonalities = self._extract_auto_seasonality(prophet_kwargs)

        self._add_seasonalities = dict()
        add_seasonality_calls = add_seasonalities if isinstance(add_seasonalities, list) else [add_seasonalities]
        for call in add_seasonality_calls:
            self._store_add_seasonality_call(seasonality_call=call)

        self.country_holidays = country_holidays
        self.prophet_kwargs = prophet_kwargs
        self.model = None

    def __str__(self):
        return 'Prophet'

    def fit(self,
            series: TimeSeries,
            future_covariates: Optional[TimeSeries] = None) -> None:

        super().fit(series, future_covariates)
        series = self.training_series

        fit_df = pd.DataFrame(data={
            'ds': series.time_index,
            'y': series.univariate_values()
        })

        self.model = prophet.Prophet(**self.prophet_kwargs)

        # add user defined seasonalities (from model creation and/or pre-fit self.add_seasonalities())
        interval_length = self._freq_to_days(series.freq_str)
        for seasonality_name, attributes in self._add_seasonalities.items():
            self.model.add_seasonality(name=seasonality_name,
                                       period=attributes['seasonal_periods'] * interval_length,
                                       fourier_order=attributes['fourier_order'])

        # add covariates
        if future_covariates is not None:
            fit_df = fit_df.merge(future_covariates.pd_dataframe(), left_on='ds', right_index=True, how='left')
            for covariate in future_covariates.columns:
                self.model.add_regressor(covariate)

        # add built-in country holidays
        if self.country_holidays is not None:
            self.model.add_country_holidays(self.country_holidays)

        execute_and_suppress_output(self.model.fit, logger, logging.WARNING, fit_df)

    def predict(self,
                n: int,
                future_covariates: Optional[TimeSeries] = None,
                num_samples: int = 1) -> TimeSeries:

        super().predict(n, future_covariates, num_samples)

        predict_df = self._generate_predict_df(n=n, future_covariates=future_covariates)

        if num_samples == 1:
            forecast = self.model.predict(predict_df)['yhat'].values
        else:
            forecast = np.expand_dims(self._stochastic_samples(predict_df, n_samples=num_samples), axis=1)

        return self._build_forecast_series(forecast)

    def _generate_predict_df(self,
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

    def _stochastic_samples(self,
                            predict_df,
                            n_samples) -> np.ndarray:
        """Returns stochastic forecast of `n_samples` samples.
        This method is a replicate of Prophet.predict() which suspends simplification of stochastic samples to
        deterministic target values."""

        # save default number of uncertainty_samples and set user-defined n_samples
        n_samples_default = self.model.uncertainty_samples
        self.model.uncertainty_samples = n_samples

        if self.model.history is None:
            raise ValueError('Model has not been fit.')

        if predict_df is None:
            predict_df = self.model.history.copy()
        else:
            if predict_df.shape[0] == 0:
                raise ValueError('Dataframe has no rows.')
            predict_df = self.model.setup_dataframe(predict_df.copy())

        predict_df['trend'] = self.model.predict_trend(predict_df)

        forecast = self.model.sample_posterior_predictive(predict_df)

        # reset default number of uncertainty_samples
        self.model.uncertainty_samples = n_samples_default
        return forecast['yhat']

    def predict_raw(self,
                    n: int,
                    future_covariates: Optional[TimeSeries] = None) -> pd.DataFrame:
        """Returns the output of the base Facebook Prophet model in form of a pandas DataFrame. Note however,
        that the output of this method is not supported for further processing with the Darts API.

        Methods of the base Prophet model can be accessed with self.model.method() (i.e. self.model.plot_components())
        """
        super().predict(n, future_covariates, num_samples=1)

        predict_df = self._generate_predict_df(n=n, future_covariates=future_covariates)

        return self.model.predict(predict_df)

    def add_seasonality(self,
                        name: str,
                        seasonal_periods: int,
                        fourier_order: int,
                        prior_scale: Optional[float] = None,
                        mode: Optional[str] = None) -> None:
        """Adds a custom seasonality to the model that reapeats after every n `seasonal_periods` timesteps.
        An example for `seasonal_periods`: If you have hourly data (frequency='H') and your seasonal cycle repeats
        after 48 hours -> `seasonal_periods=48`.

        Apart from `seasonal_periods`, this is very similar to how you would call Facebook Prophet's
        `add_seasonality()` method. For information about the parameters see:
        `The Prophet source code <https://github.com/facebook/prophet/blob/master/python/prophet/forecaster.py>`_.

        Parameters
        ----------
        name
            name of the seasonality component
        seasonal_periods
            number of timesteps after which the seasonal cycle repeats
        fourier_order
            number of Fourier components to use
        prior_scale
            optionally, a prior scale for this component
        mode
            optionally, 'additive' or 'multiplicative'
        """
        function_call = {
            'name': name,
            'seasonal_periods': seasonal_periods,
            'fourier_order': fourier_order,
            'prior_scale': prior_scale,
            'mode': mode
        }
        self._store_add_seasonality_call(seasonality_call=function_call)

    def _store_add_seasonality_call(self,
                                    seasonality_call: Optional[dict] = None) -> None:
        """Checks the validity of an add_seasonality() call and stores valid calls.
        As the actual model is only created at fitting time, and seasonalities are added pre-fit,
        the add_seasonality calls must be stored and checked on Darts' side.

        Raises
        ----------
        ValueError
            if `seasonality_call` has missing or empty mandatory keys/arguments

            if `seasonality_call` with `name` already exists.

            if `seasonality_call` has invalid keys/arguments

            if `seasonality_call` has invalid dtypes
        """

        if seasonality_call is None:
            return

        seasonality_properties = {
            'name': {'default': None, 'dtype': str},
            'seasonal_periods': {'default': None, 'dtype': int},
            'fourier_order': {'default': None, 'dtype': int},
            'prior_scale': {'default': None, 'dtype': float},
            'mode': {'default': None, 'dtype': str}
        }
        seasonality_default = {kw: seasonality_properties[kw]['default'] for kw in seasonality_properties}

        mandatory_keywords = ['name', 'seasonal_periods', 'fourier_order']

        add_seasonality_call = dict(seasonality_default, **seasonality_call)

        missing_kws = [kw for kw in mandatory_keywords if add_seasonality_call[kw] is None]
        raise_if(len(missing_kws) > 0,
                 f'Seasonality `{add_seasonality_call["name"]}` has missing mandatory keywords or empty arguments: '
                 f'{missing_kws}.',
                 logger)

        seasonality_name = add_seasonality_call['name']
        raise_if(seasonality_name in self._auto_seasonalities or seasonality_name in self._add_seasonalities,
                 f'Adding seasonality with `name={seasonality_name}` failed. A seasonality with this name already '
                 f'exists.')

        invalid_kws = [kw for kw in add_seasonality_call if kw not in seasonality_default]
        raise_if(len(invalid_kws) > 0,
                 f'Seasonality `{add_seasonality_call["name"]}` has invalid keywords: {invalid_kws}. Only the '
                 f'following arguments are supported: {list(seasonality_default)}',
                 logger)

        invalid_types = [kw for kw, value in add_seasonality_call.items()
                         if not isinstance(value, seasonality_properties[kw]['dtype'])
                         and value is not None]
        raise_if(len(invalid_types) > 0,
                 f'Seasonality `{add_seasonality_call["name"]}` has invalid value dtypes: {invalid_types} must be '
                 f'of type {[seasonality_properties[kw]["dtype"] for kw in invalid_types]}.',
                 logger)
        self._add_seasonalities[seasonality_name] = add_seasonality_call

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
    def _freq_to_days(freq: str) -> float:
        """Converts a frequency to number of days required by Facebook Prophet

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
        elif freq in ['B', 'C'] or freq.startswith(('B-', 'C-')):  # business day
            days = 1 * 7/5
        elif freq in ['D'] or freq.startswith('D-'):  # day
            days = 1.
        elif freq in ['H', 'BH', 'CBH'] or freq.startswith(('H', 'BH', 'CBH')):  # hour
            days = 1/24
        elif freq in ['T', 'min'] or freq.startswith(('H', 'BH', 'CBH')):  # minute
            days = 1/(24 * 60)
        elif freq == 'S' or freq.startswith('S'):  # second
            days = 1/seconds_per_day
        elif freq in ['L', 'ms'] or freq.startswith(('L', 'ms')):  # millisecond
            days = 1/(seconds_per_day * 10**3)
        elif freq == ['U', 'us'] or freq.startswith(('U', 'us')):  # microsecond
            days = 1/(seconds_per_day * 10**6)
        elif freq == 'N' or freq.startswith('N'):  # nanosecond
            days = 1/(seconds_per_day * 10**9)
        else:
            raise ValueError("freq {} not understood. Please report if you think this is in error.".format(freq))
        return days

    def _supports_range_index(self) -> bool:
        """Prophet does not support integer range index."""
        raise_if(True,
                 'Prophet does not support integer range index. The index of the TimeSeries must be of type '
                 'pandas.DatetimeIndex',
                 logger)
        return False
