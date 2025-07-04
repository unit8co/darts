"""
Kalman Filter Forecaster
------------------------

A model producing stochastic forecasts based on the Kalman filter.
The filter is first optionally fitted on the series (using the N4SID
identification algorithm), and then run on future time steps in order
to obtain forecasts.

This implementation accepts an optional control signal (future covariates).
"""

from typing import Optional

import numpy as np
from nfoursid.kalman import Kalman

from darts import TimeSeries
from darts.logging import get_logger
from darts.models.filtering.kalman_filter import KalmanFilter
from darts.models.forecasting.forecasting_model import (
    TransferableFutureCovariatesLocalForecastingModel,
)
from darts.utils.utils import random_method

logger = get_logger(__name__)


class KalmanForecaster(TransferableFutureCovariatesLocalForecastingModel):
    @random_method
    def __init__(
        self,
        dim_x: int = 1,
        kf: Optional[Kalman] = None,
        add_encoders: Optional[dict] = None,
        random_state: Optional[int] = None,
    ):
        """Kalman filter Forecaster

        This model uses a Kalman filter to produce forecasts. It uses a
        :class:`darts.models.filtering.kalman_filter.KalmanFilter` object
        and treats future values as missing values.

        The model can optionally receive a :class:`nfoursid.kalman.Kalman`
        object specifying the Kalman filter, or, if not specified, the filter
        will be trained using the N4SID system identification algorithm.

        Parameters
        ----------
        dim_x : int
            Size of the Kalman filter state vector.
        kf : nfoursid.kalman.Kalman
            Optionally, an instance of `nfoursid.kalman.Kalman`.
            If this is provided, the parameter dim_x is ignored. This instance will be copied for every
            call to `predict()`, so the state is not carried over from one time series to another across several
            calls to `predict()`.
            The various dimensionalities of the filter must match those of the `TimeSeries` used when
            calling `predict()`.
            If this is specified, it is still necessary to call `fit()` before calling `predict()`,
            although this will have no effect on the Kalman filter.
        add_encoders
            A large number of future covariates can be automatically generated with `add_encoders`.
            This can be done by adding multiple pre-defined index encoders and/or custom user-made functions that
            will be used as index encoders. Additionally, a transformer such as Darts' :class:`Scaler` can be added to
            transform the generated covariates. This happens all under one hood and only needs to be specified at
            model creation.
            Read :meth:`SequentialEncoder <darts.dataprocessing.encoders.SequentialEncoder>` to find out more about
            ``add_encoders``. Default: ``None``. An example showing some of ``add_encoders`` features:

            .. highlight:: python
            .. code-block:: python

                def encode_year(idx):
                    return (idx.year - 1950) / 50

                add_encoders={
                    'cyclic': {'future': ['month']},
                    'datetime_attribute': {'future': ['hour', 'dayofweek']},
                    'position': {'future': ['relative']},
                    'custom': {'future': [encode_year]},
                    'transformer': Scaler(),
                    'tz': 'CET'
                }
            ..
        random_state
            Controls the randomness for reproducible forecasting.

        Examples
        --------
        >>> from darts.datasets import AirPassengersDataset
        >>> from darts.models import KalmanForecaster
        >>> from darts.utils.timeseries_generation import datetime_attribute_timeseries
        >>> series = AirPassengersDataset().load()
        >>> # optionally, use some future covariates; e.g. the value of the month encoded as a sine and cosine series
        >>> future_cov = datetime_attribute_timeseries(series, "month", cyclic=True, add_length=6)
        >>> # increasing the size of the state vector
        >>> model = KalmanForecaster(dim_x=12)
        >>> model.fit(series, future_covariates=future_cov)
        >>> pred = model.predict(6, future_covariates=future_cov)
        >>> pred.values()
        array([[474.40680728],
               [440.51801726],
               [461.94512461],
               [494.42090089],
               [528.6436328 ],
               [590.30647185]])

        .. note::
            `Kalman example notebook <https://unit8co.github.io/darts/examples/10-Kalman-filter-examples.html>`_
            presents techniques that can be used to improve the forecasts quality compared to this simple usage
            example.
        """
        super().__init__(add_encoders=add_encoders)
        self.dim_x = dim_x
        self.kf = kf
        self.darts_kf = KalmanFilter(dim_x, kf)

    def _fit(self, series: TimeSeries, future_covariates: Optional[TimeSeries] = None):
        super()._fit(series, future_covariates)
        if self.kf is None:
            self.darts_kf.fit(series=series, covariates=future_covariates)

        return self

    def predict(
        self,
        n: int,
        series: Optional[TimeSeries] = None,
        future_covariates: Optional[TimeSeries] = None,
        num_samples: int = 1,
        predict_likelihood_parameters: bool = False,
        verbose: bool = False,
        show_warnings: bool = True,
        random_state: Optional[int] = None,
        **kwargs,
    ) -> TimeSeries:
        # we override `predict()` to pass a non-None `series`, so that historic_future_covariates
        # will be passed to `_predict()`
        series = series if series is not None else self.training_series
        return super().predict(
            n,
            series,
            future_covariates,
            num_samples,
            random_state=random_state,
            **kwargs,
        )

    @random_method
    def _predict(
        self,
        n: int,
        series: Optional[TimeSeries] = None,
        historic_future_covariates: Optional[TimeSeries] = None,
        future_covariates: Optional[TimeSeries] = None,
        num_samples: int = 1,
        predict_likelihood_parameters: bool = False,
        verbose: bool = False,
        random_state: Optional[int] = None,
    ) -> TimeSeries:
        super()._predict(
            n, series, historic_future_covariates, future_covariates, num_samples
        )
        time_index = self._generate_new_dates(n, input_series=series)
        placeholder_vals = np.zeros((n, self.training_series.width)) * np.nan
        series_future = TimeSeries(
            times=time_index,
            values=placeholder_vals,
            components=self.training_series.columns,
            copy=False,
            **self.training_series._attrs,
        )

        series = series.append(series_future)
        if historic_future_covariates is not None:
            future_covariates = historic_future_covariates.append(future_covariates)

        filtered_series = self.darts_kf.filter(
            series=series, covariates=future_covariates, num_samples=num_samples
        )

        return filtered_series[-n:]

    @property
    def supports_multivariate(self) -> bool:
        return True

    @property
    def supports_probabilistic_prediction(self) -> bool:
        return True
