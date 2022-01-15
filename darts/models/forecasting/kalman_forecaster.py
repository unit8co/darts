"""
Kalman Filter Forecaster
------------------------

A model producing stochastic forecasts based on the Kalman filter.
The filter is first optionally fitted on the series (using the N4SID
identification algorithm), and then run on future time steps in order
to obtain forecasts.

This implementation accepts an optional control signal (future covariate).
"""

from typing import Optional

import numpy as np
from nfoursid.kalman import Kalman

from darts.models.forecasting.forecasting_model import DualCovariatesForecastingModel
from darts.models.filtering.kalman_filter import KalmanFilter
from darts.timeseries import TimeSeries
from darts.logging import get_logger

logger = get_logger(__name__)


class KalmanForecaster(DualCovariatesForecastingModel):
    def __init__(self, dim_x: int = 1, kf: Optional[Kalman] = None):
        """Kalman Forecaster
        This model implements a Kalman filter over a time series.

        The key method is `KalmanFilter.filter()`.
        It considers the provided time series as containing (possibly noisy) observations z obtained from a
        (possibly noisy) linear dynamical system with hidden state x. The function `filter(series)` returns a new
        `TimeSeries` describing the distribution of the output z (without noise), as inferred by the Kalman filter from
        sequentially observing z from `series`, and the dynamics of the linear system of order dim_x.

        The method `KalmanFilter.fit()` is used to initialize the Kalman filter by estimating the state space model of
        a linear dynamical system and the covariance matrices of the process and measurement noise using the N4SID
        algorithm.

        This implementation uses Kalman from the NFourSID package. More information can be found here:
        https://nfoursid.readthedocs.io/en/latest/source/kalman.html.

        The dimensionality of the measurements z and optional control signal (covariates) u is automatically inferred
        upon calling `filter()`.

        Parameters
        ----------
        dim_x : int
            Size of the Kalman filter state vector.
        kf : nfoursid.kalman.Kalman
            Optionally, an instance of `nfoursid.kalman.Kalman`.
            If this is provided, the parameter dim_x is ignored. This instance will be copied for every
            call to `filter()`, so the state is not carried over from one time series to another across several
            calls to `filter()`.
            The various dimensionalities of the filter must match those of the `TimeSeries` used when
            calling `filter()`.
        """
        super().__init__()
        self.dim_x = dim_x
        self.kf = kf
        self.darts_kf = KalmanFilter(dim_x, kf)

    def __str__(self):
        return "Kalman Filter Forecaster (dim_x={})".format(self.dim_x)

    def _fit(self, series: TimeSeries, future_covariates: Optional[TimeSeries] = None):

        super()._fit(series, future_covariates)
        if self.kf is None:
            self.darts_kf.fit(series=series, covariates=future_covariates)

        return self

    def _predict(
        self,
        n: int,
        future_covariates: Optional[TimeSeries] = None,
        num_samples: int = 1,
    ) -> TimeSeries:

        super()._predict(n, future_covariates, num_samples)

        time_index = self._generate_new_dates(n)
        placeholder_vals = np.zeros((n, self.training_series.width)) * np.nan
        series_future = TimeSeries.from_times_and_values(time_index, placeholder_vals)
        whole_series = self.training_series.append(series_future)
        filtered_series = self.darts_kf.filter(
            whole_series, covariates=future_covariates, num_samples=num_samples
        )

        return filtered_series[-n:]

    def _is_probabilistic(self) -> bool:
        return True
