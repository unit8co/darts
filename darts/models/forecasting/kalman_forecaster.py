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

from darts.logging import get_logger
from darts.models.filtering.kalman_filter import KalmanFilter
from darts.models.forecasting.forecasting_model import DualCovariatesForecastingModel
from darts.timeseries import TimeSeries

logger = get_logger(__name__)


class KalmanForecaster(DualCovariatesForecastingModel):
    def __init__(self, dim_x: int = 1, kf: Optional[Kalman] = None):
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
        """
        super().__init__()
        self.dim_x = dim_x
        self.kf = kf
        self.darts_kf = KalmanFilter(dim_x, kf)

    def __str__(self):
        return f"Kalman Filter Forecaster (dim_x={self.dim_x})"

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
