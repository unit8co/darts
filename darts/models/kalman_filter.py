"""
Kalman Filter
-------------
"""

from abc import ABC

from typing import Optional
from filterpy.kalman import KalmanFilter as FpKalmanFilter
from copy import deepcopy
import numpy as np

from .filtering_model import FilteringModel
from ..timeseries import TimeSeries
from ..utils.utils import raise_if_not


class KalmanFilter(FilteringModel, ABC):
    def __init__(
            self, 
            dim_x: int = 1,
            x_init: Optional[np.array] = None,
            P: Optional[np.array] = None,
            Q: Optional[np.array] = None,
            R: Optional[np.array] = None,
            H: Optional[np.array] = None,
            F: Optional[np.array] = None,
            kf: Optional[FpKalmanFilter] = None
            ):
        """
        This model implements a Kalman filter over a time series (without control signal).

        The key method is `KalmanFilter.filter()`.
        It considers the provided time series as containing (possibly noisy) observations z obtained from a
        (possibly noisy) linear dynamical system with hidden state x. The function `filter(series)` returns a new
        `TimeSeries` describing the distribution of the state x, as inferred by the Kalman filter from
        sequentially observing z from `series`.
        Depending on the use case, this can be used to de-noise a series or infer the underlying hidden state of the
        data generating process (assuming notably that the dynamical system generating the data is known, as captured
        by the `F` matrix.).

        This implementation wraps around filterpy.kalman.KalmanFilter, so more information the parameters can be found
        here: https://filterpy.readthedocs.io/en/latest/kalman/KalmanFilter.html

        The dimensionality of the measurements z is automatically inferred upon calling `filter()`.
        This implementation doesn't include control signal.

        Parameters
        ----------
        dim_x : int
            Size of the Kalman filter state vector. It determines the dimensionality of the `TimeSeries`
            returned by the `filter()` function.
        x_init : ndarray (dim_x, 1), default: [0, 0, ..., 0]
            Initial state; will be updated at each time step.
        P : ndarray (dim_x, dim_x), default: identity matrix
            initial covariance matrix; will be update at each time step
        Q : ndarray (dim_x, dim_x), default: identity matrix
            Process noise covariance matrix
        R : ndarray (dim_z, dim_z), default: identity matrix
            Measurement noise covariance matrix. `dim_z` must match the dimensionality (width) of the `TimeSeries`
            used with `filter()`.
        H : ndarray (dim_z, dim_x), default: all-ones matrix
            measurement function; describes how the measurement z is obtained from the state vector x
        F : ndarray (dim_x, dim_x), default: identity matrix
            State transition matrix; describes how the state evolves from one time step to the next
            in the underlying dynamical system.
        kf : filterpy.kalman.KalmanFilter
            Optionally, an instance of `filterpy.kalman.KalmanFilter`.
            If this is provided, the other parameters are ignored. This instance will be copied for every
            call to `filter()`, so the state is not carried over from one time series to another across several
            calls to `filter()`.
            The various dimensionality in the filter must match those in the `TimeSeries` used when calling `filter()`.
        """
        super().__init__()
        if kf is None:
            self.dim_x = dim_x
            self.x_init = x_init if x_init is not None else np.zeros(self.dim_x,)
            self.P = P if P is not None else np.eye(self.dim_x)
            self.Q = Q if Q is not None else np.eye(self.dim_x)
            self.R = R
            self.H = H
            self.F = F if F is not None else np.eye(self.dim_x)
            self.kf = None
            self.kf_provided = False
        else:
            self.kf = kf
            self.kf_provided = True

    def __str__(self):
        return 'KalmanFilter(dim_x={})'.format(self.dim_x)

    def filter(self,
               series: TimeSeries,
               num_samples: int = 1):
        """
        Sequentially applies the Kalman filter on the provided series of observations.

        Parameters
        ----------
        series : TimeSeries
            The series of observations used to infer the state values according to the specified Kalman process.
            This must be a deterministic series (containing one sample).

        Returns
        -------
        TimeSeries
            A stochastic `TimeSeries` of state values, of dimension `dim_x`.
        """

        raise_if_not(series.is_deterministic, 'The input series for the Kalman filter must be '
                                              'deterministic (observations).')

        dim_z = series.width

        if not self.kf_provided:
            kf = FpKalmanFilter(dim_x=self.dim_x, dim_z=dim_z)
            kf.x = self.x_init
            kf.P = self.P
            kf.Q = self.Q
            kf.R = self.R if self.R is not None else np.eye(dim_z)
            kf.H = self.H if self.H is not None else np.ones((dim_z, self.dim_x))
            kf.F = self.F
        else:
            raise_if_not(dim_z == self.kf.dim_z, 'The provided TimeSeries dimensionality does not match '
                                                 'the filter observation dimensionality dim_z.')
            kf = deepcopy(self.kf)

        super().filter(series)
        values = series.values(copy=False)

        # For each time step, we'll sample "n_samples" from a multivariate Gaussian
        # whose mean vector and covariance matrix come from the Kalman filter.
        if num_samples == 1:
            sampled_states = np.zeros(((len(values)), self.dim_x, ))
        else:
            sampled_states = np.zeros(((len(values)), self.dim_x, num_samples))

        # process_means = np.zeros((len(values), self.dim_x))  # mean values
        # process_covariances = ...                            # covariance matrices; TODO
        for i in range(len(values)):
            obs = values[i, :]
            kf.predict()
            kf.update(obs)
            mean_vec = kf.x.reshape(self.dim_x,)

            if num_samples == 1:
                # It's actually not sampled in this case
                sampled_states[i, :] = mean_vec
            else:
                cov_matrix = kf.P
                sampled_states[i, :, :] = np.random.multivariate_normal(mean_vec, cov_matrix, size=num_samples).T

        # TODO: later on for a forecasting model we'll have to do something like
        """
        for _ in range(horizon):
            kf.predict()
            # forecasts on the observations, obtained from the state
            preds.append(kf.H.dot(kf.x))
            preds_cov.append(kf.H.dot(kf.P).dot(kf.H.T))
        """

        return TimeSeries.from_times_and_values(series.time_index, sampled_states)
