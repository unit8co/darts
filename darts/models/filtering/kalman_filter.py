"""
Kalman Filter
-------------
"""

from abc import ABC
from copy import deepcopy
from typing import Optional

import numpy as np
import pandas as pd
from nfoursid.kalman import Kalman
from nfoursid.nfoursid import NFourSID

from darts.logging import raise_if, raise_if_not
from darts.models.filtering.filtering_model import FilteringModel
from darts.timeseries import TimeSeries


class KalmanFilter(FilteringModel, ABC):
    def __init__(self, dim_x: int = 1, kf: Optional[Kalman] = None):
        """
        This model implements a Kalman filter over a time series.

        The key method is `KalmanFilter.filter()`.
        It considers the provided time series as containing (possibly noisy) observations z obtained from a
        (possibly noisy) linear dynamical system with hidden state x. The function `filter(series)` returns a new
        `TimeSeries` describing the distribution of the output z (without noise), as inferred by the Kalman filter from
        sequentially observing z from `series`, and the dynamics of the linear system of order dim_x.

        This filter also supports missing values in the observation series, in which case the underlying
        Kalman filter will carry on using its mean estimate.

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
            The dimensionalities of the filter must match those of the `TimeSeries` used when calling `filter()`.
        """
        # TODO: Add support for x_init. Needs reimplementation of NFourSID.

        super().__init__()

        if kf is None:
            self.kf = None
            self.dim_x = dim_x
            self._kf_provided = False
        else:
            self.kf = kf
            self.dim_u = kf.state_space.u_dim
            self.dim_x = kf.state_space.x_dim
            self.dim_y = kf.state_space.y_dim
            self._kf_provided = True
            if self.dim_u > 0:
                self._expect_covariates = True

    def __str__(self):
        return f"KalmanFilter(dim_x={self.dim_x})"

    def fit(
        self,
        series: TimeSeries,
        covariates: Optional[TimeSeries] = None,
        num_block_rows: Optional[int] = None,
    ) -> "KalmanFilter":
        """
        Initializes the Kalman filter using the N4SID algorithm.

        Parameters
        ----------
        series : TimeSeries
            The series of outputs (observations) used to infer the underlying state space model.
            This must be a deterministic series (containing one sample).
        covariates : Optional[TimeSeries]
            An optional series of inputs (control signal) that will also be used to infer the underlying state space
            model. This must be a deterministic series (containing one sample).
        num_block_rows : Optional[int]
            The number of block rows to use in the block Hankel matrices used in the N4SID algorithm.
            See the documentation of nfoursid.nfoursid.NFourSID for more information.
            If not provided, the dimensionality of the state space model will be used, with a maximum of 10.

        Returns
        -------
        self
            Fitted Kalman filter.
        """
        if covariates is not None:
            self._expect_covariates = True
            covariates = covariates.slice_intersect(series)
            raise_if_not(
                series.has_same_time_as(covariates),
                "The number of timesteps in the series and the covariates must match.",
            )

        # TODO: Handle multiple timeseries. Needs reimplementation of NFourSID?
        self.dim_y = series.width
        outputs = series.to_dataframe(copy=False)
        outputs.columns = [f"y_{i}" for i in outputs.columns]

        if covariates is not None:
            self.dim_u = covariates.width
            inputs = covariates.to_dataframe(copy=False)
            inputs.columns = [f"u_{i}" for i in inputs.columns]
            input_columns = list(inputs.columns)
            measurements = pd.concat([outputs, inputs], axis=1)
        else:
            measurements = outputs
            input_columns = None

        if num_block_rows is None:
            num_block_rows = max(10, self.dim_x)
        nfoursid = NFourSID(
            measurements,
            output_columns=list(outputs.columns),
            input_columns=input_columns,
            num_block_rows=num_block_rows,
        )
        nfoursid.subspace_identification()
        state_space_identified, covariance_matrix = nfoursid.system_identification(
            rank=self.dim_x
        )

        self.kf = Kalman(state_space_identified, covariance_matrix)

        return self

    def filter(
        self,
        series: TimeSeries,
        covariates: Optional[TimeSeries] = None,
        num_samples: int = 1,
    ) -> TimeSeries:
        """
        Sequentially applies the Kalman filter on the provided series of observations.

        Parameters
        ----------
        series : TimeSeries
            The series of outputs (observations) used to infer the underlying outputs according to the specified Kalman
            process. This must be a deterministic series (containing one sample).
        covariates : Optional[TimeSeries]
            An optional series of inputs (control signal), necessary if the Kalman filter was initialized with
            covariates. This must be a deterministic series (containing one sample).
        num_samples : int, default: 1
            The number of samples to generate from the inferred distribution of the output z. If this is set to 1, the
            output is a `TimeSeries` containing a single sample using the mean of the distribution.

        Returns
        -------
        TimeSeries
            A (stochastic) `TimeSeries` of the inferred output z, of the same width as the input series.
        """
        super().filter(series)

        raise_if(
            self.kf is None,
            "The Kalman filter has not been fitted yet. Call `fit()` first "
            "or provide Kalman filter in constructor.",
        )

        raise_if_not(
            series.width == self.dim_y,
            "The provided TimeSeries dimensionality does not match "
            "the output dimensionality of the Kalman filter.",
        )

        raise_if(
            covariates is not None and not self._expect_covariates,
            "Covariates were provided, but the Kalman filter was not fitted with covariates.",
        )

        if self._expect_covariates:
            raise_if(
                covariates is None,
                "The Kalman filter was fitted with covariates, but these were not provided.",
            )

            raise_if_not(
                covariates.is_deterministic,
                "The covariates must be deterministic (observations).",
            )

            covariates = covariates.slice_intersect(series)
            raise_if_not(
                series.has_same_time_as(covariates),
                "The number of timesteps in the series and the covariates must match.",
            )

        kf = deepcopy(self.kf)

        y_values = series.values(copy=False)
        if self._expect_covariates:
            u_values = covariates.values(copy=False)

            # set control signal to 0 if it contains NaNs:
            u_values = np.nan_to_num(u_values, copy=True, nan=0.0)
        else:
            u_values = np.zeros((len(y_values), 0))

        # For each time step, we'll sample "n_samples" from a multivariate Gaussian
        # whose mean vector and covariance matrix come from the Kalman filter.
        sampled_outputs = np.zeros((len(y_values), self.dim_y, num_samples))

        for i in range(len(y_values)):
            y = y_values[i, :].reshape(-1, 1)
            u = u_values[i, :].reshape(-1, 1)

            if np.isnan(y).any():
                y = None

            kf.step(y, u)
            mean_vec = kf.y_filtereds[-1].reshape(
                self.dim_y,
            )

            if num_samples == 1:
                sampled_outputs[i, :, 0] = mean_vec
            else:
                # The measurement covariance matrix is given by the sum of the covariance matrix of the
                # state estimate (transformed by C) and the covariance matrix of the measurement noise.
                cov_matrix = (
                    kf.state_space.c @ kf.p_filtereds[-1] @ kf.state_space.c.T + kf.r
                )
                sampled_outputs[i, :, :] = np.random.multivariate_normal(
                    mean_vec, cov_matrix, size=num_samples
                ).T

        return series.with_values(sampled_outputs)
