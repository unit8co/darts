import numpy as np
import pandas as pd

from abc import ABC

from typing import Optional
from filterpy.kalman import KalmanFilter

from .filtering_model import FilteringModel
from ..timeseries import TimeSeries


class Kalman(FilteringModel, ABC):

    def __init__(
            self, 
            dim_x: int = 1, 
            dim_z: int = 1, 
            x = np.array([[1.]]),
            F = np.array([[1]]),
            H = np.array([[1]]),
            P = np.array([[1000.,    0.], [   0., 1000.] ]),
            Q = 0.1,
            kf: Optional[KalmanFilter] = None
        ):
        """ Kalman filter model
        This model doesn't predict the future values, instead it predicts 
        more precise values compared to noise interfered data
        
        Parameters
        ----------
        dim_x : int
            Size of the Kalman state vector
        dim_y : int
            Size of the measurement vector
        x : numpy array
            filter state estimate
        F : numpy array
            state transistion matrix
        H : numpy array
            measurement function
        P : numpy array
            covariance matrix
        Q : float
            Process uncertainty/noise multiplicator
        kf : KalmanFilter
            Instance of Kalman Filter (for advanced configuration)
        """
        super().__init__()
        if kf is None:
            self.dim_x = dim_x
            self.dim_z = dim_z
            self.kf = KalmanFilter(dim_x=dim_x, dim_z=dim_z)
            self.kf.F = F
            self.kf.H = H
            self.kf.x = x
            self.kf.P *= 1000.
            self.kf.Q[-1,-1] *= Q
            self.kf.Q[4:,4:] *= Q
            self.kf.R = 50
        else:
            self.kf = kf

    def __str__(self):
        return 'KALMAN({},{},{})'.format(self.dim_x, self.dim_z, self.kf.x)

    def fit(self, series: TimeSeries):
        super().fit(series)

    def filter(self):
        return self.training_series.map(self._kalman_iteration)

    def _kalman_iteration(self, observation):
        self.kf.predict()
        self.kf.update(observation)
        return self.kf.x[0][0]


    @property
    def min_train_series_length(self) -> int:
        return 10
