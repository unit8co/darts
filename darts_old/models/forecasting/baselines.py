"""
Baseline Models
---------------

A collection of simple benchmark models for single uni- and multivariate series.
"""

from typing import Optional

import numpy as np

from darts import TimeSeries
from darts.logging import get_logger, raise_if_not
from darts.models.forecasting.forecasting_model import (
    LocalForecastingModel,
)

logger = get_logger(__name__)


class NaiveMean(LocalForecastingModel):
    def __init__(self):
        """Naive Mean Model

        This model has no parameter, and always predicts the
        mean value of the training series.

        Examples
        --------
        >>> from darts.datasets import AirPassengersDataset
        >>> from darts.models import NaiveMean
        >>> series = AirPassengersDataset().load()
        >>> model = NaiveMean()
        >>> model.fit(series)
        >>> pred = model.predict(6)
        >>> print(pred.values())
        [[280.29861111]
         [280.29861111]
         [280.29861111]
         [280.29861111]
         [280.29861111]
         [280.29861111]]
        """
        super().__init__()
        self.mean_val = None

    @property
    def supports_multivariate(self) -> bool:
        return True

    def fit(self, series: TimeSeries, verbose: Optional[bool] = None):
        super().fit(series, verbose=verbose)

        self.mean_val = np.mean(series.values(copy=False), axis=0)
        return self

    def predict(
        self,
        n: int,
        num_samples: int = 1,
        verbose: Optional[bool] = None,
        show_warnings: bool = True,
        random_state: Optional[int] = None,
    ):
        super().predict(n, num_samples, verbose=verbose)
        forecast = np.tile(self.mean_val, (n, 1))
        return self._build_forecast_series(forecast)


class NaiveSeasonal(LocalForecastingModel):
    def __init__(self, K: int = 1):
        """Naive Seasonal Model

        This model always predicts the value of `K` time steps ago.
        When `K=1`, this model predicts the last value of the training set.
        When `K>1`, it repeats the last `K` values of the training set.

        Parameters
        ----------
        K
            the number of last time steps of the training set to repeat

        Examples
        --------
        >>> from darts.datasets import AirPassengersDataset
        >>> from darts.models import NaiveSeasonal
        >>> series = AirPassengersDataset().load()
        # prior analysis suggested seasonality of 12
        >>> model = NaiveSeasonal(K=12)
        >>> model.fit(series)
        >>> pred = model.predict(6)
        >>> print(pred.values())
        [[417.]
         [391.]
         [419.]
         [461.]
         [472.]
         [535.]]
        """
        super().__init__()
        self.last_k_vals = None
        self.K = K

    @property
    def supports_multivariate(self) -> bool:
        return True

    @property
    def _target_window_lengths(self):
        return max(self.K, 3), 0

    def fit(self, series: TimeSeries, verbose: Optional[bool] = None):
        super().fit(series, verbose=verbose)

        raise_if_not(
            len(series) >= self.K,
            f"The time series requires at least K={self.K} points",
            logger,
        )
        self.last_k_vals = series.values(copy=False)[-self.K :, :]
        return self

    def predict(
        self,
        n: int,
        num_samples: int = 1,
        verbose: Optional[bool] = None,
        show_warnings: bool = True,
        random_state: Optional[int] = None,
    ):
        super().predict(n, num_samples, verbose=verbose)
        forecast = np.array([self.last_k_vals[i % self.K, :] for i in range(n)])
        return self._build_forecast_series(forecast)


class NaiveDrift(LocalForecastingModel):
    def __init__(self):
        """Naive Drift Model

        This model fits a line between the first and last point of the training series,
        and extends it in the future. For a training series of length :math:`T`, we have:

        .. math:: \\hat{y}_{T+h} = y_T + h\\left( \\frac{y_T - y_1}{T - 1} \\right)

        Examples
        --------
        >>> from darts.datasets import AirPassengersDataset
        >>> from darts.models import NaiveDrift
        >>> series = AirPassengersDataset().load()
        >>> model = NaiveDrift()
        >>> model.fit(series)
        >>> pred = model.predict(6)
        >>> print(pred.values())
        [[434.23776224]
         [436.47552448]
         [438.71328671]
         [440.95104895]
         [443.18881119]
         [445.42657343]]
        """
        super().__init__()

    @property
    def supports_multivariate(self) -> bool:
        return True

    def fit(self, series: TimeSeries, verbose: Optional[bool] = None):
        super().fit(series, verbose=verbose)
        assert series.n_samples == 1, "This model expects deterministic time series"
        return self

    def predict(
        self,
        n: int,
        num_samples: int = 1,
        verbose: Optional[bool] = None,
        show_warnings: bool = True,
        random_state: Optional[int] = None,
    ):
        super().predict(n, num_samples, verbose=verbose)
        first, last = (
            self.training_series.first_values(),
            self.training_series.last_values(),
        )
        slope = (last - first) / (len(self.training_series) - 1)
        last_value = last + slope * n
        forecast = np.linspace(last, last_value, num=n + 1)[1:]
        return self._build_forecast_series(forecast)


class NaiveMovingAverage(LocalForecastingModel):
    def __init__(self, input_chunk_length: int = 1):
        """Naive Moving Average Model

        This model forecasts using an autoregressive moving average (ARMA).

        Parameters
        ----------
        input_chunk_length
            The size of the sliding window used to calculate the moving average

        Examples
        --------
        >>> from darts.datasets import AirPassengersDataset
        >>> from darts.models import NaiveMovingAverage
        >>> series = AirPassengersDataset().load()
        # using the average of the last 6 months
        >>> model = NaiveMovingAverage(input_chunk_length=6)
        >>> pred = model.predict(6)
        >>> print(pred.values())
        [[503.16666667]
         [483.36111111]
         [462.9212963 ]
         [455.40817901]
         [454.47620885]
         [465.22224366]]
        """
        super().__init__()
        self.input_chunk_length = input_chunk_length
        self.rolling_window = None

    @property
    def supports_multivariate(self) -> bool:
        return True

    @property
    def _target_window_lengths(self):
        return self.input_chunk_length, 0

    def __str__(self):
        return f"NaiveMovingAverage({self.input_chunk_length})"

    def fit(self, series: TimeSeries, verbose: Optional[bool] = None):
        super().fit(series, verbose=verbose)
        raise_if_not(
            series.is_deterministic,
            "This model expects deterministic time series",
            logger,
        )

        self.rolling_window = series[-self.input_chunk_length :].values(copy=False)
        return self

    def predict(
        self,
        n: int,
        num_samples: int = 1,
        verbose: Optional[bool] = None,
        show_warnings: bool = True,
        random_state: Optional[int] = None,
    ):
        super().predict(n, num_samples, verbose=verbose)

        predictions_with_observations = np.concatenate(
            (self.rolling_window, np.zeros(shape=(n, self.rolling_window.shape[1]))),
            axis=0,
        )
        rolling_sum = sum(self.rolling_window)

        chunk_length = self.input_chunk_length
        for i in range(chunk_length, chunk_length + n):
            prediction = rolling_sum / chunk_length
            predictions_with_observations[i] = prediction
            lost_value = predictions_with_observations[i - chunk_length]
            rolling_sum += prediction - lost_value
        return self._build_forecast_series(predictions_with_observations[-n:])
