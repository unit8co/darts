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

from darts.logging import get_logger
from darts.models.filtering.kalman_filter import KalmanFilter
from darts.models.forecasting.forecasting_model import (
    TransferableFutureCovariatesLocalForecastingModel,
)
from darts.timeseries import TimeSeries

logger = get_logger(__name__)


class KalmanForecaster(TransferableFutureCovariatesLocalForecastingModel):
    def __init__(
        self,
        dim_x: int = 1,
        kf: Optional[Kalman] = None,
        add_encoders: Optional[dict] = None,
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

                add_encoders={
                    'cyclic': {'future': ['month']},
                    'datetime_attribute': {'future': ['hour', 'dayofweek']},
                    'position': {'future': ['relative']},
                    'custom': {'future': [lambda idx: (idx.year - 1950) / 50]},
                    'transformer': Scaler()
                }
            ..
        """
        super().__init__(add_encoders=add_encoders)
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

    def predict(
        self,
        n: int,
        series: Optional[TimeSeries] = None,
        future_covariates: Optional[TimeSeries] = None,
        num_samples: int = 1,
        **kwargs,
    ) -> TimeSeries:
        # we override `predict()` to pass a non-None `series`, so that historic_future_covariates
        # will be passed to `_predict()`
        series = series if series is not None else self.training_series
        return super().predict(n, series, future_covariates, num_samples, **kwargs)

    def _predict(
        self,
        n: int,
        series: Optional[TimeSeries] = None,
        historic_future_covariates: Optional[TimeSeries] = None,
        future_covariates: Optional[TimeSeries] = None,
        num_samples: int = 1,
        verbose: bool = False,
    ) -> TimeSeries:

        super()._predict(
            n, series, historic_future_covariates, future_covariates, num_samples
        )
        time_index = self._generate_new_dates(n, input_series=series)
        placeholder_vals = np.zeros((n, self.training_series.width)) * np.nan
        series_future = TimeSeries.from_times_and_values(
            time_index,
            placeholder_vals,
            columns=self.training_series.columns,
            static_covariates=self.training_series.static_covariates,
            hierarchy=self.training_series.hierarchy,
        )

        series = series.append(series_future)
        if historic_future_covariates is not None:
            future_covariates = historic_future_covariates.append(future_covariates)

        filtered_series = self.darts_kf.filter(
            series=series, covariates=future_covariates, num_samples=num_samples
        )

        return filtered_series[-n:]

    def _is_probabilistic(self) -> bool:
        return True
