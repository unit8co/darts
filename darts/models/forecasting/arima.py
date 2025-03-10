"""
ARIMA
-----

Models for ARIMA (Autoregressive integrated moving average) [1]_.
The implementations is wrapped around `statsmodels <https://github.com/statsmodels/statsmodels>`_.

References
----------
.. [1] https://wikipedia.org/wiki/Autoregressive_integrated_moving_average
"""

import sys
from collections.abc import Sequence
from typing import Literal, Optional, Union

if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:
    from typing_extensions import TypeAlias

import numpy as np
from statsmodels.tsa.arima.model import ARIMA as staARIMA

from darts.logging import get_logger
from darts.models.forecasting.forecasting_model import (
    TransferableFutureCovariatesLocalForecastingModel,
)
from darts.timeseries import TimeSeries

logger = get_logger(__name__)


IntOrIntSequence: TypeAlias = Union[int, Sequence[int]]


class ARIMA(TransferableFutureCovariatesLocalForecastingModel):
    def __init__(
        self,
        p: IntOrIntSequence = 12,
        d: int = 1,
        q: IntOrIntSequence = 0,
        seasonal_order: tuple[int, IntOrIntSequence, IntOrIntSequence, int] = (
            0,
            0,
            0,
            0,
        ),
        trend: Optional[Union[Literal["n", "c", "t", "ct"], list[int]]] = None,
        random_state: Optional[int] = None,
        add_encoders: Optional[dict] = None,
    ):
        """ARIMA
        ARIMA-type models extensible with exogenous variables (future covariates)
        and seasonal components.

        Parameters
        ----------
        p : int | Sequence[int]
            Order (number of time lags) of the autoregressive model (AR).
            If a sequence of integers, specifies the exact lags to include.
        d : int
            The order of differentiation; i.e., the number of times the data
            have had past values subtracted (I).
        q : int | Sequence[int]
            The size of the moving average window (MA).
            If a sequence of integers, specifies the exact lags to include in the window.
        seasonal_order: Tuple[int | Sequence[int], int, int | Sequence[int], int]
            The (P,D,Q,s) order of the seasonal component for the AR parameters (P),
            differences (D), MA parameters (Q) and periodicity (s). D and s are always integers,
            while P and Q may either be integers or sequence of positive integers
            specifying exactly which lag orders are included.
        trend: Literal['n', 'c', 't', 'ct'] | list[int], optional
            Parameter controlling the deterministic trend. Either a string or list of integers.
            If a string, can be 'n' for no trend, 'c' for a constant term, 't' for a linear trend in time,
            and 'ct' for a constant term and linear trend.
            If a list of integers, defines a polynomial according to `numpy.poly1d` [1]_. E.g., `[1,1,0,1]` would
            translate to :math:`a + bt + ct^3`.
            Trend term of lower order than `d + D` cannot be as they would be eliminated due to the differencing
            operation.
            Default is 'c' for models without integration, and 'n' for models with integration.
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

        Examples
        --------
        >>> from darts.datasets import AirPassengersDataset
        >>> from darts.models import ARIMA
        >>> from darts.utils.timeseries_generation import datetime_attribute_timeseries
        >>> series = AirPassengersDataset().load()
        >>> # optionally, use some future covariates; e.g. the value of the month encoded as a sine and cosine series
        >>> future_cov = datetime_attribute_timeseries(series, "month", cyclic=True, add_length=6)
        >>> # define ARIMA parameters
        >>> model = ARIMA(p=12, d=1, q=2)
        >>> model.fit(series, future_covariates=future_cov)
        >>> pred = model.predict(6, future_covariates=future_cov)
        >>> pred.values()
        array([[451.36489334],
               [416.88972829],
               [443.10520391],
               [481.07892911],
               [502.11286509],
               [555.50153984]])

        References
        ----------
        .. [1] https://numpy.org/doc/stable/reference/generated/numpy.poly1d.html
        """
        super().__init__(add_encoders=add_encoders)
        self.order = p, d, q
        self.seasonal_order = seasonal_order
        self.trend = trend
        self.model = None
        self._random_state = (
            random_state
            if random_state is None
            else np.random.RandomState(random_state)
        )

    @property
    def supports_multivariate(self) -> bool:
        return False

    def _fit(self, series: TimeSeries, future_covariates: Optional[TimeSeries] = None):
        super()._fit(series, future_covariates)

        self._assert_univariate(series)

        # storing to restore the statsmodels model results object
        self.training_historic_future_covariates = future_covariates

        m = staARIMA(
            series.values(copy=False),
            exog=future_covariates.values(copy=False) if future_covariates else None,
            order=self.order,
            seasonal_order=self.seasonal_order,
            trend=self.trend,
        )
        self.model = m.fit()

        return self

    def _predict(
        self,
        n: int,
        series: Optional[TimeSeries] = None,
        historic_future_covariates: Optional[TimeSeries] = None,
        future_covariates: Optional[TimeSeries] = None,
        num_samples: int = 1,
        verbose: bool = False,
    ) -> TimeSeries:
        if num_samples > 1 and self.trend:
            logger.warning(
                "Trends are not well supported yet for getting probabilistic forecasts with ARIMA."
                "If you run into issues, try calling fit() with num_samples=1 or removing the trend from"
                "your model."
            )

        super()._predict(
            n, series, historic_future_covariates, future_covariates, num_samples
        )

        # updating statsmodels results object state with the new ts and covariates
        if series is not None:
            self.model = self.model.apply(
                series.values(copy=False),
                exog=(
                    historic_future_covariates.values(copy=False)
                    if historic_future_covariates
                    else None
                ),
            )

        if num_samples == 1:
            forecast = self.model.forecast(
                steps=n,
                exog=(
                    future_covariates.values(copy=False) if future_covariates else None
                ),
            )
        else:
            forecast = self.model.simulate(
                nsimulations=n,
                repetitions=num_samples,
                initial_state=self.model.states.predicted[-1, :],
                random_state=self._random_state,
                anchor="end",
                exog=(
                    future_covariates.values(copy=False) if future_covariates else None
                ),
            )

        # restoring statsmodels results object state
        if series is not None:
            self.model = self.model.apply(
                self._orig_training_series.values(copy=False),
                exog=(
                    self.training_historic_future_covariates.values(copy=False)
                    if self.training_historic_future_covariates
                    else None
                ),
            )

        return self._build_forecast_series(forecast)

    @property
    def supports_probabilistic_prediction(self) -> bool:
        return True

    @property
    def min_train_series_length(self) -> int:
        return 30
