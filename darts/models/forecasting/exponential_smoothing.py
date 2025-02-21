"""
Exponential Smoothing
---------------------
"""

from typing import Any, Optional

import numpy as np
import statsmodels.tsa.holtwinters as hw

from darts.logging import get_logger
from darts.models.forecasting.forecasting_model import LocalForecastingModel
from darts.timeseries import TimeSeries
from darts.utils.utils import ModelMode, SeasonalityMode

logger = get_logger(__name__)


class ExponentialSmoothing(LocalForecastingModel):
    def __init__(
        self,
        trend: Optional[ModelMode] = ModelMode.ADDITIVE,
        damped: Optional[bool] = False,
        seasonal: Optional[SeasonalityMode] = SeasonalityMode.ADDITIVE,
        seasonal_periods: Optional[int] = None,
        random_state: int = 0,
        kwargs: Optional[dict[str, Any]] = None,
        **fit_kwargs,
    ):
        """Exponential Smoothing

        This is a wrapper around
        `statsmodels  Holt-Winters' Exponential Smoothing
        <https://www.statsmodels.org/stable/generated/statsmodels.tsa.holtwinters.ExponentialSmoothing.html>`_;
        we refer to this link for the original and more complete documentation of the parameters.

        `trend` must be a ``ModelMode`` Enum member. You can access the Enum with
         ``from darts.utils.utils import ModelMode``.
        `seasonal` must be a ``SeasonalityMode`` Enum member. You can access the Enum with
        ``from darts.utils.utils import SeasonalityMode``.

        ``ExponentialSmoothing(trend=ModelMode.NONE, seasonal=SeasonalityMode.NONE)`` corresponds to a single
        exponential smoothing.
        ``ExponentialSmoothing(trend=ModelMode.ADDITIVE, seasonal=SeasonalityMode.NONE)`` corresponds to a Holt's
        exponential smoothing.

        Please note that automatic `seasonal_period` selection (setting the `seasonal_periods` parameter equal to
        `None`) can sometimes lead to errors if the input time series is too short. In these cases we suggest to
        manually set the `seasonal_periods` parameter to a positive integer.

        Parameters
        ----------
        trend
            Type of trend component. Either ``ModelMode.ADDITIVE``, ``ModelMode.MULTIPLICATIVE``, ``ModelMode.NONE``,
            or ``None``. Defaults to ``ModelMode.ADDITIVE``.
        damped
            Should the trend component be damped. Defaults to False.
        seasonal
            Type of seasonal component. Either ``SeasonalityMode.ADDITIVE``, ``SeasonalityMode.MULTIPLICATIVE``,
            ``SeasonalityMode.NONE``, or ``None``. Defaults to ``SeasonalityMode.ADDITIVE``.
        seasonal_periods
            The number of periods in a complete seasonal cycle, e.g., 4 for quarterly data or 7 for daily
            data with a weekly cycle. If not set, inferred from frequency of the series.
        kwargs
            Some optional keyword arguments that will be used to call
            :func:`statsmodels.tsa.holtwinters.ExponentialSmoothing()`.
            See `the documentation
            <https://www.statsmodels.org/stable/generated/statsmodels.tsa.holtwinters.ExponentialSmoothing.html>`_.
        fit_kwargs
            Some optional keyword arguments that will be used to call
            :func:`statsmodels.tsa.holtwinters.ExponentialSmoothing.fit()`.
            See `the documentation
            <https://www.statsmodels.org/stable/generated/statsmodels.tsa.holtwinters.ExponentialSmoothing.fit.html>`_.

        Examples
        --------
        >>> from darts.datasets import AirPassengersDataset
        >>> from darts.models import ExponentialSmoothing
        >>> from darts.utils.utils import ModelMode, SeasonalityMode
        >>> series = AirPassengersDataset().load()
        >>> # using Holt's exponential smoothing
        >>> model = ExponentialSmoothing(trend=ModelMode.ADDITIVE, seasonal=SeasonalityMode.MULTIPLICATIVE)
        >>> model.fit(series)
        >>> pred = model.predict(6)
        >>> pred.values()
        array([[445.24283838],
               [418.22618932],
               [465.31305075],
               [494.95129261],
               [505.4770514 ],
               [573.31519186]])
        """
        super().__init__()
        self.trend = trend
        self.damped = damped
        self.seasonal = seasonal
        self.infer_seasonal_periods = seasonal_periods is None
        self.seasonal_periods = seasonal_periods
        self.constructor_kwargs = dict() if kwargs is None else kwargs
        self.fit_kwargs = fit_kwargs
        self.model = None
        np.random.seed(random_state)

    def fit(self, series: TimeSeries):
        super().fit(series)
        self._assert_univariate(series)
        series = self.training_series

        # if the model was initially created with `self.seasonal_periods=None`, make sure that
        # the model will try to automatically infer the index, otherwise it should use the
        # provided `seasonal_periods` value
        seasonal_periods_param = (
            None if self.infer_seasonal_periods else self.seasonal_periods
        )

        # set the seasonal periods parameter to a default value if it was not provided explicitly
        # and if it cannot be inferred due to the lack of a datetime index
        if self.seasonal_periods is None and series.has_range_index:
            seasonal_periods_param = 12

        hw_model = hw.ExponentialSmoothing(
            series.values(copy=False),
            trend=self.trend if self.trend is None else self.trend.value,
            damped_trend=self.damped,
            seasonal=self.seasonal if self.seasonal is None else self.seasonal.value,
            seasonal_periods=seasonal_periods_param,
            freq=series.freq if series.has_datetime_index else None,
            dates=series.time_index if series.has_datetime_index else None,
            **self.constructor_kwargs,
        )
        hw_results = hw_model.fit(**self.fit_kwargs)
        self.model = hw_results

        if self.infer_seasonal_periods:
            self.seasonal_periods = hw_model.seasonal_periods

        return self

    def predict(
        self,
        n: int,
        num_samples: int = 1,
        verbose: bool = False,
        show_warnings: bool = True,
    ):
        super().predict(n, num_samples)

        if num_samples == 1:
            forecast = self.model.forecast(n)
        else:
            forecast = np.expand_dims(
                self.model.simulate(n, repetitions=num_samples), axis=1
            )

        return self._build_forecast_series(forecast)

    @property
    def supports_multivariate(self) -> bool:
        return False

    @property
    def supports_probabilistic_prediction(self) -> bool:
        return True

    @property
    def min_train_series_length(self) -> int:
        if self.seasonal_periods is not None and self.seasonal_periods > 1:
            return 2 * self.seasonal_periods
        return 3
