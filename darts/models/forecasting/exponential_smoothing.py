"""
Exponential Smoothing
---------------------
"""

from typing import Any, Literal

import numpy as np
import statsmodels.tsa.holtwinters as hw
from sklearn.utils import check_random_state

from darts import TimeSeries
from darts.models import LinearRegressionModel
from darts.models.forecasting.forecasting_model import (
    FutureCovariatesLocalForecastingModel,
)
from darts.utils._modes import ADDITIVE, MULTIPLICATIVE, validate_mode
from darts.utils._statsmodels_utils import SM_RNG_KWARG
from darts.utils.utils import random_method


class ExponentialSmoothing(FutureCovariatesLocalForecastingModel):
    @random_method
    def __init__(
        self,
        trend: Literal["additive", "multiplicative"] | None = ADDITIVE,
        damped: bool | None = False,
        seasonal: Literal["additive", "multiplicative"] | None = ADDITIVE,
        seasonal_periods: int | None = None,
        error: str | None = "add",
        random_errors: Any | None = None,
        random_state: int | None = None,
        min_train_length: int | None = None,
        add_encoders: dict | None = None,
        kwargs: dict[str, Any] | None = None,
        **fit_kwargs,
    ):
        """Exponential Smoothing

        This is a wrapper around
        `statsmodels  Holt-Winters' Exponential Smoothing
        <https://www.statsmodels.org/stable/generated/statsmodels.tsa.holtwinters.ExponentialSmoothing.html>`__;
        we refer to this link for the original and more complete documentation of the parameters.

        ``ExponentialSmoothing(trend=None, seasonal=None)`` corresponds to a single
        exponential smoothing.
        ``ExponentialSmoothing(trend="additive", seasonal=None)`` corresponds to a Holt's
        exponential smoothing.

        Please note that automatic `seasonal_period` selection (setting the `seasonal_periods` parameter equal to
        `None`) can sometimes lead to errors if the input time series is too short. In these cases we suggest to
        manually set the `seasonal_periods` parameter to a positive integer.

        This model supports future covariates. Since Holt-Winters' Exponential Smoothing has no native support for
        exogenous variables, the target series is first regressed against the future covariates with a
        :class:`~darts.models.forecasting.linear_regression_model.LinearRegressionModel` (OLS), and the Exponential
        Smoothing model is then fitted on the in-sample residuals of this regression. At prediction time, the
        residuals are forecasted and the regression's prediction on the future covariates is added back. This approach
        is the same as the one used by :class:`~darts.models.forecasting.sf_model.StatsForecastModel`.
        The regression residuals can be zero or negative even when the original target series is strictly positive,
        so multiplicative ``trend`` / ``seasonal`` components can fail when used with future covariates.

        Parameters
        ----------
        trend
            Type of trend component: ``"additive"``, ``"multiplicative"``, or ``None`` to disable.
            Defaults to ``"additive"``.
        damped
            Should the trend component be damped. Defaults to False.
        seasonal
            Type of seasonal component: ``"additive"``, ``"multiplicative"``, or ``None`` to disable.
            Defaults to ``"additive"``.
        seasonal_periods
            The number of periods in a complete seasonal cycle, e.g., 4 for quarterly data or 7 for daily
            data with a weekly cycle. If not set, inferred from frequency of the series.
        error
            Specifies the type of error model for state space formulation to use when using predict()
            with ``num_samples > 1``. Default is `"add"`.
            Will be passed to statsmodels' :func:`simulate()` method. See the documentation `here
            <https://www.statsmodels.org/stable/generated/statsmodels.tsa.holtwinters.HoltWintersResults.simulate.html>`__
            for more information.
        random_errors
            Specifies how the random errors should be obtained, when using predict() with ``num_samples > 1``.
            Will be passed to statsmodels' :func:`simulate()` method. See the documentation `here
            <https://www.statsmodels.org/stable/generated/statsmodels.tsa.holtwinters.HoltWintersResults.simulate.html>`__
            for more information.
        random_state
            Controls the randomness for reproducible forecasting.
        min_train_length
            Optionally, set a custom minimum required training series length for this model to allow training on
            shorter input series. By default, Darts sets a conservative minimum length to avoid downstream issues.
            Changing this value might lead to such downstream issues. Default: ``None`` (keeps the default requirement).
        add_encoders
            A large number of future covariates can be automatically generated with `add_encoders`.
            This can be done by adding multiple pre-defined index encoders and/or custom user-made functions that
            will be used as index encoders. Additionally, a transformer such as Darts' :class:`Scaler` can be added to
            transform the generated covariates. This happens all under one hood and only needs to be specified at
            model creation.
            Read :meth:`SequentialEncoder <darts.dataprocessing.encoders.SequentialEncoder>` to find out more about
            ``add_encoders``. Default: ``None``.
        kwargs
            Some optional keyword arguments that will be used to call
            :func:`statsmodels.tsa.holtwinters.ExponentialSmoothing()`.
            See `the documentation
            <https://www.statsmodels.org/stable/generated/statsmodels.tsa.holtwinters.ExponentialSmoothing.html>`__.
        fit_kwargs
            Some optional keyword arguments that will be used to call
            :func:`statsmodels.tsa.holtwinters.ExponentialSmoothing.fit()`.
            See `the documentation
            <https://www.statsmodels.org/stable/generated/statsmodels.tsa.holtwinters.ExponentialSmoothing.fit.html>`__.

        Examples
        --------
        >>> from darts.datasets import AirPassengersDataset
        >>> from darts.models import ExponentialSmoothing
        >>> series = AirPassengersDataset().load()
        >>> # using Holt's exponential smoothing
        >>> model = ExponentialSmoothing(trend="additive", seasonal="multiplicative")
        >>> model.fit(series)
        >>> pred = model.predict(6)
        >>> print(pred.values())
        [[445.24283838]
         [418.22618932]
         [465.31305075]
         [494.95129261]
         [505.4770514 ]
         [573.31519186]]
        """
        super().__init__(add_encoders=add_encoders, min_train_length=min_train_length)
        self.trend = validate_mode(
            trend, param="trend", allowed={ADDITIVE, MULTIPLICATIVE, None}
        )
        self.damped = damped
        self.seasonal = validate_mode(
            seasonal, param="seasonal", allowed={ADDITIVE, MULTIPLICATIVE, None}
        )
        self.infer_seasonal_periods = seasonal_periods is None
        self.seasonal_periods = seasonal_periods
        self.error = error
        self.random_errors = random_errors
        self.constructor_kwargs = dict() if kwargs is None else kwargs
        self.fit_kwargs = fit_kwargs
        self.model = None
        # future covariates support is added through the use of a linear model
        self._linreg: LinearRegressionModel | None = None

    def _fit(
        self,
        series: TimeSeries,
        future_covariates: TimeSeries | None = None,
        verbose: bool | None = None,
    ):
        super()._fit(series, future_covariates, verbose=verbose)
        self._assert_univariate(series)
        series = self.training_series

        if future_covariates is None:
            self._linreg = None
            target = series.values(copy=False)
        else:
            # perform OLS and get in-sample residuals
            self._linreg = LinearRegressionModel(
                lags_future_covariates=[0],
                use_static_covariates=False,
            )
            self._linreg.fit(
                series, future_covariates=future_covariates, verbose=verbose
            )
            fitted_values = self._linreg.model.predict(
                X=future_covariates.slice_intersect_values(series, copy=False)[:, :, 0]
            )
            target = (series.values(copy=False).flatten() - fitted_values).reshape(
                -1, 1
            )

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
            target,
            trend=self.trend,
            damped_trend=self.damped,
            seasonal=self.seasonal,
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

    @random_method
    @random_method
    def _predict(
        self,
        n: int,
        future_covariates: TimeSeries | None = None,
        num_samples: int = 1,
        predict_likelihood_parameters: bool = False,
        verbose: bool = False,
        random_state: int | None = None,
        **kwargs,
    ):
        super()._predict(
            n,
            future_covariates=future_covariates,
            num_samples=num_samples,
            predict_likelihood_parameters=predict_likelihood_parameters,
            verbose=verbose,
        )

        if num_samples == 1:
            forecast = self.model.forecast(n)
        else:
            rng = check_random_state(random_state)

            forecast = np.expand_dims(
                self.model.simulate(
                    n,
                    repetitions=num_samples,
                    random_errors=self.random_errors,
                    error=self.error,
                    **{SM_RNG_KWARG: rng},
                ),
                axis=1,
            )

        if self._linreg is not None:
            # the model was trained on the regression residuals, add back the regression's prediction
            mu_linreg = self._linreg.predict(
                n, series=self.training_series, future_covariates=future_covariates
            )
            forecast = forecast + mu_linreg.values(copy=False).reshape(
                (n,) + (1,) * (forecast.ndim - 1)
            )

        return self._build_forecast_series(forecast)

    @property
    def supports_multivariate(self) -> bool:
        return False

    @property
    def supports_probabilistic_prediction(self) -> bool:
        return True

    @property
    def _target_window_lengths(self) -> tuple[int, int]:
        if self.seasonal_periods is not None and self.seasonal_periods > 1:
            input_length_inferred = 2 * self.seasonal_periods
        else:
            input_length_inferred = 3
        return self._min_train_input_length(input_length_inferred), 0
