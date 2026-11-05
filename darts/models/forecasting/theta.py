"""
Theta Method
------------
"""

import math
from typing import Any, Literal, Self

import numpy as np
import statsmodels.tsa.holtwinters as hw

from darts import TimeSeries
from darts.logging import get_logger, raise_log
from darts.models.forecasting.forecasting_model import LocalForecastingModel
from darts.utils._modes import (
    ADDITIVE,
    EXPONENTIAL,
    LINEAR,
    MULTIPLICATIVE,
    validate_mode,
)
from darts.utils.statistics import (
    check_seasonality,
    extract_trend_and_seasonality,
    remove_from_series,
)

logger = get_logger(__name__)
ALPHA_START = 0.2


class Theta(LocalForecastingModel):
    # .. todo: Implement OTM: Optimized Theta Method (https://arxiv.org/pdf/1503.03529.pdf)
    # .. todo: Try with something different than SES? They do that in the paper.
    def __init__(
        self,
        theta: int = 2,
        seasonality_period: int | None = None,
        season_mode: Literal["additive", "multiplicative"] | None = MULTIPLICATIVE,
        min_train_length: int | None = None,
    ):
        """
        An implementation of the Theta method with configurable `theta` parameter. See [1]_.

        The training time series is de-seasonalized according to `seasonality_period`,
        or an inferred seasonality period.

        Parameters
        ----------
        theta
            Value of the theta parameter. Defaults to 2. Cannot be set to 0.
            If `theta = 1`, then the theta method restricts to a simple exponential smoothing (SES)
        seasonality_period
            User-defined seasonality period. If not set, will be tentatively inferred from the training series upon
            calling :func:`fit()`.
        season_mode
            Type of seasonality: ``"additive"``, ``"multiplicative"``, or ``None`` to disable seasonality.
            Defaults to ``"multiplicative"``.
        min_train_length
            Optionally, set a custom minimum required training series length for this model to allow training on
            shorter input series. By default, Darts sets a conservative minimum length to avoid downstream issues.
            Changing this value might lead to such downstream issues. Default: ``None`` (keeps the default requirement).

        References
        ----------
        .. [1] `Unmasking the Theta method <https://robjhyndman.com/papers/Theta.pdf`

        Examples
        --------
        >>> from darts.datasets import AirPassengersDataset
        >>> from darts.models import Theta
        >>> series = AirPassengersDataset().load()
        >>> # using the canonical Theta method
        >>> model = Theta(theta=2)
        >>> model.fit(series)
        >>> pred = model.predict(6)
        >>> print(pred.values())
        [[442.7256909 ]
         [433.74381763]
         [494.54534585]
         [480.36937856]
         [481.06675142]
         [545.80068173]]
        """

        super().__init__(min_train_length=min_train_length)

        self.model = None
        self.coef = 1
        self.alpha = 1
        self.length = 0
        self.theta = theta
        self.is_seasonal = False
        self.seasonality = None
        self.seasonality_period = seasonality_period
        self.season_period = None
        self.season_mode = validate_mode(
            season_mode,
            param="season_mode",
            allowed={ADDITIVE, MULTIPLICATIVE, None},
        )

        if self.theta == 0:
            raise_log(ValueError("The parameter theta cannot be equal to 0."))

    def fit(self, series: TimeSeries, verbose: bool | None = False):
        super().fit(series, verbose=verbose)
        self._assert_univariate(series)

        self.length = len(series)

        # Check for statistical significance of user-defined season period
        # or infers season_period from the TimeSeries itself.
        if self.season_mode is None:
            self.season_period = 1
        else:
            self.season_period = self.seasonality_period
        if self.season_period is None:
            max_lag = len(series) // 2
            self.is_seasonal, self.season_period = check_seasonality(
                series, self.season_period, max_lag=max_lag
            )
        else:
            # force the user-defined seasonality to be considered as a true seasonal period.
            self.is_seasonal = self.season_period > 1

        new_series = series

        # Store and remove seasonality effect if there is any.
        if self.is_seasonal:
            if (new_series.univariate_values() <= 0).any() and (
                self.season_mode == MULTIPLICATIVE
            ):
                self.season_mode = ADDITIVE
                logger.warning(
                    "Time series has negative or zero values. Fallback to additive seasonality."
                )
            _, self.seasonality = extract_trend_and_seasonality(
                series, self.season_period, model=self.season_mode
            )
            new_series = remove_from_series(
                series, self.seasonality, model=self.season_mode
            )

        # SES part of the decomposition.
        self.model = hw.SimpleExpSmoothing(
            endog=new_series.values(copy=False),
            initialization_method="estimated",
        ).fit()

        # Linear Regression part of the decomposition. We select the degree one coefficient.
        b_theta = np.polyfit(
            np.array([i for i in range(0, self.length)]),
            (1.0 - self.theta) * new_series.values(copy=False),
            1,
        )[0]

        # Normalization of the coefficient b_theta.
        self.coef = b_theta / (-self.theta)

        self.alpha = self.model.params["smoothing_level"]
        if self.alpha == 0.0:  # pragma: no cover
            self.model = hw.SimpleExpSmoothing(
                endog=new_series.values(copy=False),
                initialization_method="known",
                initial_level=ALPHA_START,
            ).fit()
            self.alpha = self.model.params["smoothing_level"]

        return self

    def predict(
        self,
        n: int,
        num_samples: int = 1,
        verbose: bool | None = None,
        show_warnings: bool = True,
        random_state: int | None = None,
    ) -> "TimeSeries":
        super().predict(n, num_samples, verbose=verbose)

        # Forecast of the SES part.
        forecast = self.model.forecast(n)

        # Forecast of the Linear Regression part.
        drift = self.coef * np.array([
            i + (1 - (1 - self.alpha) ** self.length) / self.alpha for i in range(0, n)
        ])

        # Combining the two forecasts
        forecast += drift

        # Re-apply the seasonal trend of the TimeSeries
        if self.is_seasonal:
            replicated_seasonality = np.tile(
                self.seasonality.to_series()[-self.season_period :],
                math.ceil(n / self.season_period),
            )[:n]
            if self.season_mode == MULTIPLICATIVE:
                forecast *= replicated_seasonality
            elif self.season_mode == ADDITIVE:
                forecast += replicated_seasonality

        return self._build_forecast_series(forecast)

    @property
    def supports_multivariate(self) -> bool:
        return False

    @property
    def _target_window_lengths(self) -> tuple[int, int]:
        if (
            self.season_mode is not None
            and self.seasonality_period
            and self.seasonality_period > 1
        ):
            input_length_inferred = 2 * self.seasonality_period
        else:
            input_length_inferred = 3
        return self._min_train_input_length(input_length_inferred), 0


class FourTheta(LocalForecastingModel):
    def __init__(
        self,
        theta: int = 2,
        seasonality_period: int | None = None,
        season_mode: Literal["additive", "multiplicative"] | None = MULTIPLICATIVE,
        model_mode: Literal["additive", "multiplicative"] = ADDITIVE,
        trend_mode: Literal["linear", "exponential"] = LINEAR,
        normalization: bool = True,
        min_train_length: int | None = None,
    ):
        """
        An implementation of the 4Theta method with configurable `theta` parameter.

        See M4 competition `solution <https://github.com/Mcompetitions/M4-methods/blob/master/4Theta%20method.R>`__.

        The training time series is de-seasonalized according to `seasonality_period`,
        or an inferred seasonality period.

        When called with `theta = X`, `model_mode = "additive"` and `trend_mode = "linear"`,
        this model is equivalent to calling `Theta(theta=X)`.

        Parameters
        ----------
        theta
            Value of the theta parameter. Defaults to 2.
            If theta = 1, then the fourtheta method restricts to a simple exponential smoothing (SES).
            If theta = 0, then the fourtheta method restricts to a simple `trend_mode` regression.
        seasonality_period
            User-defined seasonality period. If not set, will be tentatively inferred from the training series upon
            calling `fit()`.
        model_mode
            Type of model combining the Theta lines: ``"additive"`` or ``"multiplicative"``.
            Defaults to ``"additive"``.
        season_mode
            Type of seasonality: ``"additive"``, ``"multiplicative"``, or ``None`` to disable seasonality.
            Defaults to ``"multiplicative"``.
        trend_mode
            Type of trend to fit: ``"linear"`` or ``"exponential"``.
            Defaults to ``"linear"``.
        normalization
            If `True`, the data is normalized so that the mean is 1. Defaults to `True`.
        min_train_length
            Optionally, set a custom minimum required training series length for this model to allow training on
            shorter input series. By default, Darts sets a conservative minimum length to avoid downstream issues.
            Changing this value might lead to such downstream issues. Default: ``None`` (keeps the default requirement).

        Notes
        -----
        Even though this model is an improvement of :class:`Theta`, it is a naive
        implementation of the algorithm, which can potentially be slower.

        Examples
        --------
        >>> from darts.datasets import AirPassengersDataset
        >>> from darts.models import FourTheta
        >>> series = AirPassengersDataset().load()
        >>> model = FourTheta(theta=2)
        >>> model.fit(series)
        >>> pred = model.predict(6)
        >>> print(pred.values())
        [[443.3949283 ]
         [434.39769555]
         [495.28886231]
         [481.08962991]
         [481.78610361]
         [546.61463773]]
        """

        super().__init__(min_train_length=min_train_length)

        self.model = None
        self.drift = None
        self.mean = 1
        self.length = 0
        self.theta = theta
        self.is_seasonal = False
        self.seasonality = None
        self.seasonality_period = seasonality_period
        self.season_period = None
        self.model_mode = validate_mode(
            model_mode,
            param="model_mode",
            allowed={ADDITIVE, MULTIPLICATIVE},
        )
        self.season_mode = validate_mode(
            season_mode,
            param="season_mode",
            allowed={ADDITIVE, MULTIPLICATIVE, None},
        )
        self.trend_mode = validate_mode(
            trend_mode,
            param="trend_mode",
            allowed={LINEAR, EXPONENTIAL},
        )
        self.wses = 0 if self.theta == 0 else (1 / theta)
        self.wdrift = 1 - self.wses
        self.fitted_values = None
        self.normalization = normalization

    def fit(self, series, verbose: bool | None = False):
        super().fit(series, verbose=verbose)

        self.length = len(series)
        # normalization of data
        if self.normalization:
            self.mean = series.to_dataframe(copy=False).mean().mean()
            if np.isclose(self.mean, 0):
                raise_log(
                    ValueError(
                        "The mean value of the provided series is too close to zero to perform normalization."
                    ),
                )
            new_ts = series / self.mean
        else:
            new_ts = series

        # Check for statistical significance of user-defined season period
        # or infers season_period from the TimeSeries itself.
        if self.season_mode is None:
            self.season_period = 1
        else:
            self.season_period = self.seasonality_period
        if self.season_period is None:
            max_lag = len(series) // 2
            self.is_seasonal, self.season_period = check_seasonality(
                series, self.season_period, max_lag=max_lag
            )
        else:
            # force the user-defined seasonality to be considered as a true seasonal period.
            self.is_seasonal = self.season_period > 1

        if (new_ts.univariate_values() <= 0).any():
            self.model_mode = ADDITIVE
            self.trend_mode = LINEAR
            if self.season_mode == MULTIPLICATIVE:
                self.season_mode = ADDITIVE
            logger.warning(
                "Time series has negative values. Fallback to additive and linear model"
            )

        # Store and remove seasonality effect if there is any.
        if self.is_seasonal:
            _, self.seasonality = extract_trend_and_seasonality(
                new_ts, self.season_period, model=self.season_mode
            )
            new_ts = remove_from_series(
                new_ts, self.seasonality, model=self.season_mode
            )

        ts_values = new_ts.univariate_values()
        if (ts_values <= 0).any() and (
            self.model_mode != ADDITIVE or self.trend_mode != LINEAR
        ):
            self.model_mode = ADDITIVE
            self.trend_mode = LINEAR
            logger.warning(
                "Time series has negative values. Fallback to additive and linear model"
            )

        # Drift part of the decomposition
        if self.trend_mode == LINEAR:
            linreg = ts_values
        else:
            linreg = np.log(ts_values)
        self.drift = np.poly1d(np.polyfit(np.arange(self.length), linreg, 1))
        theta0_in = self.drift(np.arange(self.length))
        if self.trend_mode == EXPONENTIAL:
            theta0_in = np.exp(theta0_in)

        if (theta0_in > 0).all() and self.model_mode == MULTIPLICATIVE:
            theta_t = (ts_values**self.theta) * (theta0_in ** (1 - self.theta))
        else:
            if self.model_mode == MULTIPLICATIVE:
                logger.warning("Negative Theta line. Fallback to additive model")
                self.model_mode = ADDITIVE
            theta_t = self.theta * ts_values + (1 - self.theta) * theta0_in

        # SES part of the decomposition.
        self.model = hw.SimpleExpSmoothing(
            endog=theta_t,
            initialization_method="estimated",
        ).fit()
        theta2_in = self.model.fittedvalues

        if (theta2_in > 0).all() and self.model_mode == MULTIPLICATIVE:
            self.fitted_values = theta2_in**self.wses * theta0_in**self.wdrift
        else:
            if self.model_mode == MULTIPLICATIVE:
                self.model_mode = ADDITIVE
                logger.warning("Negative Theta line. Fallback to additive model")
                theta_t = self.theta * ts_values + (1 - self.theta) * theta0_in
                self.model = hw.SimpleExpSmoothing(
                    endog=theta_t,
                    initialization_method="estimated",
                ).fit()
                theta2_in = self.model.fittedvalues
            self.fitted_values = self.wses * theta2_in + self.wdrift * theta0_in
        if self.is_seasonal:
            if self.season_mode == ADDITIVE:
                self.fitted_values += self.seasonality.univariate_values(copy=False)
            elif self.season_mode == MULTIPLICATIVE:
                self.fitted_values *= self.seasonality.univariate_values(copy=False)
        # Fitted values are the results of the fit of the model on the train series. A good fit of the model
        # will lead to fitted_values similar to ts. But one cannot see if it overfits.
        if self.normalization:
            self.fitted_values *= self.mean

        return self

    def predict(
        self,
        n: int,
        num_samples: int = 1,
        verbose: bool | None = None,
        show_warnings: bool = True,
        random_state: int | None = None,
    ) -> TimeSeries:
        super().predict(n, num_samples, verbose=verbose)

        # Forecast of the SES part.
        forecast = self.model.forecast(n)

        # Forecast of the Linear Regression part.
        drift = self.drift(np.arange(self.length, self.length + n))
        if self.trend_mode == EXPONENTIAL:
            drift = np.exp(drift)

        if self.model_mode == ADDITIVE:
            forecast = self.wses * forecast + self.wdrift * drift
        else:
            forecast = forecast**self.wses * drift**self.wdrift

        # Re-apply the seasonal trend of the TimeSeries
        if self.is_seasonal:
            replicated_seasonality = np.tile(
                self.seasonality.to_series()[-self.season_period :],
                math.ceil(n / self.season_period),
            )[:n]
            if self.season_mode == MULTIPLICATIVE:
                forecast *= replicated_seasonality
            else:
                forecast += replicated_seasonality

        if self.normalization:
            forecast *= self.mean

        return self._build_forecast_series(forecast)

    @classmethod
    def select_best_model(
        cls,
        ts: TimeSeries,
        thetas: list[int] | None = None,
        m: int | None = None,
        normalization: bool = True,
        n_jobs: int = 1,
    ) -> tuple[Self, dict[str, Any], float]:
        """
        Performs a grid search over all hyper parameters to select the best model,
        using the fitted values on the training series `ts`.


        Uses 'LocalForecastingModel.gridsearch' with 'use_fitted_values=True' and 'metric=metrics.mae`.

        Parameters
        ----------
        ts
            The TimeSeries on which the model will be tested.
        thetas
            A list of thetas to loop over. Defaults to [1, 2, 3].
        m
            Optionally, the season used to decompose the time series.
        normalization
            If `True`, the data is normalized so that the mean is 1. Defaults to `True`.
        n_jobs
            The number of jobs to run in parallel. Parallel jobs are created only when there are two or more theta
            values to be evaluated. Each job will instantiate, train, and evaluate a different instance of the model.
            Defaults to `1` (sequential). Setting the parameter to `-1` means using all the available cores.

        Returns
        -------
        FourTheta
            The best performing model on the time series.
        """
        # Only import if needed
        from darts.metrics import mae

        if thetas is None:
            thetas = [1, 2, 3]
        if (ts.values(copy=False) <= 0).any():
            drift_mode = [LINEAR]
            model_mode = [ADDITIVE]
            season_mode = [ADDITIVE]
            logger.warning(
                "The given TimeSeries has negative values. The method will only test "
                "linear trend and additive modes."
            )
        else:
            season_mode = [ADDITIVE, MULTIPLICATIVE, None]
            model_mode = [ADDITIVE, MULTIPLICATIVE]
            drift_mode = [LINEAR, EXPONENTIAL]

        theta = cls.gridsearch(
            {
                "theta": thetas,
                "model_mode": model_mode,
                "season_mode": season_mode,
                "trend_mode": drift_mode,
                "seasonality_period": [m],
                "normalization": [normalization],
            },
            ts,
            use_fitted_values=True,
            metric=mae,
            n_jobs=n_jobs,
        )
        return theta

    @property
    def supports_multivariate(self) -> bool:
        return False

    @property
    def _target_window_lengths(self) -> tuple[int, int]:
        if (
            self.season_mode is not None
            and self.seasonality_period
            and self.seasonality_period > 1
        ):
            input_length_inferred = 2 * self.seasonality_period
        else:
            input_length_inferred = 3
        return self._min_train_input_length(input_length_inferred), 0
