"""
BATS and TBATS
--------------

(T)BATS models [1]_ stand for

* (Trigonometric)
* Box-Cox
* ARMA errors
* Trend
* Seasonal components

They are appropriate to model "complex
seasonal time series such as those with multiple
seasonal periods, high frequency seasonality,
non-integer seasonality and dual-calendar effects" [1]_.

References
----------
.. [1] https://robjhyndman.com/papers/ComplexSeasonality.pdf
"""

from typing import List, Optional

import numpy as np
from scipy.special import inv_boxcox
from tbats import TBATS as tbats_TBATS

from darts.logging import get_logger
from darts.models.forecasting.forecasting_model import ForecastingModel
from darts.timeseries import TimeSeries

logger = get_logger(__name__)


class TBATS(ForecastingModel):
    def __init__(
        self,
        seasonal_periods: Optional[List[int]] = "freq",
        use_arma_errors: Optional[bool] = None,
        use_box_cox: Optional[bool] = None,
        use_trend: Optional[bool] = None,
        use_damped_trend: Optional[bool] = None,
        random_state: int = 0,
        **kwargs,
    ):

        """TBATS

        This is a wrapper around
        `tbats TBATS model
        <https://github.com/intive-DataScience/tbats>`_;
        we refer to this link for the documentation on the parameters.

        Parameters
        ----------
        seasonal_periods
            A list of seasonal periods. If ``None``, no seasonality will be set.
            If set to ``"freq"``, a single "naive" seasonality
            based on the series frequency will be used (e.g. [12] for monthly series).
        use_arma_errors
            Whether to use ARMA errors (``None``: try with and without)
        use_box_cox
            Whether to use BoxCox transform (``None``: try with and without)
        use_trend
            Whether to use trend (``None``: try with and without)
        use_damped_trend
            Whether to use damped trend (``None``: try with and without)
        kwargs
            Other optional keyword arguments that will be used to call
            :class:`tbats.TBATS`.
        """
        super().__init__()
        self.seasonal_periods = seasonal_periods
        self.use_arma_errors = use_arma_errors
        self.use_box_cox = use_box_cox
        self.use_trend = use_trend
        self.use_damped_trend = use_damped_trend
        self.tbats_kwargs = kwargs

        self.infer_seasonal_periods = seasonal_periods == "freq"
        self.model = None
        np.random.seed(random_state)

    def __str__(self):
        return (
            f"TBATS(periods={self.seasonal_periods}, arma_errs={self.use_arma_errors}, "
            f"boxcox={self.use_box_cox}, trend={self.use_trend}, damped_trend={self.use_damped_trend}"
        )

    @staticmethod
    def _infer_naive_seasonality(series: TimeSeries):
        """
        Infer a naive seasonality based on the frequency
        """
        if series.has_range_index:
            return [12]
        elif series.freq_str == "B":
            return [5]
        elif series.freq_str == "D":
            return [7]
        elif series.freq_str == "W":
            return [52]
        elif series.freq_str in ["MS", "M"]:
            return [12]
        elif series.freq_str == ["Q", "BQ", "QS", "BQS"]:
            return [4]
        elif series.freq_str == ["H"]:
            return [24]
        return None

    @staticmethod
    def _darts_calculate_confidence_intervals(model, predictions, n_samples):
        """
        This function is drawn from Model._calculate_confidence_intervals() in tbats.
        We have to implement our own version here in order to compute the samples before
        the inverse boxcox transform.
        """
        F = model.matrix.make_F_matrix()
        g = model.matrix.make_g_vector()
        w = model.matrix.make_w_vector()

        c = np.asarray([1.0] * len(predictions))
        f_running = np.identity(F.shape[1])
        for step in range(1, len(predictions)):
            c[step] = w @ f_running @ g
            f_running = f_running @ F
        variance_multiplier = np.cumsum(c * c)

        base_variance_boxcox = np.sum(model.resid_boxcox * model.resid_boxcox) / len(
            model.y
        )
        variance_boxcox = base_variance_boxcox * variance_multiplier
        std_boxcox = np.sqrt(variance_boxcox)

        # get the samples before inverse boxcoxing
        samples = np.random.normal(
            loc=model._boxcox(predictions),
            scale=std_boxcox,
            size=(n_samples, len(predictions)),
        ).T
        samples = np.expand_dims(samples, axis=1)

        # apply inverse boxcox if needed
        boxcox_lambda = model.params.box_cox_lambda
        if boxcox_lambda is not None:
            samples = inv_boxcox(samples, boxcox_lambda)

        return samples

    def fit(self, series: TimeSeries):
        super().fit(series)
        series = self.training_series

        if self.infer_seasonal_periods:
            seasonal_periods = TBATS._infer_naive_seasonality(series)
        else:
            seasonal_periods = self.seasonal_periods

        model = tbats_TBATS(
            seasonal_periods=seasonal_periods,
            use_arma_errors=self.use_arma_errors,
            use_box_cox=self.use_box_cox,
            use_trend=self.use_trend,
            use_damped_trend=self.use_damped_trend,
            show_warnings=False,
            **self.tbats_kwargs,
        )
        fitted_model = model.fit(series.values())
        self.model = fitted_model

        return self

    def predict(self, n, num_samples=1):
        super().predict(n, num_samples)

        yhat = self.model.forecast(steps=n)
        if num_samples == 1:
            samples = yhat.view(len(yhat), 1)
        else:
            samples = TBATS._darts_calculate_confidence_intervals(
                self.model, yhat, num_samples
            )
        return self._build_forecast_series(samples)

    def _is_probabilistic(self) -> bool:
        return True

    @property
    def min_train_series_length(self) -> int:
        if isinstance(self.seasonal_periods, int) and self.seasonal_periods > 1:
            return 2 * self.seasonal_periods
        return 3
