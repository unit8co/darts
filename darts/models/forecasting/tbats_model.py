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

from abc import ABC, abstractmethod
from typing import Optional, Union

import numpy as np
from scipy.special import inv_boxcox
from tbats import BATS as tbats_BATS
from tbats import TBATS as tbats_TBATS

from darts.logging import get_logger
from darts.models.forecasting.forecasting_model import LocalForecastingModel
from darts.timeseries import TimeSeries

logger = get_logger(__name__)


def _seasonality_from_freq(series: TimeSeries):
    """
    Infer a naive seasonality based on the frequency
    """

    if series.has_range_index:
        return None

    freq = series.freq_str

    if freq in ["B", "C"]:
        return [5]
    elif freq == "D":
        return [7]
    elif freq == "W" or freq.startswith("W-"):
        return [52]
    elif freq in [
        "M",
        "BM",
        "CBM",
        "SM",
        "LWOM",
        "WOM",
    ] or freq.startswith(("M", "BM", "BS", "CBM", "SM", "LWOM", "WOM")):
        return [12]  # month
    elif freq in ["Q", "BQ", "REQ"] or freq.startswith(("Q", "BQ", "REQ")):
        return [4]  # quarter
    else:
        freq_lower = freq.lower()
        if freq_lower in ["h", "bh", "cbh"]:
            return [24]  # hour
        elif freq_lower in ["t", "min"]:
            return [60]  # minute
        elif freq_lower == "s":
            return [60]  # second
    return None


def _compute_samples(model, predictions, n_samples):
    """
    This function is drawn from Model._calculate_confidence_intervals() in tbats.
    We have to implement our own version here in order to compute the samples before
    the inverse boxcox transform.
    """

    # In the deterministic case we return the analytic mean
    if n_samples == 1:
        return np.expand_dims(predictions, axis=1)

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


class _BaseBatsTbatsModel(LocalForecastingModel, ABC):
    def __init__(
        self,
        use_box_cox: Optional[bool] = None,
        box_cox_bounds: tuple = (0, 1),
        use_trend: Optional[bool] = None,
        use_damped_trend: Optional[bool] = None,
        seasonal_periods: Optional[Union[str, list]] = "freq",
        use_arma_errors: Optional[bool] = True,
        show_warnings: bool = False,
        n_jobs: Optional[int] = None,
        multiprocessing_start_method: Optional[str] = "spawn",
        random_state: int = 0,
    ):
        """
        This is a wrapper around
        `tbats
        <https://github.com/intive-DataScience/tbats>`_.

        This implementation also provides naive frequency inference (when "freq"
        is provided for ``seasonal_periods``),
        as well as Darts-compatible sampling of the resulting normal distribution.

        For convenience, the tbats documentation of the parameters is reported here.

        Parameters
        ----------
        use_box_cox
            If Box-Cox transformation of original series should be applied.
            When ``None`` both cases shall be considered and better is selected by AIC.
        box_cox_bounds
            Minimal and maximal Box-Cox parameter values.
        use_trend
            Indicates whether to include a trend or not.
            When ``None``, both cases shall be considered and the better one is selected by AIC.
        use_damped_trend
            Indicates whether to include a damping parameter in the trend or not.
            Applies only when trend is used.
            When ``None``, both cases shall be considered and the better one is selected by AIC.
        seasonal_periods
            Length of each of the periods (amount of observations in each period).
            TBATS accepts int and float values here.
            BATS accepts only int values.
            When ``None`` or empty array, non-seasonal model shall be fitted.
            If set to ``"freq"``, a single "naive" seasonality
            based on the series frequency will be used (e.g. [12] for monthly series).
            In this latter case, the seasonality will be recomputed every time the model is fit.
        use_arma_errors
            When True BATS will try to improve the model by modelling residuals with ARMA.
            Best model will be selected by AIC.
            If ``False``, ARMA residuals modeling will not be considered.
        show_warnings
            If warnings should be shown or not.
        n_jobs
            How many jobs to run in parallel when fitting BATS model.
            When not provided BATS shall try to utilize all available cpu cores.
        multiprocessing_start_method
            How threads should be started.
            See https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods
        random_state
            Sets the underlying random seed at model initialization time.

        Examples
        --------
        >>> from darts.datasets import AirPassengersDataset
        >>> from darts.models import TBATS # or BATS
        >>> series = AirPassengersDataset().load()
        >>> # based on preliminary analysis, the series contains a trend
        >>> model = TBATS(use_trend=True)
        >>> model.fit(series)
        >>> pred = model.predict(6)
        >>> pred.values()
        array([[448.29856017],
               [439.42215052],
               [507.73465028],
               [493.03751671],
               [498.85885374],
               [564.64871897]])
        """
        super().__init__()

        self.kwargs = {
            "use_box_cox": use_box_cox,
            "box_cox_bounds": box_cox_bounds,
            "use_trend": use_trend,
            "use_damped_trend": use_damped_trend,
            "seasonal_periods": seasonal_periods,
            "use_arma_errors": use_arma_errors,
            "show_warnings": show_warnings,
            "n_jobs": n_jobs,
            "multiprocessing_start_method": multiprocessing_start_method,
        }

        self.seasonal_periods = seasonal_periods
        self.infer_seasonal_periods = seasonal_periods == "freq"
        self.model = None
        np.random.seed(random_state)

    @abstractmethod
    def _create_model(self):
        pass

    def fit(self, series: TimeSeries):
        super().fit(series)
        self._assert_univariate(series)
        series = self.training_series

        if self.infer_seasonal_periods:
            seasonality = _seasonality_from_freq(series)
            self.kwargs["seasonal_periods"] = seasonality
            self.seasonal_periods = seasonality

        model = self._create_model()
        fitted_model = model.fit(series.values())
        self.model = fitted_model

        return self

    def predict(
        self,
        n: int,
        num_samples: int = 1,
        verbose: bool = False,
        show_warnings: bool = True,
    ):
        super().predict(n, num_samples)

        yhat = self.model.forecast(steps=n)
        samples = _compute_samples(self.model, yhat, num_samples)

        return self._build_forecast_series(samples)

    @property
    def supports_multivariate(self) -> bool:
        return False

    @property
    def supports_probabilistic_prediction(self) -> bool:
        return True

    @property
    def min_train_series_length(self) -> int:
        if (
            isinstance(self.seasonal_periods, list)
            and len(self.seasonal_periods) > 0
            and max(self.seasonal_periods) > 1
        ):
            return 2 * max(self.seasonal_periods)
        return 3


class TBATS(_BaseBatsTbatsModel):
    def _create_model(self):
        return tbats_TBATS(**self.kwargs)


class BATS(_BaseBatsTbatsModel):
    def _create_model(self):
        return tbats_BATS(**self.kwargs)
