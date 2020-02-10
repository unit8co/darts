import statsmodels.tsa.holtwinters as hw
from ..timeseries import TimeSeries
from .autoregressive_model import AutoRegressiveModel
import numpy as np
import math
from .statistics import check_seasonality, extract_trend_and_seasonality, remove_seasonality


class Theta(AutoRegressiveModel):
    """
    An implementation of the Theta method with variable value of the `theta` parameter.

    :param theta: User-defined value for the theta parameter. Default to 0.

    .. todo: Implement OTM: Optimized Theta Method (https://arxiv.org/pdf/1503.03529.pdf)
    .. todo: From the OTM, set theta_2 = 2-theta_1 to recover our generalization - but we have an explicit formula.
    .. todo: Try with something different than SES? They do that in the paper.
    """
    def __init__(self, theta: int = 0):
        super().__init__()

        self.model = None
        self.coef = 1
        self.alpha = 1
        self.length = 0
        self.theta = theta
        self.is_seasonal = False
        self.seasonality = None
        self.season_period = 0

        # Remark on the values of the theta parameter:
        # - if theta = 1, then the theta method restricts to a simple exponential smoothing (SES)
        # - if theta = 2, the formula for self.coef below fails, hence it is forbidden.

        if self.theta == 2:
            raise ValueError('The parameter theta cannot be equal to 2.')

    def fit(self, ts, season_period: int = None):
        """
        Fits the Theta method to the TimeSeries `ts`.


        The model decomposition is defined by the parameters `theta`, and the TimeSeries `ts`
        is de-seasonalized according to `season_period`.

        :param ts: The TimeSeries to fit.
        :param season_period: User-defined seasonality period. Default to None.
        """
        super().fit(ts)

        self.length = len(ts)

        # Check for statistical significance of user-defined season period
        # or infers season_period from the TimeSeries itself.
        self.is_seasonal, self.season_period = check_seasonality(ts, season_period)

        new_ts = ts

        # Store and remove seasonality effect if there is any.
        if self.is_seasonal:
            _, self.seasonality = extract_trend_and_seasonality(ts, self.season_period)
            new_ts = remove_seasonality(ts, self.season_period)

        # SES part of the decomposition.
        self.model = hw.SimpleExpSmoothing(new_ts.values()).fit()

        # Linear Regression part of the decomposition. We select the degree one coefficient.
        b_theta = np.polyfit(np.array([i for i in range(0, self.length)]), (1.0 - self.theta) * new_ts.values(), 1)[0]

        # Normalization of the coefficient b_theta.
        self.coef = b_theta / (2.0 - self.theta)

        self.alpha = self.model.params["smoothing_level"]

    def predict(self, n: int) -> 'TimeSeries':
        """
        Uses the fitted model to predict `n` values in the future.

        :param n: The length of the horizon to predict over.
        :return: A new TimeSeries containing the `n` prediction points.
        """
        super().predict(n)

        # Forecast of the SES part.
        forecast = self.model.forecast(n)

        # Forecast of the Linear Regression part.
        drift = self.coef * np.array([i + (1 - (1 - self.alpha) ** self.length) / self.alpha for i in range(0, n)])

        # Combining the two forecasts
        forecast += drift

        # Re-apply the seasonal trend of the TimeSeries
        if self.is_seasonal:

            replicated_seasonality = np.tile(self.seasonality.pd_series()[-self.season_period:],
                                             math.ceil(n / self.season_period))[:n]
            forecast *= replicated_seasonality

        return self._build_forecast_series(forecast)
