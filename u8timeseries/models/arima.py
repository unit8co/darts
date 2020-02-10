from .autoregressive_model import AutoRegressiveModel
from statsmodels.tsa.arima_model import ARMA, ARIMA
from pmdarima import auto_arima
from ..timeseries import TimeSeries


class Arima(AutoRegressiveModel):
    """
    Implementation of an ARIMA model.

    Currently a wrapper around the statsmodel implementation.

    :param p: An integer representing the lag order.
    :param d: An integer for the order of differentiation.
    :param q: An interger for the size of the moving average window.
    """

    def __init__(self, p: int = 12, d: int = 1, q: int = 0):
        super().__init__()
        self.p = p
        self.d = d
        self.q = q
        self.model = None

    def __str__(self):
        return 'ARIMA({},{},{})'.format(self.p, self.d, self.q)

    def fit(self, series: TimeSeries):
        super().fit(series)

        m = ARIMA(series.values(),
                  order=(self.p, self.d, self.q)) if self.d > 0 else ARMA(series.values(), order=(self.p, self.q))
        self.model = m.fit(disp=0)

    def predict(self, n):
        super().predict(n)
        forecast = self.model.forecast(steps=n)[0]
        return self._build_forecast_series(forecast)


class AutoArima(AutoRegressiveModel):

    def __init__(self, start_p=1, max_p=12, start_q=0, max_q=12, max_P=2, max_Q=2, start_P=1, start_Q=1,
                 start_d=0, max_d=2, max_D=1, max_order=30, seasonal=True, stepwise=True, approximation=False,
                 error_action='ignore', trace=False, suppress_warnings=True):

        super().__init__()

        self.start_p = start_p
        self.max_p = max_p
        self.start_q = start_q
        self.max_q = max_q
        self.max_P = max_P
        self.max_Q = max_Q
        self.start_P = start_P
        self.start_Q = start_Q
        self.start_d = start_d
        self.max_d = max_d
        self.max_D = max_D
        self.max_order = max_order
        self.seasonal = seasonal
        self.stepwise = stepwise
        self.approximation = approximation
        self.error_action = error_action
        self.trace = trace
        self.suppress_warnings = suppress_warnings
        self.model = None

    def __str__(self):
        return 'auto-ARIMA'

    def fit(self, series: TimeSeries):
        super().fit(series)
        self.model = auto_arima(series.values(),
                                start_p=self.start_p,
                                max_p=self.max_p,
                                start_q=self.start_q,
                                max_q=self.max_q,
                                max_P=self.max_P,
                                max_Q=self.max_Q,
                                start_P=self.start_P,
                                start_Q=self.start_Q,
                                start_d=self.start_d,
                                max_d=self.max_d,
                                max_D=self.max_D,
                                max_order=self.max_order,
                                seasonal=self.seasonal,
                                stepwise=self.stepwise,
                                approximation=self.approximation,
                                error_action=self.error_action,
                                trace=self.trace,
                                suppress_warnings=self.suppress_warnings)

    def predict(self, n):
        super().predict(n)
        forecast = self.model.predict(n_periods=n)
        return self._build_forecast_series(forecast)
