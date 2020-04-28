from ..timeseries import TimeSeries
from ..custom_logging import get_logger, raise_log
from sklearn.preprocessing import MinMaxScaler, StandardScaler

logger = get_logger(__name__)


class Transformer:
    """
    Wrapper class for using transformers / scalers (typically from scikit-learn) on TimeSeries data
    """

    def __init__(self, transformer=MinMaxScaler(feature_range=(0, 1))):
        def _raise():
            raise_log(ValueError(
                      'The provided transformer object must have fit(), transform() and inverse_transform() methods'),
                      logger)

        if (not callable(getattr(transformer, "fit", None)) or
            not callable(getattr(transformer, "transform", None)) or
            not callable(getattr(transformer, "inverse_transform", None))):
                _raise()

        self.transformer = transformer
        self.train_series = None
        self._fit_called = False

    def fit(self, series: TimeSeries) -> 'Transformer':
        self.transformer.fit(series.values().reshape((-1, 1)))
        self.train_series = series
        self._fit_called = True
        return self

    def transform(self, series: TimeSeries) -> TimeSeries:
        """
        Returns a new time series, transformed with this (fitted) transformer.
        Currently, this does not handle series with confidence intervals (they are discarded).
        :param series:
        :return: a new time series, transformed with this (fitted) transformer.

        # TODO log warn when the series has some CIs.
        """
        assert self._fit_called, 'fit() must be called before transform()'
        return TimeSeries.from_times_and_values(series.time_index(),
                                                self.transformer.transform(series.values().
                                                                           reshape((-1, 1))).reshape((-1,)))

    def fit_transform(self, series: TimeSeries) -> TimeSeries:
        return self.fit(series).transform(series)

    def inverse_transform(self, series: TimeSeries) -> TimeSeries:
        return TimeSeries.from_times_and_values(series.time_index(),
                                                self.transformer.inverse_transform(series.values().
                                                                                   reshape((-1, 1))).reshape((-1,)))
