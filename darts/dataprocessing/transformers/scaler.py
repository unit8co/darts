"""
Scaler
------
"""
from darts.dataprocessing.transformers import FittableDataTransformer, InvertibleDataTransformer

from darts.timeseries import TimeSeries
from darts.logging import get_logger, raise_log
from sklearn.preprocessing import MinMaxScaler

logger = get_logger(__name__)


class Scaler(FittableDataTransformer[TimeSeries], InvertibleDataTransformer[TimeSeries]):
    def __init__(self, scaler=MinMaxScaler(feature_range=(0, 1)), name="Scaler"):
        """
        Generic wrapper class for using scalers that implement `fit()`, `transform()` and
        `inverse_transform()` methods (typically from scikit-learn) on `TimeSeries`.

        Parameters
        ----------
        scaler
            The scaler to transform the data.
            It must provide the `fit()`, `transform()` and `inverse_transform()` methods.
            Default: `sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1))`; this
            will scale all the values of a time series between 0 and 1.
        name
            A specific name for the scaler
        """
        super().__init__(name)

        if (not callable(getattr(scaler, "fit", None)) or not callable(getattr(scaler, "transform", None))
                or not callable(getattr(scaler, "inverse_transform", None))): # noqa W503
            raise_log(ValueError(
                      'The provided transformer object must have fit(), transform() and inverse_transform() methods'),
                      logger)

        self.transformer = scaler
        self.train_series = None

    def fit(self, series: TimeSeries) -> 'Scaler':
        """
        Fits this scaler to the provided time series data

        Parameters
        ----------
        series
            The time series to fit the scaler on

        Returns
        -------
            Fitted scaler (self)
        """
        super().fit(series)
        self.transformer.fit(series.values().reshape((-1, series.width)))
        self.train_series = series
        return self

    def transform(self, series: TimeSeries, *args, **kwargs) -> TimeSeries:
        """
        Returns a new time series, transformed with this (fitted) scaler.
        This does not handle series with confidence intervals - the intervals are discarded.

        Parameters
        ----------
        series
            The time series to transform

        Returns
        -------
        TimeSeries
            A new time series, transformed with this (fitted) scaler.
        """
        super().transform(series, *args, **kwargs)
        return TimeSeries.from_times_and_values(series.time_index(),
                                                self.transformer.transform(series.values().
                                                                           reshape((-1, series.width))),
                                                series.freq())

    def inverse_transform(self, series: TimeSeries, *args, **kwargs) -> TimeSeries:
        """
        Performs the inverse transformation on a time series

        Parameters
        ----------
        series
            The time series to inverse transform

        Returns
        -------
        TimeSeries
            The inverse transform
        """
        super().inverse_transform(series, *args, **kwargs)
        return TimeSeries.from_times_and_values(series.time_index(),
                                                self.transformer.inverse_transform(series.values().
                                                                                   reshape((-1, series.width))),
                                                series.freq())
