"""
Scaler
------
"""
from joblib import Parallel, delayed
from typing import Union, Sequence
import sklearn.base

from darts.dataprocessing.transformers import FittableDataTransformer, InvertibleDataTransformer
from darts.utils import _build_tqdm_iterator

from darts.timeseries import TimeSeries
from darts.logging import get_logger, raise_log
from sklearn.preprocessing import MinMaxScaler

logger = get_logger(__name__)


class Scaler(FittableDataTransformer[TimeSeries], InvertibleDataTransformer[TimeSeries]):
    def __init__(self,
                 scaler=MinMaxScaler(feature_range=(0, 1)),
                 name="Scaler",
                 n_jobs: int = 1,
                 verbose: bool = False):
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
        n_jobs
            The number of jobs to run in parallel. Defaults to `1`. `-1` means using all processors
        verbose
            Optionally, whether to print progress
        """
        super().__init__(name)

        if (not callable(getattr(scaler, "fit", None)) or not callable(getattr(scaler, "transform", None))
                or not callable(getattr(scaler, "inverse_transform", None))): # noqa W503
            raise_log(ValueError(
                      'The provided transformer object must have fit(), transform() and inverse_transform() methods'),
                      logger)

        self.transformer = scaler
        self.train_series = None
        self.transformer_instances = None
        self._n_jobs = n_jobs
        self._verbose = verbose

    def _get_new_scaler(self):
        return sklearn.base.clone(self.transformer)

    def fit(self, series: Union[TimeSeries, Sequence[TimeSeries]]) -> 'Scaler':
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
        self.train_series = series

        if isinstance(series, TimeSeries):
            self.transformer.fit(series.values().reshape((-1, series.width)))
        else:

            def train_new_scaler(series):
                scaler = self._get_new_scaler().fit(series.values().reshape((-1, series.width)))
                return scaler

            iterator = _build_tqdm_iterator(series, verbose=self._verbose, desc="Fitting {}".format(self.name))

            self.transformer_instances = Parallel(n_jobs=self._n_jobs, prefer="threads")(delayed(train_new_scaler)
                                                                                         (series)
                                                                                         for series in iterator)

        return self

    def transform(self,
                  series: Union[TimeSeries, Sequence[TimeSeries]],
                  *args,
                  **kwargs) -> Union[TimeSeries, Sequence[TimeSeries]]:
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
        
        def _transform_series(series, transformer):
            return TimeSeries.from_times_and_values(series.time_index(),
                                                    transformer.transform(series.values().
                                                                          reshape((-1, series.width))),
                                                    series.freq())

        if (isinstance(series, TimeSeries)):
            return _transform_series(series, self.transformer)
        else:
            iterator = _build_tqdm_iterator(zip(series, self.transformer_instances),
                                            verbose=self._verbose,
                                            total=len(series),
                                            desc="Applying {}".format(self.name))

            transformed_series = Parallel(n_jobs=self._n_jobs)(delayed(_transform_series)(series, transformer)
                                                               for series, transformer in iterator)
            return transformed_series

    def inverse_transform(self,
                          series: Union[TimeSeries, Sequence[TimeSeries]],
                          *args,
                          **kwargs) -> Union[TimeSeries, Sequence[TimeSeries]]:
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

        def _inverse_transform_series(series, transformer):
            return TimeSeries.from_times_and_values(series.time_index(),
                                                    transformer.inverse_transform(series.values().
                                                                                  reshape((-1, series.width))),
                                                    series.freq())

        if (isinstance(series, TimeSeries)):
            return _inverse_transform_series(series, self.transformer)
        else:
            iterator = _build_tqdm_iterator(zip(series, self.transformer_instances),
                                            verbose=self._verbose,
                                            total=len(series),
                                            desc="Reverse applying {}".format(self.name))

            transformed_series = Parallel(n_jobs=self._n_jobs)(delayed(_inverse_transform_series)
                                                               (series, transformer)
                                                               for series, transformer in iterator)
            return transformed_series

