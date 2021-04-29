"""
Scaler
------
"""
from typing import Sequence, Iterator, Any, Tuple
from darts.dataprocessing.transformers import InvertibleDataTransformer, FittableDataTransformer
from darts.timeseries import TimeSeries
from darts.logging import get_logger, raise_log
from sklearn.preprocessing import MinMaxScaler
from copy import deepcopy

logger = get_logger(__name__)


class Scaler(InvertibleDataTransformer, FittableDataTransformer):
    def __init__(self,
                 scaler=None,
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
            The number of jobs to run in parallel (in case the transformer is handling a Sequence[TimeSeries]).
            Defaults to `1` (sequential). `-1` means using all the available processors.
            Note: for a small amount of data, the parallelisation overhead could end up increasing the total
            required amount of time.
        verbose
            Optionally, whether to print operations progress
        """

        def _scaler_ts_transform(series: TimeSeries, transformer) -> TimeSeries:
            return TimeSeries.from_times_and_values(series.time_index(),
                                                    transformer.transform(series.values().
                                                                          reshape((-1, series.width))),
                                                    series.freq())

        def _scaler_ts_inverse_transform(series: TimeSeries, transformer, *args, **kwargs) -> TimeSeries:
            return TimeSeries.from_times_and_values(series.time_index(),
                                                    transformer.inverse_transform(series.values().
                                                                                  reshape((-1, series.width))),
                                                    series.freq())

        def _scaler_ts_fit(series: TimeSeries, transformer, *args, **kwargs) -> Any:
            # fit_parameter will receive the transformer object instance
            scaler = transformer.fit(series.values().reshape((-1, series.width)))
            return scaler

        super().__init__(ts_transform=_scaler_ts_transform,
                         ts_inverse_transform=_scaler_ts_inverse_transform,
                         ts_fit=_scaler_ts_fit,
                         name=name,
                         n_jobs=n_jobs,
                         verbose=verbose)

        if scaler is None:
            scaler = MinMaxScaler(feature_range=(0, 1))

        if (not callable(getattr(scaler, "fit", None)) or not callable(getattr(scaler, "transform", None))
                or not callable(getattr(scaler, "inverse_transform", None))):  # noqa W503
            raise_log(ValueError(
                      'The provided transformer object must have fit(), transform() and inverse_transform() methods'),
                      logger)

        self.transformer = scaler
        self.transformer_instances = None

    def _fit_iterator(self, series: Sequence[TimeSeries]) -> Iterator[Tuple[TimeSeries, Any]]:
        # generator which returns deep copies of the 'scaler' argument
        new_scaler_gen = (deepcopy(self.transformer) for _ in range(len(series)))
        return zip(series, new_scaler_gen)

    def _transform_iterator(self, series: Sequence[TimeSeries]) -> Iterator[Tuple[TimeSeries, Any]]:
        # since '_ts_fit()' returns the scaler objects, the 'fit()' call will save transformers instance into
        # self._fitted_params
        return zip(series, self._fitted_params)

    def _inverse_transform_iterator(self, series: Sequence[TimeSeries]) -> Iterator[Tuple[TimeSeries, Any]]:
        # the same self._fitted_params will be used also for the 'ts_inverse_transform()'
        return zip(series, self._fitted_params)
