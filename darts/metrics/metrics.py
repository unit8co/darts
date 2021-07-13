"""
Metrics
-------

Some metrics to compare time series.
"""

import numpy as np
from ..timeseries import TimeSeries
from darts.utils import _parallel_apply, _build_tqdm_iterator
from ..utils.statistics import check_seasonality
from ..logging import raise_if_not, get_logger, raise_log
from warnings import warn
from typing import Optional, Callable, Sequence, Union, Tuple
from inspect import signature
from functools import wraps


logger = get_logger(__name__)


"""
Note: for new metrics added to this module to be able to leverage the two decorators, it is required both having
the `actual_series` and `pred_series` parameters, and not having other `Sequence` as args (since these decorators
don't “unpack“ parameters different from `actual_series` and `pred_series`). In those cases, the new metric must take
care of dealing with Sequence[TimeSeries] and multivariate TimeSeries on its own (See mase() implementation).
"""


def multi_ts_support(func):
    """
    This decorator further adapts the metrics that took as input two univariate/multivariate `TimeSeries` instances,
    adding support for equally-sized sequences of `TimeSeries` instances. The decorator computes the pairwise metric for
    `TimeSeries` with the same indices, and returns a float value that is computed as a function of all the
    pairwise metrics using a `inter_reduction` subroutine passed as argument to the metric function.

    If a 'Sequence[TimeSeries]' is passed as input, this decorator provides also parallelisation of the metric
    evaluation regarding different `TimeSeries` (if the `n_jobs` parameter is not set 1).
    """

    @wraps(func)
    def wrapper_multi_ts_support(*args, **kwargs):
        actual_series = kwargs['actual_series'] if 'actual_series' in kwargs else args[0]
        pred_series = kwargs['pred_series'] if 'pred_series' in kwargs else args[0] if 'actual_series' in kwargs \
            else args[1]

        n_jobs = kwargs.pop('n_jobs', signature(func).parameters['n_jobs'].default)
        verbose = kwargs.pop('verbose', signature(func).parameters['verbose'].default)

        raise_if_not(isinstance(n_jobs, int), "n_jobs must be an integer")
        raise_if_not(isinstance(verbose, bool), "verbose must be a bool")

        actual_series = [actual_series] if not isinstance(actual_series, Sequence) else actual_series
        pred_series = [pred_series] if not isinstance(pred_series, Sequence) else pred_series

        raise_if_not(len(actual_series) == len(pred_series),
                     "The two TimeSeries sequences must have the same length.", logger)

        num_series_in_args = int('actual_series' not in kwargs) + int('pred_series' not in kwargs)
        kwargs.pop('actual_series', 0)
        kwargs.pop('pred_series', 0)

        iterator = _build_tqdm_iterator(iterable=zip(actual_series, pred_series),
                                        verbose=verbose,
                                        total=len(actual_series))

        value_list = _parallel_apply(iterator=iterator,
                                     fn=func,
                                     n_jobs=n_jobs,
                                     fn_args=args[num_series_in_args:],
                                     fn_kwargs=kwargs)

        # in case the reduction is not reducing the metrics sequence to a single value, e.g., if returning the
        # np.ndarray of values with the identity function, we must handle the single TS case, where we should
        # return a single value instead of a np.array of len 1

        if len(value_list) == 1:
            value_list = value_list[0]

        if 'inter_reduction' in kwargs:
            return kwargs['inter_reduction'](value_list)
        else:
            return signature(func).parameters['inter_reduction'].default(value_list)

    return wrapper_multi_ts_support


def multivariate_support(func):
    """
    This decorator transforms a metric function that takes as input two univariate TimeSeries instances
    into a function that takes two equally-sized multivariate TimeSeries instances, computes the pairwise univariate
    metrics for components with the same indices, and returns a float value that is computed as a function of all the
    univariate metrics using a `reduction` subroutine passed as argument to the metric function.
    """
    @wraps(func)
    def wrapper_multivariate_support(*args, **kwargs):
        # we can avoid checks about args and kwargs since the input is adjusted by the previous decorator
        actual_series = args[0]
        pred_series = args[1]

        raise_if_not(actual_series.width == pred_series.width,
                     "The two TimeSeries instances must have the same width.", logger)

        value_list = []
        for i in range(actual_series.width):
            value_list.append(func(actual_series.univariate_component(i), pred_series.univariate_component(i),
                              *args[2:], **kwargs))  # [2:] since we already know the first two arguments are the series
        if 'reduction' in kwargs:
            return kwargs['reduction'](value_list)
        else:
            return signature(func).parameters['reduction'].default(value_list)
    return wrapper_multivariate_support


def _get_values_or_raise(series_a: TimeSeries,
                         series_b: TimeSeries,
                         intersect: bool) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns the numpy values of two time series. If intersect is true, considers only their time intersection.
    Raises a ValueError if the two time series (or their intersection) do not have the same time index.

    For stochastic series, return the median sample value
    """
    raise_if_not(series_a.width == series_b.width, " The two time series must have the same number of components",
                 logger)

    raise_if_not(isinstance(intersect, bool), "The intersect parameter must be a bool")

    series_a_common = series_a.slice_intersect(series_b) if intersect else series_a
    series_b_common = series_b.slice_intersect(series_a) if intersect else series_b

    raise_if_not(series_a_common.has_same_time_as(series_b_common), 'The two time series (or their intersection) '
                                                                    'must have the same time index.'
                                                                    '\nFirst series: {}\nSecond series: {}'.format(
                                                                    series_a.time_index, series_b.time_index),
                 logger)

    # TODO: we may want to change this...
    series_a_det = series_a_common if series_a_common.is_deterministic else series_a_common.quantile_timeseries(quantile=0.5)
    series_b_det = series_b_common if series_b_common.is_deterministic else series_b_common.quantile_timeseries(quantile=0.5)

    return series_a_det.univariate_values(), series_b_det.univariate_values()


def _remove_nan_union(array_a: np.ndarray,
                      array_b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns the two inputs arrays where all elements are deleted that have an index that corresponds to
    a NaN value in either of the two input arrays.
    """

    isnan_mask = np.logical_or(np.isnan(array_a), np.isnan(array_b))
    return np.delete(array_a, isnan_mask), np.delete(array_b, isnan_mask)


@multi_ts_support
@multivariate_support
def mae(actual_series: Union[TimeSeries, Sequence[TimeSeries]],
        pred_series: Union[TimeSeries, Sequence[TimeSeries]],
        intersect: bool = True,
        *,
        reduction: Callable[[np.ndarray], float] = np.mean,
        inter_reduction: Callable[[np.ndarray], Union[float, np.ndarray]] = lambda x: x,
        n_jobs: int = 1,
        verbose: bool = False) -> Union[float, np.ndarray]:
    """ Mean Absolute Error (MAE).

    For two time series :math:`y^1` and :math:`y^2` of length :math:`T`, it is computed as

    .. math:: \\frac{1}{T}\\sum_{t=1}^T{(|y^1_t - y^2_t|)}.

    If any of the series is stochastic (containing several samples), the median sample value is considered.

    Parameters
    ----------
    actual_series
        The `TimeSeries` or `Sequence[TimeSeries]` of actual values.
    pred_series
        The `TimeSeries` or `Sequence[TimeSeries]` of predicted values.
    intersect
        For time series that are overlapping in time without having the same time index, setting `intersect=True`
        will consider the values only over their common time interval (intersection in time).
    reduction
        Function taking as input a `np.ndarray` and returning a scalar value. This function is used to aggregate
        the metrics of different components in case of multivariate `TimeSeries` instances.
    inter_reduction
        Function taking as input a `np.ndarray` and returning either a scalar value or a `np.ndarray`.
        This function can be used to aggregate the metrics of different series in case the metric is evaluated on a
        `Sequence[TimeSeries]`. Defaults to the identity function, which returns the pairwise metrics for each pair
        of `TimeSeries` received in input. Example: `inter_reduction=np.mean`, will return the average of the pairwise
        metrics.
    n_jobs
        The number of jobs to run in parallel. Parallel jobs are created only when a `Sequence[TimeSeries]` is
        passed as input, parallelising operations regarding different `TimeSeries`. Defaults to `1`
        (sequential). Setting the parameter to `-1` means using all the available processors.
    verbose
        Optionally, whether to print operations progress

    Returns
    -------
    float
        The Mean Absolute Error (MAE)

"""

    y1, y2 = _get_values_or_raise(actual_series, pred_series, intersect)
    y1, y2 = _remove_nan_union(y1, y2)
    return np.mean(np.abs(y1 - y2))


@multi_ts_support
@multivariate_support
def mse(actual_series: Union[TimeSeries, Sequence[TimeSeries]],
        pred_series: Union[TimeSeries, Sequence[TimeSeries]],
        intersect: bool = True,
        *,
        reduction: Callable[[np.ndarray], float] = np.mean,
        inter_reduction: Callable[[np.ndarray], Union[float, np.ndarray]] = lambda x: x,
        n_jobs: int = 1,
        verbose: bool = False) -> Union[float, np.ndarray]:
    """ Mean Squared Error (MSE).

    For two time series :math:`y^1` and :math:`y^2` of length :math:`T`, it is computed as

    .. math:: \\frac{1}{T}\\sum_{t=1}^T{(y^1_t - y^2_t)^2}.

    If any of the series is stochastic (containing several samples), the median sample value is considered.

    Parameters
    ----------
    actual_series
        The `TimeSeries` or `Sequence[TimeSeries]` of actual values.
    pred_series
        The `TimeSeries` or `Sequence[TimeSeries]` of predicted values.
    intersect
        For time series that are overlapping in time without having the same time index, setting `intersect=True`
        will consider the values only over their common time interval (intersection in time).
    reduction
        Function taking as input a `np.ndarray` and returning a scalar value. This function is used to aggregate
        the metrics of different components in case of multivariate `TimeSeries` instances.
    inter_reduction
        Function taking as input a `np.ndarray` and returning either a scalar value or a `np.ndarray`.
        This function can be used to aggregate the metrics of different series in case the metric is evaluated on a
        `Sequence[TimeSeries]`. Defaults to the identity function, which returns the pairwise metrics for each pair
        of `TimeSeries` received in input. Example: `inter_reduction=np.mean`, will return the average of the pairwise
        metrics.
    n_jobs
        The number of jobs to run in parallel. Parallel jobs are created only when a `Sequence[TimeSeries]` is
        passed as input, parallelising operations regarding different `TimeSeries`. Defaults to `1`
        (sequential). Setting the parameter to `-1` means using all the available processors.
    verbose
        Optionally, whether to print operations progress

    Returns
    -------
    float
        The Mean Squared Error (MSE)
    """

    y_true, y_pred = _get_values_or_raise(actual_series, pred_series, intersect)
    y_true, y_pred = _remove_nan_union(y_true, y_pred)
    return np.mean((y_true - y_pred)**2)


@multi_ts_support
@multivariate_support
def rmse(actual_series: Union[TimeSeries, Sequence[TimeSeries]],
         pred_series: Union[TimeSeries, Sequence[TimeSeries]],
         intersect: bool = True,
         *,
         reduction: Callable[[np.ndarray], float] = np.mean,
         inter_reduction: Callable[[np.ndarray], Union[float, np.ndarray]] = lambda x: x,
         n_jobs: int = 1,
         verbose: bool = False) -> Union[float, np.ndarray]:
    """ Root Mean Squared Error (RMSE).

    For two time series :math:`y^1` and :math:`y^2` of length :math:`T`, it is computed as

    .. math:: \\sqrt{\\frac{1}{T}\\sum_{t=1}^T{(y^1_t - y^2_t)^2}}.

    If any of the series is stochastic (containing several samples), the median sample value is considered.

    Parameters
    ----------
    actual_series
        The `TimeSeries` or `Sequence[TimeSeries]` of actual values.
    pred_series
        The `TimeSeries` or `Sequence[TimeSeries]` of predicted values.
    intersect
        For time series that are overlapping in time without having the same time index, setting `intersect=True`
        will consider the values only over their common time interval (intersection in time).
    reduction
        Function taking as input a `np.ndarray` and returning a scalar value. This function is used to aggregate
        the metrics of different components in case of multivariate `TimeSeries` instances.
    inter_reduction
        Function taking as input a `np.ndarray` and returning either a scalar value or a `np.ndarray`.
        This function can be used to aggregate the metrics of different series in case the metric is evaluated on a
        `Sequence[TimeSeries]`. Defaults to the identity function, which returns the pairwise metrics for each pair
        of `TimeSeries` received in input. Example: `inter_reduction=np.mean`, will return the average of the pairwise
        metrics.
    n_jobs
        The number of jobs to run in parallel. Parallel jobs are created only when a `Sequence[TimeSeries]` is
        passed as input, parallelising operations regarding different `TimeSeries`. Defaults to `1`
        (sequential). Setting the parameter to `-1` means using all the available processors.
    verbose
        Optionally, whether to print operations progress

    Returns
    -------
    float
        The Root Mean Squared Error (RMSE)
    """
    return np.sqrt(mse(actual_series, pred_series, intersect))


@multi_ts_support
@multivariate_support
def rmsle(actual_series: Union[TimeSeries, Sequence[TimeSeries]],
          pred_series: Union[TimeSeries, Sequence[TimeSeries]],
          intersect: bool = True,
          *,
          reduction: Callable[[np.ndarray], float] = np.mean,
          inter_reduction: Callable[[np.ndarray], Union[float, np.ndarray]] = lambda x: x,
          n_jobs: int = 1,
          verbose: bool = False) -> Union[float, np.ndarray]:
    """ Root Mean Squared Log Error (RMSLE).

    For two time series :math:`y^1` and :math:`y^2` of length :math:`T`, it is computed as

    .. math:: \\sqrt{\\frac{1}{T}\\sum_{t=1}^T{\\left(\\log{(y^1_t + 1)} - \\log{(y^2_t + 1)}\\right)^2}},

    using the natural logarithm.

    If any of the series is stochastic (containing several samples), the median sample value is considered.

    Parameters
    ----------
    actual_series
        The `TimeSeries` or `Sequence[TimeSeries]` of actual values.
    pred_series
        The `TimeSeries` or `Sequence[TimeSeries]` of predicted values.
    intersect
        For time series that are overlapping in time without having the same time index, setting `intersect=True`
        will consider the values only over their common time interval (intersection in time).
    reduction
        Function taking as input a `np.ndarray` and returning a scalar value. This function is used to aggregate
        the metrics of different components in case of multivariate `TimeSeries` instances.
    inter_reduction
        Function taking as input a `np.ndarray` and returning either a scalar value or a `np.ndarray`.
        This function can be used to aggregate the metrics of different series in case the metric is evaluated on a
        `Sequence[TimeSeries]`. Defaults to the identity function, which returns the pairwise metrics for each pair
        of `TimeSeries` received in input. Example: `inter_reduction=np.mean`, will return the average of the pairwise
        metrics.
    n_jobs
        The number of jobs to run in parallel. Parallel jobs are created only when a `Sequence[TimeSeries]` is
        passed as input, parallelising operations regarding different `TimeSeries`. Defaults to `1`
        (sequential). Setting the parameter to `-1` means using all the available processors.
    verbose
        Optionally, whether to print operations progress

    Returns
    -------
    float
        The Root Mean Squared Log Error (RMSLE)
    """

    y1, y2 = _get_values_or_raise(actual_series, pred_series, intersect)
    y1, y2 = _remove_nan_union(y1, y2)
    y1, y2 = np.log(y1 + 1), np.log(y2 + 1)
    return np.sqrt(np.mean((y1 - y2)**2))


@multi_ts_support
@multivariate_support
def coefficient_of_variation(actual_series: Union[TimeSeries, Sequence[TimeSeries]],
                             pred_series: Union[TimeSeries, Sequence[TimeSeries]],
                             intersect: bool = True,
                             *,
                             reduction: Callable[[np.ndarray], float] = np.mean,
                             inter_reduction: Callable[[np.ndarray], Union[float, np.ndarray]] = lambda x: x,
                             n_jobs: int = 1,
                             verbose: bool = False) -> Union[float, np.ndarray]:
    """ Coefficient of Variation (percentage).

    Given a time series of actual values :math:`y_t` and a time series of predicted values :math:`\\hat{y}_t`,
    it is a percentage value, computed as

    .. math:: 100 \\cdot \\text{RMSE}(y_t, \\hat{y}_t) / \\bar{y_t},

    where :math:`\\text{RMSE}()` denotes the root mean squared error, and
    :math:`\\bar{y_t}` is the average of :math:`y_t`.

    Currently this only supports deterministic series (made of one sample).

    Parameters
    ----------
    actual_series
        The `TimeSeries` or `Sequence[TimeSeries]` of actual values.
    pred_series
        The `TimeSeries` or `Sequence[TimeSeries]` of predicted values.
    intersect
        For time series that are overlapping in time without having the same time index, setting `intersect=True`
        will consider the values only over their common time interval (intersection in time).
    reduction
        Function taking as input a `np.ndarray` and returning a scalar value. This function is used to aggregate
        the metrics of different components in case of multivariate `TimeSeries` instances.
    inter_reduction
        Function taking as input a `np.ndarray` and returning either a scalar value or a `np.ndarray`.
        This function can be used to aggregate the metrics of different series in case the metric is evaluated on a
        `Sequence[TimeSeries]`. Defaults to the identity function, which returns the pairwise metrics for each pair
        of `TimeSeries` received in input. Example: `inter_reduction=np.mean`, will return the average of the pairwise
        metrics.
    n_jobs
        The number of jobs to run in parallel. Parallel jobs are created only when a `Sequence[TimeSeries]` is
        passed as input, parallelising operations regarding different `TimeSeries`. Defaults to `1`
        (sequential). Setting the parameter to `-1` means using all the available processors.
    verbose
        Optionally, whether to print operations progress

    Returns
    -------
    float
        The Coefficient of Variation
    """

    return 100 * rmse(actual_series, pred_series, intersect) / actual_series.mean().mean()


@multi_ts_support
@multivariate_support
def mape(actual_series: Union[TimeSeries, Sequence[TimeSeries]],
         pred_series: Union[TimeSeries, Sequence[TimeSeries]],
         intersect: bool = True,
         *,
         reduction: Callable[[np.ndarray], float] = np.mean,
         inter_reduction: Callable[[np.ndarray], Union[float, np.ndarray]] = lambda x: x,
         n_jobs: int = 1,
         verbose: bool = False) -> Union[float, np.ndarray]:
    """ Mean Absolute Percentage Error (MAPE).

    Given a time series of actual values :math:`y_t` and a time series of predicted values :math:`\\hat{y}_t`
    both of length :math:`T`, it is a percentage value computed as

    .. math:: 100 \\cdot \\frac{1}{T} \\sum_{t=1}^{T}{\\left| \\frac{y_t - \\hat{y}_t}{y_t} \\right|}.

    Note that it will raise a `ValueError` if :math:`y_t = 0` for some :math:`t`. Consider using
    the Mean Absolute Scaled Error (MASE) in these cases.

    If any of the series is stochastic (containing several samples), the median sample value is considered.

    Parameters
    ----------
    actual_series
        The `TimeSeries` or `Sequence[TimeSeries]` of actual values.
    pred_series
        The `TimeSeries` or `Sequence[TimeSeries]` of predicted values.
    intersect
        For time series that are overlapping in time without having the same time index, setting `intersect=True`
        will consider the values only over their common time interval (intersection in time).
    reduction
        Function taking as input a `np.ndarray` and returning a scalar value. This function is used to aggregate
        the metrics of different components in case of multivariate `TimeSeries` instances.
    inter_reduction
        Function taking as input a `np.ndarray` and returning either a scalar value or a `np.ndarray`.
        This function can be used to aggregate the metrics of different series in case the metric is evaluated on a
        `Sequence[TimeSeries]`. Defaults to the identity function, which returns the pairwise metrics for each pair
        of `TimeSeries` received in input. Example: `inter_reduction=np.mean`, will return the average of the pairwise
        metrics.
    n_jobs
        The number of jobs to run in parallel. Parallel jobs are created only when a `Sequence[TimeSeries]` is
        passed as input, parallelising operations regarding different `TimeSeries`. Defaults to `1`
        (sequential). Setting the parameter to `-1` means using all the available processors.
    verbose
        Optionally, whether to print operations progress

    Raises
    ------
    ValueError
        If the actual series contains some zeros.

    Returns
    -------
    float
        The Mean Absolute Percentage Error (MAPE)
    """

    y_true, y_hat = _get_values_or_raise(actual_series, pred_series, intersect)
    y_true, y_hat = _remove_nan_union(y_true, y_hat)
    raise_if_not((y_true != 0).all(), 'The actual series must be strictly positive to compute the MAPE.', logger)
    return 100. * np.mean(np.abs((y_true - y_hat) / y_true))


@multi_ts_support
@multivariate_support
def smape(actual_series: Union[TimeSeries, Sequence[TimeSeries]],
          pred_series: Union[TimeSeries, Sequence[TimeSeries]],
          intersect: bool = True,
          *,
          reduction: Callable[[np.ndarray], float] = np.mean,
          inter_reduction: Callable[[np.ndarray], Union[float, np.ndarray]] = lambda x: x,
          n_jobs: int = 1,
          verbose: bool = False) -> Union[float, np.ndarray]:
    """ symmetric Mean Absolute Percentage Error (sMAPE).

    Given a time series of actual values :math:`y_t` and a time series of predicted values :math:`\\hat{y}_t`
    both of length :math:`T`, it is a percentage value computed as

    .. math::
        200 \\cdot \\frac{1}{T}
        \\sum_{t=1}^{T}{\\frac{\\left| y_t - \\hat{y}_t \\right|}{\\left| y_t \\right| + \\left| \\hat{y}_t \\right|} }.

    Note that it will raise a `ValueError` if :math:`\\left| y_t \\right| + \\left| \\hat{y}_t \\right| = 0`
     for some :math:`t`. Consider using the Mean Absolute Scaled Error (MASE) in these cases.

    If any of the series is stochastic (containing several samples), the median sample value is considered.

    Parameters
    ----------
    actual_series
        The `TimeSeries` or `Sequence[TimeSeries]` of actual values.
    pred_series
        The `TimeSeries` or `Sequence[TimeSeries]` of predicted values.
    intersect
        For time series that are overlapping in time without having the same time index, setting `intersect=True`
        will consider the values only over their common time interval (intersection in time).
    reduction
        Function taking as input a `np.ndarray` and returning a scalar value. This function is used to aggregate
        the metrics of different components in case of multivariate `TimeSeries` instances.
    inter_reduction
        Function taking as input a `np.ndarray` and returning either a scalar value or a `np.ndarray`.
        This function can be used to aggregate the metrics of different series in case the metric is evaluated on a
        `Sequence[TimeSeries]`. Defaults to the identity function, which returns the pairwise metrics for each pair
        of `TimeSeries` received in input. Example: `inter_reduction=np.mean`, will return the average of the pairwise
        metrics.
    n_jobs
        The number of jobs to run in parallel. Parallel jobs are created only when a `Sequence[TimeSeries]` is
        passed as input, parallelising operations regarding different `TimeSeries`. Defaults to `1`
        (sequential). Setting the parameter to `-1` means using all the available processors.
    verbose
        Optionally, whether to print operations progress

    Raises
    ------
    ValueError
        If the actual series and the pred series contains some zeros at the same time index.

    Returns
    -------
    float
        The symmetric Mean Absolute Percentage Error (sMAPE)
    """

    y_true, y_hat = _get_values_or_raise(actual_series, pred_series, intersect)
    y_true, y_hat = _remove_nan_union(y_true, y_hat)
    raise_if_not(np.logical_or(y_true != 0, y_hat != 0).all(),
                 'The actual series must be strictly positive to compute the sMAPE.', logger)
    return 200. * np.mean(np.abs(y_true - y_hat) / (np.abs(y_true) + np.abs(y_hat)))


# mase cannot leverage multivariate and multi_ts with the decorator since also the `insample` is a Sequence[TimeSeries]
def mase(actual_series: Union[TimeSeries, Sequence[TimeSeries]],
         pred_series: Union[TimeSeries, Sequence[TimeSeries]],
         insample: Union[TimeSeries, Sequence[TimeSeries]],
         m: Optional[int] = 1,
         intersect: bool = True,
         *,
         reduction: Callable[[np.ndarray], float] = np.mean,
         inter_reduction: Callable[[np.ndarray], Union[float, np.ndarray]] = lambda x: x,
         n_jobs: int = 1,
         verbose: bool = False) -> Union[float, np.ndarray]:
    """ Mean Absolute Scaled Error (MASE).

    See `Mean absolute scaled error wikipedia page <https://en.wikipedia.org/wiki/Mean_absolute_scaled_error>`_
    for details about the MASE and how it is computed.

    If any of the series is stochastic (containing several samples), the median sample value is considered.

    Parameters
    ----------
    actual_series
        The `TimeSeries` or `Sequence[TimeSeries]` of actual values.
    pred_series
        The `TimeSeries` or `Sequence[TimeSeries]` of predicted values.
    insample
        The training series used to forecast `pred_series` .
        This series serves to compute the scale of the error obtained by a naive forecaster on the training data.
    m
        Optionally, the seasonality to use for differencing.
        `m=1` corresponds to the non-seasonal MASE, whereas `m>1` corresponds to seasonal MASE.
        If `m=None`, it will be tentatively inferred
        from the auto-correlation function (ACF). It will fall back to a value of 1 if this fails.
    intersect
        For time series that are overlapping in time without having the same time index, setting `intersect=True`
        will consider the values only over their common time interval (intersection in time).
    reduction
        Function taking as input a `np.ndarray` and returning a scalar value. This function is used to aggregate
        the metrics of different components in case of multivariate `TimeSeries` instances.
    inter_reduction
        Function taking as input a `np.ndarray` and returning either a scalar value or a `np.ndarray`.
        This function can be used to aggregate the metrics of different series in case the metric is evaluated on a
        `Sequence[TimeSeries]`. Defaults to the identity function, which returns the pairwise metrics for each pair
        of `TimeSeries` received in input. Example: `inter_reduction=np.mean`, will return the average of the pairwise
        metrics.
    n_jobs
        The number of jobs to run in parallel. Parallel jobs are created only when a `Sequence[TimeSeries]` is
        passed as input, parallelising operations regarding different `TimeSeries`. Defaults to `1`
        (sequential). Setting the parameter to `-1` means using all the available processors.
    verbose
        Optionally, whether to print operations progress

    Raises
    ------
    ValueError
        If the `insample` series is periodic ( :math:`X_t = X_{t-m}` )

    Returns
    -------
    float
        The Mean Absolute Scaled Error (MASE)
    """

    def _multivariate_mase(actual_series: TimeSeries,
                           pred_series: TimeSeries,
                           insample: TimeSeries,
                           m: int,
                           intersect: bool,
                           reduction: Callable[[np.ndarray], float]):

        raise_if_not(actual_series.width == pred_series.width,
                     "The two TimeSeries instances must have the same width.", logger)
        raise_if_not(actual_series.width == insample.width,
                     "The insample TimeSeries must have the same width as the other series.", logger)
        raise_if_not(insample.end_time() + insample.freq == pred_series.start_time(),
                     "The pred_series must be the forecast of the insample series", logger)

        insample_ = insample.quantile_timeseries(quantile=0.5) if insample.is_stochastic else insample

        value_list = []
        for i in range(actual_series.width):
            # old implementation of mase on univariate TimeSeries
            if m is None:
                test_season, m = check_seasonality(insample)
                if not test_season:
                    warn("No seasonality found when computing MASE. Fixing the period to 1.", UserWarning)
                    m = 1

            y_true, y_hat = _get_values_or_raise(actual_series.univariate_component(i),
                                                 pred_series.univariate_component(i),
                                                 intersect)

            x_t = insample_.univariate_component(i).values()
            errors = np.abs(y_true - y_hat)
            scale = np.mean(np.abs(x_t[m:] - x_t[:-m]))
            raise_if_not(not np.isclose(scale, 0), "cannot use MASE with periodical signals", logger)
            value_list.append(np.mean(errors / scale))

        return reduction(value_list)

    if isinstance(actual_series, TimeSeries):
        raise_if_not(isinstance(pred_series, TimeSeries), "Expecting pred_series to be TimeSeries")
        raise_if_not(isinstance(insample, TimeSeries), "Expecting insample to be TimeSeries")
        return _multivariate_mase(actual_series=actual_series,
                                  pred_series=pred_series,
                                  insample=insample,
                                  m=m,
                                  intersect=intersect,
                                  reduction=reduction)

    elif isinstance(actual_series, Sequence) and isinstance(actual_series[0], TimeSeries):

        raise_if_not(isinstance(pred_series, Sequence) and isinstance(pred_series[0], TimeSeries),
                     "Expecting pred_series to be a Sequence[TimeSeries]")
        raise_if_not(isinstance(insample, Sequence) and isinstance(insample[0], TimeSeries),
                     "Expecting insample to be a Sequence[TimeSeries]")
        raise_if_not(len(pred_series) == len(actual_series) and len(pred_series) == len(insample),
                     "The TimeSeries sequences must have the same length.", logger)

        raise_if_not(isinstance(n_jobs, int), "n_jobs must be an integer")
        raise_if_not(isinstance(verbose, bool), "verbose must be a bool")

        iterator = _build_tqdm_iterator(iterable=zip(actual_series, pred_series, insample),
                                        verbose=verbose,
                                        total=len(actual_series))

        value_list = _parallel_apply(iterator=iterator,
                                     fn=_multivariate_mase,
                                     n_jobs=n_jobs,
                                     fn_args=dict(),
                                     fn_kwargs={
                                         "m": m,
                                         "intersect": intersect,
                                         "reduction": reduction
                                     })
        return inter_reduction(value_list)
    else:
        raise_log(ValueError("Input type not supported, only TimeSeries and Sequence[TimeSeries] are accepted."))


@multi_ts_support
@multivariate_support
def ope(actual_series: Union[TimeSeries, Sequence[TimeSeries]],
        pred_series: Union[TimeSeries, Sequence[TimeSeries]],
        intersect: bool = True,
        *,
        reduction: Callable[[np.ndarray], float] = np.mean,
        inter_reduction: Callable[[np.ndarray], Union[float, np.ndarray]] = lambda x: x,
        n_jobs: int = 1,
        verbose: bool = False) -> Union[float, np.ndarray]:
    """ Overall Percentage Error (OPE).

    Given a time series of actual values :math:`y_t` and a time series of predicted values :math:`\\hat{y}_t`
    both of length :math:`T`, it is a percentage value computed as

    .. math:: 100 \\cdot \\left| \\frac{\\sum_{t=1}^{T}{y_t}
              - \\sum_{t=1}^{T}{\\hat{y}_t}}{\\sum_{t=1}^{T}{y_t}} \\right|.

    If any of the series is stochastic (containing several samples), the median sample value is considered.

    Parameters
    ----------
    actual_series
        The `TimeSeries` or `Sequence[TimeSeries]` of actual values.
    pred_series
        The `TimeSeries` or `Sequence[TimeSeries]` of predicted values.
    intersect
        For time series that are overlapping in time without having the same time index, setting `intersect=True`
        will consider the values only over their common time interval (intersection in time).
    reduction
        Function taking as input a `np.ndarray` and returning a scalar value. This function is used to aggregate
        the metrics of different components in case of multivariate `TimeSeries` instances.
    inter_reduction
        Function taking as input a `np.ndarray` and returning either a scalar value or a `np.ndarray`.
        This function can be used to aggregate the metrics of different series in case the metric is evaluated on a
        `Sequence[TimeSeries]`. Defaults to the identity function, which returns the pairwise metrics for each pair
        of `TimeSeries` received in input. Example: `inter_reduction=np.mean`, will return the average of the pairwise
        metrics.
    n_jobs
        The number of jobs to run in parallel. Parallel jobs are created only when a `Sequence[TimeSeries]` is
        passed as input, parallelising operations regarding different `TimeSeries`. Defaults to `1`
        (sequential). Setting the parameter to `-1` means using all the available processors.
    verbose
        Optionally, whether to print operations progress

    Raises
    ------
    ValueError
        If :math:`\\sum_{t=1}^{T}{y_t} = 0`.

    Returns
    -------
    float
        The Overall Percentage Error (OPE)
    """

    y_true, y_pred = _get_values_or_raise(actual_series, pred_series, intersect)
    y_true, y_pred = _remove_nan_union(y_true, y_pred)
    y_true_sum, y_pred_sum = np.sum(y_true), np.sum(y_pred)
    raise_if_not(y_true_sum > 0, 'The series of actual value cannot sum to zero when computing OPE.', logger)
    return np.abs((y_true_sum - y_pred_sum) / y_true_sum) * 100.


@multi_ts_support
@multivariate_support
def marre(actual_series: Union[TimeSeries, Sequence[TimeSeries]],
          pred_series: Union[TimeSeries, Sequence[TimeSeries]],
          intersect: bool = True,
          *,
          reduction: Callable[[np.ndarray], float] = np.mean,
          inter_reduction: Callable[[np.ndarray], Union[float, np.ndarray]] = lambda x: x,
          n_jobs: int = 1,
          verbose: bool = False) -> Union[float, np.ndarray]:
    """ Mean Absolute Ranged Relative Error (MARRE).

    Given a time series of actual values :math:`y_t` and a time series of predicted values :math:`\\hat{y}_t`
    both of length :math:`T`, it is a percentage value computed as

    .. math:: 100 \\cdot \\frac{1}{T} \\sum_{t=1}^{T} {\\left| \\frac{y_t - \\hat{y}_t} {\\max_t{y_t} -
              \\min_t{y_t}} \\right|}

    If any of the series is stochastic (containing several samples), the median sample value is considered.

    Parameters
    ----------
    actual_series
        The `TimeSeries` or `Sequence[TimeSeries]` of actual values.
    pred_series
        The `TimeSeries` or `Sequence[TimeSeries]` of predicted values.
    intersect
        For time series that are overlapping in time without having the same time index, setting `intersect=True`
        will consider the values only over their common time interval (intersection in time).
    reduction
        Function taking as input a `np.ndarray` and returning a scalar value. This function is used to aggregate
        the metrics of different components in case of multivariate `TimeSeries` instances.
    inter_reduction
        Function taking as input a `np.ndarray` and returning either a scalar value or a `np.ndarray`.
        This function can be used to aggregate the metrics of different series in case the metric is evaluated on a
        `Sequence[TimeSeries]`. Defaults to the identity function, which returns the pairwise metrics for each pair
        of `TimeSeries` received in input. Example: `inter_reduction=np.mean`, will return the average of the pairwise
        metrics.
    n_jobs
        The number of jobs to run in parallel. Parallel jobs are created only when a `Sequence[TimeSeries]` is
        passed as input, parallelising operations regarding different `TimeSeries`. Defaults to `1`
        (sequential). Setting the parameter to `-1` means using all the available processors.
    verbose
        Optionally, whether to print operations progress

    Raises
    ------
    ValueError
        If :math:`\\max_t{y_t} = \\min_t{y_t}`.

    Returns
    -------
    float
        The Mean Absolute Ranged Relative Error (MARRE)
    """

    y_true, y_hat = _get_values_or_raise(actual_series, pred_series, intersect)
    y_true, y_hat = _remove_nan_union(y_true, y_hat)
    raise_if_not(y_true.max() > y_true.min(), 'The difference between the max and min values must be strictly'
                 'positive to compute the MARRE.', logger)
    true_range = y_true.max() - y_true.min()
    return 100. * np.mean(np.abs((y_true - y_hat) / true_range))


@multi_ts_support
@multivariate_support
def r2_score(actual_series: Union[TimeSeries, Sequence[TimeSeries]],
             pred_series: Union[TimeSeries, Sequence[TimeSeries]],
             intersect: bool = True,
             *,
             reduction: Callable[[np.ndarray], float] = np.mean,
             inter_reduction: Callable[[np.ndarray], Union[float, np.ndarray]] = lambda x: x,
             n_jobs: int = 1,
             verbose: bool = False) -> Union[float, np.ndarray]:
    """ Coefficient of Determination :math:`R^2`.

    See `Coefficient of determination wikipedia page <https://en.wikipedia.org/wiki/Coefficient_of_determination>`_
    for details about the :math:`R^2` score and how it is computed.
    Please note that this metric is not symmetric, `series1` should correspond to the ground truth series,
    whereas `series2` should correspond to the predicted series.

    If any of the series is stochastic (containing several samples), the median sample value is considered.

    Parameters
    ----------
    actual_series
        The `TimeSeries` or `Sequence[TimeSeries]` of actual values.
    pred_series
        The `TimeSeries` or `Sequence[TimeSeries]` of predicted values.
    intersect
        For time series that are overlapping in time without having the same time index, setting `intersect=True`
        will consider the values only over their common time interval (intersection in time).
    reduction
        Function taking as input a `np.ndarray` and returning a scalar value. This function is used to aggregate
        the metrics of different components in case of multivariate `TimeSeries` instances.
    inter_reduction
        Function taking as input a `np.ndarray` and returning either a scalar value or a `np.ndarray`.
        This function can be used to aggregate the metrics of different series in case the metric is evaluated on a
        `Sequence[TimeSeries]`. Defaults to the identity function, which returns the pairwise metrics for each pair
        of `TimeSeries` received in input. Example: `inter_reduction=np.mean`, will return the average of the pairwise
        metrics.
    n_jobs
        The number of jobs to run in parallel. Parallel jobs are created only when a `Sequence[TimeSeries]` is
        passed as input, parallelising operations regarding different `TimeSeries`. Defaults to `1`
        (sequential). Setting the parameter to `-1` means using all the available processors.
    verbose
        Optionally, whether to print operations progress

    Returns
    -------
    float
        The Coefficient of Determination :math:`R^2`
    """
    y1, y2 = _get_values_or_raise(actual_series, pred_series, intersect)
    y1, y2 = _remove_nan_union(y1, y2)
    ss_errors = np.sum((y1 - y2) ** 2)
    y_hat = y1.mean()
    ss_tot = np.sum((y1 - y_hat) ** 2)
    return 1 - ss_errors / ss_tot
