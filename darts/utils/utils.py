"""
Additional util functions
-------------------------
"""
import pandas as pd

from ..timeseries import TimeSeries
from ..logging import raise_log, get_logger, raise_if_not, raise_if
from typing import List, Callable, TypeVar, Union
from IPython import get_ipython
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook
from functools import wraps
from types import SimpleNamespace
from inspect import signature, Parameter, getcallargs

logger = get_logger(__name__)


# TODO: we do not check the time index here
def retain_period_common_to_all(series: List[TimeSeries]) -> List[TimeSeries]:
    """
    Trims all series in the provided list, if necessary, so that the returned time series have
    a common span (corresponding to largest time sub-interval common to all series).

    Parameters
    ----------
    series
        The list of series to consider.

    Raises
    ------
    ValueError
        If no common time sub-interval exists

    Returns
    -------
    List[TimeSeries]
        A list of series, where each series have the same span
    """

    last_first = max(map(lambda s: s.start_time(), series))
    first_last = min(map(lambda s: s.end_time(), series))

    if last_first >= first_last:
        raise_log(ValueError('The provided time series must have nonzero overlap'), logger)

    return list(map(lambda s: s.slice(last_first, first_last), series))


def _build_tqdm_iterator(iterable, verbose):
    """
    Build an iterable, possibly using tqdm (either in notebook or regular mode)

    Parameters
    ----------
    iterable
    verbose

    Returns
    -------

    """

    def _isnotebook():
        try:
            shell = get_ipython().__class__.__name__
            if shell == 'ZMQInteractiveShell':
                return True  # Jupyter notebook or qtconsole
            elif shell == 'TerminalInteractiveShell':
                return False  # Terminal running IPython
            else:
                return False  # Other type (?)
        except NameError:
            return False  # Probably standard Python interpreter

    if verbose:
        if _isnotebook():
            iterator = tqdm_notebook(iterable)
        else:
            iterator = tqdm(iterable)
    else:
        iterator = iterable
    return iterator


# Types for sanity checks decorator
A = TypeVar('A')
B = TypeVar('B')
T = TypeVar('T')


def _with_sanity_checks(*sanity_check_methods: str) -> Callable[[Callable[[A, B], T]], Callable[[A, B], T]]:
    """
    Decorator allowing to specify some sanity check method(s) to be used on a class method.
    The decorator guarantees that args and kwargs from the method to sanitize will be available in the
    sanity check methods as specified in the sanitized method's signature, irrespective of how it was called.

    Parameters
    ----------
    *sanity_check_methods
        one or more sanity check methods that will be called with all the parameter of the decorated method.

    Returns
    -------
    A Callable corresponding to the decorated method.

    Examples
    --------
    class Model:
        def _a_sanity_check(self, *args, **kwargs):
            raise_if_not(kwargs['b'] == kwargs['c'], 'b must equal c', logger)

        @_with_sanity_checks("_a_sanity_check")
        def fit(self, a, b=0, c=0):
            # at this point we can safely assume that 'b' and 'c' are equal...
            ...
    """
    def decorator(method_to_sanitize: Callable[[A, B], T]) -> Callable[[A, B], T]:
        @wraps(method_to_sanitize)
        def sanitized_method(self, *args: A, **kwargs: B) -> T:
            for sanity_check_method in sanity_check_methods:
                # Convert all arguments into keyword arguments
                all_as_kwargs = getcallargs(method_to_sanitize, self, *args, **kwargs)

                # Then separate args from kwargs according to the function's signature
                only_args = all_as_kwargs.copy()
                only_kwargs = all_as_kwargs.copy()

                for param_name, param in signature(method_to_sanitize).parameters.items():
                    if param.default == Parameter.empty and param.kind != Parameter.VAR_POSITIONAL:
                        only_kwargs.pop(param_name)
                    else:
                        only_args.pop(param_name)

                only_args.pop('self')

                getattr(self, sanity_check_method)(*only_args.values(), **only_kwargs)
            return method_to_sanitize(self, *only_args.values(), **only_kwargs)
        return sanitized_method
    return decorator


def _backtest_general_checks(series, kwargs):
    """
    Performs checks common to ForecastingModel and RegressionModel backtest() methods

    Parameters
    ----------
    series
        Either training_series when called from ForecastingModel, or target_series if called from RegressionModel
    signature_params
        A dictionary of the signature parameters of the calling method, to get the default values
        Typically would be signature(self.backtest).parameters
    kwargs
        Params specified by the caller of backtest(), they take precedence over the arguments' default values
    """

    # parse kwargs
    n = SimpleNamespace(**kwargs)

    # check forecast horizon
    forecast_horizon = n.forecast_horizon

    raise_if_not(forecast_horizon > 0, 'The provided forecasting horizon must be a positive integer.', logger)

    # check start parameter
    if hasattr(n, 'start'):
        if isinstance(n.start, float):
            raise_if_not(n.start >= 0.0 and n.start < 1.0, '`start` should be between 0.0 and 1.0.', logger)
        elif isinstance(n.start, pd.Timestamp):
            raise_if(n.start not in series, '`start` timestamp must be an entry in the time series\' time index')
            raise_if(n.start == series.end_time(), '`start` timestamp is the last timestamp of the series', logger)
        elif isinstance(n.start, int):
            raise_if_not(n.start >= 0, logger)
            raise_if(n.start > len(series), '`start` index should be smaller than length of the series', logger)
        else:
            raise_log(TypeError("`start` needs to be either `float`, `int` or `pd.Timestamp`"), logger)

    start = _get_timestamp_at_point(n.start, series)

    raise_if(start == series.start_time(), '`start` corresponds to the first timestamp of the series, '
             'resulting in empty training set')

    # check that trim_to_series and start together form a valid combination
    trim_to_series = n.trim_to_series

    if trim_to_series:
        raise_if_not(start + series.freq() * forecast_horizon in series,
                     '`start` timestamp is too late in the series to make any predictions with'
                     '`trim_to_series` set to `True`.', logger)


def _get_timestamp_at_point(point: Union[pd.Timestamp, float, int], series: TimeSeries) -> pd.Timestamp:
    """
    Converts a point into a pandas.Timestamp in the time series

    Parameters
    ----------
    point
        This parameter supports 3 different data types: `float`, `int` and `pandas.Timestamp`.
        In the case of `float`, the parameter will be treated as the proportion of the time series
        that should lie before the point.
        In the case of `int`, the parameter will be treated as an integer index to the time index of
        `series`.
        In case of `pandas.Timestamp`, `point` will be returned as is
    series
        The time series to index in
    """
    if isinstance(point, float):
        point_index = int((len(series.time_index()) - 1) * point)
        timestamp = series.time_index()[point_index]
    elif isinstance(point, int):
        timestamp = series[point].start_time()
    else:
        timestamp = point
    return timestamp
