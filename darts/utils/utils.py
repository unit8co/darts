"""
Additional util functions
-------------------------
"""

from ..timeseries import TimeSeries
from ..logging import raise_log, get_logger
from typing import List, Callable, TypeVar
from IPython import get_ipython
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook
from functools import wraps

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
    """ Decorator allowing to specify some sanity check method(s) to be used on a class method.

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
                getattr(self, sanity_check_method)(*args, **kwargs)
            return method_to_sanitize(self, *args, **kwargs)
        return sanitized_method
    return decorator
