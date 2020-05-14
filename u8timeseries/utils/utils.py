"""
Additional util functions
-------------------------
"""

from ..timeseries import TimeSeries
from ..logging import raise_log, get_logger
from typing import List
from IPython import get_ipython
from tqdm import tqdm, tqdm_notebook

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
