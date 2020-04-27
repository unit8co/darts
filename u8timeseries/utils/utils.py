from ..timeseries import TimeSeries
from ..custom_logging import raise_log, get_logger
from typing import List
from IPython import get_ipython
from tqdm import tqdm, tqdm_notebook

logger = get_logger(__name__)

def retain_period_common_to_all(series: List[TimeSeries]) -> List[TimeSeries]:
    """
    Trims all series in the provided list, if necessary, so that the return time series have
    the same time index (corresponding to largest duration common to all series).

    Raises an error if no such time index exists.
    :param series:
    :return:
    """

    last_first = max(map(lambda s: s.start_time(), series))
    first_last = min(map(lambda s: s.end_time(), series))

    if last_first >= first_last:
        raise_log(ValueError('The provided time series must have nonzero overlap'), logger)

    return list(map(lambda s: s.slice(last_first, first_last), series))


def build_tqdm_iterator(iterable, verbose):
    """
    Build an iterable, possibly using tqdm (either in notebook or regular mode)
    :param iterable:
    :param verbose:
    :return:
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

