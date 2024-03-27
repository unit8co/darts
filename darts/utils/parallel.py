"""
Parallel execution util functions
-------------------------
"""

from typing import Callable, Iterator, List, Tuple

from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook

from darts.logging import get_logger

try:
    from IPython import get_ipython
except ModuleNotFoundError:
    get_ipython = None

logger = get_logger(__name__)


def _build_tqdm_iterator(iterable, verbose, **kwargs):
    """
    Build an iterable, possibly using tqdm (either in notebook or regular mode)

    Parameters
    ----------
    iterable
    verbose
    total
        Length of the iterator, helps in cases where tqdm is not detecting the total length.

    Returns
    -------
    """

    def _isnotebook():
        if get_ipython is None:
            return False
        try:
            shell = get_ipython().__class__.__name__
            if shell == "ZMQInteractiveShell":
                return True  # Jupyter notebook or qtconsole
            elif shell == "TerminalInteractiveShell":
                return False  # Terminal running IPython
            else:
                return False  # Other type (?)
        except NameError:
            return False  # Probably standard Python interpreter

    if verbose:
        if _isnotebook():
            iterator = tqdm_notebook(iterable, **kwargs)
        else:
            iterator = tqdm(iterable, **kwargs)

    else:
        iterator = iterable
    return iterator


def _parallel_apply(
    iterator: Iterator[Tuple], fn: Callable, n_jobs: int, fn_args, fn_kwargs
) -> List:
    """
    Utility function that parallelise the execution of a function over an Iterator

    Parameters
    ----------
    iterator (Iterator[Tuple])
        Iterator which returns tuples of input value to feed to fn. Constant `args` and `kwargs` should passed through
        `fn_args` and  `fn_kwargs` respectively.
    fn (Callable)
        The function to be parallelized.
    n_jobs (int)
        The number of jobs to run in parallel. Defaults to `1` (sequential). Setting the parameter to `-1` means using
        all the available processors.
        Note: for a small amount of data, the parallelisation overhead could end up increasing the total
        required amount of time.
    fn_args
        Additional arguments for each `fn()` call
    fn_kwargs
        Additional keyword arguments for each `fn()` call

    """

    returned_data = Parallel(n_jobs=n_jobs)(
        delayed(fn)(*sample, *fn_args, **fn_kwargs) for sample in iterator
    )
    return returned_data
