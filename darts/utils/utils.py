"""
Additional util functions
-------------------------
"""

from enum import Enum
from functools import wraps
from inspect import Parameter, getcallargs, signature
from typing import Callable, Iterator, List, Optional, Tuple, TypeVar, Union

import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook

from darts.logging import get_logger, raise_if, raise_if_not, raise_log

try:
    from IPython import get_ipython
except ModuleNotFoundError:
    get_ipython = None

logger = get_logger(__name__)


# Enums
class SeasonalityMode(Enum):
    MULTIPLICATIVE = "multiplicative"
    ADDITIVE = "additive"
    NONE = None


class TrendMode(Enum):
    LINEAR = "linear"
    EXPONENTIAL = "exponential"


class ModelMode(Enum):
    MULTIPLICATIVE = "multiplicative"
    ADDITIVE = "additive"
    NONE = None


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


# Types for sanity checks decorator: T is the output of the method to sanitize
T = TypeVar("T")


def _with_sanity_checks(
    *sanity_check_methods: str,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator allowing to specify some sanity check method(s) to be used on a class method.
    The decorator guarantees that args and kwargs from the method to sanitize will be available in the
    sanity check methods as specified in the sanitized method's signature, irrespective of how it was called.
    TypeVar `T` corresponds to the output of the method that the sanity checks are performed for.

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

    def decorator(method_to_sanitize: Callable[..., T]) -> Callable[..., T]:
        @wraps(method_to_sanitize)
        def sanitized_method(self, *args, **kwargs) -> T:
            only_args, only_kwargs = {}, {}
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

                only_args.pop("self")

                getattr(self, sanity_check_method)(*only_args.values(), **only_kwargs)
            return method_to_sanitize(self, *only_args.values(), **only_kwargs)

        return sanitized_method

    return decorator


def _parallel_apply(iterator: Iterator[Tuple], fn: Callable, n_jobs: int, fn_args, fn_kwargs) -> List:
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

    returned_data = Parallel(n_jobs=n_jobs)(delayed(fn)(*sample, *fn_args, **fn_kwargs) for sample in iterator)
    return returned_data


def _check_quantiles(quantiles):
    raise_if_not(
        all([0 < q < 1 for q in quantiles]),
        "All provided quantiles must be between 0 and 1.",
    )

    # we require the median to be present and the quantiles to be symmetric around it,
    # for correctness of sampling.
    median_q = 0.5
    raise_if_not(median_q in quantiles, "median quantile `q=0.5` must be in `quantiles`")
    is_centered = [
        -1e-6 < (median_q - left_q) + (median_q - right_q) < 1e-6 for left_q, right_q in zip(quantiles, quantiles[::-1])
    ]
    raise_if_not(
        all(is_centered),
        "quantiles lower than `q=0.5` need to share same difference to `0.5` as quantiles " "higher than `q=0.5`",
    )


def slice_index(
    index: Union[pd.RangeIndex, pd.DatetimeIndex],
    start: Union[int, pd.Timestamp],
    end: Union[int, pd.Timestamp],
) -> Union[pd.RangeIndex, pd.DatetimeIndex]:
    """
    Returns a new Index with the same type as the input `index`, containing the values between `start`
    and `end` included. If start and end are not in the index, the closest values are used instead.
    The start and end values can be either integers (in which case they are interpreted as indices),
    or pd.Timestamps (in which case they are interpreted as actual timestamps).


    Parameters
    ----------
    index
        The index to slice.
    start
        The start of the returned index.
    end
        The end of the returned index.

    Returns
    -------
    Union[pd.RangeIndex, pd.DatetimeIndex]
        A new index with the same type as the input `index`, but with only the values between `start` and `end`
        included.
    """

    if type(start) is not type(end):
        raise_log(
            ValueError("start and end values must be of the same type (either both integers or both pd.Timestamps)"),
            logger,
        )

    if isinstance(start, pd.Timestamp) and isinstance(index, pd.RangeIndex):
        raise_log(
            ValueError(
                "start and end values are a pd.Timestamp, but time_index is a RangeIndex. "
                "Please provide an integer start value."
            ),
            logger,
        )
    if isinstance(start, int) and isinstance(index, pd.DatetimeIndex):
        raise_log(
            ValueError(
                "start and end value are integer, but time_index is a RangeIndex. "
                "Please provide an integer end value."
            ),
            logger,
        )

    start_idx = index.get_indexer(generate_index(start, length=1), method="nearest")[0]
    end_idx = index.get_indexer(generate_index(end, length=1), method="nearest")[0]

    return index[start_idx : end_idx + 1]


def drop_before_index(
    index: Union[pd.RangeIndex, pd.DatetimeIndex],
    split_point: Union[int, pd.Timestamp],
) -> Union[pd.RangeIndex, pd.DatetimeIndex]:
    """
    Drops everything before the provided time `split_point` (excluded) from the index.

    Parameters
    ----------
    index
        The index to drop values from.
    split_point
        The timestamp that indicates cut-off time.

    Returns
    -------
    Union[pd.RangeIndex, pd.DatetimeIndex]
        A new index with values before `split_point` dropped.
    """
    return slice_index(index, split_point, index[-1])


def drop_after_index(
    index: Union[pd.RangeIndex, pd.DatetimeIndex],
    split_point: Union[int, pd.Timestamp],
) -> Union[pd.RangeIndex, pd.DatetimeIndex]:
    """
    Drops everything after the provided time `split_point` (excluded) from the index.

    Parameters
    ----------
    index
        The index to drop values from.
    split_point
        The timestamp that indicates cut-off time.

    Returns
    -------
    Union[pd.RangeIndex, pd.DatetimeIndex]
        A new index with values after `split_point` dropped.
    """

    return slice_index(index, index[0], split_point)


def n_steps_between(
    end: Union[pd.Timestamp, int],
    start: Union[pd.Timestamp, int],
    freq: Union[pd.DateOffset, int, str],
) -> int:
    """Get the number of time steps with a given frequency `freq` between `end` and `start`.
    Works for both integers and time stamps.

    * if `end`, `start`, `freq` are all integers, we can simple divide the difference by the frequency.
    * if `freq` is a pandas Dateoffset with non-ambiguous timedelate (e.g. "d", "h", ..., and not "M", "Y", ...),
        we can simply divide by the frequency
    * otherwise, we take the period difference between the two time stamps.

    Parameters
    ----------
    end
        The end pandas Timestamp / integer.
    start
        The start pandas Timestamp / integer.
    freq
        The frequency / step size.

    Returns
    -------
    int
        The number of steps/periods between `end` and `start` with a given frequency `freq`.

    Examples
    --------
    >>> n_steps_between(start=pd.Timestamp("2000-01-01"), end=pd.Timestamp("2000-03-01"), freq="M")
    2
    >>> n_steps_between(start=0, end=2, freq=1)
    2
    >>> n_steps_between(start=0, end=2, freq=2)
    1
    """
    freq = pd.tseries.frequencies.to_offset(freq) if isinstance(freq, str) else freq
    valid_int = isinstance(start, int) and isinstance(end, int) and isinstance(freq, int)
    valid_time = isinstance(start, pd.Timestamp) and isinstance(end, pd.Timestamp) and isinstance(freq, pd.DateOffset)
    if not (valid_int or valid_time):
        raise_log(
            ValueError(
                "Either `start` and `end` must be pandas Timestamps and `freq` a pandas Dateoffset, "
                "or all `start`, `end`, `freq` must be integers."
            ),
            logger=logger,
        )
    # Series frequency represents a non-ambiguous timedelta value (not ‘M’, ‘Y’ or ‘y’)
    if pd.to_timedelta(freq, errors="coerce") is not pd.NaT:
        n_steps = (end - start) // freq
    else:
        period_alias = pd.tseries.frequencies.get_period_alias(freq.freqstr)
        if period_alias is None:
            raise_log(
                ValueError(
                    f"Cannot infer period alias for `freq={freq}`. " f"Is it a valid pandas offset/frequency alias?"
                ),
                logger=logger,
            )
        # Create a temporary DatetimeIndex to extract the actual start index.
        n_steps = (end.to_period(period_alias) - start.to_period(period_alias)).n
    return n_steps


def generate_index(
    start: Optional[Union[pd.Timestamp, int]] = None,
    end: Optional[Union[pd.Timestamp, int]] = None,
    length: Optional[int] = None,
    freq: Union[str, int, pd.DateOffset] = None,
    name: str = None,
) -> Union[pd.DatetimeIndex, pd.RangeIndex]:
    """Returns an index with a given start point and length. Either a pandas DatetimeIndex with given frequency
    or a pandas RangeIndex. The index starts at

    Parameters
    ----------
    start
        The start of the returned index. If a pandas Timestamp is passed, the index will be a pandas
        DatetimeIndex. If an integer is passed, the index will be a pandas RangeIndex index. Works only with
        either `length` or `end`.
    end
        Optionally, the end of the returned index. Works only with either `start` or `length`. If `start` is
        set, `end` must be of same type as `start`. Else, it can be either a pandas Timestamp or an integer.
    length
        Optionally, the length of the returned index. Works only with either `start` or `end`.
    freq
        The time difference between two adjacent entries in the returned index. In case `start` is a timestamp,
        a DateOffset alias is expected; see
        `docs <https://pandas.pydata.org/pandas-docs/stable/user_guide/TimeSeries.html#dateoffset-objects>`_.
        By default, "D" (daily) is used.
        If `start` is an integer, `freq` will be interpreted as the step size in the underlying RangeIndex.
        The freq is optional for generating an integer index (if not specified, 1 is used).
    name
        Optionally, an index name.
    """
    constructors = [
        arg_name for arg, arg_name in zip([start, end, length], ["start", "end", "length"]) if arg is not None
    ]
    raise_if(
        len(constructors) != 2,
        "index can only be generated with exactly two of the following parameters: [`start`, `end`, `length`]. "
        f"Observed parameters: {constructors}. For generating an index with `end` and `length` consider setting "
        f"`start` to None.",
        logger,
    )
    raise_if(
        end is not None and start is not None and type(start) is not type(end),
        "index generation with `start` and `end` requires equal object types of `start` and `end`",
        logger,
    )

    if isinstance(start, pd.Timestamp) or isinstance(end, pd.Timestamp):
        index = pd.date_range(
            start=start,
            end=end,
            periods=length,
            freq="D" if freq is None else freq,
            name=name,
        )
    else:  # int
        step = 1 if freq is None else freq
        index = pd.RangeIndex(
            start=start if start is not None else end - step * length + step,
            stop=end + step if end is not None else start + step * length,
            step=step,
            name=name,
        )
    return index
