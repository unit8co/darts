"""
Utils for Pytorch and its usage
-------------------------------
"""
from typing import Callable, TypeVar, Any
from inspect import signature

from sklearn.utils import check_random_state
from torch.random import fork_rng, manual_seed

from ..logging import raise_if_not, get_logger

T = TypeVar('T')
logger = get_logger(__name__)

MAX_TORCH_SEED_VALUE = (1 << 63) - 1


def _is_method(func: Callable[..., Any]) -> bool:
    """ Check if the specified function is a method.

    Parameters
    ----------
    func
        the function to inspect.

    Returns
    -------
    bool
        true if `func` is a method, false otherwise.
    """
    spec = signature(func)
    if len(spec.parameters) > 0:
        if list(spec.parameters.keys())[0] == 'self':
            return True
    return False


def random_method(decorated: Callable[..., T]) -> Callable[..., T]:
    """ Decorator usable on any method within a class that will provide an isolated torch random context.

    The decorator will store a `_random_instance` property on the object in order to persist successive calls to the RNG

    Parameters
    ----------
    decorated
        A method to be run in an isolated torch random context.

    """
    # check that @random_method has been applied to a method.
    raise_if_not(_is_method(decorated), "@random_method can only be used on methods.", logger)

    def decorator(self, *args, **kwargs) -> T:
        if "random_state" in kwargs.keys() or hasattr(self, "_random_instance"):
            if "random_state" in kwargs.keys():
                self._random_instance = check_random_state(kwargs["random_state"])

            with fork_rng():
                manual_seed(self._random_instance.randint(0, high=MAX_TORCH_SEED_VALUE))
                decorated(self, *args, **kwargs)
        else:
            decorated(self, *args, **kwargs)

    return decorator
