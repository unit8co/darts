"""
utils related to Pytorch and its usage.
"""
from typing import Callable, TypeVar
from inspect import getargspec

from ..logging import raise_if_not, get_logger

T = TypeVar('T')
logger = get_logger(__name__)


def random_method(decorated: Callable[..., T]) -> Callable[..., T]:
    """ Decorator usable on any method within a class that will provide an isolated torch random context.

    The decorator will store a `_random_state` property on the object in order to persist successive calls to the RNG.

    Parameters
    ----------
    decorated
        A method to be run in an isolated torch random context.

    """
    # check that @random_method has been applied to a method.
    spec = getargspec(decorated)
    is_method = spec.args and spec.args[0] == 'self'
    raise_if_not(is_method, "@random_method can only be used on methods.", logger)

    def decorator(self, *args, **kwargs) -> T:
        return decorated(self, *args, **kwargs)

    return decorator
