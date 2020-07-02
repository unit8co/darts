"""
utils related to Pytorch and its usage.
"""
from typing import Callable, TypeVar
T = TypeVar('T')


def random_method(decorated: Callable[..., T]) -> Callable[..., T]:
    """ Decorator usable on any method within a class that will provide an isolated torch random context.

    The decorator will store a `_random_state` property on the object in order to persist successive calls to the RNG.

    Parameters
    ----------
    decorated
        A method to be run in an isolated torch random context.

    """
    # TODO: add a check for the decorated to be a method!!!

    def decorator(self, *args, **kwargs) -> T:
        return decorated(self, *args, **kwargs)

    return decorator
