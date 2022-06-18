"""
Utils for Pytorch and its usage
-------------------------------
"""

from functools import wraps
from inspect import signature
from typing import Any, Callable, TypeVar

import torch.nn as nn
import torch.nn.functional as F
from numpy.random import randint
from sklearn.utils import check_random_state
from torch import Tensor
from torch.random import fork_rng, manual_seed

from darts.logging import get_logger, raise_if_not

T = TypeVar("T")
logger = get_logger(__name__)

MAX_TORCH_SEED_VALUE = (1 << 31) - 1  # to accommodate 32-bit architectures
MAX_NUMPY_SEED_VALUE = (1 << 31) - 1


class MonteCarloDropout(nn.Dropout):
    """
    Defines Monte Carlo dropout Module as defined
    in the paper https://arxiv.org/pdf/1506.02142.pdf.
    In summary, This technique uses the regular dropout
    which can be interpreted as a Bayesian approximation of
    a well-known probabilistic model: the Gaussian process.
    We can treat the many different networks
    (with different neurons dropped out) as Monte Carlo samples
    from the space of all available models. This provides mathematical
    grounds to reason about the model’s uncertainty and, as it turns out,
    often improves its performance.
    """

    # We need to init it to False as some models may start by
    # a validation round, in which case MC dropout is disabled.
    mc_dropout_enabled: bool = False

    def train(self, mode: bool = True):
        # NOTE: we could use the line below if self.mc_dropout_rate represented
        # a rate to be applied at inference time, and self.applied_rate the
        # actual rate to be used in self.forward(). However, the original paper
        # considers the same rate for training and inference; we also stick to this.

        # self.applied_rate = self.p if mode else self.mc_dropout_rate

        if mode:  # in train mode, keep dropout as is
            self.mc_dropout_enabled = True
        # in eval mode, bank on the mc_dropout_enabled flag
        # mc_dropout_enabled is set equal to "mc_dropout" param given to predict()

    def forward(self, input: Tensor) -> Tensor:
        # NOTE: we could use the following line in case a different rate
        # is used for inference:
        # return F.dropout(input, self.applied_rate, True, self.inplace)

        return F.dropout(input, self.p, self.mc_dropout_enabled, self.inplace)


def _is_method(func: Callable[..., Any]) -> bool:
    """Check if the specified function is a method.

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
        if list(spec.parameters.keys())[0] == "self":
            return True
    return False


def random_method(decorated: Callable[..., T]) -> Callable[..., T]:
    """Decorator usable on any method within a class that will provide an isolated torch random context.

    The decorator will store a `_random_instance` property on the object in order to persist successive calls to the RNG

    Parameters
    ----------
    decorated
        A method to be run in an isolated torch random context.

    """
    # check that @random_method has been applied to a method.
    raise_if_not(
        _is_method(decorated), "@random_method can only be used on methods.", logger
    )

    @wraps(decorated)
    def decorator(self, *args, **kwargs) -> T:
        if "random_state" in kwargs.keys():
            self._random_instance = check_random_state(kwargs["random_state"])
        elif not hasattr(self, "_random_instance"):
            self._random_instance = check_random_state(
                randint(0, high=MAX_NUMPY_SEED_VALUE)
            )

        with fork_rng():
            manual_seed(self._random_instance.randint(0, high=MAX_TORCH_SEED_VALUE))
            return decorated(self, *args, **kwargs)

    return decorator
