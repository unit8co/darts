"""
Utils for Pytorch and its usage
-------------------------------
"""

from collections.abc import Callable
from functools import wraps
from typing import TypeVar

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from sklearn.utils import check_random_state
from torch import Tensor
from torch.random import fork_rng, manual_seed

from darts.logging import get_logger, raise_log
from darts.utils.utils import MAX_NUMPY_SEED_VALUE, MAX_TORCH_SEED_VALUE, _is_method

T = TypeVar("T")
logger = get_logger(__name__)


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

    # mc dropout is deactivated at init; see `MonteCarloDropout.mc_dropout_enabled` for more info
    _mc_dropout_enabled = False

    def forward(self, input: Tensor) -> Tensor:
        # NOTE: we could use the following line in case a different rate
        # is used for inference:
        # return F.dropout(input, self.applied_rate, True, self.inplace)
        return F.dropout(input, self.p, self.mc_dropout_enabled, self.inplace)

    @property
    def mc_dropout_enabled(self) -> bool:
        # mc dropout is only activated on `PLForecastingModule.on_predict_start()`
        # otherwise, it is activated based on the `model.training` flag.
        return self._mc_dropout_enabled or self.training


def random_method(decorated: Callable[..., T]) -> Callable[..., T]:
    """Decorator usable on any method within a class that will provide an isolated torch random context.

    The decorator will store a `_random_instance` property on the object in order to persist successive calls to the RNG

    Parameters
    ----------
    decorated
        A method to be run in an isolated torch random context.
    """
    # check that @random_method has been applied to a method.
    if not _is_method(decorated):
        raise_log(ValueError("@random_method can only be used on methods."), logger)

    @wraps(decorated)
    def decorator(self, *args, **kwargs) -> T:
        store_instance = False
        random_instance = None
        if "random_state" in kwargs.keys() and kwargs["random_state"] is not None:
            # get random state from model constructor or `predict()`
            random_instance = check_random_state(kwargs["random_state"])
            if not hasattr(self, "_random_instance"):
                # store random instance when called from model constructor
                store_instance = True
        elif not hasattr(self, "_random_instance"):
            # get random state for first time from other method
            store_instance = True
            random_instance = check_random_state(
                np.random.randint(0, high=MAX_NUMPY_SEED_VALUE)
            )

        # if no random instance is provided, use the one stored in the class
        if random_instance is None:
            random_instance = self._random_instance

        if store_instance:
            self._random_instance = random_instance

        # handle the randomness
        with fork_rng():
            manual_seed(random_instance.randint(0, high=MAX_TORCH_SEED_VALUE))
            return decorated(self, *args, **kwargs)

    return decorator
