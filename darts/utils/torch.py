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
from numpy.random import SeedSequence
from sklearn.utils import check_random_state
from torch import Tensor
from torch.random import fork_rng, manual_seed

from darts.logging import raise_log
from darts.utils.utils import MAX_NUMPY_SEED_VALUE, MAX_TORCH_SEED_VALUE, _is_method

T = TypeVar("T")

_DATALOADER_SHUFFLE_ENTROPY = sum(map(ord, "dataloader_shuffle"))


def _derive_dataloader_shuffle_seed(seed_key: int | np.random.RandomState) -> int:
    """Derive a stable torch DataLoader shuffle seed without advancing ``_random_instance``."""
    if isinstance(seed_key, np.random.RandomState):
        seed_key = seed_key.get_state()[1][0]
    entropy = [int(seed_key), _DATALOADER_SHUFFLE_ENTROPY]
    return int(SeedSequence(entropy).generate_state(1)[0]) % MAX_TORCH_SEED_VALUE


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
        raise_log(ValueError("@random_method can only be used on methods."))

    @wraps(decorated)
    def decorator(self, *args, **kwargs) -> T:
        store_instance = False
        random_instance = None
        dataloader_seed_key = kwargs.get("random_state")
        if "random_state" in kwargs.keys() and kwargs["random_state"] is not None:
            # get random state from model constructor or `predict()`
            random_instance = check_random_state(kwargs["random_state"])
            if not hasattr(self, "_random_instance"):
                # store random instance when called from model constructor
                store_instance = True
        elif not hasattr(self, "_random_instance"):
            # get random state for first time from other method
            store_instance = True
            dataloader_seed_key = np.random.randint(0, high=MAX_NUMPY_SEED_VALUE)
            random_instance = check_random_state(dataloader_seed_key)

        # if no random instance is provided, use the one stored in the class
        if random_instance is None:
            random_instance = self._random_instance

        if store_instance:
            self._random_instance = random_instance
            if (
                not hasattr(self, "_dataloader_shuffle_seed")
                and dataloader_seed_key is not None
            ):
                self._dataloader_shuffle_seed = _derive_dataloader_shuffle_seed(
                    dataloader_seed_key
                )

        # When resuming from a checkpoint, PyTorch Lightning restores the training
        # state (including loop progress) from the `.ckpt`. Reseeding torch here would
        # break continuity with an uninterrupted `fit()` run.
        if getattr(
            self, "load_ckpt_path", None
        ) is not None and decorated.__name__.startswith("fit"):
            return decorated(self, *args, **kwargs)

        # handle the randomness
        with fork_rng():
            manual_seed(random_instance.randint(0, high=MAX_TORCH_SEED_VALUE))
            return decorated(self, *args, **kwargs)

    return decorator
