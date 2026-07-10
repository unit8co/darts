"""
Utils for Pytorch and its usage
-------------------------------
"""

from __future__ import annotations

import gc
import re
from collections.abc import Callable
from functools import wraps
from typing import TypeVar

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.utils import check_random_state
from torch import Tensor
from torch.random import fork_rng, manual_seed

from darts.logging import get_logger, raise_log
from darts.utils.utils import MAX_NUMPY_SEED_VALUE, MAX_TORCH_SEED_VALUE, _is_method

logger = get_logger(__name__)

T = TypeVar("T")


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


# ---------------------------------------------------------------------------
# Memory error detection
# ---------------------------------------------------------------------------

_MEMORY_ERROR_PATTERNS = [
    re.compile(r"out of memory", re.IGNORECASE),
    re.compile(r"CUDNN_STATUS_NOT_SUPPORTED"),
    re.compile(r"DefaultCPUAllocator: can't allocate memory"),
    re.compile(r"Invalid buffer size", re.IGNORECASE),
    re.compile(r"not enough memory", re.IGNORECASE),
    re.compile(r"Allocation on device", re.IGNORECASE),
    re.compile(r"allocat.*failed", re.IGNORECASE),
]


def is_memory_error(exception: BaseException) -> bool:
    """Check whether an exception is related to device/CPU memory exhaustion.

    Broader than PyTorch Lightning's ``is_oom_error`` which only matches a
    narrow set of CUDA and CPU OOM messages.  This additionally catches
    buffer-size errors raised by kernels such as
    ``nn.functional.scaled_dot_product_attention`` and generic allocation
    failures across all backends (CUDA, MPS, XPU, CPU).
    """
    if hasattr(torch.cuda, "OutOfMemoryError") and isinstance(
        exception, torch.cuda.OutOfMemoryError
    ):
        return True

    if not isinstance(exception, RuntimeError):
        return False
    msg = str(exception)
    return any(p.search(msg) for p in _MEMORY_ERROR_PATTERNS)


# ---------------------------------------------------------------------------
# Garbage collection / cache clearing
# ---------------------------------------------------------------------------


def garbage_collection(device_type: str | None = None) -> None:
    """Run Python GC and free device allocator caches.

    Parameters
    ----------
    device_type
        One of ``"cpu"``, ``"cuda"``, ``"mps"``, etc.  When ``None`` the
        function only runs ``gc.collect()``.
    """
    gc.collect()
    if device_type is None or device_type == "cpu":
        return

    try:
        if hasattr(torch, "accelerator") and hasattr(torch.accelerator, "memory"):
            torch.accelerator.memory.empty_cache()
        elif device_type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif device_type == "mps" and hasattr(torch, "mps"):
            torch.mps.empty_cache()
    except RuntimeError:
        pass
