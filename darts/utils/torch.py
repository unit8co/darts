"""
Utils for Pytorch and its usage
-------------------------------
"""

from __future__ import annotations

import ctypes
import gc
import os
import platform
import re
import resource
import sys
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


# ---------------------------------------------------------------------------
# Memory budget helpers
# ---------------------------------------------------------------------------

_SIZE_UNITS = {
    "B": 1,
    "KB": 1024,
    "MB": 1024**2,
    "GB": 1024**3,
    "TB": 1024**4,
}


def parse_memory_budget(budget: int | float | str) -> int:
    """Parse a human-readable memory budget into bytes.

    Accepts:
    - ``int`` / ``float``: interpreted as bytes.
    - ``str``: e.g. ``"8GB"``, ``"4096MB"``, ``"0.5TB"``, ``"1024"`` (bytes).

    Returns
    -------
    int
        The budget in bytes.
    """
    if isinstance(budget, (int | float)):
        return int(budget)
    budget = budget.strip().upper()
    for suffix, multiplier in sorted(_SIZE_UNITS.items(), key=lambda kv: -len(kv[0])):
        if budget.endswith(suffix):
            return int(float(budget[: -len(suffix)].strip()) * multiplier)
    return int(float(budget))


# ---------------------------------------------------------------------------
# MemoryMonitor – backend-agnostic device memory profiling
# ---------------------------------------------------------------------------


def _get_total_system_memory() -> int:
    """Return total physical RAM in bytes (best-effort, cross-platform)."""
    if sys.platform == "win32":
        try:

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(stat)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return int(stat.ullTotalPhys)
        except Exception:
            return 0
    else:
        try:
            return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        except (ValueError, OSError):
            return 0


def _get_process_rss() -> int:
    """Return the current process RSS in bytes."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    rss = usage.ru_maxrss
    if platform.system() == "Linux":
        rss *= 1024  # Linux reports KB, macOS/BSD report bytes
    return rss


class MemoryMonitor:
    """Backend-agnostic device memory monitor.

    Provides the four operations required for memory-budget-based batch size
    scaling:

    * ``reset_peak_memory()`` – reset the peak-memory watermark.
    * ``get_peak_memory()``  – read peak memory since last reset (bytes).
    * ``get_total_memory()`` – total device memory capacity (bytes).
    * ``empty_cache()``      – release allocator-cached memory.

    The class automatically dispatches to the best available API:

    1. ``torch.accelerator.memory`` (PyTorch >= 2.10, all accelerators).
    2. Device-specific APIs (``torch.cuda``, ``torch.mps``) for older PyTorch.
    3. Process-level RSS monitoring for CPU.
    """

    def __init__(self, device: torch.device) -> None:
        self._device = device
        dtype = device.type

        if dtype == "cpu":
            self._backend = "cpu"
        elif (
            hasattr(torch, "accelerator")
            and hasattr(torch.accelerator, "memory")
            and self._accelerator_has_stats()
        ):
            self._backend = "accelerator"
        elif dtype == "cuda" and torch.cuda.is_available():
            self._backend = "cuda"
        elif dtype == "mps" and hasattr(torch, "mps"):
            self._backend = "mps"
        else:
            raise_log(
                ValueError(
                    f"Memory-budget scaling is not supported for device type "
                    f"'{dtype}'. Use mode='power' or mode='binsearch' instead."
                ),
            )

        # For delta-based backends (MPS fallback, CPU) we track memory
        # manually between reset / read cycles.
        self._baseline: int = 0
        self._peak: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset_peak_memory(self) -> None:
        """Reset peak memory tracking."""
        if self._backend == "accelerator":
            torch.accelerator.memory.reset_peak_memory_stats()
        elif self._backend == "cuda":
            torch.cuda.reset_peak_memory_stats(self._device)
        elif self._backend == "mps":
            self._baseline = torch.mps.current_allocated_memory()
            self._peak = self._baseline
        elif self._backend == "cpu":
            self._baseline = _get_process_rss()
            self._peak = self._baseline

    def get_peak_memory(self) -> int:
        """Return peak memory allocated (bytes) since last reset."""
        if self._backend == "accelerator":
            return torch.accelerator.memory.max_memory_allocated()
        elif self._backend == "cuda":
            return torch.cuda.max_memory_allocated(self._device)
        elif self._backend == "mps":
            current = torch.mps.current_allocated_memory()
            self._peak = max(self._peak, current)
            return self._peak - self._baseline
        elif self._backend == "cpu":
            current = _get_process_rss()
            self._peak = max(self._peak, current)
            return self._peak - self._baseline
        return 0

    def get_total_memory(self) -> int:
        """Return total device memory capacity (bytes)."""
        if self._backend == "accelerator":
            try:
                _free, total = torch.accelerator.memory.get_memory_info()
                if total > 0:
                    return total
            except Exception:
                pass
        if self._backend == "cuda" or (
            self._backend == "accelerator" and self._device.type == "cuda"
        ):
            try:
                _free, total = torch.cuda.mem_get_info(self._device)
                return total
            except Exception:
                pass
        if self._backend == "mps" or (
            self._backend == "accelerator" and self._device.type == "mps"
        ):
            if hasattr(torch.mps, "recommended_max_memory"):
                try:
                    return torch.mps.recommended_max_memory()
                except Exception:
                    pass
            if hasattr(torch.mps, "driver_allocated_memory"):
                return _get_total_system_memory()
        if self._backend == "cpu":
            return _get_total_system_memory()
        return 0

    def empty_cache(self) -> None:
        """Free cached memory held by the allocator."""
        garbage_collection(self._device.type)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _accelerator_has_stats(self) -> bool:
        """Check whether ``torch.accelerator.memory`` actually returns stats
        for the current device (some backends register the API but don't
        implement the underlying C++ methods)."""
        try:
            stats = torch.accelerator.memory.memory_stats()
            return len(stats) > 0
        except Exception:
            return False
