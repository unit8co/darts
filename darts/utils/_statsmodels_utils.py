"""Internal utilities for statsmodels compatibility across versions."""

import contextlib
from importlib.metadata import version

_sm_150_or_above = True
with contextlib.suppress(Exception):
    _statsmodel_rng_min_version = (0, 15, 0)
    _sm_version = version("statsmodels")
    _sm_version = tuple(int(el) for el in _sm_version.split(".")[:3])
    _sm_150_or_above = _sm_version >= _statsmodel_rng_min_version

SM_RNG_KWARG = "rng" if _sm_150_or_above else "random_state"
