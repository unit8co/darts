"""
Model Serialization Utilities
-----------------------------

Persistence helpers for Darts models.  Torch-specific safe loading for ``.ckpt`` and
``.pt`` files lives in :mod:`darts.utils.serialization.torch`; shared torch-free helpers
(allowlist logic, restricted unpickler) in :mod:`darts.utils.serialization.base`.
"""
