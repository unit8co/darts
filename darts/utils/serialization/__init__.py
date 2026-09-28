"""
Model Serialization Utilities
-----------------------------

Persistence helpers for Darts models.

:mod:`darts.utils.serialization.base` is torch-free. It holds shared reference helpers
and the state codec (primitives, NumPy, pandas, TimeSeries, and importable classes)
for torch and future non-torch model storage.
:mod:`darts.utils.serialization.torch` loads Lightning ``.ckpt`` checkpoints.
:mod:`darts.utils.serialization.wrapper` is torch-only. It writes and reads the
``TorchForecastingModel`` ``.pt`` wrapper with ``torch.save`` / ``torch.load``.
"""
