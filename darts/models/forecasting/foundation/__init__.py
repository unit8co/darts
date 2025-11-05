"""
Foundation Models Package

This package provides infrastructure for time series foundation models in Darts.

Foundation models are pre-trained on massive datasets (100B+ time points) enabling:
- Zero-shot forecasting without training
- Few-shot learning via in-context examples
- Parameter-efficient fine-tuning (PEFT) with LoRA

Architecture
------------
All foundation models inherit from FoundationForecastingModel, which provides:

- **Lazy loading**: Models download on first use, not at import
- **Device management**: Automatic CUDA/MPS/CPU detection
- **Unified pattern**: Consistent .model property and load_model()
- **PEFT support**: LoRA fine-tuning infrastructure

Available Models
----------------
- TimesFMModel : Google's decoder-only transformer (200M parameters)
  - Univariate only, no covariate support
  - Max context: 16384 tokens
  - Source: HuggingFace Hub

- ChronosModel : Amazon's T5-based probabilistic forecaster (120M parameters)
  - Multivariate with past/future covariate support
  - Max context: 8192 tokens
  - Source: S3 bucket

Base Classes
------------
- FoundationForecastingModel : Base class with lazy loading and PEFT support

Examples
--------
Zero-shot forecasting:

>>> from darts.models.foundation import TimesFMModel
>>> model = TimesFMModel()
>>> forecast = model.predict(n=12, series=my_series)

With device selection:

>>> model = ChronosModel(device="cuda")
>>> forecast = model.predict(n=12, series=my_series)

Fine-tuning with LoRA:

>>> model = TimesFMModel(lora_config={"r": 8, "lora_alpha": 16})
>>> model.fit(series=training_data, epochs=10)
>>> forecast = model.predict(n=12)

References
----------
.. [1] Das et al., "A decoder-only foundation model for time-series forecasting",
       ICML 2024. https://arxiv.org/abs/2310.10688
.. [2] Hugging Face PEFT: https://huggingface.co/docs/peft
"""

from darts.models.forecasting.foundation.base import FoundationForecastingModel
from darts.utils.utils import NotImportedModule

__all__ = ["FoundationForecastingModel"]

# Try to import ChronosModel (requires chronos-forecasting at usage time, not import time)
try:
    from darts.models.forecasting.foundation.chronos import ChronosModel

    __all__.append("ChronosModel")
except ImportError:
    ChronosModel = NotImportedModule(module_name="chronos-forecasting", warn=False)

# Try to import TimesFMModel (requires torch at import time)
try:
    from darts.models.forecasting.foundation.timesfm import TimesFMModel

    __all__.append("TimesFMModel")
except ImportError:
    TimesFMModel = NotImportedModule(module_name="torch", warn=False)
