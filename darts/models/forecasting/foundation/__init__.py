"""
Foundation Models Package

This package provides infrastructure for time series foundation models in Darts.

Foundation models are pre-trained on massive datasets (100B+ time points) enabling:
- Zero-shot forecasting without training
- Few-shot learning via in-context examples
- Parameter-efficient fine-tuning (PEFT) with LoRA

Available Models
----------------
- TimesFMModel : Google's decoder-only transformer (200M parameters)

Planned Models
--------------
- ChronosModel : Amazon's T5-based probabilistic forecaster (120M parameters)

Base Classes
------------
- FoundationForecastingModel : Base class for foundation models with PEFT support

Utilities
---------
- peft_utils : LoRA and PEFT helper functions
- device_utils : Device detection and memory management

Examples
--------
Zero-shot forecasting:

>>> from darts.models.foundation import TimesFMModel
>>> model = TimesFMModel()
>>> forecast = model.predict(n=12, series=my_series)

Fine-tuning with LoRA:

>>> model = TimesFMModel(lora_config={"r": 8, "lora_alpha": 16})
>>> model.fit(series=training_data, epochs=10)

References
----------
.. [1] Das et al., "A decoder-only foundation model for time-series forecasting",
       ICML 2024. https://arxiv.org/abs/2310.10688
.. [2] Hugging Face PEFT: https://huggingface.co/docs/peft
"""

from darts.models.forecasting.foundation.base import FoundationForecastingModel
from darts.models.forecasting.foundation.timesfm import TimesFMModel

__all__ = [
    "FoundationForecastingModel",
    "TimesFMModel",
]
