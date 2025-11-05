"""
Foundation Forecasting Model Base Class

Base class for pre-trained time series foundation models.
"""

from typing import Dict, Optional

import torch

from darts.logging import get_logger
from darts.models.forecasting.forecasting_model import GlobalForecastingModel

logger = get_logger(__name__)


class FoundationForecastingModel(GlobalForecastingModel):
    """Base class for pre-trained time series foundation models.

    Foundation models are pre-trained on large datasets and support
    zero-shot forecasting without training.

    Parameters
    ----------
    device : str, optional
        Device to use ("cuda", "mps", "cpu"). Auto-detected if None.
    lora_config : dict, optional
        LoRA configuration for fine-tuning. When provided, enables
        parameter-efficient fine-tuning via PEFT.

    Examples
    --------
    Zero-shot forecasting:

    >>> from darts.models import ChronosModel
    >>> model = ChronosModel()
    >>> forecast = model.predict(n=24, series=train)

    With fine-tuning:

    >>> model = ChronosModel(lora_config={"r": 8, "lora_alpha": 16})
    >>> model.fit(series=train, num_steps=1000)
    >>> forecast = model.predict(n=24)

    Notes
    -----
    - Models lazy-load on first use (no download at import)
    - fit() is optional for zero-shot usage
    - Subclasses implement load_model() to load pretrained weights
    """

    def __init__(
        self,
        device: Optional[str] = None,
        lora_config: Optional[Dict] = None,
        **kwargs
    ):
        """Initialize foundation model.

        Parameters
        ----------
        device : str, optional
            Device to use. Auto-detected if None.
        lora_config : dict, optional
            LoRA configuration for fine-tuning.
        **kwargs
            Passed to GlobalForecastingModel.
        """
        super().__init__(**kwargs)
        self._fit_called = True  # Pre-trained models ready immediately

        # Auto-detect device if not specified
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = device
        self.lora_config = lora_config

        # Lazy loading state
        self._model = None
        self._is_loaded = False

    @property
    def model(self):
        """Lazy-load model on first access."""
        if not self._is_loaded:
            logger.info(f"Loading {self.__class__.__name__}...")
            self._model = self.load_model()
            self._is_loaded = True
            logger.info("Model loaded")
        return self._model

    def load_model(self):
        """Load pretrained model.

        Subclasses override to load model from HuggingFace, S3, etc.

        Returns
        -------
        model
            Loaded pretrained model.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement load_model()"
        )

    def fit(self, series, past_covariates=None, future_covariates=None, **kwargs):
        """Fit the model.

        For zero-shot usage (lora_config=None), validates inputs without training.
        For fine-tuning (lora_config provided), trains PEFT adapters.

        Parameters
        ----------
        series : TimeSeries or list
            Training series.
        past_covariates : TimeSeries or list, optional
            Past covariates.
        future_covariates : TimeSeries or list, optional
            Future covariates.
        **kwargs
            Training parameters (num_steps, learning_rate, etc.)

        Returns
        -------
        self
            Fitted model.
        """
        super().fit(series)

        if self.lora_config is not None:
            # Fine-tuning path - subclasses implement _apply_peft() and _train_with_peft()
            logger.info("Applying PEFT and fine-tuning")
            if hasattr(self, '_apply_peft'):
                self._apply_peft()
            if hasattr(self, '_train_with_peft'):
                self._train_with_peft(series, past_covariates, future_covariates, **kwargs)

        return self

    @property
    def supports_probabilistic_prediction(self) -> bool:
        """Foundation models support probabilistic forecasting."""
        return True
