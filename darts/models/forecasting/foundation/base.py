"""
Foundation Forecasting Model Base Class

This module provides the base class for time series foundation models in Darts.
Foundation models are pre-trained on massive datasets and support zero-shot forecasting,
few-shot learning, and parameter-efficient fine-tuning (PEFT).
"""

import logging
from abc import abstractmethod
from typing import Dict, List, Optional, Union

from darts import TimeSeries
from darts.logging import get_logger
from darts.models.forecasting.forecasting_model import GlobalForecastingModel

logger = get_logger(__name__)


class FoundationForecastingModel(GlobalForecastingModel):
    """
    Base class for foundation models with optional PEFT support.

    Foundation models are pre-trained on massive time series datasets, enabling:

    - **Zero-shot forecasting**: Direct prediction without calling fit()
    - **Few-shot learning**: In-context learning from example series
    - **Fine-tuning**: Parameter-efficient adaptation via PEFT (LoRA, Prefix Tuning)

    The fit() method is **optional** for zero-shot usage but required for fine-tuning.
    When `lora_config` is provided, fit() applies PEFT adapters and trains them.

    Parameters
    ----------
    lora_config : dict, optional
        LoRA configuration following Hugging Face PEFT pattern.
        When provided, enables parameter-efficient fine-tuning.

        Example configuration:
            {
                "r": 8,                          # LoRA rank
                "lora_alpha": 16,                # Scaling factor
                "target_modules": ["qkv_proj"],  # Layers to adapt
                "lora_dropout": 0.05,            # Dropout probability
            }

        See https://huggingface.co/docs/peft for details.

    Examples
    --------
    Zero-shot forecasting (no training):

    >>> from darts.models.foundation import TimesFMModel
    >>> model = TimesFMModel()
    >>> forecast = model.predict(n=12, series=my_series)

    Fine-tuning with LoRA:

    >>> model = TimesFMModel(
    ...     lora_config={"r": 8, "lora_alpha": 16}
    ... )
    >>> model.fit(series=training_data, epochs=10)
    >>> forecast = model.predict(n=12)

    Notes
    -----
    Foundation models follow different patterns than traditional Darts models:

    - fit() is optional for zero-shot usage
    - predict() can be called without prior fit() call
    - Fine-tuning uses PEFT to train <1% of parameters efficiently

    References
    ----------
    .. [1] Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models",
           ICLR 2022. https://arxiv.org/abs/2106.09685
    .. [2] Hugging Face PEFT library. https://huggingface.co/docs/peft
    """

    def __init__(self, lora_config: Optional[Dict] = None, **kwargs):
        """
        Initialize foundation forecasting model.

        Parameters
        ----------
        lora_config : dict, optional
            LoRA configuration for parameter-efficient fine-tuning.
        **kwargs
            Additional arguments passed to GlobalForecastingModel.
        """
        super().__init__(**kwargs)
        self.lora_config = lora_config
        self._peft_model = None
        self._is_peft_applied = False

    def fit(
        self,
        series: Union[TimeSeries, List[TimeSeries]],
        past_covariates: Optional[Union[TimeSeries, List[TimeSeries]]] = None,
        future_covariates: Optional[Union[TimeSeries, List[TimeSeries]]] = None,
        **kwargs
    ) -> "FoundationForecastingModel":
        """
        Fit the foundation model.

        For zero-shot models (lora_config=None), this validates inputs and loads
        the pre-trained model without training.

        When lora_config is provided, this applies PEFT adapters and trains them
        on the provided series.

        Parameters
        ----------
        series : TimeSeries or List[TimeSeries]
            Training time series.
        past_covariates : TimeSeries or List[TimeSeries], optional
            Past covariates (if supported by model).
        future_covariates : TimeSeries or List[TimeSeries], optional
            Future covariates (if supported by model).
        **kwargs
            Additional training parameters (epochs, learning_rate, etc.)

        Returns
        -------
        self
            Fitted model instance.
        """
        if self.lora_config is not None:
            logger.info("Applying PEFT configuration and fine-tuning model")
            self._apply_peft()
            return self._train_with_peft(series, past_covariates, future_covariates, **kwargs)
        else:
            logger.info("Zero-shot mode: fit() validates inputs without training")
            return self._zero_shot_fit(series, past_covariates, future_covariates, **kwargs)

    @abstractmethod
    def _apply_peft(self) -> None:
        """
        Apply PEFT configuration to the base model.

        This method should:
        1. Load the pre-trained base model if not already loaded
        2. Apply LoRA adapters using Hugging Face PEFT library
        3. Set self._peft_model to the adapter-enhanced model
        4. Set self._is_peft_applied = True

        Raises
        ------
        NotImplementedError
            If PEFT is not yet implemented for this model.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not yet support PEFT fine-tuning"
        )

    @abstractmethod
    def _train_with_peft(
        self,
        series: Union[TimeSeries, List[TimeSeries]],
        past_covariates: Optional[Union[TimeSeries, List[TimeSeries]]],
        future_covariates: Optional[Union[TimeSeries, List[TimeSeries]]],
        **kwargs
    ) -> "FoundationForecastingModel":
        """
        Train the PEFT adapters on the provided data.

        Parameters
        ----------
        series : TimeSeries or List[TimeSeries]
            Training time series.
        past_covariates : TimeSeries or List[TimeSeries], optional
            Past covariates.
        future_covariates : TimeSeries or List[TimeSeries], optional
            Future covariates.
        **kwargs
            Training parameters (epochs, learning_rate, etc.)

        Returns
        -------
        self
            Model with trained PEFT adapters.

        Raises
        ------
        NotImplementedError
            If PEFT training is not yet implemented for this model.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not yet support PEFT training"
        )

    @abstractmethod
    def _zero_shot_fit(
        self,
        series: Union[TimeSeries, List[TimeSeries]],
        past_covariates: Optional[Union[TimeSeries, List[TimeSeries]]],
        future_covariates: Optional[Union[TimeSeries, List[TimeSeries]]],
        **kwargs
    ) -> "FoundationForecastingModel":
        """
        Validate inputs for zero-shot inference without training.

        This method should:
        1. Validate input series format
        2. Load pre-trained model if not already loaded
        3. Store series for later use in predict()
        4. Return self without modifying model weights

        Parameters
        ----------
        series : TimeSeries or List[TimeSeries]
            Validation/reference time series.
        past_covariates : TimeSeries or List[TimeSeries], optional
            Past covariates.
        future_covariates : TimeSeries or List[TimeSeries], optional
            Future covariates.
        **kwargs
            Additional parameters (ignored in zero-shot mode).

        Returns
        -------
        self
            Validated model ready for prediction.
        """
        pass
