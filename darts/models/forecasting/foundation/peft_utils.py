"""
PEFT Utilities for Foundation Models

This module provides utilities for Parameter-Efficient Fine-Tuning (PEFT) following
Hugging Face standards. Compatible with: LoRA, Prefix Tuning, Adapter layers, etc.
"""

import logging
from typing import Dict, List, Optional, Union

import torch.nn as nn

from darts.logging import get_logger

logger = get_logger(__name__)


def create_lora_model(
    base_model: nn.Module,
    lora_config: Dict,
    target_modules: Optional[List[str]] = None
):
    """
    Apply LoRA adapters to a base model using Hugging Face PEFT pattern.

    This function wraps the base model with Low-Rank Adaptation (LoRA) layers,
    enabling parameter-efficient fine-tuning. Only the LoRA adapter weights are
    trainable, while the base model weights remain frozen.

    Parameters
    ----------
    base_model : torch.nn.Module
        Pre-trained foundation model to adapt.
    lora_config : dict
        LoRA configuration with keys:
            - r (int): LoRA rank (typical values: 4, 8, 16)
            - lora_alpha (int): Scaling factor (typical: 2*r)
            - lora_dropout (float): Dropout probability (typical: 0.05)
            - bias (str): Bias handling - "none", "all", or "lora_only"
    target_modules : list of str, optional
        Module names to apply LoRA adapters to.
        If None, automatically detects appropriate modules.

    Returns
    -------
    peft_model : PeftModel
        Model with LoRA adapters applied. Only adapter weights are trainable.

    Examples
    --------
    >>> from darts.models.foundation.peft_utils import create_lora_model
    >>> base_model = load_pretrained_model()
    >>> lora_config = {"r": 8, "lora_alpha": 16, "lora_dropout": 0.05}
    >>> peft_model = create_lora_model(base_model, lora_config)
    >>> print(f"Trainable params: {count_trainable_params(peft_model)}")

    References
    ----------
    .. [1] Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models",
           ICLR 2022. https://arxiv.org/abs/2106.09685
    .. [2] Hugging Face PEFT: https://huggingface.co/docs/peft
    """
    try:
        from peft import get_peft_model, LoraConfig, TaskType
    except ImportError:
        raise ImportError(
            "PEFT library is required for fine-tuning. "
            "Install with: pip install peft"
        )

    # Auto-detect target modules if not provided
    if target_modules is None:
        target_modules = auto_detect_lora_targets(base_model)
        logger.info(f"Auto-detected LoRA target modules: {target_modules}")

    # Create LoRA configuration
    config = LoraConfig(
        r=lora_config.get("r", 8),
        lora_alpha=lora_config.get("lora_alpha", 16),
        target_modules=target_modules,
        lora_dropout=lora_config.get("lora_dropout", 0.05),
        bias=lora_config.get("bias", "none"),
        task_type=TaskType.CAUSAL_LM,
    )

    # Apply PEFT to create adapter-enhanced model
    peft_model = get_peft_model(base_model, config)

    # Log parameter efficiency
    trainable, total = count_trainable_params(peft_model)
    logger.info(
        f"PEFT applied: {trainable:,} trainable parameters "
        f"({100.0 * trainable / total:.2f}% of {total:,} total)"
    )

    return peft_model


def auto_detect_lora_targets(model: nn.Module) -> List[str]:
    """
    Auto-detect appropriate modules for LoRA adaptation.

    Searches for attention projection layers (query, key, value, output)
    and fully-connected layers commonly used in transformers.

    Parameters
    ----------
    model : torch.nn.Module
        Model to analyze for LoRA targets.

    Returns
    -------
    target_modules : list of str
        Module names suitable for LoRA adaptation.

    Notes
    -----
    Common patterns detected:
    - Attention: q_proj, k_proj, v_proj, qkv_proj, o_proj
    - Feed-forward: fc1, fc2, dense
    - Model-specific: up_proj, down_proj, gate_proj
    """
    target_modules = []

    # Common attention module patterns
    attention_patterns = [
        "q_proj", "k_proj", "v_proj",  # Separate Q, K, V
        "qkv_proj",                     # Fused QKV (TimesFM 2.5+)
        "o_proj", "out_proj",           # Output projection
        "c_attn", "c_proj",             # GPT-style
    ]

    # Common feed-forward patterns
    ff_patterns = [
        "fc1", "fc2",                   # Standard MLPs
        "dense", "intermediate",        # BERT-style
        "up_proj", "down_proj", "gate_proj"  # LLaMA-style
    ]

    all_patterns = attention_patterns + ff_patterns

    # Search for matching module names
    for name, module in model.named_modules():
        # Check if module name contains any pattern
        module_basename = name.split(".")[-1]  # Get last component
        if module_basename in all_patterns:
            if module_basename not in target_modules:
                target_modules.append(module_basename)

    if not target_modules:
        logger.warning(
            "Could not auto-detect LoRA target modules. "
            "Please specify target_modules manually."
        )
        # Fallback: target common linear layers
        target_modules = ["q_proj", "v_proj"]

    return target_modules


def count_trainable_params(model: nn.Module) -> tuple:
    """
    Count trainable and total parameters in a model.

    Parameters
    ----------
    model : torch.nn.Module
        Model to analyze.

    Returns
    -------
    trainable : int
        Number of trainable parameters.
    total : int
        Total number of parameters.
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def validate_peft_applied(model: nn.Module) -> bool:
    """
    Verify that PEFT has been correctly applied to a model.

    Checks that:
    1. Model has trainable parameters
    2. Some parameters are frozen (base model)
    3. Parameter count is reasonable for PEFT (<10% trainable)

    Parameters
    ----------
    model : torch.nn.Module
        Model to validate.

    Returns
    -------
    is_valid : bool
        True if PEFT appears correctly applied.

    Raises
    ------
    ValueError
        If validation fails with details about the issue.
    """
    trainable, total = count_trainable_params(model)

    if trainable == 0:
        raise ValueError("No trainable parameters found. PEFT may not be applied.")

    if trainable == total:
        raise ValueError(
            "All parameters are trainable. "
            "Expected frozen base model with trainable adapters."
        )

    trainable_percent = 100.0 * trainable / total
    if trainable_percent > 10.0:
        logger.warning(
            f"PEFT typically trains <1% of parameters, but found {trainable_percent:.1f}% trainable. "
            "This may indicate incorrect PEFT configuration."
        )

    logger.info(f"PEFT validation passed: {trainable:,}/{total:,} parameters trainable")
    return True


def save_lora_adapters(model, save_path: str) -> None:
    """
    Save only the LoRA adapter weights (not the base model).

    This creates a small checkpoint file containing only the trained adapter
    weights, typically a few megabytes vs gigabytes for the full model.

    Parameters
    ----------
    model : PeftModel
        Model with trained LoRA adapters.
    save_path : str
        Path to save adapter weights.

    Notes
    -----
    To use the saved adapters later:
    >>> base_model = load_pretrained_model()
    >>> peft_model = PeftModel.from_pretrained(base_model, save_path)
    """
    try:
        from peft import PeftModel
    except ImportError:
        raise ImportError("PEFT library required. Install with: pip install peft")

    if not isinstance(model, PeftModel):
        raise TypeError("Model must be a PeftModel to save adapters")

    model.save_pretrained(save_path)
    logger.info(f"LoRA adapters saved to {save_path}")


def load_lora_adapters(base_model, adapter_path: str):
    """
    Load previously trained LoRA adapters onto a base model.

    Parameters
    ----------
    base_model : torch.nn.Module
        Pre-trained base model.
    adapter_path : str
        Path to saved adapter weights.

    Returns
    -------
    peft_model : PeftModel
        Model with adapters loaded.
    """
    try:
        from peft import PeftModel
    except ImportError:
        raise ImportError("PEFT library required. Install with: pip install peft")

    peft_model = PeftModel.from_pretrained(base_model, adapter_path)
    logger.info(f"LoRA adapters loaded from {adapter_path}")
    return peft_model
