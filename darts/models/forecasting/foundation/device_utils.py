"""
Device Utilities for Foundation Models

Common utilities for device detection, checkpoint loading, and memory management
shared across foundation model implementations.
"""

import logging
from typing import Optional

import torch

from darts.logging import get_logger

logger = get_logger(__name__)


def auto_detect_device() -> str:
    """
    Automatically detect the best available device for model inference.

    Detection priority: CUDA (NVIDIA GPUs) > MPS (Apple Silicon) > CPU

    Returns
    -------
    device : str
        Device string suitable for PyTorch ("cuda", "mps", or "cpu").

    Examples
    --------
    >>> device = auto_detect_device()
    >>> print(f"Using device: {device}")
    Using device: mps

    Notes
    -----
    MPS (Metal Performance Shaders) provides GPU acceleration on Apple Silicon
    (M1, M2, M3 chips). It's significantly faster than CPU for transformer models.
    """
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"CUDA available: Using {gpu_name}")
    elif torch.backends.mps.is_available():
        device = "mps"
        logger.info("MPS available: Using Apple Silicon GPU")
    else:
        device = "cpu"
        logger.info("No GPU available: Using CPU (this will be slower)")

    return device


def get_device_memory_info(device: str) -> dict:
    """
    Get memory information for the specified device.

    Parameters
    ----------
    device : str
        Device identifier ("cuda", "cuda:0", "mps", "cpu").

    Returns
    -------
    memory_info : dict
        Dictionary with keys:
            - total : Total memory in GB
            - allocated : Currently allocated memory in GB
            - free : Free memory in GB (total - allocated)

    Notes
    -----
    Memory reporting is only available for CUDA devices. MPS and CPU return
    estimated values.
    """
    memory_info = {"total": 0.0, "allocated": 0.0, "free": 0.0}

    if device.startswith("cuda"):
        # CUDA devices provide accurate memory reporting
        total = torch.cuda.get_device_properties(device).total_memory
        allocated = torch.cuda.memory_allocated(device)
        memory_info["total"] = total / (1024**3)  # Convert to GB
        memory_info["allocated"] = allocated / (1024**3)
        memory_info["free"] = memory_info["total"] - memory_info["allocated"]
    elif device == "mps":
        # MPS doesn't expose memory APIs, estimate based on system
        import psutil
        mem = psutil.virtual_memory()
        memory_info["total"] = mem.total / (1024**3)
        memory_info["free"] = mem.available / (1024**3)
        memory_info["allocated"] = 0.0  # Cannot track MPS allocation
    else:
        # CPU memory
        import psutil
        mem = psutil.virtual_memory()
        memory_info["total"] = mem.total / (1024**3)
        memory_info["free"] = mem.available / (1024**3)
        memory_info["allocated"] = (mem.total - mem.available) / (1024**3)

    return memory_info


def recommend_batch_size(model_size_mb: float, device: str) -> int:
    """
    Recommend batch size based on model size and available device memory.

    Parameters
    ----------
    model_size_mb : float
        Model size in megabytes.
    device : str
        Target device ("cuda", "mps", "cpu").

    Returns
    -------
    batch_size : int
        Recommended batch size for inference.

    Notes
    -----
    Conservative recommendations to avoid OOM errors. Actual optimal batch size
    may be higher depending on sequence length and other factors.
    """
    memory_info = get_device_memory_info(device)
    free_gb = memory_info["free"]

    # Rule of thumb: model + activations should use <80% of free memory
    usable_gb = free_gb * 0.8
    model_gb = model_size_mb / 1024

    # Estimate activation memory (conservative: 4x model size per batch item)
    activation_per_item_gb = model_gb * 4

    if usable_gb < model_gb:
        logger.warning(
            f"Limited memory: {free_gb:.1f}GB free, model needs {model_gb:.1f}GB. "
            "Consider using a smaller model or CPU offloading."
        )
        return 1

    # Calculate batch size
    batch_size = max(1, int((usable_gb - model_gb) / activation_per_item_gb))

    # Cap at reasonable limits
    if device == "cpu":
        batch_size = min(batch_size, 4)   # CPU is slower, smaller batches
    elif device == "mps":
        batch_size = min(batch_size, 16)  # MPS has memory limits
    else:
        batch_size = min(batch_size, 32)  # CUDA can handle larger batches

    return batch_size


def validate_device_compatibility(device: str, model_name: str) -> bool:
    """
    Validate that the requested device is compatible with the model.

    Some models have known issues on certain devices (e.g., MPS compatibility).

    Parameters
    ----------
    device : str
        Requested device.
    model_name : str
        Model identifier or name.

    Returns
    -------
    is_compatible : bool
        True if device is compatible, False otherwise.

    Raises
    ------
    RuntimeError
        If device is incompatible with known workarounds.
    """
    # Known compatibility issues
    mps_incompatible_ops = [
        "fused_qkv",  # Some models use fused QKV that MPS doesn't support well
    ]

    if device == "mps":
        # Check PyTorch version for MPS support
        import torch
        pytorch_version = tuple(int(x) for x in torch.__version__.split(".")[:2])
        if pytorch_version < (2, 0):
            raise RuntimeError(
                f"MPS requires PyTorch 2.0+, but found {torch.__version__}. "
                "Upgrade with: pip install --upgrade torch"
            )

        # Model-specific MPS warnings
        if "timesfm" in model_name.lower() and pytorch_version < (2, 1):
            logger.warning(
                "TimesFM on MPS works best with PyTorch 2.1+. "
                "You may experience slower performance."
            )

    return True


def clear_device_cache(device: str) -> None:
    """
    Clear device memory cache to free up space.

    Useful between training runs or when switching models.

    Parameters
    ----------
    device : str
        Device to clear cache for.
    """
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
        logger.info("CUDA cache cleared")
    elif device == "mps":
        torch.mps.empty_cache()
        logger.info("MPS cache cleared")
    else:
        logger.debug("No cache to clear for CPU")


def load_checkpoint_lazy(
    model_name: str,
    cache_dir: Optional[str] = None,
    device: Optional[str] = None
) -> dict:
    """
    Lazy-load model checkpoint from Hugging Face Hub.

    Downloads only if not already cached. Provides progress updates.

    Parameters
    ----------
    model_name : str
        Hugging Face model identifier (e.g., "google/timesfm-2.5-200m-pytorch").
    cache_dir : str, optional
        Directory to cache downloads. Defaults to Hugging Face cache.
    device : str, optional
        Device to load checkpoint onto. If None, leaves on CPU for flexibility.

    Returns
    -------
    checkpoint : dict
        Loaded model checkpoint (state_dict, config, etc.).

    Examples
    --------
    >>> checkpoint = load_checkpoint_lazy("google/timesfm-2.5-200m-pytorch")
    >>> model.load_state_dict(checkpoint['state_dict'])

    Notes
    -----
    Uses Hugging Face's automatic caching. Subsequent calls are near-instant.
    """
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is required for checkpoint loading. "
            "Install with: pip install huggingface-hub"
        )

    logger.info(f"Loading checkpoint for {model_name}")

    # Download entire model repository (checkpoint + config)
    local_path = snapshot_download(
        repo_id=model_name,
        cache_dir=cache_dir,
        resume_download=True,
    )

    logger.info(f"Checkpoint loaded from {local_path}")

    # Load checkpoint to specified device
    if device is None:
        device = "cpu"  # Safe default for lazy loading

    checkpoint_path = f"{local_path}/pytorch_model.bin"

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True  # Security: only load tensors, not arbitrary code
    )

    return checkpoint
