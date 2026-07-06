"""Shared test configuration and mocks for foundation models.

Every model-specific test file (``test_chronos2.py``, ``test_tirex.py``, …)
as well as the cross-model ``test_foundation.py`` should import definitions
from here instead of duplicating them.
"""

import shutil
from pathlib import Path

import torch

# ── Artefact paths ──────────────────────────────────────────────────────────
ARTEFACTS_DIR = Path(__file__).parent / "artefacts"

CHRONOS2_TINY_DIR = (ARTEFACTS_DIR / "chronos2" / "tiny_chronos2").absolute()
CHRONOS2_TINY_MAX_CONTEXT_LENGTH = 21
CHRONOS2_TINY_MAX_PREDICTION_LENGTH = 77

PATCHTST_FM_TINY_DIR = (ARTEFACTS_DIR / "patchtstfm" / "tiny_patchtst_fm").absolute()
PATCHTST_FM_TINY_CONTEXT_LENGTH = 128

# ── HuggingFace mock download (Chronos-2 tiny artefact) ────────────────────
HF_HUB_DOWNLOAD_PATCH_TARGET = (
    "darts.models.components.huggingface_connector.hf_hub_download"
)


def mock_hf_hub_download(
    repo_id: str,
    filename: str,
    revision: str | None,
    local_dir: str | Path | None,
    **kwargs,
):
    """Drop-in replacement for ``hf_hub_download`` that serves files from
    the tiny Chronos-2 artefact directory."""
    path = CHRONOS2_TINY_DIR / filename
    if local_dir is None:
        return str(path)
    dest_path = Path(local_dir) / filename
    shutil.copy(path, dest_path)
    return str(dest_path)


# ── TiRex stub ──────────────────────────────────────────────────────────────
TIREX_QUANTILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
TIREX_LOAD_MODEL_PATCH_TARGET = "darts.models.forecasting.tirex_model.load_model"


class TiRexStub:
    """Lightweight stub emulating the ``tirex`` pipeline API so that
    ``TiRexModel`` can run without downloading the real weights.

    Provides ``_forecast_quantiles(context, prediction_length)`` which
    returns deterministic quantile forecasts based on simple arithmetic.
    """

    def _forecast_quantiles(self, context, prediction_length: int, **_kwargs):
        B, H, Q = int(context.shape[0]), int(prediction_length), len(TIREX_QUANTILES)
        mean = torch.arange(
            1, H + 1, dtype=torch.float32, device=context.device
        ).repeat(B, 1)
        quantiles = torch.zeros((B, H, Q), dtype=torch.float32, device=context.device)
        for qi, q in enumerate(TIREX_QUANTILES):
            quantiles[:, :, qi] = mean + (float(q) - 0.5)
        return quantiles, mean
