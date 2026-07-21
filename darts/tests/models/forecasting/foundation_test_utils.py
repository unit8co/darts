"""Shared test configuration and mocks for foundation models.

Every model-specific test file (``test_chronos2.py``, ``test_tirex.py``, …)
as well as the cross-model ``test_foundation.py`` should import definitions
from here instead of duplicating them.
"""

import contextlib
import functools
import shutil
from pathlib import Path
from unittest.mock import patch

import torch

# ── Artefact paths ──────────────────────────────────────────────────────────
ARTEFACTS_DIR = Path(__file__).parent / "artefacts"

CHRONOS2_TINY_DIR = (ARTEFACTS_DIR / "chronos2" / "tiny_chronos2").absolute()
CHRONOS2_TINY_MAX_CONTEXT_LENGTH = 21
CHRONOS2_TINY_MAX_PREDICTION_LENGTH = 77

PATCHTST_FM_TINY_DIR = (ARTEFACTS_DIR / "patchtstfm" / "tiny_patchtst_fm").absolute()
PATCHTST_FM_TINY_CONTEXT_LENGTH = 128

TIMESFM2P5_TINY_DIR = (ARTEFACTS_DIR / "timesfm2p5" / "tiny_timesfm2p5").absolute()
TIMESFM2P5_TINY_MAX_CONTEXT_LENGTH = 64
TIMESFM2P5_TINY_MAX_PREDICTION_LENGTH = 8

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


# ── TimesFM 2.5 tiny model ─────────────────────────────────────────────────
#
# The production ``_TimesFM2p5Module`` uses a hardcoded
# ``_TimesFM2p5_200M_Definition`` class attribute for its architecture.
# To avoid modifying production code, we construct a tiny definition at
# test time and patch it in via ``timesfm2p5_tiny_context()``.

from darts.models.components.timesfm2p5_submodels import (
    _ResidualBlockConfig,
    _StackedTransformersConfig,
    _TransformerConfig,
)
from darts.models.forecasting.timesfm2p5_model import (
    _TimesFM2p5_200M_Definition,
    _TimesFM2p5Module,
)

_TIMESFM2P5_PATCH_DEF_TARGET = (
    "darts.models.forecasting.timesfm2p5_model._TimesFM2p5_200M_Definition"
)


class _TinyTimesFM2p5Definition(_TimesFM2p5_200M_Definition):
    """Subclass that overrides ``context_limit`` (a class variable)."""

    context_limit = TIMESFM2P5_TINY_MAX_CONTEXT_LENGTH


_D = 8
_NUM_Q = 10  # 9 quantiles + 1 mean (same as production)

TIMESFM2P5_TINY_CONFIG = _TinyTimesFM2p5Definition(
    input_patch_len=4,
    output_patch_len=TIMESFM2P5_TINY_MAX_PREDICTION_LENGTH,
    output_quantile_len=16,
    tokenizer=_ResidualBlockConfig(
        input_dims=4 * 2,
        hidden_dims=_D,
        output_dims=_D,
        use_bias=True,
        activation="swish",
    ),
    stacked_transformers=_StackedTransformersConfig(
        num_layers=1,
        transformer=_TransformerConfig(
            model_dims=_D,
            hidden_dims=_D,
            num_heads=2,
            attention_norm="rms",
            feedforward_norm="rms",
            qk_norm="rms",
            use_bias=False,
            use_rotary_position_embeddings=True,
            ff_activation="swish",
            fuse_qkv=True,
        ),
    ),
    output_projection_point=_ResidualBlockConfig(
        input_dims=_D,
        hidden_dims=_D,
        output_dims=TIMESFM2P5_TINY_MAX_PREDICTION_LENGTH * _NUM_Q,
        use_bias=False,
        activation="swish",
    ),
    output_projection_quantiles=_ResidualBlockConfig(
        input_dims=_D,
        hidden_dims=_D,
        output_dims=16 * _NUM_Q,
        use_bias=False,
        activation="swish",
    ),
)


def _mock_timesfm2p5_hf_hub_download(
    repo_id: str,
    filename: str,
    revision: str | None,
    local_dir: str | Path | None,
    **kwargs,
):
    """Serves files from the tiny TimesFM 2.5 artefact directory."""
    path = TIMESFM2P5_TINY_DIR / filename
    if local_dir is None:
        return str(path)
    dest_path = Path(local_dir) / filename
    shutil.copy(path, dest_path)
    return str(dest_path)


class timesfm2p5_tiny_context:
    """Patches TimesFM 2.5 to use a tiny architecture.

    Can be used as a **decorator** or a **context manager**::

        @timesfm2p5_tiny_context()
        def test_something(self):
            model = TimesFM2p5Model(...)

        # -- or --

        with timesfm2p5_tiny_context():
            model = TimesFM2p5Model(...)

    Patches three things:

    1. ``_TimesFM2p5Module.config`` class attribute -> tiny definition
    2. ``_TimesFM2p5_200M_Definition()`` constructor calls -> tiny definition
    3. ``hf_hub_download`` -> serves tiny artefact files
    """

    def __enter__(self):
        self._stack = contextlib.ExitStack()
        self._stack.enter_context(
            patch.object(_TimesFM2p5Module, "config", TIMESFM2P5_TINY_CONFIG)
        )
        self._stack.enter_context(
            patch(_TIMESFM2P5_PATCH_DEF_TARGET, return_value=TIMESFM2P5_TINY_CONFIG)
        )
        self._stack.enter_context(
            patch(
                HF_HUB_DOWNLOAD_PATCH_TARGET,
                side_effect=_mock_timesfm2p5_hf_hub_download,
            )
        )
        return self

    def __exit__(self, *exc_info):
        return self._stack.__exit__(*exc_info)

    def __call__(self, fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with type(self)():
                return fn(*args, **kwargs)

        return wrapper
