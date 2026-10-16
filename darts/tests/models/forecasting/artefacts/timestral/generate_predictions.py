"""Regenerate Timestral-5M reference predictions using only the upstream package.

Run from the repository root in the development environment:

    uv run python darts/tests/models/forecasting/artefacts/timestral/generate_predictions.py

Install the upstream source pinned in pyproject.toml first. This script uses no
Darts models, datasets or conversion helpers. The archive contains the inputs,
all output quantiles, and provenance so tests do not run a second forecaster.
Only the config and weights are downloaded; neither is committed to Darts.
"""

import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from timestral import TimestralForecaster

MODEL_ID = "mistralai/Timestral-5M"
MODEL_REVISION = "bdd20cfc15a3fea0e4cb5ccef37f8d3ca7aa335f"
SOURCE_REVISION = "a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff"


def main():
    torch.set_num_threads(1)
    forecaster = TimestralForecaster.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, device="cpu"
    )
    rng = np.random.default_rng(123)
    target = rng.normal(size=(96, 2)).cumsum(axis=0).astype(np.float32)
    covariates = rng.normal(size=(96 + 1025, 2)).cumsum(axis=0).astype(np.float32)
    arrays = dict(target=target, covariates=covariates)
    for horizon in (17, 1025):
        forecast = forecaster.forecast(
            context=target[-64:].T[None],
            prediction_length=horizon,
            past_covariates=covariates[32:96].T[None],
            known_covariates=covariates[32 : 96 + horizon].T[None],
        )
        arrays[f"prediction_{horizon}"] = forecast.quantiles[0].permute(1, 0, 2).numpy()

    # Univariate categorical encoding, missing data, an unseen future category,
    # and a shifted forecast. A missing shift must not move future covariates.
    target_cat = target[:, :1].copy()
    target_cat[45:48] = np.nan
    codes = (np.arange(96 + 3 + 17) % 3).astype(np.float32)[:, None]
    codes[41] = np.nan
    codes[101:104] = 9
    known = codes[32:].copy()
    known[64:67] = np.nan
    forecast = forecaster.forecast(
        context=target_cat[-64:].T[None],
        prediction_length=20,
        categorical_past_covariates=codes[32:96].T[None],
        categorical_known_covariates=known.T[None],
    )
    arrays.update(
        categorical_target=target_cat,
        categorical_covariates=codes,
        categorical_prediction=forecast.quantiles[0, :, 3:].permute(1, 0, 2).numpy(),
        quantile_levels=np.asarray(forecaster.quantile_levels),
    )
    provenance = dict(
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        source_revision=SOURCE_REVISION,
        package_versions={
            name: version(name) for name in ("timestral", "torch", "numpy")
        },
        device="cpu",
        dtype="float32",
        context_length=64,
        symmetric_averaging=True,
        clip_nonnegative=True,
    )
    directory = Path(__file__).parent
    np.savez_compressed(directory / "timestral_prediction.npz", **arrays)
    (directory / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
