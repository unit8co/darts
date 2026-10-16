"""Timestral adapter contract and numerical fidelity tests.

Routine tests use a tiny real network and local safetensors. Only the slow
released-checkpoint test accesses the Hub; no dataset downloads are needed.
"""

import copy
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from darts.tests.conftest import TIMESTRAL_AVAILABLE, TORCH_AVAILABLE, tfm_kwargs

if not TORCH_AVAILABLE or not TIMESTRAL_AVAILABLE:
    pytest.skip("Torch and Timestral are required", allow_module_level=True)

import torch
from timestral import TimestralForecaster

from darts import TimeSeries
from darts.models import TimestralModel
from darts.tests.models.forecasting.foundation_test_utils import tiny_timestral_dir
from darts.utils.likelihood_models import GaussianLikelihood, QuantileRegression


def make_series(length=48, components=1, offset=0):
    rng = np.random.default_rng(123)
    values = rng.normal(size=(length, components)).cumsum(axis=0) + offset
    return TimeSeries.from_times_and_values(
        pd.date_range("2024-01-01", periods=length, freq="h"),
        values.astype(np.float32),
        columns=[f"v{i}" for i in range(components)],
    )


def make_model(**kwargs):
    return TimestralModel(**{
        "input_chunk_length": 23,
        "output_chunk_length": 7,
        "local_dir": tiny_timestral_dir(),
        **copy.deepcopy(tfm_kwargs),
        **kwargs,
    })


def upstream_forecast(
    directory,
    target,
    past=None,
    known=None,
    horizon=7,
    shift=0,
    categorical=False,
    context_length=23,
    **kwargs,
):
    """Independent TimeSeries conversion for comparison with the public upstream API."""

    def tensor(values):
        return torch.from_numpy(values.T.copy()).unsqueeze(0)

    context = target.values()[-context_length:]
    inputs = {}
    for cov, kind in ((past, "past"), (known, "known")):
        if cov is None:
            continue
        history = cov.slice(
            target.time_index[-len(context)], target.end_time()
        ).values()
        if kind == "known":
            future = cov.slice_n_points_after(
                target.end_time() + (shift + 1) * target.freq, horizon
            ).values()
            history = np.concatenate((
                history,
                np.full((shift, cov.n_components), np.nan, dtype=np.float32),
                future,
            ))
        if categorical:
            inputs[f"{kind}_covariates"] = tensor(history[:, :1])
            inputs[f"categorical_{kind}_covariates"] = tensor(history[:, 1:])
        else:
            inputs[f"{kind}_covariates"] = tensor(history)
    result = TimestralForecaster.from_pretrained(directory).forecast(
        tensor(context), prediction_length=horizon + shift, **inputs, **kwargs
    )
    return result.quantiles[0, :, shift:].permute(1, 0, 2).numpy()


@pytest.mark.parametrize("components", [1, 3])
@pytest.mark.parametrize(
    "covariates", ["none", "past", "future", "both", "categorical"]
)
@pytest.mark.parametrize("horizon,shift", [(7, 0), (19, 3)])
def test_fidelity(components, covariates, horizon, shift):
    """Joint variates, missing values, partial patches, shifts and multiple decode blocks."""
    target = make_series(components=components)
    values = target.values()
    values[25:28, 0] = np.nan
    target = target.with_values(values)
    cov = make_series(96, 2, offset=5)
    values = cov.values()
    values[:, 1] = np.arange(len(cov)) % 3
    values[32, :] = np.nan
    # Include a category first seen in the horizon.
    values[49:52, 1] = 10
    cov = cov.with_values(values)
    past = cov if covariates in ("past", "both", "categorical") else None
    known = cov if covariates in ("future", "both", "categorical") else None
    categorical = covariates == "categorical"
    model = make_model(
        output_chunk_length=horizon,
        output_chunk_shift=shift,
        likelihood=QuantileRegression([0.1, 0.5, 0.9]),
        categorical_past_covariates="v1" if categorical else None,
        categorical_future_covariates=["v1"] if categorical else None,
    )
    with patch("pytorch_lightning.Trainer.fit") as train:
        model.fit(target, past_covariates=past, future_covariates=known)
        train.assert_not_called()
    actual = model.predict(horizon, predict_likelihood_parameters=True)
    expected = upstream_forecast(
        tiny_timestral_dir(), target, past, known, horizon, shift, categorical
    )
    np.testing.assert_allclose(
        actual.values().reshape(expected.shape), expected, rtol=1e-5, atol=1e-5
    )
    assert actual.start_time() == target.end_time() + (shift + 1) * target.freq
    assert np.isfinite(actual.values()).all()
    assert (np.diff(expected, axis=-1) >= 0).all()


def test_median_sampling_and_autoregression():
    target = make_series(components=2)
    model = make_model()
    model.fit(target)
    actual = model.predict(7)
    expected = upstream_forecast(tiny_timestral_dir(), target)[:, :, 1]
    np.testing.assert_allclose(actual.values(), expected, rtol=1e-5, atol=1e-5)
    assert actual.components.equals(target.components)
    assert not model.supports_probabilistic_prediction
    assert model.predict(18).n_components == 2

    probabilistic = make_model(
        likelihood=QuantileRegression([0.1, 0.5, 0.9]), random_state=42
    )
    probabilistic.fit(target)
    samples = probabilistic.predict(18, num_samples=10, random_state=17)
    assert samples.all_values().shape == (18, 2, 10)
    assert np.isfinite(samples.all_values()).all()
    assert np.any(np.std(samples.all_values(), axis=2) > 0)
    np.testing.assert_array_equal(
        samples.all_values(),
        probabilistic.predict(18, num_samples=10, random_state=17).all_values(),
    )


def test_batch_independence():
    targets = [make_series(48, 2), make_series(37, 2, offset=100)]
    model = make_model(input_chunk_length=(1, 23))
    model.fit(targets)
    batched = model.predict(11, series=targets, batch_size=2)
    separate = [model.predict(11, series=target) for target in targets]
    for actual, expected in zip(batched, separate):
        np.testing.assert_allclose(
            actual.values(), expected.values(), rtol=1e-5, atol=1e-5
        )
        assert actual.time_index.equals(expected.time_index)


def test_covariate_window_and_options():
    target = make_series()
    cov = make_series(96, 2)
    model = make_model(
        output_chunk_shift=3,
        setup_kwargs={"per_shot_horizon_patches": 1},
        inference_kwargs={"symmetric_averaging": False, "clip_nonnegative": False},
        categorical_past_covariates="v1",
        categorical_future_covariates="v1",
    )
    model.fit(target, past_covariates=cov, future_covariates=cov)
    forecaster = model.model.forecaster
    assert forecaster.per_shot_horizon_patches == 1
    with patch.object(forecaster, "forecast", wraps=forecaster.forecast) as call:
        model.predict(7)
    args = call.call_args.kwargs
    np.testing.assert_array_equal(args["context"][0].T.numpy(), target.values()[-23:])
    assert args["prediction_length"] == 10
    assert args["symmetric_averaging"] is False
    assert args["clip_nonnegative"] is False
    np.testing.assert_array_equal(
        args["past_covariates"][0, 0].numpy(), cov.values()[25:48, 0]
    )
    np.testing.assert_array_equal(
        args["categorical_past_covariates"][0, 0].numpy(), cov.values()[25:48, 1]
    )
    for name, column in (("known_covariates", 0), ("categorical_known_covariates", 1)):
        values = args[name][0, 0].numpy()
        np.testing.assert_array_equal(values[:23], cov.values()[25:48, column])
        assert np.isnan(values[23:26]).all()
        np.testing.assert_array_equal(values[26:], cov.values()[51:58, column])
    with pytest.raises(ValueError, match="output_chunk_shift"):
        model.predict(8)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"enable_finetuning": True}, "fine-tuning"),
        ({"enable_finetuning": {"unfreeze": ["*"]}}, "fine-tuning"),
        ({"likelihood": GaussianLikelihood()}, "QuantileRegression"),
        ({"likelihood": QuantileRegression([0.2, 0.5, 0.8])}, "subset"),
        ({"input_chunk_length": 65}, "cannot exceed"),
        ({"input_chunk_length": (1, 65)}, "cannot exceed"),
        ({"pl_trainer_kwargs": {"precision": "64-true"}}, "float32"),
        ({"setup_kwargs": {"device": "cpu"}}, "Unsupported setup_kwargs"),
        ({"setup_kwargs": {"max_context": 32}}, "Unsupported setup_kwargs"),
        ({"inference_kwargs": {"context": []}}, "Unsupported inference_kwargs"),
        (
            {"inference_kwargs": {"prediction_length": 99}},
            "Unsupported inference_kwargs",
        ),
        ({"categorical_past_covariates": ["v0", "v0"]}, "unique component"),
        ({"categorical_future_covariates": [0]}, "unique component"),
    ],
)
def test_invalid_parameters(kwargs, match):
    with pytest.raises(ValueError, match=match):
        make_model(**kwargs)


def test_dtype_validation():
    model = make_model()
    target = make_series()
    with pytest.raises(ValueError, match="float32 input"):
        model.fit(target.astype(np.float64))
    model.fit(target)
    with pytest.raises(ValueError, match="float32 input"):
        model.predict(7, series=target.astype(np.float64))


def test_custom_trainer_precision():
    from pytorch_lightning import Trainer

    model = make_model()
    model.fit(make_series())
    trainer = Trainer(precision="64-true", **tfm_kwargs["pl_trainer_kwargs"])
    with pytest.raises(ValueError, match="precision='32-true'"):
        model.predict(7, trainer=trainer)


def test_categorical_fit_from_dataset_requires_names():
    from darts.utils.data import SequentialTorchTrainingDataset

    dataset = SequentialTorchTrainingDataset(
        series=[make_series()],
        past_covariates=[make_series(96, 2)],
        input_chunk_length=23,
        output_chunk_length=0,
    )
    with pytest.raises(ValueError, match=r"require fit\(\) with TimeSeries"):
        make_model(categorical_past_covariates="v1").fit_from_dataset(dataset)


@pytest.mark.parametrize("kind", ["past", "future"])
def test_covariate_coverage_validation(kind):
    target = make_series()
    cov = make_series(96)
    model = make_model()
    model.fit(target, **{f"{kind}_covariates": cov})
    # Both past and future covariates need to cover the forecast origin when
    # Darts rolls the context forward for a second prediction chunk.
    with pytest.raises(ValueError):
        model.predict(14, **{f"{kind}_covariates": cov[:48]})


@pytest.mark.parametrize("kind", ["past", "future"])
def test_categorical_schema_validation(kind):
    target = make_series()
    cov = make_series(96, 2)
    model = make_model(**{f"categorical_{kind}_covariates": "v1"})
    with pytest.raises(ValueError, match="requires"):
        model.fit(target)
    with pytest.raises(ValueError, match="Unknown categorical"):
        model.fit(target, **{f"{kind}_covariates": cov[["v0"]]})
    model.fit(target, **{f"{kind}_covariates": cov})
    with pytest.raises(ValueError, match="same component names and order"):
        model.predict(7, **{f"{kind}_covariates": cov[["v1", "v0"]]})
    with pytest.raises(ValueError, match="same component names and order"):
        make_model(**{f"categorical_{kind}_covariates": "v1"}).fit(
            [target, target], **{f"{kind}_covariates": [cov, cov[["v1", "v0"]]]}
        )


def test_all_categorical_covariates():
    target = make_series()
    cov = make_series(96).with_values((np.arange(96) % 3).astype(np.float32)[:, None])
    model = make_model(
        categorical_past_covariates="v0", categorical_future_covariates="v0"
    )
    model.fit(target, past_covariates=cov, future_covariates=cov)
    assert np.isfinite(model.predict(7).values()).all()


@pytest.mark.parametrize("categorical", [False, True])
def test_historical_forecasts_and_encoders(categorical):
    target = make_series(64)
    model = make_model(
        input_chunk_length=12,
        output_chunk_length=4,
        add_encoders={"cyclic": {"future": ["hour"]}},
        categorical_future_covariates="v1" if categorical else None,
    )
    covariates = make_series(96, 2) if categorical else None
    model.fit(target, future_covariates=covariates)
    options = dict(
        series=target,
        forecast_horizon=4,
        start=40,
        start_format="position",
        stride=8,
        retrain=False,
        last_points_only=False,
        future_covariates=covariates,
    )
    optimized = model.historical_forecasts(**options, enable_optimization=True)
    ordinary = model.historical_forecasts(**options, enable_optimization=False)
    for actual, expected in zip(optimized, ordinary):
        assert actual.time_index.equals(expected.time_index)
        np.testing.assert_allclose(
            actual.values(), expected.values(), rtol=1e-5, atol=1e-5
        )


def test_save_load_without_hub(tmp_path):
    target = make_series()
    cov = make_series(96, 2)
    model = make_model(categorical_future_covariates="v1")
    model.fit(target, future_covariates=cov)
    expected = model.predict(7)
    path = str(tmp_path / "timestral.pt")
    model.save(path)
    with patch(
        "darts.models.components.huggingface_connector.hf_hub_download",
        side_effect=AssertionError("offline"),
    ):
        restored = TimestralModel.load(path)
        np.testing.assert_allclose(
            restored.predict(7).values(), expected.values(), rtol=1e-6, atol=1e-6
        )
        assert restored.model.forecaster.model is restored.model.timestral
        clean_path = str(tmp_path / "clean.pt")
        model.save(clean_path, clean=True)
        clean = TimestralModel.load(clean_path)
        np.testing.assert_allclose(
            clean.predict(7, series=target, future_covariates=cov).values(),
            expected.values(),
            rtol=1e-6,
            atol=1e-6,
        )
        weights = make_model(categorical_future_covariates="v1")
        weights.load_weights(path)
        np.testing.assert_allclose(
            weights.predict(7, series=target, future_covariates=cov).values(),
            expected.values(),
            rtol=1e-6,
            atol=1e-6,
        )
        with pytest.raises(ValueError, match="same component names and order"):
            weights.predict(7, series=target, future_covariates=cov[["v1", "v0"]])


def test_hub_revision_and_local_config(tmp_path):
    directory = Path(tiny_timestral_dir())
    with patch(
        "darts.models.components.huggingface_connector.hf_hub_download",
        side_effect=lambda filename, **_: str(directory / filename),
    ) as download:
        model = make_model(
            local_dir=None,
            hub_model_name="test/timestral",
            hub_model_revision="revision",
        )
        assert download.call_args.kwargs == dict(
            repo_id="test/timestral",
            filename="config.json",
            revision="revision",
            local_dir=None,
        )
        model.fit(make_series())
        assert download.call_args.kwargs["filename"] == "model.safetensors"
    config = json.loads((directory / "config.json").read_text())
    config["covariate_enabled"] = False
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="does not support past covariates"):
        make_model(local_dir=tmp_path).fit(make_series(), past_covariates=make_series())


@pytest.mark.parametrize("accelerator", ["cpu", "cuda", "mps"])
def test_accelerator(accelerator, tmp_path):
    if accelerator == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    trainer = copy.deepcopy(tfm_kwargs["pl_trainer_kwargs"])
    trainer["accelerator"] = accelerator
    model = make_model(pl_trainer_kwargs=trainer)
    if accelerator == "mps":
        with pytest.raises(ValueError, match="Timestral requires CPU or CUDA"):
            model.fit(make_series())
            model.predict(7)
        return
    model.fit(make_series())
    prediction = model.predict(7)
    assert model.model.forecaster.device.type == accelerator
    path = str(tmp_path / "device.pt")
    model.save(path)
    restored = TimestralModel.load(
        path, pl_trainer_kwargs=tfm_kwargs["pl_trainer_kwargs"]
    )
    actual = restored.predict(7)
    assert restored.model.forecaster.device.type == "cpu"
    np.testing.assert_allclose(
        actual.values(), prediction.values(), rtol=1e-3, atol=1e-3
    )


@pytest.mark.slow
@pytest.mark.parametrize("horizon", [17, 1025])
def test_released_checkpoint_fidelity(horizon):
    """Compare against upstream 5M, including the 1,024-step decode boundary."""
    target = make_series(96, 2)
    cov = make_series(96 + horizon, 2)
    model_id = "mistralai/Timestral-5M"
    revision = "bdd20cfc15a3fea0e4cb5ccef37f8d3ca7aa335f"
    model = TimestralModel(
        input_chunk_length=64,
        output_chunk_length=horizon,
        hub_model_name=model_id,
        hub_model_revision=revision,
        likelihood=QuantileRegression([0.1, 0.5, 0.9]),
        **tfm_kwargs,
    )
    model.fit(target, past_covariates=cov, future_covariates=cov)
    actual = (
        model.predict(horizon, predict_likelihood_parameters=True)
        .values()
        .reshape(horizon, 2, 3)
    )
    forecaster = TimestralForecaster.from_pretrained(model_id, revision=revision)
    expected = forecaster.forecast(
        target.values()[-64:].T[None],
        prediction_length=horizon,
        past_covariates=cov.values()[32:96].T[None],
        known_covariates=cov.values()[32 : 96 + horizon].T[None],
    )
    indices = [expected.quantile_levels.index(q) for q in (0.1, 0.5, 0.9)]
    expected = expected.quantiles[0].permute(1, 0, 2).numpy()[..., indices]
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
