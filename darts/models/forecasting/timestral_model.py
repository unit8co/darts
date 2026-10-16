"""
Timestral: Zero-Shot Multivariate Forecasting
---------------------------------------------

Timestral supports joint multivariate forecasting with past and future covariates.

See the `Foundation Model Examples
<https://unit8co.github.io/darts/examples/25-FoundationModel-examples.html>`__.
"""

import os
from collections.abc import Sequence
from copy import deepcopy
from typing import Any

import numpy as np
import torch
from pytorch_lightning import Trainer
from timestral import TimestralConfig, TimestralForecaster
from timestral import TimestralModel as TimestralNetwork

from darts import TimeSeries
from darts.logging import raise_log
from darts.models.forecasting.foundation_model import FoundationModel
from darts.models.forecasting.pl_forecasting_module import PLForecastingModule
from darts.utils.data import TorchInferenceDataset, TorchTrainingDataset
from darts.utils.data.torch_datasets.utils import (
    InputChunkLength,
    PLModuleInput,
    PLModuleOutput,
    TorchInferenceSample,
    TorchTrainingSample,
)
from darts.utils.likelihood_models.torch import QuantileRegression


class _TimestralModule(PLForecastingModule):
    """Adapt the upstream forecaster to Darts' mixed-covariate Lightning interface.

    Each batch member is an independent forecasting task; its target components
    are forecast jointly. The registered network owns the parameters, while the
    forecaster handles preprocessing and block decoding without gradient updates.
    """

    _quantile_indices: torch.Tensor

    def __init__(
        self,
        config: dict[str, Any],
        setup_kwargs: dict[str, Any],
        inference_kwargs: dict[str, Any],
        categorical_indices: dict[str, list[int]],
        **kwargs,
    ):
        """Create the network from configuration without downloading weights.

        Parameters
        ----------
        config
            Serialized Timestral checkpoint configuration.
        setup_kwargs
            Additional arguments to ``TimestralForecaster``.
        inference_kwargs
            Additional arguments to ``TimestralForecaster.forecast()``.
        categorical_indices
            Indices of categorical components in the past and future covariates.
        **kwargs
            Additional arguments to ``PLForecastingModule``.
        """
        super().__init__(**kwargs)
        # Register the network, not the plain Python forecaster, so Lightning owns
        # its device and state dict. Reconstruct from config without Hub access on load.
        self.timestral = TimestralNetwork(TimestralConfig.from_dict(config))
        # Darts already slices the initial context. Retain the checkpoint's cap
        # while upstream feeds medians back across decode blocks.
        self.forecaster = TimestralForecaster(self.timestral, **setup_kwargs)
        self._inference_kwargs = inference_kwargs
        self._categorical_indices = categorical_indices
        quantiles = self.likelihood.quantiles if self.likelihood else [0.5]
        self.register_buffer(
            "_quantile_indices",
            torch.tensor([self.forecaster.quantile_levels.index(q) for q in quantiles]),
        )

    def on_predict_start(self) -> None:
        if self.device.type == "mps":
            raise_log(
                ValueError(
                    "Timestral requires CPU or CUDA; MPS does not support its float64 scaling."
                )
            )
        if self.trainer.precision not in (32, "32", "32-true"):
            raise_log(ValueError("Timestral requires float32 / precision='32-true'."))
        # Match upstream loading: recompute rotary frequencies on the execution device.
        self.timestral.transformer.rope.rebuild_inv_freq()

    def _split_covariates(
        self, values: torch.Tensor | None, kind: str
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if values is None:
            return None, None
        categorical = self._categorical_indices[kind]
        numeric = [i for i in range(values.shape[-1]) if i not in categorical]
        return (
            values[..., numeric].transpose(1, 2) if numeric else None,
            values[..., categorical].transpose(1, 2) if categorical else None,
        )

    def forward(self, x_in: PLModuleInput) -> PLModuleOutput:
        # Dimension notation:
        # B: batch size; L: context length; T: output chunk length; S: output shift;
        # C: target components; P/F: past/future covariate components;
        # Q: checkpoint quantiles; N: selected quantiles (1 for median predictions).
        if next(self.timestral.parameters()).dtype != torch.float32:
            raise_log(ValueError("Timestral requires float32 / precision='32-true'."))
        target = x_in.past_target  # (B, L, C)
        # Split numeric and categorical rows, transposing (B, L, P) -> (B, P, L).
        past, categorical_past = self._split_covariates(x_in.past_covariates, "past")
        known = x_in.future_covariates  # (B, T, F)
        if known is not None:
            # Darts omits future covariates inside the output shift. Keep that
            # gap missing rather than moving known future values into the past.
            gap = known.new_full(
                (target.shape[0], self.output_chunk_shift, known.shape[-1]), torch.nan
            )
            # History + missing shift + forecast window: (B, L + S + T, F).
            known = torch.cat((x_in.historic_future_covariates, gap, known), dim=1)
        known, categorical_known = self._split_covariates(known, "future")
        forecast = self.forecaster.forecast(
            context=target.transpose(1, 2),
            prediction_length=self.output_chunk_length + self.output_chunk_shift,
            past_covariates=past,
            known_covariates=known,
            categorical_past_covariates=categorical_past,
            categorical_known_covariates=categorical_known,
            **self._inference_kwargs,
        )
        # (B, C, S + T, Q) -> (B, S + T, C, Q) -> (B, T, C, N).
        prediction = forecast.quantiles.permute(0, 2, 1, 3)
        prediction = prediction[:, self.output_chunk_shift :].index_select(
            -1, self._quantile_indices
        )
        return PLModuleOutput(prediction=prediction)


class TimestralModel(FoundationModel):
    def __init__(
        self,
        input_chunk_length: InputChunkLength,
        output_chunk_length: int,
        output_chunk_shift: int = 0,
        likelihood: QuantileRegression | None = None,
        hub_model_name: str = "mistralai/Timestral-209M",
        hub_model_revision: str | None = None,
        local_dir: str | os.PathLike | None = None,
        categorical_past_covariates: str | list[str] | None = None,
        categorical_future_covariates: str | list[str] | None = None,
        setup_kwargs: dict[str, Any] | None = None,
        inference_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ):
        """Timestral foundation model for zero-shot probabilistic forecasting.

        Wraps Mistral's optional `timestral <https://github.com/mistralai/timestral>`_ package.
        Target components are forecast jointly; different series in a batch remain independent.
        ``fit()`` initializes the pretrained model without training. Fine-tuning is not supported.
        The 5M, 28M, 123M and 209M checkpoints share this interface.

        .. note::
            Install ``darts[torch]`` and the optional Timestral package as described in the
            installation guide. Timestral code is Apache-2.0 licensed; pretrained weights have a
            separate `Modified MIT license <https://huggingface.co/mistralai/Timestral-209M/blob/main/LICENSE>`_.

        .. note::
            Use float32 input series (``series.astype("float32")``) and 32-bit trainer precision.
            NaNs represent missing observations. Static covariates are not used. CPU and CUDA are
            supported; MPS is not, because upstream computes scaling statistics in float64.
            Automatic device selection uses CUDA when available, otherwise CPU.

        Parameters
        ----------
        input_chunk_length
            Context length, or ``(min_length, max_length)`` for variable-length inference. The maximum
            must not exceed the checkpoint's context length (32,768 for the released checkpoints).
        output_chunk_length
            Number of steps per Darts prediction chunk. ``predict(n)`` uses Darts autoregression when
            ``n > output_chunk_length``. Timestral also decodes long chunks internally, feeding back
            block medians; its default 1,024-step decode block is not a maximum forecast horizon.
        output_chunk_shift
            Gap between context and returned forecast. Known-future covariates in this gap are treated
            as missing. As with other Darts models, a shift prevents Darts autoregression.
        likelihood
            ``None`` returns the median. Otherwise, use ``QuantileRegression`` with a subset of the
            checkpoint's quantiles. ``predict_likelihood_parameters=True`` returns those quantiles;
            ``num_samples > 1`` uses Darts' quantile interpolation sampling.
        hub_model_name
            Hugging Face checkpoint ID. Default: ``"mistralai/Timestral-209M"``. Other released sizes
            are ``"mistralai/Timestral-5M"``, ``"mistralai/Timestral-28M"`` and ``"mistralai/Timestral-123M"``.
        hub_model_revision
            Optional checkpoint branch, tag or commit hash.
        local_dir
            Existing local directory containing ``config.json`` and ``model.safetensors``. Missing
            files are downloaded there from the Hub. ``None`` uses the Hugging Face cache.
        categorical_past_covariates
            Past-covariate component name or list of names to treat as categories. Values must be
            numeric category codes, with NaN for missing values. Use consistent codes across history
            and future; do not scale categorical components. All other components are numeric.
        categorical_future_covariates
            Future-covariate component name or list of names to treat as categories. For univariate
            targets, upstream encodes categories against only the current target context. For
            multivariate targets it uses raw codes. Unseen categories use upstream's fallback.
            Named categorical covariates require ``fit()`` before using dataset-level prediction.
        setup_kwargs
            Optional Timestral forecaster settings: ``per_shot_horizon_patches``, ``quantile_real_cap_k``
            and ``extreme_value_threshold``. Context length and device are controlled by Darts.
        inference_kwargs
            Optional ``symmetric_averaging`` and ``clip_nonnegative`` flags, both defaulting to True
            upstream. These are forwarded on each internal forecast, not to Darts ``predict()``.
        **kwargs
            Standard :class:`~darts.models.forecasting.foundation_model.FoundationModel` options,
            including ``batch_size``, ``add_encoders``, ``random_state`` and ``pl_trainer_kwargs``.
            Select CPU/GPU through ``pl_trainer_kwargs``. ``enable_finetuning`` must be False or None.

        Examples
        --------
        >>> from darts.datasets import AirPassengersDataset
        >>> from darts.models import TimestralModel
        >>> series = AirPassengersDataset().load().astype("float32")
        >>> model = TimestralModel(input_chunk_length=48, output_chunk_length=12)
        >>> model.fit(series)
        >>> forecast = model.predict(12)
        """
        if kwargs.get("enable_finetuning") not in (None, False):
            raise_log(ValueError("Timestral does not support fine-tuning."))
        if likelihood is not None and not isinstance(likelihood, QuantileRegression):
            raise_log(
                ValueError("Timestral only supports a QuantileRegression likelihood.")
            )
        precision = (kwargs.get("pl_trainer_kwargs") or {}).get("precision", "32-true")
        if precision not in (32, "32", "32-true"):
            raise_log(ValueError("Timestral requires float32 / precision='32-true'."))
        self._setup_kwargs = dict(setup_kwargs or {})
        self._inference_kwargs = dict(inference_kwargs or {})
        for name, options, allowed in (
            (
                "setup_kwargs",
                self._setup_kwargs,
                {
                    "per_shot_horizon_patches",
                    "quantile_real_cap_k",
                    "extreme_value_threshold",
                },
            ),
            (
                "inference_kwargs",
                self._inference_kwargs,
                {"symmetric_averaging", "clip_nonnegative"},
            ),
        ):
            if unknown := options.keys() - allowed:
                raise_log(
                    ValueError(
                        f"Unsupported {name}: {sorted(unknown)}. Allowed: {sorted(allowed)}."
                    )
                )
        self._categorical_names = {}
        self._covariate_components = {}
        self._categorical_indices = {"past": [], "future": []}
        for kind, names in (
            ("past", categorical_past_covariates),
            ("future", categorical_future_covariates),
        ):
            names = [names] if isinstance(names, str) else names
            if names is not None and (
                not isinstance(names, list)
                or not all(isinstance(n, str) for n in names)
                or len(names) != len(set(names))
            ):
                raise_log(
                    ValueError(
                        f"categorical_{kind}_covariates must contain unique component names."
                    )
                )
            self._categorical_names[kind] = names or []

        super().__init__(**kwargs)
        self._timestral_config = self.hf_connector.load_config()
        config = TimestralConfig.from_dict(self._timestral_config)
        if self.input_chunk_length > config.context_length:
            raise_log(
                ValueError(f"input_chunk_length cannot exceed {config.context_length}.")
            )
        if likelihood is not None and not set(likelihood.quantiles).issubset(
            config.quantile_knots
        ):
            raise_log(
                ValueError(f"Quantiles must be a subset of {config.quantile_knots}.")
            )

    def _validate_covariate_components(
        self,
        past_covariates: Sequence[TimeSeries] | None,
        future_covariates: Sequence[TimeSeries] | None,
    ) -> None:
        config = TimestralConfig.from_dict(self._timestral_config)
        for kind, covariates in (
            ("past", past_covariates),
            ("future", future_covariates),
        ):
            names = self._categorical_names[kind]
            if covariates is None:
                if names:
                    raise_log(
                        ValueError(
                            f"categorical_{kind}_covariates requires {kind}_covariates."
                        )
                    )
                continue
            if not config.covariate_enabled or (
                kind == "future" and not config.supports_known_covariates
            ):
                raise_log(
                    ValueError(
                        f"This Timestral checkpoint does not support {kind} covariates."
                    )
                )
            if not names:
                continue
            components = tuple(covariates[0].components)
            if not set(names).issubset(components):
                raise_log(
                    ValueError(
                        f"Unknown categorical_{kind}_covariates: {set(names) - set(components)}."
                    )
                )
            expected = self._covariate_components.get(kind, components)
            if any(tuple(cov.components) != expected for cov in covariates):
                raise_log(
                    ValueError(
                        f"{kind}_covariates must keep the same component names and order."
                    )
                )
            self._covariate_components[kind] = components
            self._categorical_indices[kind] = [components.index(name) for name in names]

    def _build_train_dataset(
        self,
        series: Sequence[TimeSeries],
        past_covariates: Sequence[TimeSeries] | None,
        future_covariates: Sequence[TimeSeries] | None,
        **kwargs,
    ) -> TorchTrainingDataset:
        self._validate_covariate_components(past_covariates, future_covariates)
        return super()._build_train_dataset(
            series=series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            **kwargs,
        )

    def _build_inference_dataset(
        self,
        n: int,
        series: Sequence[TimeSeries],
        past_covariates: Sequence[TimeSeries] | None,
        future_covariates: Sequence[TimeSeries] | None,
        **kwargs,
    ) -> TorchInferenceDataset:
        self._validate_covariate_components(past_covariates, future_covariates)
        return super()._build_inference_dataset(
            n=n,
            series=series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            **kwargs,
        )

    def _verify_dtypes(
        self, ds_output: TorchTrainingSample | TorchInferenceSample
    ) -> None:
        super()._verify_dtypes(ds_output)
        if ds_output.past_target.dtype != np.float32:
            raise_log(
                ValueError(
                    'Timestral requires float32 input series. Use series.astype("float32").'
                )
            )

    @staticmethod
    def _init_trainer(trainer_params: dict, max_epochs: int | None = None) -> Trainer:
        # Apply this at trainer construction so it also covers clean checkpoints,
        # which intentionally discard trainer_params, and cross-device loading.
        trainer_params = dict(trainer_params)
        accelerator = trainer_params.get("accelerator", "auto")
        if accelerator == "auto":
            trainer_params["accelerator"] = (
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        elif accelerator == "mps":
            raise_log(
                ValueError(
                    "Timestral requires CPU or CUDA; MPS does not support its float64 scaling."
                )
            )
        return FoundationModel._init_trainer(trainer_params, max_epochs)

    def _restore_checkpoint_metadata(self, tfm_save: "TimestralModel") -> None:
        super()._restore_checkpoint_metadata(tfm_save)
        # load_weights() recreates the module before seeing any TimeSeries. Restore
        # the categorical column mapping from the validated wrapper checkpoint.
        self._covariate_components = deepcopy(tfm_save._covariate_components)
        self._categorical_indices = deepcopy(tfm_save._categorical_indices)

    def _create_model(self, train_sample: TorchTrainingSample) -> PLForecastingModule:
        for kind, names in self._categorical_names.items():
            if names and kind not in self._covariate_components:
                raise_log(
                    ValueError(
                        "Named categorical covariates require fit() with TimeSeries inputs."
                    )
                )
        self.pl_module_params.update(
            config=self._timestral_config,
            setup_kwargs=self._setup_kwargs,
            inference_kwargs=self._inference_kwargs,
            categorical_indices=self._categorical_indices,
        )
        module = _TimestralModule(**self.pl_module_params)
        self.hf_connector.load_model_weights(module.timestral)
        return module
