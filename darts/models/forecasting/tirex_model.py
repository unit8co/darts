"""
TiRex: Zero-Shot Forecasting
----------------------------

TiRex can be used the same way as other foundation models (e.g. Chronos2), with the exception
that it does not support any type of covariates.

For detailed examples and tutorials, check out the following notebooks:

* `Chronos-2 Foundation Model Examples
  <https://unit8co.github.io/darts/examples/25-Chronos-2-examples.html>`__
* `Fine-Tuning Examples
  <https://unit8co.github.io/darts/examples/27-Torch-and-Foundation-Model-Fine-Tuning-examples.html>`__
"""

import os
from typing import Any

import torch
from tirex import load_model

from darts.logging import get_logger, raise_log
from darts.models.forecasting.foundation_model import FoundationModel
from darts.models.forecasting.pl_forecasting_module import PLForecastingModule
from darts.utils.likelihood_models.torch import QuantileRegression

logger = get_logger(__name__)


class _TiRexModule(PLForecastingModule):
    """PyTorch Lightning module wrapping a pre-loaded TiRex pipeline.

    Adapts TiRex's forecasting interface to Darts' PLForecastingModule API.
    Multivariate inputs are supported by folding components into the batch dimension.
    Covariates are not supported.

    During fine-tuning, the gradient-enabled ``_forecast_tensor`` path is used,
    bypassing the ``@torch.inference_mode()`` decorator on ``_forecast_quantiles``.
    Loss is computed over all 9 pre-trained quantiles; only user-specified quantiles
    are returned at prediction time.
    """

    _user_quantile_indices: torch.Tensor

    def __init__(
        self,
        tirex_kwargs: dict[str, Any],
        all_quantiles: tuple[float, ...],
        enable_finetuning: bool | dict = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        tirex_kwargs["device"] = str(self.device)
        self.tirex_pipeline = load_model(**tirex_kwargs)

        self.future_len = (self.output_chunk_length or 0) + self.output_chunk_shift
        # bool(dict) is True for a non-empty dict; _setup_finetuning() handles the actual
        # parameter freeze/unfreeze pattern — here we only need a flag for the forward path
        self._enable_finetuning = bool(enable_finetuning)

        user_q: list[float] = (
            self.likelihood.quantiles
            if isinstance(self.likelihood, QuantileRegression)
            else [0.5]
        )
        self.register_buffer(
            "_user_quantile_indices",
            torch.tensor(
                [all_quantiles.index(q) for q in user_q],
                dtype=torch.long,
                device=self.device,
            ),
        )

        if enable_finetuning:
            # loss is computed over all pre-trained quantiles to preserve the distribution;
            # user-specified quantiles are selected at prediction time
            self._finetuning_likelihood = QuantileRegression(list(all_quantiles))
        else:
            self._finetuning_likelihood = None

    def forward(self, x_in, *args, **kwargs):
        """Forward pass returning quantile predictions shaped ``(batch, time, n_targets, n_quantiles)``.

        During training with fine-tuning enabled, all 9 pre-trained quantiles are returned
        for the loss. At prediction time, only user-specified quantiles are returned.
        """
        x_past, _, _ = x_in
        # fold target components into batch dim for multivariate support: (B,L,C) -> (B*C,L)
        x_past = x_past.transpose(1, 2).flatten(start_dim=0, end_dim=1)

        if self.training and self._enable_finetuning:
            # call _forecast_tensor directly to keep gradients flowing — _forecast_quantiles
            # is decorated with @torch.inference_mode() which would block backprop
            # output: (B*C, Q, H) -> swapaxes -> (B*C, H, Q), then slice output_chunk_shift
            q_sel = self.tirex_pipeline._forecast_tensor(
                context=x_past,
                prediction_length=self.future_len,
            ).swapaxes(1, 2)[:, self.output_chunk_shift :, :]
        else:
            quantiles, _ = self.tirex_pipeline._forecast_quantiles(
                context=x_past,
                prediction_length=self.future_len,
                output_device=x_past.device,
            )
            # (B*C, H, Q) -> slice shift -> select user quantiles -> (B*C, T, N)
            q_sel = quantiles[:, self.output_chunk_shift :, :].index_select(
                dim=-1, index=self._user_quantile_indices
            )

        # unfold batch dim and permute to Darts' output shape: (B, T, C, N or Q)
        return q_sel.unflatten(dim=0, sizes=(-1, self.n_targets)).permute(0, 2, 1, 3)

    def _compute_loss(self, output, target, criterion, sample_weight):
        if self.training and self._enable_finetuning:
            return self._finetuning_likelihood.compute_loss(
                output, target, sample_weight
            )
        return super()._compute_loss(output, target, criterion, sample_weight)


class TiRexModel(FoundationModel):
    _DEFAULT_QUANTILES: tuple[float, ...] = (
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
    )
    _MAX_PREDICTION_LENGTH = 2048

    def __init__(
        self,
        input_chunk_length: int,
        output_chunk_length: int,
        output_chunk_shift: int = 0,
        accept_license: bool = False,
        likelihood: QuantileRegression | None = None,
        hub_model_name: str = "NX-AI/TiRex",
        hub_model_revision: str | None = None,
        local_dir: str | os.PathLike | None = None,
        tirex_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ):
        """
        TiRex foundation model for zero-shot time series forecasting.

        This is a Darts wrapper around the TiRex model introduced in Auer et al. (2025) [1]_. The implementation
        delegates all forecasting logic and weight loading to the optional
        `tirex-ts <https://pypi.org/project/tirex-ts>`_ package while exposing a standard :class:`TorchForecastingModel`
        interface.

        TiRex is a pre-trained foundation model designed for zero-shot forecasting across both short and long horizons.

        This model supports either univariate or multivariate time series, but does not support covariates.
        For multivariate time series, the model is applied independently to each component.

        By default, the model is deterministic (median forecast only). To enable probabilistic forecasts, pass a
        :class:`~darts.utils.likelihood_models.torch.QuantileRegression` instance to the ``likelihood`` parameter.
        It is recommended to call :func:`predict()` with ``predict_likelihood_parameters=True`` or ``num_samples >> 1``
        to get meaningful results.

        For more details on the TiRex model, see the original paper [1]_, the `TiRex base repo <https://github.com/NX-AI/tirex>`_,
        `model card <https://huggingface.co/NX-AI/TiRex>`_, and `docs <https://nx-ai.github.io/tirex/>`_.

        .. note::
            TiRex is distributed under the `NXAI Community License <https://github.com/NX-AI/tirex/blob/main/LICENSE>`_.
            You must explicitly acknowledge this license by passing ``accept_license=True`` when constructing the model.

        .. note::
            Partial fine-tuning is supported via
            ``enable_finetuning={"unfreeze": ["tirex_pipeline.output_patch_embedding*", ...]}``.
            Fine-tuning requires ``tirex_kwargs={"backend": "torch"}``; only the last sLSTM blocks
            and the output head are gradient-safe (see notebook for recommended configurations).
            Full fine-tuning (``enable_finetuning=True``) is **not supported** — backpropagation
            through the early sLSTM blocks produces NaN gradients.

        Parameters
        ----------
        input_chunk_length
            Number of time steps in the past to take as a model input (per chunk). Applies to the target
            series.
        output_chunk_length
            Number of time steps predicted at once (per chunk) by the internal model. It is not the same as forecast
            horizon `n` used in `predict()`, which is the desired number of prediction points generated using
            either a one-shot- or autoregressive forecast. Setting `n <= output_chunk_length` prevents auto-regression.
            For TiRex, `output_chunk_length + output_chunk_shift` must be less than or equal to ``2048``.
        output_chunk_shift
            Optionally, the number of steps to shift the start of the output chunk into the future (relative to the
            input chunk end). This will create a gap between the input and output. Predictions will start
            `output_chunk_shift` steps after the end of the target `series`. If `output_chunk_shift` is set, the model
            cannot generate autoregressive predictions (`n > output_chunk_length`). Default: ``0``.
        likelihood
            The likelihood model to be used for probabilistic forecasts. Must be ``None`` or an instance of
            :class:`~darts.utils.likelihood_models.torch.QuantileRegression`. Requested quantiles must be a subset of
            TiRex's default quantiles: [0.1, 0.2, ..., 0.9]. Default: ``None`` (deterministic; median forecast only).
        accept_license
            Must be set to ``True`` to confirm acceptance of the NXAI Community License. Default: ``False``.
        hub_model_name
            The model ID on HuggingFace Hub. Default: ``"NX-AI/TiRex"``.
        hub_model_revision
            The model version to use. This can be a branch name, tag name, or commit hash. Default: ``None``, which
            will use the default branch from ``hub_model_name``.
        local_dir
            Optional local directory to load the pre-downloaded model. If specified and the directory is empty, the
            model will be downloaded from HuggingFace Hub and saved to this directory. Default: ``None``, which will
            use a cache directory managed by ``huggingface_hub`` instead.
        tirex_kwargs
            Additional keyword arguments forwarded to ``tirex.load_model()``.
        **kwargs
            Optional arguments to initialize the pytorch_lightning.Module, pytorch_lightning.Trainer, and
            Darts' :class:`TorchForecastingModel`.

        batch_size
            Number of time series (input and output sequences) used in each training pass. Default: ``32``.
        random_state
            Controls the randomness of reproducible forecasting.
        pl_trainer_kwargs
            By default :class:`TorchForecastingModel` creates a PyTorch Lightning Trainer with several useful presets
            that performs the training, validation and prediction processes. These presets include automatic
            checkpointing, tensorboard logging, setting the torch device and more.
            With ``pl_trainer_kwargs`` you can add additional kwargs to instantiate the PyTorch Lightning trainer
            object. Check the `PL Trainer documentation
            <https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html>`__ for more information about the
            supported kwargs. Default: ``None``.
            Running on GPU(s) is also possible using ``pl_trainer_kwargs`` by specifying keys ``"accelerator",
            "devices", and "auto_select_gpus"``. Some examples for setting the devices inside the ``pl_trainer_kwargs``
            dict:

            - ``{"accelerator": "cpu"}`` for CPU,
            - ``{"accelerator": "gpu", "devices": [i]}`` to use only GPU ``i`` (``i`` must be an integer),
            - ``{"accelerator": "gpu", "devices": -1, "auto_select_gpus": True}`` to use all available GPUS.

            For more info, see here:
            https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#trainer-flags , and
            https://pytorch-lightning.readthedocs.io/en/stable/accelerators/gpu_basic.html#train-on-multiple-gpus

            With parameter ``"callbacks"`` you can add custom or PyTorch-Lightning built-in callbacks to Darts'
            :class:`TorchForecastingModel`. Below is an example for adding EarlyStopping to the training process.
            The model will stop training early if the validation loss `val_loss` does not improve beyond
            specifications. For more information on callbacks, visit:
            `PyTorch Lightning Callbacks
            <https://pytorch-lightning.readthedocs.io/en/stable/extensions/callbacks.html>`__

            .. highlight:: python
            .. code-block:: python

                from pytorch_lightning.callbacks.early_stopping import EarlyStopping

                # stop training when validation loss does not decrease more than 0.05 (`min_delta`) over
                # a period of 5 epochs (`patience`)
                my_stopper = EarlyStopping(
                    monitor="val_loss",
                    patience=5,
                    min_delta=0.05,
                    mode='min',
                )

                pl_trainer_kwargs={"callbacks": [my_stopper]}
            ..

            Note that you can also use a custom PyTorch Lightning Trainer for training and prediction with optional
            parameter ``trainer`` in :func:`fit()` and :func:`predict()`.
        show_warnings
            whether to show warnings raised from PyTorch Lightning. Useful to detect potential issues of
            your forecasting use case. Default: ``False``.

        References
        ----------
        .. [1] A. Auer et al., "TiRex: Zero-Shot Forecasting Across Long and Short Horizons with Enhanced In-Context
                Learning", NeurIPS 2025. https://arxiv.org/abs/2505.23719.

        Examples
        --------
        >>> from darts.models import TiRexModel
        >>> from darts.datasets import AirPassengersDataset
        >>> series = AirPassengersDataset().load().astype("float32")
        >>> # you must explicitly accept the license to use the model
        >>> model = TiRexModel(
        ...     input_chunk_length=12,
        ...     output_chunk_length=6,
        ...     accept_license=True,
        ... )
        >>> model.fit(series)
        >>> pred = model.predict(n=6)
        >>> print(pred.all_values())
        [[[440.2505 ]]
        [[444.44373]]
        [[447.36362]]
        [[451.50375]]
        [[458.05853]]
        [[461.98694]]]

        Probabilistic forecasting:

        >>> from darts.utils.likelihood_models import QuantileRegression
        >>> model = TiRexModel(
        ...     input_chunk_length=12,
        ...     output_chunk_length=6,
        ...     likelihood=QuantileRegression(quantiles=[0.1, 0.5, 0.9]),
        ...     accept_license=True,
        ... )
        >>> model.fit(series)
        >>> pred = model.predict(n=6, predict_likelihood_parameters=True)
        """
        if not accept_license:
            raise_log(
                ValueError(
                    "TiRex is distributed under the NXAI Community License. "
                    "Set `accept_license=True` to confirm you have reviewed and accept the terms: "
                    "https://github.com/NX-AI/tirex/blob/main/LICENSE"
                ),
                logger,
            )
        print(
            "TiRex is licensed under the NXAI Community License — "
            "https://github.com/NX-AI/tirex/blob/main/LICENSE\n"
            "Fine-tuned weights are derivative works subject to the same terms."
        )

        if likelihood is not None:
            if not isinstance(likelihood, QuantileRegression):
                raise_log(
                    ValueError(
                        "Only QuantileRegression likelihood is supported for TiRexModel."
                    ),
                    logger,
                )
            req = tuple(float(q) for q in likelihood.quantiles)
            if not set(req).issubset(set(self._DEFAULT_QUANTILES)):
                raise_log(
                    ValueError(
                        "Requested quantiles must be a subset of TiRex quantiles "
                        f"{self._DEFAULT_QUANTILES}."
                    ),
                    logger,
                )

        if output_chunk_length + output_chunk_shift > self._MAX_PREDICTION_LENGTH:
            raise_log(
                ValueError(
                    f"TiRex supports a maximum prediction length of {self._MAX_PREDICTION_LENGTH} per call. "
                    f"Please ensure `output_chunk_length + output_chunk_shift <= {self._MAX_PREDICTION_LENGTH}`."
                ),
                logger,
            )

        super().__init__(**kwargs)

        tirex_kwargs = tirex_kwargs or {}
        hf_kwargs = {
            **(
                {"revision": hub_model_revision}
                if hub_model_revision is not None
                else {}
            ),
            **({"local_dir": local_dir} if local_dir is not None else {}),
            **(tirex_kwargs.get("hf_kwargs", {})),
        }
        if "path" in tirex_kwargs:
            raise_log(
                ValueError(
                    "The `path` argument for loading the TiRex model should be passed via `hub_model_name`,"
                    "not `tirex_kwargs`."
                ),
                logger,
            )
        self.tirex_kwargs = {
            "path": hub_model_name,
            **({"hf_kwargs": hf_kwargs} if hf_kwargs else {}),
            **tirex_kwargs,
        }

    @property
    def supports_past_covariates(self) -> bool:
        return False

    @property
    def supports_future_covariates(self) -> bool:
        return False

    def _create_model(self, train_sample) -> PLForecastingModule:
        pl_module_params = self.pl_module_params or {}
        # enable_finetuning is injected into pl_module_params by the base class;
        # _TiRexModule accepts it as an explicit parameter and converts dict form to bool
        return _TiRexModule(
            tirex_kwargs=self.tirex_kwargs,
            all_quantiles=self._DEFAULT_QUANTILES,
            **pl_module_params,
        )
