"""
TiRex: Zero-Shot Forecasting
----------------------------

For detailed examples and tutorials, see:

* `TiRex Foundation Model Examples
  <https://unit8co.github.io/darts/examples/28-TiRex-examples.html>`__

"""

import os
from dataclasses import dataclass
from typing import Any

import torch

from darts.logging import get_logger, raise_log
from darts.models.forecasting.foundation_model import FoundationModel
from darts.models.forecasting.pl_forecasting_module import PLForecastingModule
from darts.utils.likelihood_models.torch import QuantileRegression

logger = get_logger(__name__)


def _require_tirex():
    """
    Import and return the TiRex `load_model` entry point from the optional
    `tirex-ts` dependency.

    This helper ensures that the TiRex integration remains optional and that
    Darts itself does not depend on `tirex-ts`. If the package is not installed,
    an informative ImportError is raised.

    Returns
    -------
    Callable
        The `load_model` function from the `tirex` package.

    Raises
    ------
    ImportError
        If `tirex-ts` is not installed.
    """
    try:
        from tirex import load_model

    except ModuleNotFoundError:
        raise_log(
            ImportError(
                "Optional dependency `tirex-ts` is required to use TiRexModel. "
                "Install it with `pip install tirex-ts` (it provides the `tirex` Python package; "
                "extras include `tirex-ts[hfdataset]`, `tirex-ts[gluonts]`)."
            ),
            logger,
        )

    return load_model


@dataclass(frozen=True)
class _TiRexQuantiles:
    # TiRex returns 9 quantiles by default (0.1..0.9).
    quantiles: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


class _TiRexModule(PLForecastingModule):
    _user_quantile_indices: torch.Tensor
    """
    PyTorch Lightning module wrapping a pre-loaded TiRex pipeline.

    This module adapts TiRex's zero-shot forecasting interface to Darts'
    `PLForecastingModule` API. It does not implement any trainable layers
    itself. Instead, it delegates all forecasting logic to the TiRex pipeline
    obtained via `tirex.load_model()`.

    The module:

    - Accepts batched past target tensors from Darts
    - Calls `tirex_pipeline.forecast()`
    - Selects user-requested quantiles (or median)
    - Returns outputs in Darts' expected shape:
      `(batch_size, time, n_targets, n_quantiles)`

    Notes
    -----
    - Mulitvariate is currently not supported.
    - Covariates are currently not supported.
    - Fine-tuning is not supported; this is a zero-shot wrapper.
    """

    def __init__(
        self,
        tirex_kwargs: dict[str, Any],
        all_quantiles: tuple[float, ...],
        **kwargs,
    ):
        super().__init__(**kwargs)

        load_model = _require_tirex()

        # ensure that the TiRex pipeline is loaded on the same device as the PL module
        tirex_kwargs["device"] = str(self.device)
        # load the TiRex pipeline (this will download weights if not cached locally)
        self.tirex_pipeline = load_model(**tirex_kwargs)

        self.future_len = (self.output_chunk_length or 0) + self.output_chunk_shift

        # gather indices of user-specified quantiles (used at prediction time)
        user_q: list[float] = (
            self.likelihood.quantiles
            if isinstance(self.likelihood, QuantileRegression)
            else [0.5]
        )
        user_quantile_indices = [all_quantiles.index(q) for q in user_q]
        self.register_buffer(
            "_user_quantile_indices",
            torch.tensor(user_quantile_indices, dtype=torch.long, device=self.device),
        )

    def forward(self, x_in, *args, **kwargs):
        """
        TiRex forward pass.

        Parameters
        ----------
        x_in
            Tuple `(x_past, x_future, x_static)` as provided by Darts.
            - `x_past`: tensor of shape `(batch_size, input_chunk_length, n_targets)`
            - `x_future`: expected to be `None` (covariates unsupported)
            - `x_static`: ignored

        Returns
        -------
        torch.Tensor
            Tensor of shape `(batch_size, output_chunk_length, n_targets, n_quantiles)`
            containing the selected quantile predictions.
            If deterministic (`likelihood=None`), only the median (0.5) quantile
            is returned.
        """
        # Dimension notation in comments below:
        #   B: batch size
        #   L: input chunk length
        #   T: output chunk length
        #   S: output chunk shift
        #   H: forecast horizon = T + S
        #   C: target components
        #   Q = 9: likelihood quantiles returned by TiRex
        #   N: likelihood quantiles (user-specified, 1 if deterministic)

        # `x_past`: (B, L, C)
        x_past, _, _ = x_in
        # fold target component into batch dimension to support multivariate forecasting
        # -> (B, C, L) -> (B * C, L)
        x_past = x_past.transpose(1, 2).flatten(start_dim=0, end_dim=1)

        # TiRex should forecast output_chunk_shift + output_chunk_length steps,
        # then we slice away the shift to match Darts' output_chunk_length.

        # `quantiles`: (B * C, H, Q)
        quantiles, _ = self.tirex_pipeline._forecast_quantiles(
            context=x_past,
            prediction_length=self.future_len,
            output_device=x_past.device,
        )

        # slice away output_chunk_shift -> (B * C, T, Q)
        quantiles = quantiles[:, self.output_chunk_shift :, :]

        # select user-requested quantiles (or median) -> (B * C, T, N)
        q_sel: torch.Tensor = quantiles.index_select(
            dim=-1, index=self._user_quantile_indices
        )

        # unfold batch dimension to separate target components -> (B, C, T, N)
        q_sel = q_sel.unflatten(dim=0, sizes=(-1, self.n_targets))

        # permute to Darts' expected output shape -> (B, T, C, N)
        q_sel = q_sel.permute(0, 2, 1, 3)

        return q_sel


class TiRexModel(FoundationModel):
    # TiRex quantiles returned by default (0.1..0.9)
    _DEFAULT_QUANTILES = _TiRexQuantiles().quantiles
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
        backend: str | None = None,
        compile: bool | None = None,
        hf_kwargs: dict[str, Any] | None = None,
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

        .. warning::
            Fine-tuning TiRex is not supported in Darts. Visit `TiRex Docs <https://nx-ai.github.io/tirex/>`_ for
            details.

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
        backend
            Optional inference engine argument passed to ``tirex.load_model()``, either ``"torch"`` or ``"cuda"``.
            If set to ``"cuda"``, `xlstm` package must be installed and the model will use custom CUDA kernels.
            Otherwise, torch backend will be used. Default: ``None`` (torch backend).
        compile
            Optional compilation flag passed to ``tirex.load_model()``. If ``True``, the model will be compiled with
            `torch.compile()`. Default: ``None`` (no compilation).
        hf_kwargs
            Optional HuggingFace Hub arguments passed to ``tirex.load_model()`` for loading the model weights.
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
        >>> # load data in float32 format
        >>> series = AirPassengersDataset().load().astype("float32")
        >>> # you must explicitly accept the license to use the model
        >>> model = TiRexModel(
        ...     input_chunk_length=12,
        ...     output_chunk_length=6,
        ...     accept_license=True,
        ... )
        >>> # fit the model (data is validated but TiRex is not actually trained)
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
        >>> pred = model.predict(n=6, num_samples=50)
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

        if likelihood is not None:
            if not isinstance(likelihood, QuantileRegression):
                raise_log(
                    ValueError(
                        "Only QuantileRegression likelihood is supported for TiRexModel."
                    ),
                    logger,
                )

            # validate that requested quantiles are a subset of TiRex quantiles
            req = tuple(float(q) for q in likelihood.quantiles)
            if not set(req).issubset(set(self._DEFAULT_QUANTILES)):
                raise_log(
                    ValueError(
                        "Requested quantiles must be a subset of TiRex quantiles "
                        f"{self._DEFAULT_QUANTILES}."
                    ),
                    logger,
                )

        # TiRex supports up to 2048 steps per single forecast call.
        if output_chunk_length + output_chunk_shift > self._MAX_PREDICTION_LENGTH:
            raise_log(
                ValueError(
                    "TiRex supports a maximum prediction length of "
                    f"{self._MAX_PREDICTION_LENGTH} per call. "
                    "Please ensure `output_chunk_length + output_chunk_shift <= {self._MAX_PREDICTION_LENGTH}`."
                ),
                logger,
            )

        if kwargs.get("enable_finetuning", False) not in (None, False):
            raise_log(
                ValueError("Fine-tuning is not supported for TiRexModel."),
                logger,
            )

        super().__init__(**kwargs)

        hf_kwargs = {
            **(
                {"revision": hub_model_revision}
                if hub_model_revision is not None
                else {}
            ),
            **({"local_dir": local_dir} if local_dir is not None else {}),
            **(hf_kwargs or {}),
        }
        self.tirex_kwargs = {
            "path": hub_model_name,
            **({"backend": backend} if backend is not None else {}),
            **({"compile": compile} if compile is not None else {}),
            **({"hf_kwargs": hf_kwargs} if hf_kwargs else {}),
            **(tirex_kwargs or {}),
        }

    @property
    def supports_past_covariates(self) -> bool:
        return False

    @property
    def supports_future_covariates(self) -> bool:
        return False

    def _create_model(self, train_sample) -> PLForecastingModule:
        pl_module_params = self.pl_module_params or {}

        return _TiRexModule(
            tirex_kwargs=self.tirex_kwargs,
            all_quantiles=self._DEFAULT_QUANTILES,
            **pl_module_params,
        )
