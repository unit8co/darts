"""
TiRex: Zero-Shot Forecasting across Long and Short Horizons
-----------------------------------------------------------

Darts wrapper for the pre-trained forecasting model TiRex introduced in
https://arxiv.org/abs/2505.23719.
The implementation is built around `tirex-ts <https://pypi.org/project/tirex-ts/>`.
The TiRex base repo <https://github.com/NX-AI/tirex>,
model card <https://huggingface.co/NX-AI/TiRex> and
docs <https://nx-ai.github.io/tirex/> provide more details.

Note: TiRex is released under the NXAI Community License. See
<https://github.com/NX-AI/tirex/blob/main/LICENSE> for details.
Users must explicitly acknowledge the license by passing `accept_license=True` when
constructing `TiRexModel`.

"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

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
        from tirex import load_model  # type: ignore

    except Exception as e:  # pragma: no cover
        raise_log(
            ImportError(
                "Optional dependency `tirex-ts` is required to use TiRexModel. "
                "Install it with `pip install tirex-ts` (it provides the `tirex` Python package; "
                "extras include `tirex-ts[hfdataset]`, `tirex-ts[gluonts]`)."
            ),
            logger,
        )
        raise e

    return load_model


@dataclass(frozen=True)
class _TiRexQuantiles:
    # TiRex returns 9 quantiles by default (0.1..0.9).
    quantiles: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


class _TiRexModule(PLForecastingModule):
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
        tirex_pipeline,
        all_quantiles: tuple[float, ...],
        **kwargs,
    ):
        # `kwargs` must include PLForecastingModule args (incl. output_chunk_length,
        # output_chunk_shift, likelihood, etc.)
        super().__init__(**kwargs)
        # Do not store the TiRex pipeline inside Lightning hyperparameters/checkpoints.
        # It is re-loadable via `tirex.load_model(...)` and can be large / non-serializable.
        self.save_hyperparameters(ignore=["tirex_pipeline"])

        self.tirex_pipeline = tirex_pipeline
        self.register_buffer(
            "_all_quantiles", torch.tensor(all_quantiles, dtype=torch.float32)
        )

        if self.likelihood is None:
            median_idx = (self._all_quantiles == 0.5).nonzero(as_tuple=True)[0]
            if len(median_idx) != 1:
                raise ValueError(
                    "Expected exactly one median quantile (0.5) in TiRex quantiles."
                )
            self.register_buffer(
                "_user_quantile_indices", median_idx.to(dtype=torch.long)
            )
        else:
            user_q = torch.tensor(self.likelihood.quantiles, dtype=torch.float32)
            indices: list[int] = []
            for q in user_q.tolist():
                matches = (
                    self._all_quantiles == torch.tensor(q, dtype=torch.float32)
                ).nonzero(as_tuple=True)[0]
                if len(matches) != 1:
                    raise ValueError(
                        f"Requested quantile {q} is not available in TiRex quantiles {all_quantiles}."
                    )
                indices.append(int(matches.item()))
            self.register_buffer(
                "_user_quantile_indices", torch.tensor(indices, dtype=torch.long)
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
        # PLModuleInput is typically a tuple: (x_past, x_future, x_static)
        x_past, x_future, _ = x_in

        # TiRex initial integration: no covariates
        if x_future is not None:
            # some datasets may provide an empty tensor; tolerate that
            if not (torch.is_tensor(x_future) and x_future.numel() == 0):
                raise ValueError("TiRexModel does not support future covariates.")

        # x_past: (B, T, C). Enforce univariate.
        if x_past.shape[-1] != 1:
            raise ValueError("TiRexModel currently supports univariate targets only.")

        context = x_past[..., 0]  # (B, T)

        # TiRex should forecast output_chunk_shift + output_chunk_length steps,
        # then we slice away the shift to match Darts' output_chunk_length.
        future_len = self.output_chunk_shift + self.output_chunk_length

        quantiles, mean = self.tirex_pipeline.forecast(
            context=context,
            prediction_length=future_len,
        )

        # Support both numpy arrays and torch tensors (tirex-ts supports both).
        if not torch.is_tensor(quantiles):
            quantiles = torch.as_tensor(quantiles)
        if not torch.is_tensor(mean):
            mean = torch.as_tensor(mean)

        quantiles = quantiles.to(device=context.device, dtype=torch.float32)
        mean = mean.to(device=context.device, dtype=torch.float32)

        # Expect TiRex outputs: quantiles (B, H, Q), mean (B, H)
        if quantiles.ndim != 3:
            raise ValueError(
                f"Unexpected TiRex quantiles shape: {tuple(quantiles.shape)}"
            )
        if mean.ndim != 2:
            raise ValueError(f"Unexpected TiRex mean shape: {tuple(mean.shape)}")

        # slice away output_chunk_shift
        quantiles = quantiles[:, self.output_chunk_shift : future_len, :]

        # select user-requested quantiles (or median)
        idx = self._user_quantile_indices
        if idx.device != quantiles.device:
            idx = idx.to(device=quantiles.device)
        idx = idx.to(dtype=torch.long).contiguous().clone()
        q_sel = quantiles.index_select(dim=-1, index=idx)

        # Darts expects output shape: (B, H, n_targets, n_quantiles)
        return q_sel.unsqueeze(2)


class TiRexModel(FoundationModel):
    # TiRex quantiles returned by default (0.1..0.9)
    _DEFAULT_QUANTILES = _TiRexQuantiles().quantiles
    _MAX_PREDICTION_LENGTH = 2048

    def __init__(
        self,
        input_chunk_length: int,
        output_chunk_length: int,
        output_chunk_shift: int = 0,
        model_name: str = "NX-AI/TiRex",
        likelihood: QuantileRegression | None = None,
        accept_license: bool = False,
        device: str | None = None,
        backend: str | None = None,
        compile: bool | None = None,
        **tirex_kwargs,
    ):
        """
        TiRex foundation model for zero-shot time series forecasting.

        This is a Darts wrapper around the TiRex model introduced in Auer et al. (2025) [1]_. The implementation
        delegates all forecasting logic and weight loading to the optional `tirex-ts` package while exposing a standard
        :class:`TorchForecastingModel` interface.

        TiRex is a pre-trained foundation model designed for zero-shot forecasting across both short and long horizons.

        This model supports either univariate or multivariate time series, but does not support covariates.
        For multivariate time series, the model is applied independently to each component.

        By default, the model is deterministic (median forecast only). To enable probabilistic forecasts, pass a
        :class:`~darts.utils.likelihood_models.torch.QuantileRegression` instance to the ``likelihood`` parameter.
        It is recommended to call :func:`predict()` with ``predict_likelihood_parameters=True`` or ``num_samples >> 1``
        to get meaningful results.

        .. note::
            TiRex is distributed under the `NXAI Community License <https://github.com/NX-AI/tirex/blob/main/LICENSE>`_.
            You must explicitly acknowledge this license by passing ``accept_license=True`` when constructing the model.

        .. warning::
            Fine-tuning is not supported in Darts. Visit `TiRex Docs <https://nx-ai.github.io/tirex/>`_ for details.

        Parameters
        ----------
        model_name
            Identifier passed to `tirex.load_model()`. Default: ``"NX-AI/TiRex"``.
        input_chunk_length
            Number of time steps in the past to take as a model input (per chunk). Applies to the target
            series.
        output_chunk_length
            Number of time steps predicted at once (per chunk) by the internal model. It is not the same as forecast
            horizon `n` used in `predict()`, which is the desired number of prediction points generated using
            either a one-shot- or autoregressive forecast. Setting `n <= output_chunk_length` prevents auto-regression.
            For TiRex, `output_chunk_length + output_chunk_shift` must be less than or equal to 2,048.
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
        device
            Optional device passed to `tirex.load_model()`.
        backend
            Optional backend passed to `tirex.load_model()`.
        compile
            Optional compilation flag passed to `tirex.load_model()`.
        add_encoders
            Optional encoders passed to :class:`FoundationModel`.
        tirex_kwargs
            Additional keyword arguments forwarded to `tirex.load_model()`.

        References
        ----------
        .. [1] A. Auer et al., "TiRex: Zero-Shot Forecasting Across Long and Short Horizons with Enhanced In-Context
        Learning", NeurIPS 2025. https://arxiv.org/abs/2505.23719.

        Examples
        --------
        >>> from darts.models import TiRexModel
        >>> from darts.utils.likelihood_models.torch import QuantileRegression
        >>> from darts.datasets import AirPassengersDataset

        >>> series = AirPassengersDataset().load().astype("float32")
        >>> train, test = series.split_after(0.72)

        >>> model = TiRexModel(
        ...     accept_license=True,
        ... )
        >>> model.fit(train)
        >>> forecast = model.predict(n=len(test), series=train)

        Probabilistic forecasting:

        >>> model = TiRexModel(
        ...     likelihood=QuantileRegression(quantiles=[0.1, 0.5, 0.9]),
        ...     accept_license=True,
        ... )
        >>> model.fit(train)
        >>> forecast = model.predict(n=len(test), series=train, num_samples=50)
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

        if likelihood is not None and not isinstance(likelihood, QuantileRegression):
            raise_log(
                ValueError(
                    "Only QuantileRegression likelihood is supported for TiRexModel."
                ),
                logger,
            )

        # validate that requested quantiles are a subset of TiRex quantiles
        if likelihood is not None:
            req = tuple(float(q) for q in likelihood.quantiles)
            if not set(req).issubset(set(self._DEFAULT_QUANTILES)):
                raise_log(
                    ValueError(
                        "Requested quantiles must be a subset of TiRex quantiles "
                        f"{self._DEFAULT_QUANTILES}."
                    ),
                    logger,
                )

        # Validate chunk lengths. Darts uses these values to build training/prediction batches.
        if input_chunk_length <= 0:
            raise_log(
                ValueError("`input_chunk_length` must be a positive integer."), logger
            )
        if output_chunk_length <= 0:
            raise_log(
                ValueError("`output_chunk_length` must be a positive integer."), logger
            )
        if output_chunk_shift < 0:
            raise_log(
                ValueError("`output_chunk_shift` must be a non-negative integer."),
                logger,
            )

        # TiRex supports up to 2048 steps per single forecast call.
        if output_chunk_length + output_chunk_shift > self._MAX_PREDICTION_LENGTH:
            raise_log(
                ValueError(
                    "TiRex supports a maximum prediction length of "
                    f"{self._MAX_PREDICTION_LENGTH} per call. "
                    "Please ensure `output_chunk_length + output_chunk_shift <= 2048`."
                ),
                logger,
            )

        super().__init__(
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            output_chunk_shift=output_chunk_shift,
            likelihood=likelihood,
        )

        # Make sure these keys exist even if the parent class does not populate them.
        # (TorchForecastingModel accesses them before `fit()`.)
        if self.pl_module_params is None:
            self.pl_module_params = {}
        self.pl_module_params.setdefault("input_chunk_length", input_chunk_length)
        self.pl_module_params.setdefault("output_chunk_length", output_chunk_length)
        self.pl_module_params.setdefault("output_chunk_shift", output_chunk_shift)

        self.model_name = model_name
        self.device = device
        self.backend = backend
        self.compile = compile
        self.tirex_kwargs = tirex_kwargs

    @property
    def supports_multivariate(self) -> bool:
        return False

    @property
    def supports_past_covariates(self) -> bool:
        return False

    @property
    def supports_future_covariates(self) -> bool:
        return False

    def _create_model(self, train_sample) -> PLForecastingModule:
        pl_module_params = self.pl_module_params or {}

        load_model = _require_tirex()

        kwargs = dict(self.tirex_kwargs)
        if self.device is not None:
            kwargs["device"] = self.device
        if self.backend is not None:
            kwargs["backend"] = self.backend
        if self.compile is not None:
            kwargs["compile"] = self.compile

        tirex_pipeline = load_model(self.model_name, **kwargs)

        return _TiRexModule(
            tirex_pipeline=tirex_pipeline,
            all_quantiles=self._DEFAULT_QUANTILES,
            **pl_module_params,
        )

    def fit(
        self,
        series,
        past_covariates=None,
        future_covariates=None,
        val_series=None,
        val_past_covariates=None,
        val_future_covariates=None,
        trainer=None,
        verbose=None,
        epochs: int = 0,
        max_samples_per_ts=None,
        dataloader_kwargs=None,
        sample_weight=None,
        val_sample_weight=None,
        stride: int = 1,
        load_best: bool = False,
    ):
        # enforce initial integration constraints early
        if (
            past_covariates is not None
            or future_covariates is not None
            or val_past_covariates is not None
            or val_future_covariates is not None
        ):
            raise_log(
                ValueError("TiRexModel currently does not support covariates."), logger
            )

        # univariate-only
        series_list = [series] if not isinstance(series, Sequence) else list(series)
        if any(s.n_components != 1 for s in series_list):
            raise_log(
                ValueError("TiRexModel currently supports univariate series only."),
                logger,
            )

        if val_series is not None:
            val_series_list = (
                [val_series]
                if not isinstance(val_series, Sequence)
                else list(val_series)
            )
            if any(s.n_components != 1 for s in val_series_list):
                raise_log(
                    ValueError(
                        "TiRexModel currently supports univariate validation series only."
                    ),
                    logger,
                )

        return super().fit(
            series=series,
            past_covariates=None,
            future_covariates=None,
            val_series=val_series,
            val_past_covariates=None,
            val_future_covariates=None,
            trainer=trainer,
            verbose=verbose,
            epochs=epochs,
            max_samples_per_ts=max_samples_per_ts,
            dataloader_kwargs=dataloader_kwargs,
            sample_weight=sample_weight,
            val_sample_weight=val_sample_weight,
            stride=stride,
            load_best=load_best,
        )
