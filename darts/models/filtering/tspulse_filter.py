"""
TSPulse Filter
--------------
"""

import os
from numbers import Integral
from typing import Any

import numpy as np
import torch

from darts import TimeSeries
from darts.logging import raise_log
from darts.models.filtering.filtering_model import FilteringModel
from darts.utils.utils import _maybe_cast_array_dtype


class TSPulseFilter(FilteringModel):
    """Zero-shot time-series reconstruction and imputation with TSPulse.

    This filter wraps IBM Granite's pretrained `TSPulse-R1 model
    <https://huggingface.co/ibm-granite/granite-timeseries-tspulse-r1>`_.
    :meth:`filter` reconstructs all values in a deterministic series. The
    reconstructed series can be combined with
    :class:`~darts.ad.anomaly_model.FilteringAnomalyModel` and a prediction-based
    scorer to produce simple time-domain reconstruction-residual anomaly scores.
    :meth:`impute` only replaces missing values and leaves observed values unchanged.

    The optional ``granite-tsfm`` package (Python 3.11 through 3.13) is required
    when ``model`` is not supplied. TSPulse is loaded lazily on the first call
    because its number of input channels is determined by the width of the input
    series.

    Parameters
    ----------
    hub_model_name
        Hugging Face model identifier or local model directory.
    hub_model_revision
        Model revision to load. The default is the revision used by IBM's zero-shot
        imputation example.
    model
        An optional pre-loaded ``TSPulseForReconstruction``-compatible model. This is
        mainly useful for offline use and testing. The model must expose
        ``config.context_length`` and return ``reconstruction_outputs``. A supplied
        upstream TSPulse model must have ``config.mask_type == "user"`` so that it
        uses the missing-value mask passed by this filter. Reload upstream models
        with ``TSPulseForReconstruction.from_pretrained(..., mask_type="user")``.
    batch_size
        Number of windows reconstructed per model call.
    stride
        Distance between consecutive context windows. If ``None``, non-overlapping
        windows are used, with one overlapping final window when needed. Smaller
        values produce more overlapping reconstructions at a higher inference cost.
        Must not exceed the model context length.
    device
        Torch device used for inference. Defaults to ``"cpu"``.
    model_kwargs
        Additional keyword arguments passed to
        ``TSPulseForReconstruction.from_pretrained()``. ``num_input_channels``,
        ``mask_type``, ``return_dict``, ``dtype``, and ``torch_dtype`` are managed by
        this wrapper and cannot be overridden.

    Notes
    -----
    This initial integration supports zero-shot reconstruction and imputation with
    IBM's dual-head imputation checkpoint. It does not reproduce IBM's official
    patchwise-masked, all-head anomaly-detection pipeline. Fine-tuning and the model's
    frequency/forecast anomaly-score ensemble are not included. Input series must
    contain at least the model's context length (512 values for the default
    checkpoint), as required by the upstream imputation model. Only float32 TSPulse
    checkpoints are supported. Every component must contain at least one observed
    value in every context window so the model can determine its location and scale.
    """

    def __init__(
        self,
        hub_model_name: str | os.PathLike = (
            "ibm-granite/granite-timeseries-tspulse-r1"
        ),
        hub_model_revision: str | None = "tspulse-hybrid-dualhead-512-p8-r1",
        model: Any | None = None,
        batch_size: int = 32,
        stride: int | None = None,
        device: str | torch.device | None = None,
        model_kwargs: dict[str, Any] | None = None,
    ):
        super().__init__()

        if (
            not isinstance(batch_size, Integral)
            or isinstance(batch_size, bool)
            or batch_size <= 0
        ):
            raise_log(ValueError("`batch_size` must be a positive integer."))
        if stride is not None and (
            not isinstance(stride, Integral) or isinstance(stride, bool) or stride <= 0
        ):
            raise_log(ValueError("`stride` must be a positive integer or `None`."))

        model_kwargs = dict(model_kwargs or {})
        reserved_kwargs = {
            "num_input_channels",
            "mask_type",
            "return_dict",
            "dtype",
            "torch_dtype",
        }.intersection(model_kwargs)
        if reserved_kwargs:
            reserved = ", ".join(sorted(reserved_kwargs))
            raise_log(
                ValueError(
                    f"The following `model_kwargs` are managed by TSPulseFilter: {reserved}."
                )
            )

        self.hub_model_name = str(hub_model_name)
        self.hub_model_revision = hub_model_revision
        self.batch_size = int(batch_size)
        self.stride = int(stride) if stride is not None else None
        self.device = torch.device(device or "cpu")
        self.model_kwargs = model_kwargs

        self._model = model
        self._model_was_supplied = model is not None
        self._model_width: int | None = None

    def __str__(self) -> str:
        return f"TSPulseFilter(model={self.hub_model_name})"

    def filter(self, series: TimeSeries) -> TimeSeries:
        """Reconstruct every value in ``series`` with TSPulse.

        Parameters
        ----------
        series
            Deterministic univariate or multivariate series to reconstruct. NaN values
            are passed to TSPulse as unobserved values. Infinite values are not
            supported. The series must contain at least ``config.context_length``
            values, and each component must have at least one observed value in every
            context window.

        Returns
        -------
        TimeSeries
            A deterministic reconstruction with the same time index, components, and
            metadata as ``series``.
        """
        super().filter(series)
        values = series.values(copy=False)
        if np.isinf(values).any():
            raise_log(ValueError("TSPulseFilter does not support infinite values."))
        model = self._get_model(series.width)
        context_length = self._context_length(model)
        if len(series) < context_length:
            raise_log(
                ValueError(
                    "TSPulseFilter requires at least the model context length "
                    f"({context_length}) of input values, found {len(series)}."
                )
            )

        model_dtype = self._model_dtype(model)
        if model_dtype != torch.float32:
            raise_log(
                ValueError(
                    "TSPulseFilter only supports float32 models, found "
                    f"{str(model_dtype).removeprefix('torch.')}."
                )
            )
        finite_values = values[~np.isnan(values)]
        dtype_max = torch.finfo(model_dtype).max
        if np.any((finite_values > dtype_max) | (finite_values < -dtype_max)):
            dtype_name = str(model_dtype).removeprefix("torch.")
            raise_log(
                ValueError(
                    "TSPulseFilter input values must be within the finite "
                    f"{dtype_name} range used by the model."
                )
            )

        stride = context_length if self.stride is None else self.stride
        if stride > context_length:
            raise_log(
                ValueError(
                    f"`stride` must not exceed the model context length "
                    f"({context_length}), found {stride}."
                )
            )

        starts = self._window_starts(len(series), context_length, stride)
        self._validate_observed_context(
            values=values,
            starts=starts,
            context_length=context_length,
            components=series.components,
        )
        reconstructed = np.zeros((len(series), series.width), dtype=np.float64)
        counts = np.zeros((len(series), series.width), dtype=np.int64)

        model.eval()
        with torch.inference_mode():
            for batch_start in range(0, len(starts), self.batch_size):
                batch_starts = starts[batch_start : batch_start + self.batch_size]
                past_values, observed_mask, valid_lengths = self._make_batch(
                    values=values,
                    starts=batch_starts,
                    context_length=context_length,
                )
                output = model(
                    past_values=past_values.to(
                        device=self.device,
                        dtype=model_dtype,
                    ),
                    past_observed_mask=observed_mask.to(self.device),
                    return_loss=False,
                    return_dict=True,
                )
                batch_reconstruction = self._reconstruction_values(output)

                expected_shape = tuple(past_values.shape)
                if tuple(batch_reconstruction.shape) != expected_shape:
                    raise_log(
                        ValueError(
                            "TSPulse returned reconstruction shape "
                            f"{tuple(batch_reconstruction.shape)}, expected {expected_shape}."
                        )
                    )

                batch_reconstruction = (
                    batch_reconstruction.detach().cpu().to(dtype=torch.float64).numpy()
                )
                for window, start, valid_length in zip(
                    batch_reconstruction, batch_starts, valid_lengths
                ):
                    stop = start + valid_length
                    reconstructed[start:stop] += window[:valid_length]
                    counts[start:stop] += 1

        if np.any(counts == 0):
            raise_log(RuntimeError("TSPulse windowing left values unreconstructed."))
        reconstructed /= counts

        return TimeSeries(
            times=series.time_index,
            values=_maybe_cast_array_dtype(reconstructed, series.dtype),
            components=series.components,
            copy=False,
            **series._attrs,
        )

    def impute(self, series: TimeSeries) -> TimeSeries:
        """Replace only NaN values in ``series`` with TSPulse reconstructions.

        Observed values are copied exactly. If the input contains no NaN values, a copy
        is returned without loading or running TSPulse.

        Parameters
        ----------
        series
            Deterministic univariate or multivariate series to impute. When missing
            values are present, the series must contain at least
            ``config.context_length`` values, and each component must have at least
            one observed value in every context window.

        Returns
        -------
        TimeSeries
            An imputed series with the same time index, components, and metadata as
            ``series``.
        """
        super().filter(series)
        values = series.values(copy=False)
        if np.isinf(values).any():
            raise_log(ValueError("TSPulseFilter does not support infinite values."))
        missing = np.isnan(values)
        if not missing.any():
            return series.copy()

        reconstructed = self.filter(series).values(copy=False)
        imputed = np.where(missing, reconstructed, values)
        return TimeSeries(
            times=series.time_index,
            values=_maybe_cast_array_dtype(imputed, series.dtype),
            components=series.components,
            copy=False,
            **series._attrs,
        )

    def _get_model(self, width: int):
        if self._model is not None and self._model_width in (None, width):
            configured_width = getattr(self._model.config, "num_input_channels", width)
            if configured_width != width:
                raise_log(
                    ValueError(
                        f"The supplied TSPulse model expects {configured_width} input "
                        f"channels, but the series has width {width}."
                    )
                )
            mask_type = getattr(self._model.config, "mask_type", None)
            if mask_type is not None and mask_type != "user":
                raise_log(
                    ValueError(
                        "A supplied upstream TSPulse model must be configured with "
                        "`config.mask_type='user'` so it uses the missing-value "
                        "mask. Reload it with "
                        "`TSPulseForReconstruction.from_pretrained(..., "
                        "mask_type='user')`."
                    )
                )
            self._model_width = width
            return self._prepare_model(self._model)

        if self._model_was_supplied:
            raise_log(
                ValueError(
                    f"The supplied TSPulse model was initialized for width "
                    f"{self._model_width}, but the series has width {width}."
                )
            )

        try:
            from tsfm_public.models.tspulse import TSPulseForReconstruction
        except ImportError as exc:
            raise ModuleNotFoundError(
                "TSPulseFilter requires the optional `granite-tsfm>=0.3.6` "
                "dependency on Python 3.11 through 3.13. Install it with "
                "`pip install granite-tsfm`."
            ) from exc

        load_kwargs = {
            "num_input_channels": width,
            "mask_type": "user",
            **self.model_kwargs,
        }
        if self.hub_model_revision is not None:
            load_kwargs["revision"] = self.hub_model_revision
        self._model = TSPulseForReconstruction.from_pretrained(
            self.hub_model_name,
            **load_kwargs,
        )
        self._model_width = width
        return self._prepare_model(self._model)

    def _prepare_model(self, model):
        return model.to(self.device)

    @staticmethod
    def _model_dtype(model) -> torch.dtype:
        for parameter in model.parameters():
            if parameter.is_floating_point():
                return parameter.dtype
        for buffer in model.buffers():
            if buffer.is_floating_point():
                return buffer.dtype
        return torch.float32

    @staticmethod
    def _context_length(model) -> int:
        context_length = getattr(model.config, "context_length", None)
        if (
            not isinstance(context_length, Integral)
            or isinstance(context_length, bool)
            or context_length <= 0
        ):
            raise_log(
                ValueError(
                    "The TSPulse model must define a positive integer "
                    "`config.context_length`."
                )
            )
        return int(context_length)

    @staticmethod
    def _window_starts(
        series_length: int, context_length: int, stride: int
    ) -> list[int]:
        if series_length <= context_length:
            return [0]
        final_start = series_length - context_length
        starts = list(range(0, final_start + 1, stride))
        if starts[-1] != final_start:
            starts.append(final_start)
        return starts

    @staticmethod
    def _validate_observed_context(
        values: np.ndarray,
        starts: list[int],
        context_length: int,
        components,
    ) -> None:
        for start in starts:
            stop = min(start + context_length, len(values))
            has_observed_context = np.any(~np.isnan(values[start:stop]), axis=0)
            if not np.all(has_observed_context):
                missing_components = ", ".join(
                    repr(str(component))
                    for component in components[~has_observed_context]
                )
                raise_log(
                    ValueError(
                        "TSPulseFilter requires observed context for every component "
                        "in every context window. Component(s) "
                        f"{missing_components} have no observed values in the window "
                        f"at positions [{start}, {stop}). Provide at least one "
                        "observed value for each component/window before filtering or "
                        "imputation."
                    )
                )

    @staticmethod
    def _make_batch(
        values: np.ndarray,
        starts: list[int],
        context_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        batch = np.zeros(
            (len(starts), context_length, values.shape[1]), dtype=values.dtype
        )
        observed = np.zeros_like(batch, dtype=bool)
        valid_lengths = []

        for idx, start in enumerate(starts):
            valid_length = min(context_length, len(values) - start)
            window = values[start : start + valid_length]
            window_observed = ~np.isnan(window)
            batch[idx, :valid_length] = np.nan_to_num(window, nan=0.0)
            observed[idx, :valid_length] = window_observed
            valid_lengths.append(valid_length)

        return torch.from_numpy(batch), torch.from_numpy(observed), valid_lengths

    @staticmethod
    def _reconstruction_values(output) -> torch.Tensor:
        if isinstance(output, dict):
            reconstruction = output.get("reconstruction_outputs")
        else:
            reconstruction = getattr(output, "reconstruction_outputs", None)
        if reconstruction is None:
            raise_log(
                ValueError("TSPulse output must contain `reconstruction_outputs`.")
            )
        if not isinstance(reconstruction, torch.Tensor):
            raise_log(
                ValueError(
                    "TSPulse `reconstruction_outputs` must be a `torch.Tensor`, "
                    f"found {type(reconstruction).__name__}."
                )
            )
        return reconstruction
