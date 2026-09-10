"""
Torch-free ONNX inference utilities.

Load a companion ``*.onnx.spec.json`` into :class:`OnnxModelSpec`, then either
run a full horizon with :func:`run_onnx_prediction` or drive a custom loop with
:func:`prepare_onnx_inputs` and :func:`extract_point_forecast`.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from darts import TimeSeries

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


@dataclass
class OnnxModelSpec:
    """Metadata describing an exported ONNX graph and its forecasting windows.

    Written next to the ``.onnx`` file as ``*.onnx.spec.json``. Feature names
    match module fields (``past_target``, optional covariates). Recurrent graphs
    add flattened ``state_in_*`` / ``state_out_*`` tensors.

    Parameters
    ----------
    input_chunk_length
        Target history consumed per inference window.
    output_chunk_length
        Steps produced per graph call (``1`` for stepwise RNN cells).
    uses_past_covariates
        Whether the model was fitted with past covariates.
    uses_future_covariates
        Whether the model was fitted with future covariates.
    uses_static_covariates
        Whether the model was fitted with static covariates.
    feature_input_names
        Graph inputs that are features (excludes ``state_in_*``).
    input_names
        All graph input names, including state.
    output_names
        All graph output names (``prediction`` plus ``state_out_*``).
    state_input_names
        Flattened recurrent state inputs, if any.
    state_output_names
        Flattened recurrent state outputs, if any.
    state_input_shapes
        Shapes of ``state_input_names`` (zeros on the first step).
    stepwise_state
        If ``True``, the graph is a 1-step cell that must be warmed up over
        ``input_chunk_length`` before forecasting (e.g. ``RNNModel``).
    """

    input_chunk_length: int
    output_chunk_length: int
    uses_past_covariates: bool
    uses_future_covariates: bool
    uses_static_covariates: bool
    feature_input_names: list[str]
    input_names: list[str]
    output_names: list[str]
    state_input_names: list[str] = field(default_factory=list)
    state_output_names: list[str] = field(default_factory=list)
    state_input_shapes: list[list[int]] | None = None
    stepwise_state: bool = False

    @classmethod
    def from_model(cls, model: Any) -> OnnxModelSpec:
        """Build a spec from a fitted :class:`~darts.models.forecasting.torch_forecasting_model.TorchForecastingModel`.

        Uses covariate flags and chunk lengths from ``model``. Does not inspect
        an ONNX session, so recurrent ``state_*`` fields stay empty.

        Parameters
        ----------
        model
            A fitted torch forecasting model.

        Returns
        -------
        OnnxModelSpec
            Feature-only spec (no ``state_*`` names).
        """
        feature_names = ["past_target"]
        if model.uses_past_covariates:
            feature_names.append("past_covariates")
        if model.uses_future_covariates:
            feature_names.extend(["historic_future_covariates", "future_covariates"])
        if model.uses_static_covariates:
            feature_names.append("static_covariates")
        return cls(
            input_chunk_length=model.input_chunk_length,
            output_chunk_length=model.output_chunk_length,
            uses_past_covariates=model.uses_past_covariates,
            uses_future_covariates=model.uses_future_covariates,
            uses_static_covariates=model.uses_static_covariates,
            feature_input_names=feature_names,
            input_names=list(feature_names),
            output_names=["prediction"],
        )

    @classmethod
    def from_session(
        cls,
        session: Any,
        *,
        input_chunk_length: int,
        output_chunk_length: int,
        uses_past_covariates: bool = False,
        uses_future_covariates: bool = False,
        uses_static_covariates: bool = False,
    ) -> OnnxModelSpec:
        """Build a spec from an ONNX Runtime session and known hyperparameters.

        Discovers input/output names and treats ``state_in_*`` / ``state_out_*``
        as recurrent state. Sets ``stepwise_state`` when state is present and
        ``past_target`` has time dimension ``1``.

        Parameters
        ----------
        session
            An ONNX Runtime ``InferenceSession``.
        input_chunk_length
            Target history length used when the model was trained.
        output_chunk_length
            Forecast steps produced per graph call.
        uses_past_covariates
            Whether past covariates are required.
        uses_future_covariates
            Whether future covariates are required.
        uses_static_covariates
            Whether static covariates are required.

        Returns
        -------
        OnnxModelSpec
            Spec including any recurrent state I/O found on the session.
        """
        input_names = [inp.name for inp in session.get_inputs()]
        output_names = [out.name for out in session.get_outputs()]
        state_input_names = [
            name for name in input_names if name.startswith("state_in_")
        ]
        state_output_names = [
            name for name in output_names if name.startswith("state_out_")
        ]
        feature_input_names = [
            name for name in input_names if not name.startswith("state_in_")
        ]
        state_input_shapes = []
        for name in state_input_names:
            for inp in session.get_inputs():
                if inp.name == name:
                    state_input_shapes.append([
                        dim if isinstance(dim, int) else 1 for dim in inp.shape
                    ])
        past_target_time = None
        for inp in session.get_inputs():
            if inp.name == "past_target" and len(inp.shape) >= 2:
                past_target_time = inp.shape[1]
                break
        stepwise_state = bool(state_input_names) and past_target_time == 1
        return cls(
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            uses_past_covariates=uses_past_covariates,
            uses_future_covariates=uses_future_covariates,
            uses_static_covariates=uses_static_covariates,
            feature_input_names=feature_input_names,
            input_names=input_names,
            output_names=output_names,
            state_input_names=state_input_names,
            state_output_names=state_output_names,
            state_input_shapes=state_input_shapes or None,
            stepwise_state=stepwise_state,
        )

    def save_json(self, path: str | Path) -> None:
        """Write the spec to a JSON file.

        Parameters
        ----------
        path
            Destination path (typically ``<model>.onnx.spec.json``).
        """
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load_json(cls, path: str | Path) -> OnnxModelSpec:
        """Load a spec from a JSON file.

        Parameters
        ----------
        path
            Path produced by :meth:`save_json`.

        Returns
        -------
        OnnxModelSpec
            The deserialized spec.
        """
        with open(path, encoding="utf-8") as f:
            return cls(**json.load(f))


def prepare_onnx_inputs(
    series: TimeSeries,
    spec: OnnxModelSpec,
    past_covariates: TimeSeries | None = None,
    future_covariates: TimeSeries | None = None,
    *,
    future_horizon: int | None = None,
) -> dict[str, np.ndarray]:
    """Slice input features for a single ONNX inference window.

    Returns a dictionary mapping ONNX input names to numpy arrays with a leading
    batch dimension. Only includes feature inputs present in ``spec``.
    Recurrent ``state_*`` inputs must be provided separately (or via
    :func:`run_onnx_prediction`).

    Parameters
    ----------
    series
        Target series; the last ``input_chunk_length`` values become
        ``past_target``.
    spec
        Graph / window metadata.
    past_covariates
        Past covariates aligned with ``series``, if used.
    future_covariates
        Future covariates covering the historic window and ``future_horizon``.
    future_horizon
        How many future covariate steps to include. Defaults to
        ``output_chunk_length``.

    Returns
    -------
    dict[str, np.ndarray]
        Feature tensors keyed by ONNX input name, each with a batch dimension.
    """
    ocl = spec.output_chunk_length
    horizon = future_horizon if future_horizon is not None else ocl

    past_start = series.end_time() - (spec.input_chunk_length - 1) * series.freq
    past_end = series.end_time()
    future_start = past_end + 1 * series.freq
    future_end = past_end + horizon * series.freq

    inputs: dict[str, np.ndarray] = {}
    dtype = series.dtype

    if "past_target" in spec.feature_input_names:
        inputs["past_target"] = np.expand_dims(
            series[past_start:past_end].values(), axis=0
        ).astype(dtype)

    if spec.uses_past_covariates and past_covariates is not None:
        if "past_covariates" in spec.feature_input_names:
            inputs["past_covariates"] = np.expand_dims(
                past_covariates[past_start:past_end].values(), axis=0
            ).astype(dtype)

    if spec.uses_future_covariates and future_covariates is not None:
        if "historic_future_covariates" in spec.feature_input_names:
            inputs["historic_future_covariates"] = np.expand_dims(
                future_covariates[past_start:past_end].values(), axis=0
            ).astype(dtype)
        if "future_covariates" in spec.feature_input_names:
            inputs["future_covariates"] = np.expand_dims(
                future_covariates[future_start:future_end].values(), axis=0
            ).astype(dtype)

    if spec.uses_static_covariates and series.has_static_covariates:
        if "static_covariates" in spec.feature_input_names:
            inputs["static_covariates"] = np.expand_dims(
                series.static_covariates_values(), axis=0
            ).astype(dtype)

    return inputs


def _extract_future_past_covariates(
    past_covariates: TimeSeries,
    series: TimeSeries,
    horizon: int,
) -> np.ndarray | None:
    """Future segment of past covariates for auto-regressive window updates."""
    if horizon <= 0:
        return None
    past_end = series.end_time()
    future_start = past_end + 1 * series.freq
    future_end = past_end + horizon * series.freq
    return np.expand_dims(
        past_covariates[future_start:future_end].values(), axis=0
    ).astype(series.dtype)


def _np_roll(arr: np.ndarray, shift: int, axis: int) -> np.ndarray:
    """NumPy equivalent of ``torch.roll`` used in ``_get_batch_prediction``."""
    return np.roll(arr, -shift, axis=axis)


def extract_point_forecast(ort_outputs: Sequence[np.ndarray]) -> np.ndarray:
    """Extract a point forecast from raw ONNX output tensors.

    Drops the batch dimension. If the prediction is
    ``(batch, time, components, n_likelihood_params)``, keeps the first
    likelihood parameter (e.g. Gaussian :math:`\\mu`).

    Parameters
    ----------
    ort_outputs
        Sequence returned by ``session.run``; ``ort_outputs[0]`` is
        ``prediction``.

    Returns
    -------
    np.ndarray
        Array of shape ``(time, components)``.
    """
    pred = np.asarray(ort_outputs[0])
    if pred.ndim == 4:
        pred = pred[0, ..., 0]
    elif pred.ndim == 3:
        pred = pred[0]
    if pred.ndim == 1:
        pred = pred[:, np.newaxis]
    return pred


def _squeeze_prediction_batch(pred: np.ndarray) -> np.ndarray:
    """Convert raw ONNX prediction to ``(batch, time, components)``.

    4-D outputs keep the first likelihood parameter (``[..., 0]``).
    """
    pred = np.asarray(pred)
    if pred.ndim == 4:
        return pred[..., 0]
    if pred.ndim == 2:
        return pred[:, np.newaxis, :]
    return pred


def _zero_state(spec: OnnxModelSpec, dtype) -> list[np.ndarray]:
    """Allocate zero tensors matching ``spec.state_input_shapes``."""
    if spec.state_input_shapes is None:
        raise ValueError(
            "`state_input_shapes` must be set on `OnnxModelSpec` for recurrent models."
        )
    return [np.zeros(shape, dtype=dtype) for shape in spec.state_input_shapes]


def _run_onnx_step(
    session: Any,
    spec: OnnxModelSpec,
    feature_arrays: Mapping[str, np.ndarray],
    state: list[np.ndarray] | None = None,
) -> tuple[np.ndarray, list[np.ndarray] | None]:
    """Run one ONNX graph call and return the squeezed prediction plus next state.

    Missing recurrent state is initialized to zeros (same as ``hx=None``).
    """
    session_input_names = [inp.name for inp in session.get_inputs()]
    ort_inputs = {
        name: feature_arrays[name]
        for name in spec.feature_input_names
        if name in feature_arrays and name in session_input_names
    }
    if spec.state_input_names:
        if state is None:
            state = _zero_state(spec, next(iter(feature_arrays.values())).dtype)
        for name, value in zip(spec.state_input_names, state):
            if name in session_input_names:
                ort_inputs[name] = value
    outputs = session.run(spec.output_names, ort_inputs)
    prediction = _squeeze_prediction_batch(outputs[0])
    next_state = None
    if spec.state_output_names:
        output_by_name = dict(zip(spec.output_names, outputs))
        next_state = [output_by_name[name] for name in spec.state_output_names]
    return prediction, next_state


def _format_onnx_forecast(batch_predictions: list[np.ndarray], n: int) -> np.ndarray:
    """Concatenate auto-regressive chunks and trim to ``(n, components, 1)``."""
    predictions = np.concatenate(batch_predictions, axis=1)
    result = predictions[0, :n, :]
    if result.ndim == 2:
        result = result[:, :, np.newaxis]
    return result


def run_onnx_prediction(
    n: int,
    session: Any,
    spec: OnnxModelSpec,
    series: TimeSeries,
    past_covariates: TimeSeries | None = None,
    future_covariates: TimeSeries | None = None,
    *,
    roll_size: int | None = None,
) -> np.ndarray:
    """Run ONNX inference for ``n`` steps after ``series`` end.

    Mirrors
    :meth:`~darts.models.forecasting.pl_forecasting_module.PLForecastingModule._get_batch_prediction`
    for deterministic models with ``num_samples=1``.

    Feed-forward graphs receive full windows and roll them when
    ``n > output_chunk_length``. Stepwise recurrent graphs
    (``spec.stepwise_state``) are 1-step cells: this warms up over
    ``input_chunk_length`` historical targets, then continues in the same
    auto-regressive loop using the last prediction as the next ``past_target``.

    Likelihood models export raw distribution parameters. This helper uses the
    first parameter as the point forecast (e.g. Gaussian :math:`\\mu`) and feeds
    it back when auto-regressing. That is not the same as ``predict()``, which
    samples. For ``n <= output_chunk_length``, compare against
    ``predict(predict_likelihood_parameters=True)`` (first parameter).

    Reversible instance norm is inside the exported graph, so norm / denorm
    match torch ``predict()``.

    Parameters
    ----------
    n
        Forecast horizon (steps after ``series`` end).
    session
        An ONNX Runtime ``InferenceSession`` for the exported graph.
    spec
        Graph / window metadata, typically from ``*.onnx.spec.json``.
    series
        Target series; the last ``input_chunk_length`` values are consumed.
    past_covariates
        Past covariates aligned with ``series``, if the model uses them.
    future_covariates
        Future covariates covering the historic window and the horizon, if used.
    roll_size
        Predicted steps committed per auto-regressive iteration. Defaults to
        ``output_chunk_length``. Forced to ``1`` for stepwise graphs.

    Returns
    -------
    np.ndarray
        Point forecast of shape ``(n, n_components, 1)``, matching
        :meth:`~darts.timeseries.TimeSeries.all_values` for a deterministic
        series.
    """
    if roll_size is None:
        roll_size = spec.output_chunk_length
    elif not 0 < roll_size <= spec.output_chunk_length:
        raise ValueError(
            "`roll_size` must be an integer between 1 and `output_chunk_length`."
        )

    icl = spec.input_chunk_length
    ocl = spec.output_chunk_length
    min_n = n if n >= ocl else ocl

    horizon = min_n + ocl
    feature_arrays = prepare_onnx_inputs(
        series=series,
        spec=spec,
        past_covariates=past_covariates,
        future_covariates=future_covariates,
        future_horizon=horizon,
    )

    past_target = feature_arrays["past_target"]
    past_covariates_arr = feature_arrays.get("past_covariates")
    historic_future_covariates = feature_arrays.get("historic_future_covariates")
    future_covariates_arr = feature_arrays.get("future_covariates")
    static_covariates = feature_arrays.get("static_covariates")

    def _feature_dict(
        future_cov_slice: np.ndarray | None,
        overrides: Mapping[str, np.ndarray] | None = None,
    ) -> dict[str, np.ndarray]:
        arrays = {
            "past_target": past_target,
            "past_covariates": past_covariates_arr,
            "historic_future_covariates": historic_future_covariates,
            "future_covariates": future_cov_slice,
            "static_covariates": static_covariates,
        }
        if overrides:
            arrays.update(overrides)
        return {
            name: value
            for name in spec.feature_input_names
            if (value := arrays[name]) is not None
        }

    state = None
    update_historic = True
    if spec.stepwise_state:

        def _warmup_future_cov(t: int) -> np.ndarray | None:
            if historic_future_covariates is not None and t + 1 < icl:
                return historic_future_covariates[:, t + 1 : t + 2, :]
            if future_covariates_arr is not None:
                idx = t + 1 - icl
                return future_covariates_arr[:, idx : idx + 1, :]
            return None

        # encode history into state, one target step at a time
        pred = None
        for t in range(icl):
            overrides: dict[str, np.ndarray] = {
                "past_target": past_target[:, t : t + 1, :],
            }
            if past_covariates_arr is not None:
                overrides["past_covariates"] = past_covariates_arr[:, t : t + 1, :]
            if historic_future_covariates is not None:
                overrides["historic_future_covariates"] = historic_future_covariates[
                    :, t : t + 1, :
                ]
            pred, state = _run_onnx_step(
                session,
                spec,
                _feature_dict(_warmup_future_cov(t), overrides),
                state=state,
            )

        if pred is None:
            raise ValueError(
                "`input_chunk_length` must be >= 1 for stepwise ONNX inference."
            )

        # remaining steps use the last prediction and last historic covariate pads
        out = pred
        past_target = pred[:, -1:, :]
        if past_covariates_arr is not None:
            past_covariates_arr = past_covariates_arr[:, -1:, :]
        if historic_future_covariates is not None:
            historic_future_covariates = historic_future_covariates[:, -1:, :]
        future_past_covariates = None
        update_historic = False
        roll_size = 1
        batch_predictions = [out]
        prediction_length = roll_size
    else:
        future_past_covariates = None
        if spec.uses_past_covariates and past_covariates is not None and n > ocl:
            future_past_covariates = _extract_future_past_covariates(
                past_covariates, series, min_n
            )

        future_cov_slice = (
            future_covariates_arr[:, :roll_size, :]
            if future_covariates_arr is not None
            else None
        )
        out, state = _run_onnx_step(session, spec, _feature_dict(future_cov_slice))
        batch_predictions = [out[:, :roll_size, :]]
        prediction_length = roll_size

    while prediction_length < min_n:
        if prediction_length + ocl > min_n:
            spillover = prediction_length + ocl - min_n
            roll_size -= spillover
            prediction_length -= spillover
            batch_predictions[-1] = batch_predictions[-1][:, :roll_size, :]

        past_target = _np_roll(past_target, roll_size, axis=1)
        if past_covariates_arr is not None:
            past_covariates_arr = _np_roll(past_covariates_arr, roll_size, axis=1)
        if historic_future_covariates is not None:
            historic_future_covariates = _np_roll(
                historic_future_covariates, roll_size, axis=1
            )

        if icl >= roll_size:
            past_target[:, -roll_size:, :] = out[:, :roll_size, :]
        else:
            past_target[:, :, :] = out[:, -icl:, :]

        if icl >= roll_size:
            left_past, right_past = prediction_length - roll_size, prediction_length
        else:
            left_past, right_past = prediction_length - icl, prediction_length

        if past_covariates_arr is not None and future_past_covariates is not None:
            if icl >= roll_size:
                past_covariates_arr[:, -roll_size:, :] = future_past_covariates[
                    :, left_past:right_past, :
                ]
            else:
                past_covariates_arr[:, :, :] = future_past_covariates[
                    :, left_past:right_past, :
                ]

        if (
            update_historic
            and historic_future_covariates is not None
            and future_covariates_arr is not None
        ):
            if icl >= roll_size:
                historic_future_covariates[:, -roll_size:, :] = future_covariates_arr[
                    :, left_past:right_past, :
                ]
            else:
                historic_future_covariates[:, :, :] = future_covariates_arr[
                    :, left_past:right_past, :
                ]

        left_future = right_past
        right_future = right_past + ocl
        future_cov_slice = (
            future_covariates_arr[:, left_future:right_future, :]
            if future_covariates_arr is not None
            else None
        )

        out, state = _run_onnx_step(
            session,
            spec,
            _feature_dict(future_cov_slice),
            state=state,
        )
        batch_predictions.append(out)
        prediction_length += ocl

    return _format_onnx_forecast(batch_predictions, n)
