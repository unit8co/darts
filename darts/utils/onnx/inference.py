"""
ONNX Inference
--------------

Torch-free ONNX inference utilities: Export a torch model to ONNX format and then:

- run torch-free forecasting with :func:`run_onnx_prediction`
- or drive a custom / advanced loop with :func:`prepare_onnx_inputs`
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from darts import TimeSeries
from darts.logging import raise_log
from darts.typing import TimeSeriesLike
from darts.utils.likelihood_models.base import Likelihood, LikelihoodType
from darts.utils.timeseries_generation import _build_forecast_series_from_schema
from darts.utils.ts_utils import get_series_seq_type, series2seq
from darts.utils.utils import _build_tqdm_iterator

ONNX_SPEC_METADATA_KEY = "darts.onnx_spec"


@dataclass
class OnnxModelSpec:
    """Metadata describing an exported ONNX graph and its forecasting windows.

    Embedded in the exported ``.onnx`` file under ``darts.onnx_spec``.
    Feature names match module fields (``past_target``, optional covariates).
    Recurrent graphs add flattened ``state_in_*`` / ``state_out_*`` tensors.

    Parameters
    ----------
    input_chunk_length
        Target history consumed per inference window.
    output_chunk_length
        Steps produced per graph call (``1`` for stepwise RNN cells).
    output_chunk_shift
        Steps that inference start is shifted into the future.
    uses_past_covariates
        Whether the model was fitted with past covariates.
    uses_future_covariates
        Whether the model was fitted with future covariates.
    uses_static_covariates
        Whether the model was fitted with static covariates.
    likelihood_parameter_names
        The likelihood parameter names if the model was trained with a
        likelihood.
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
    output_chunk_shift: int
    uses_past_covariates: bool
    uses_future_covariates: bool
    uses_static_covariates: bool
    likelihood_parameter_names: list[str] | None
    feature_input_names: list[str]
    input_names: list[str]
    output_names: list[str]
    state_input_names: list[str] = field(default_factory=list)
    state_output_names: list[str] = field(default_factory=list)
    state_input_shapes: list[list[int]] | None = None
    stepwise_state: bool = False

    @classmethod
    def from_session(cls, session: Any) -> OnnxModelSpec:
        """Load the spec embedded in an ONNX Runtime inference session.

        Parameters
        ----------
        session
            An ONNX Runtime ``InferenceSession`` for a model exported with
            :meth:`~darts.models.forecasting.torch_forecasting_model.TorchForecastingModel.to_onnx`.

        Returns
        -------
        OnnxModelSpec
            The deserialized spec.

        Raises
        ------
        ValueError
            If the model does not contain :data:`ONNX_SPEC_METADATA_KEY` metadata.
        """
        meta = session.get_modelmeta().custom_metadata_map
        return cls(**json.loads(meta[ONNX_SPEC_METADATA_KEY]))


def prepare_onnx_inputs(
    series: TimeSeries,
    spec: OnnxModelSpec,
    past_covariates: TimeSeries | None = None,
    future_covariates: TimeSeries | None = None,
    n: int | None = None,
) -> dict[str, np.ndarray]:
    """Slice input features for a single ONNX inference window.

    Returns a dictionary mapping ONNX input names to numpy arrays with a leading
    batch dimension. Only includes feature inputs present in ``spec``.
    Recurrent ``state_*`` inputs must be provided separately (or via
    :func:`run_onnx_prediction`).

    Parameters
    ----------
    series
        The series or sequence of series, representing the history of the target series whose
        future is to be predicted. If specified, the method returns the forecasts of these
        series. Otherwise, the method returns the forecast of the (single) training series.
    spec
        Graph / window metadata, typically from :meth:`OnnxModelSpec.from_session`.
    past_covariates
        Optionally, the past-observed covariates series needed as inputs for the model.
        They must match the covariates used for training in terms of dimension.
    future_covariates
        Optionally, the future-known covariates series needed as inputs for the model.
        They must match the covariates used for training in terms of dimension.
    n
        The number of time steps after the end of the target series for which to produce predictions.

    Returns
    -------
    dict[str, np.ndarray]
        Feature tensors keyed by ONNX input name, each with a batch dimension.
    """
    ocl = spec.output_chunk_length
    ocs = spec.output_chunk_shift
    if n is None or n < ocl:
        min_n = ocl
    else:
        min_n = n

    freq = series.freq
    past_start = series.end_time() - (spec.input_chunk_length - 1) * freq
    past_end = series.end_time()
    future_start = past_end + (ocs + 1) * freq
    future_end = past_end + min_n * freq

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
            if min_n > ocl:
                future_past_end = future_start + (min_n - ocl - 1) * freq
                inputs["future_past_covariates"] = np.expand_dims(
                    past_covariates[future_start:future_past_end].values(), axis=0
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


def run_onnx_prediction(
    n: int,
    session: Any,
    series: TimeSeriesLike,
    past_covariates: TimeSeriesLike | None = None,
    future_covariates: TimeSeriesLike | None = None,
    roll_size: int | None = None,
    verbose: bool = True,
) -> TimeSeriesLike:
    """Run ONNX inference for ``n`` steps after ``series`` end.

    Mirrors ``TorchForecastingModel.predict()`` for deterministic models with
    ``num_samples=1`` and probabilistic models with
    ``predict_likelihood_parameters=True``.

    Autoregressive forecasts are only supported for deterministic models.

    .. note::
        Requires ONNX dependencies to be installed:

        - `onnxruntime>=1.24.1` for inference
        - `onnx>=1.0.0` and `onnxscript>=0.7.0` for exporting to ONNX

    Example for exporting a :class:`DLinearModel` to ONNX format and torch-free forcasting:

    .. highlight:: python
    .. code-block:: python

        # train torch model and export to ONNX format
        from darts.datasets import AirPassengersDataset
        from darts.models import DLinearModel

        series = AirPassengersDataset().load().astype("f")
        model = DLinearModel(input_chunk_length=12, output_chunk_length=1)
        model.fit(series, epochs=1)
        onnx_filename = "my_model.onnx"
        model.to_onnx(onnx_filename)

        # run torch-free inference
        import onnxruntime as ort
        from darts.utils.onnx import run_onnx_prediction

        session = ort.InferenceSession(onnx_filename)
        forecast = run_onnx_prediction(n=3, session=session, series=series)
    ..

    Parameters
    ----------
    n
        The number of time steps after the end of the target series for which to produce predictions.
    session
        An ONNX Runtime ``InferenceSession`` for the exported graph.
    series
        The series or sequence of series, representing the history of the target series whose
        future is to be predicted. If specified, the method returns the forecasts of these
        series. Otherwise, the method returns the forecast of the (single) training series.
    past_covariates
        Optionally, the past-observed covariates series needed as inputs for the model.
        They must match the covariates used for training in terms of dimension.
    future_covariates
        Optionally, the future-known covariates series needed as inputs for the model.
        They must match the covariates used for training in terms of dimension.
    roll_size
        For self-consuming predictions, i.e. ``n > output_chunk_length``, determines how many
        outputs of the model are fed back into it at every iteration of feeding the predicted target
        (and optionally future covariates) back into the model. If this parameter is not provided,
        it will be set ``output_chunk_length`` by default. Forced to ``1`` for stepwise graphs.
    verbose
        Whether to display the forecast progress.

    Returns
    -------
    TimeSeriesLike
        The forecasted series or sequence of series.
    """
    # model / graph metadata
    spec = OnnxModelSpec.from_session(session)

    if roll_size is None:
        roll_size = spec.output_chunk_length
    elif not 0 < roll_size <= spec.output_chunk_length:
        raise ValueError(
            "`roll_size` must be an integer between 1 and `output_chunk_length`."
        )

    ocl = spec.output_chunk_length
    ocs = spec.output_chunk_shift

    likelihood_parameters = spec.likelihood_parameter_names
    likelihood = None
    if likelihood_parameters is not None:
        if n > ocl:
            raise_log(
                ValueError(
                    "Cannot generate auto-regressive predictions `n > output_chunk_length` "
                    "when model was fitted with a likelihood."
                ),
            )

        # create a generic likelihood for component naming (the type is not important here)
        likelihood = Likelihood(
            likelihood_type=LikelihoodType.Quantile,
            parameter_names=likelihood_parameters,
        )

    series_seq_type = get_series_seq_type(series)
    series = series2seq(series)
    past_covariates = series2seq(past_covariates)
    future_covariates = series2seq(future_covariates)

    iterator = _build_tqdm_iterator(
        iterable=series,
        verbose=verbose,
        total=len(series),
        desc="Generating Forecasts",
    )
    predictions: list[TimeSeries] = []
    for idx, series_i in enumerate(iterator):
        # create forecast `TimeSeries`
        feature_arrays = prepare_onnx_inputs(
            n=n,
            series=series_i,
            spec=spec,
            past_covariates=past_covariates[idx]
            if past_covariates is not None
            else None,
            future_covariates=future_covariates[idx]
            if future_covariates is not None
            else None,
        )
        # shape (batch size = 1, horizon, components * likelihood parameters)
        prediction_arr = _get_batch_prediction(
            n=n,
            feature_arrays=feature_arrays,
            session=session,
            spec=spec,
            roll_size=roll_size,
        )
        prediction_series = _build_forecast_series_from_schema(
            values=prediction_arr[0],
            schema=series_i.schema(copy=False),
            pred_start=series_i.end_time() + (ocs + 1) * series_i.freq,
            predict_likelihood_parameters=likelihood is not None,
            likelihood_component_names_fn=likelihood.component_names
            if likelihood is not None
            else None,
            copy=False,
        )
        predictions.append(prediction_series)
    return series2seq(predictions, series_seq_type)


def _get_batch_prediction(
    n: int,
    feature_arrays: dict[str, np.ndarray],
    session: Any,
    spec: OnnxModelSpec,
    roll_size: int,
) -> np.ndarray:
    """Generate ONNX forecasts for one batch feature array.

    Mirrors ``darts.models.forecasting.pl_forecasting_module.PLForecastingModule._get_batch_prediction``.
    """
    icl = spec.input_chunk_length
    ocl = spec.output_chunk_length
    # predict at least `output_chunk_length` points, so that we use the most recent target values
    min_n = n if n >= ocl else ocl

    past_target = feature_arrays["past_target"]
    past_covariates_arr = feature_arrays.get("past_covariates")
    future_past_covariates = feature_arrays.get("future_past_covariates")
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
        # recurrent models work step-wise; first warm up over the input chunk,
        # then follow the regular forecast path below

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

        # remaining steps use the last prediction and last historic covariate pads
        out = pred
        past_target = pred[:, -1:, :]
        if past_covariates_arr is not None:
            past_covariates_arr = past_covariates_arr[:, -1:, :]
        if historic_future_covariates is not None:
            historic_future_covariates = historic_future_covariates[:, -1:, :]
        update_historic = False
        roll_size = 1
        batch_predictions = [out]
        prediction_length = roll_size
    else:
        future_cov_slice = (
            future_covariates_arr[:, :ocl, :]
            if future_covariates_arr is not None
            else None
        )
        out, state = _run_onnx_step(session, spec, _feature_dict(future_cov_slice))
        batch_predictions = [out[:, :roll_size, :]]
        prediction_length = roll_size

    while prediction_length < min_n:
        # we want the last prediction to end exactly at `min_n` into the future.
        # this means we may have to truncate the previous prediction and step
        # back the roll size for the last chunk
        if prediction_length + ocl > min_n:
            spillover = prediction_length + ocl - min_n
            roll_size -= spillover
            prediction_length -= spillover
            batch_predictions[-1] = batch_predictions[-1][:, :roll_size, :]

        # roll over past input tensors to contain the latest target and covariates
        past_target = np.roll(past_target, -roll_size, axis=1)
        if past_covariates_arr is not None:
            past_covariates_arr = np.roll(past_covariates_arr, -roll_size, axis=1)
        if historic_future_covariates is not None:
            historic_future_covariates = np.roll(
                historic_future_covariates, -roll_size, axis=1
            )

        # update target input to include next `roll_size` predictions
        if icl >= roll_size:
            past_target[:, -roll_size:, :] = out[:, :roll_size, :]
        else:
            past_target[:, :, :] = out[:, -icl:, :]

        # set left and right boundaries for extracting future elements
        if icl >= roll_size:
            left_past, right_past = prediction_length - roll_size, prediction_length
        else:
            left_past, right_past = prediction_length - icl, prediction_length

        # update past covariates to include next `roll_size` future past covariates elements
        if past_covariates_arr is not None and future_past_covariates is not None:
            if icl >= roll_size:
                past_covariates_arr[:, -roll_size:, :] = future_past_covariates[
                    :, left_past:right_past, :
                ]
            else:
                past_covariates_arr[:, :, :] = future_past_covariates[
                    :, left_past:right_past, :
                ]

        # update historic future covariates to include next `roll_size` future covariates elements
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

    # bring predictions into desired format and drop unnecessary values
    batch_predictions = np.concatenate(batch_predictions, axis=1)
    batch_predictions = batch_predictions[:, :n, :]
    return batch_predictions


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
            # allocate zero tensors matching ``spec.state_input_shapes``
            dtype = next(iter(feature_arrays.values())).dtype
            state = [np.zeros(shape, dtype=dtype) for shape in spec.state_input_shapes]
        for name, value in zip(spec.state_input_names, state):
            if name in session_input_names:
                ort_inputs[name] = value
    outputs = session.run(spec.output_names, ort_inputs)
    prediction = outputs[0]

    # move likelihood parameters to component dimension (c1_p1, ..., c1_pn, ..., cn_p1, ..., cn_pn)
    # auto-regression is not allowed with likelihood models; hence, no dimensionality issues
    prediction = np.asarray(prediction).reshape(prediction.shape[:2] + (-1,))

    next_state = None
    if spec.state_output_names:
        output_by_name = dict(zip(spec.output_names, outputs))
        next_state = [output_by_name[name] for name in spec.state_output_names]
    return prediction, next_state
