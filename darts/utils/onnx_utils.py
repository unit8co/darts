"""
ONNX Utils (backward-compatible re-exports).

Prefer :mod:`darts.utils.onnx` for new code. ``prepare_onnx_inputs`` here still
accepts a fitted ``model=`` in place of ``spec=``.
"""

from darts.utils.onnx.export import prepare_onnx_export
from darts.utils.onnx.inference import (
    OnnxModelSpec,
    extract_point_forecast,
    run_onnx_prediction,
)
from darts.utils.onnx.inference import (
    prepare_onnx_inputs as _prepare_onnx_inputs,
)

__all__ = [
    "OnnxModelSpec",
    "extract_point_forecast",
    "prepare_onnx_export",
    "prepare_onnx_inputs",
    "run_onnx_prediction",
]


def prepare_onnx_inputs(
    series=None,
    past_covariates=None,
    future_covariates=None,
    *,
    model=None,
    spec=None,
    **kwargs,
):
    """Slice input features for ONNX inference.

    Accepts either an :class:`~darts.utils.onnx.inference.OnnxModelSpec` or a
    fitted torch forecasting ``model`` (legacy API).

    Parameters
    ----------
    series
        Target series (required).
    past_covariates
        Past covariates, if the model uses them.
    future_covariates
        Future covariates, if the model uses them.
    model
        Fitted torch forecasting model; used to build a spec when ``spec`` is
        omitted.
    spec
        Explicit graph / window metadata. Required if ``model`` is omitted.
    **kwargs
        Forwarded to :func:`darts.utils.onnx.inference.prepare_onnx_inputs`
        (e.g. ``future_horizon``).

    Returns
    -------
    dict[str, np.ndarray]
        Feature tensors keyed by ONNX input name.
    """
    if series is None:
        raise TypeError("`series` is required.")
    if spec is None:
        if model is None:
            raise TypeError("Either `spec` or `model` must be provided.")
        spec = OnnxModelSpec.from_model(model)
    return _prepare_onnx_inputs(
        series=series,
        spec=spec,
        past_covariates=past_covariates,
        future_covariates=future_covariates,
        **kwargs,
    )
