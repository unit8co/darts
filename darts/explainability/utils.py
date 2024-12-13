from collections.abc import Sequence
from typing import Optional, Union

from darts import TimeSeries
from darts.logging import get_logger, raise_if, raise_if_not, raise_log
from darts.models.forecasting.forecasting_model import ForecastingModel
from darts.utils.statistics import stationarity_tests
from darts.utils.ts_utils import series2seq

logger = get_logger(__name__)


def process_input(
    model: ForecastingModel,
    input_type: str,
    series: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
    past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
    future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
    fallback_series: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
    fallback_past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
    fallback_future_covariates: Optional[
        Union[TimeSeries, Sequence[TimeSeries]]
    ] = None,
    check_component_names: bool = False,
    requires_input: bool = True,
    requires_covariates_encoding: bool = False,
    test_stationarity: bool = False,
):
    """Helper function to process and check either of the background or foreground series input to
    `_ForecastingModelExplainer`.

    If no input was provided (`series`, `past/future_covariates`), the fallback will be used for downstream tasks.
    Will raise an error if both input and fallback are not available.

    The fallback is dependent on the input type ("background" or "foreground"):
    - for background `input_type`: fallback are the series saved in fitted forecasting model
    - for foreground `input_type`: fallback are the background series from `_ForecastingModelExplainer`

    Parameters
    ----------
    model
        any `ForecastingModel`.
    input_type
        the type of input series. Either "background" or "foreground"
    series
         Optionally, one or a sequence of target `TimeSeries`.
    past_covariates
         Optionally, one or a sequence of past covariates `TimeSeries`.
    future_covariates
         Optionally, one or a sequence of future covariates `TimeSeries`.
    fallback_series
        Optionally, one or a sequence of target `TimeSeries` to fall back to in case `series` was not provided.
    fallback_past_covariates
        Optionally, one or a sequence of past covariates `TimeSeries` to fall back to in case `past_covariates` was
        not provided.
    fallback_future_covariates
        Optionally, one or a sequence of future covariates `TimeSeries` to fall back to in case `future_covariates` was
        not provided.
    check_component_names
        Whether to enforce that, in the case of multiple time series, all series of the same type (target or
        *_covariates) must have the same component names.
    requires_input
        Whether the input is required. If `True`, raises an error if no input was provided.
    requires_covariates_encoding
        Whether to apply the model's encoders to the input covariates. This should only be `True` if the
        Explainer will not call model methods `fit()` or `predict()` directly.
    test_stationarity
        Whether to raise a warning if not all components from the target `series` are stationary.
    """
    if input_type not in ["background", "foreground"]:
        raise_log(
            ValueError(
                f"Unknown `input_type='{input_type}'`. Must be one of ['background', 'foreground']."
            ),
            logger,
        )

    # if any input is given, treat it as if the input was required
    if (
        series is not None
        or past_covariates is not None
        or future_covariates is not None
    ):
        requires_input = True

    # if `series` was not passed, use the fallback input
    # - for background input type: fallback are the series saved in fitted forecasting model
    # - for foreground input type: fallback are the background series from `_ForecastingModelExplainer`
    if series is None:
        raise_if(
            (past_covariates is not None) or (future_covariates is not None),
            f"Supplied {input_type} past or future covariates but no {input_type} series. Please also provide "
            f"`{input_type}_series`.",
            logger,
        )
        if requires_input and fallback_series is None:
            error_msg = (
                "`model` was fit on multiple time series."
                if input_type == "background"
                else "no `background_series` was provided at `Explainer` creation"
            )
            raise_log(
                ValueError(f"`{input_type}_series` must be provided {error_msg}"),
                logger,
            )
        series = fallback_series
        past_covariates = fallback_past_covariates
        future_covariates = fallback_future_covariates
    # otherwise use the passed input, and generate the covariate encodings (they will be removed again later on
    # if `requires_covariates_encoding=False`)
    else:
        if model.encoders.encoding_available:
            past_covariates, future_covariates = model.generate_fit_encodings(
                series=series,
                past_covariates=past_covariates,
                future_covariates=future_covariates,
            )

    series = series2seq(series)
    past_covariates = series2seq(past_covariates)
    future_covariates = series2seq(future_covariates)

    (
        target_components,
        static_covariates_components,
        past_covariates_components,
        future_covariates_components,
    ) = get_component_names(
        series=series,
        past_covariates=past_covariates,
        future_covariates=future_covariates,
    )

    _check_valid_input(
        model,
        input_type,
        series,
        past_covariates,
        future_covariates,
        target_components,
        past_covariates_components,
        future_covariates_components,
        check_component_names=check_component_names,
        requires_input=requires_input,
        test_stationarity=test_stationarity,
    )

    # make sure to remove any encodings from covariates if downstream tasks require covariates without encodings
    if not requires_covariates_encoding and model.encoders.encoding_available:
        if past_covariates is not None and model.encoders.past_encoders:
            cov = past_covariates[0]
            encoded = model.encoders.past_components
            drop_cols = cov.components[cov.components.isin(encoded)]
            if not drop_cols.empty and len(drop_cols) == cov.n_components:
                past_covariates = None
            elif not drop_cols.empty:
                past_covariates = [
                    cov[cov.components.drop(drop_cols).tolist()]
                    for cov in past_covariates
                ]
        if future_covariates is not None and model.encoders.future_encoders:
            cov = future_covariates[0]
            encoded = model.encoders.future_components
            drop_cols = cov.components[cov.components.isin(encoded)]
            if not drop_cols.empty and len(drop_cols) == cov.n_components:
                future_covariates = None
            elif not drop_cols.empty:
                future_covariates = [
                    cov[cov.components.drop(drop_cols).tolist()]
                    for cov in future_covariates
                ]
    return (
        series,
        past_covariates,
        future_covariates,
        target_components,
        static_covariates_components,
        past_covariates_components,
        future_covariates_components,
    )


def process_horizons_and_targets(
    horizons: Optional[Union[int, Sequence[int]]] = None,
    fallback_horizon: Optional[int] = None,
    target_components: Optional[Union[str, Sequence[str]]] = None,
    fallback_target_components: Optional[Sequence[str]] = None,
    check_component_names: bool = False,
) -> tuple[Sequence[int], Sequence[str]]:
    """Processes the input horizons and target component names.

    horizons
        Optionally, an integer or sequence of integers representing the future time steps to be explained.
        `1` corresponds to the first timestamp being forecasted.
        All values must be `<=output_chunk_length` of the explained forecasting model.
    fallback_horizon
        Optionally, a horizon to fall back to in case `horizons` was not provided.
    target_components
        Optionally, a string or sequence of strings with the target components to explain.
    fallback_target_components
         Optionally, a sequence of strings to fall back to in case `target_components` was not provided.
    check_component_names
        Whether to enforce that the target components are in `fallback_target_component`.
    """

    if target_components is not None:
        if isinstance(target_components, str):
            target_components = [target_components]
        if check_component_names and fallback_target_components is not None:
            invalid_components = [
                target_name
                for target_name in target_components
                if target_name not in fallback_target_components
            ]
            raise_if(
                len(invalid_components) > 0,
                "Invalid `target_components`. The following components are not in the components of the "
                f"`background_series`: {invalid_components}. Provide some valid components from: "
                f"{fallback_target_components}.",
                logger,
            )
    else:
        target_components = fallback_target_components

    if horizons is not None:
        if isinstance(horizons, int):
            horizons = [horizons]

        if fallback_horizon is not None:
            raise_if(
                max(horizons) > fallback_horizon,
                "At least one of the `horizons` is larger than `output_chunk_length`.",
            )
        raise_if(min(horizons) < 1, "All `horizons` must be `>=1`.")
    else:
        horizons = range(1, fallback_horizon + 1)

    return horizons, target_components


def get_component_names(
    series: Sequence[TimeSeries],
    past_covariates: Optional[Sequence[TimeSeries]] = None,
    future_covariates: Optional[Sequence[TimeSeries]] = None,
    idx: int = 0,
) -> tuple[list[str], Optional[list[str]], Optional[list[str]], Optional[list[str]]]:
    """Extract and return the components of target series, static covariate, past and future covariates series.

    Parameters
    ----------
    model
        any `ForecastingModel`.
    series
         A sequence of target `TimeSeries`.
    past_covariates
         Optionally, a sequence of past covariates `TimeSeries`.
    future_covariates
         Optionally, a sequence of future covariates `TimeSeries`.
    idx
        the index of the input sequences to extract the components from.
    """
    target_components = series[idx].components.tolist()

    # covariates
    static_covariates = series[idx].static_covariates
    sc_components = (
        static_covariates.columns.tolist() if static_covariates is not None else []
    )
    pc_components = (
        past_covariates[idx].components.tolist() if past_covariates is not None else []
    )
    fc_components = (
        future_covariates[idx].components.tolist()
        if future_covariates is not None
        else []
    )

    # set to None if not available
    sc_components = sc_components if sc_components else None
    pc_components = pc_components if pc_components else None
    fc_components = fc_components if fc_components else None
    return target_components, sc_components, pc_components, fc_components


def _check_valid_input(
    model,
    input_type: str,
    series: Sequence[TimeSeries],
    past_covariates: Optional[Sequence[TimeSeries]],
    future_covariates: Optional[Sequence[TimeSeries]],
    target_components: Optional[list[str]],
    past_covariates_components: Optional[list[str]],
    future_covariates_components: Optional[list[str]],
    check_component_names: bool = False,
    requires_input: bool = False,
    test_stationarity: bool = False,
):
    """Checks that the input is valid"""
    if test_stationarity and series is not None:
        if not _test_stationarity(series):
            logger.warning(
                "At least one component of the target series is not stationary. "
                "Beware of wrong interpretation of the chosen explainability."
            )

    if input_type not in ["background", "foreground"]:
        raise_log(
            ValueError(
                f"Unknown `input_type='{input_type}'`. Must be one of ['background', 'foreground']."
            ),
            logger,
        )
    if past_covariates is not None:
        raise_if_not(
            len(series) == len(past_covariates),
            f"The number of {input_type} series and past covariates must be the same.",
            logger,
        )

    if future_covariates is not None:
        raise_if_not(
            len(series) == len(future_covariates),
            f"The number of {input_type} series and future covariates must be the same.",
            logger,
        )

    if requires_input:
        raise_if(
            model.uses_past_covariates and past_covariates is None,
            f"A {input_type} past covariates is not provided, but the model requires past covariates.",
            logger,
        )
        raise_if(
            model.uses_future_covariates and future_covariates is None,
            f"A {input_type} future covariates is not provided, but the model requires future covariates.",
            logger,
        )

    if not check_component_names:
        return

    # ensure we have the same names between TimeSeries (if list of). Important to ensure homogeneity
    # for explained features.
    for idx in range(len(series)):
        raise_if_not(
            all([
                series[idx].columns.to_list() == target_components,
                (
                    past_covariates[idx].columns.to_list() == past_covariates_components
                    if past_covariates is not None
                    else True
                ),
                (
                    future_covariates[idx].columns.to_list()
                    == future_covariates_components
                    if future_covariates is not None
                    else True
                ),
            ]),
            "Columns names must be identical between TimeSeries list components (multi-TimeSeries).",
        )


def _test_stationarity(series: Union[TimeSeries, Sequence[TimeSeries]]):
    return all([(stationarity_tests(bs[c]) for c in bs.components) for bs in series])
