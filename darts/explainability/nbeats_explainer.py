"""
NBEATSModel Explainer
---------------------

The `NBEATSExplainer` uses a trained interpretable :class:`NBEATSModel
<darts.models.forecasting.nbeats.NBEATSModel>` (``generic_architecture=False``) and extracts the forecast
contributions of its trend and seasonality stacks.

The contributions can be extracted using the :class:`NBEATSExplainabilityResult
<darts.explainability.explainability_result.NBEATSExplainabilityResult>` returned by
:func:`explain() <NBEATSExplainer.explain>`.
"""

from collections.abc import Sequence
from typing import cast

import numpy as np

from darts import TimeSeries
from darts.explainability.explainability import _ForecastingModelExplainer
from darts.explainability.explainability_result import NBEATSExplainabilityResult
from darts.logging import get_logger, raise_log
from darts.models import NBEATSModel
from darts.models.forecasting.nbeats import _NBEATSModule
from darts.typing import TimeSeriesLike
from darts.utils.ts_utils import SeriesType, get_series_seq_type

logger = get_logger(__name__)

_STACK_NAMES = ["trend", "seasonality"]


class NBEATSExplainer(_ForecastingModelExplainer):
    model: NBEATSModel

    def __init__(
        self,
        model: NBEATSModel,
        background_series: TimeSeriesLike | None = None,
        background_past_covariates: TimeSeriesLike | None = None,
    ):
        """
        Explainer class for the interpretable `NBEATSModel`.

        The interpretable N-BEATS architecture consists of one trend stack and one seasonality stack. The explainer
        returns the contribution of each of the two stacks to the forecast. Their sum is the model forecast.

        **Definitions**

        - A background series is a `TimeSeries` that is used as a default for generating the explainability result
          (if no `foreground` is passed to :func:`explain() <NBEATSExplainer.explain>`).
        - A foreground series is a `TimeSeries` that can be passed to :func:`explain() <NBEATSExplainer.explain>` to
          use instead of the background for generating the explainability result.

        Parameters
        ----------
        model
            The fitted `NBEATSModel` to be explained. Must use ``generic_architecture=False``, must be deterministic
            (no `likelihood`) and must not use reversible instance normalization.
        background_series
            Optionally, a series or list of series to use as a default target series for the explanations.
            Optional if `model` was trained on a single target series. By default, it is the `series` used at fitting
            time. Mandatory if `model` was trained on multiple (sequence of) target series.
        background_past_covariates
            Optionally, a past covariates series or list of series to use as a default past covariates series
            for the explanations. The same requirements apply as for `background_series`.

        Examples
        --------
        >>> from darts.datasets import AirPassengersDataset
        >>> from darts.explainability import NBEATSExplainer
        >>> from darts.models import NBEATSModel
        >>> series = AirPassengersDataset().load().astype("float32")
        >>> model = NBEATSModel(
        >>>     input_chunk_length=24,
        >>>     output_chunk_length=12,
        >>>     generic_architecture=False,
        >>> )
        >>> model.fit(series)
        >>> # create the explainer and generate explanations
        >>> explainer = NBEATSExplainer(model)
        >>> results = explainer.explain()
        >>> trend = results.get_trend()
        >>> seasonality = results.get_seasonality()
        """
        if not isinstance(model, NBEATSModel):
            raise_log(ValueError("`model` must be a Darts `NBEATSModel`."))
        if model.generic_architecture:
            raise_log(
                ValueError(
                    "`NBEATSExplainer` requires the interpretable architecture: create the `NBEATSModel` with "
                    "`generic_architecture=False`."
                ),
            )
        super().__init__(
            model,
            background_series=background_series,
            background_past_covariates=background_past_covariates,
            requires_background=True,
            requires_covariates_encoding=False,
            check_component_names=False,
            test_stationarity=False,
        )
        if model.likelihood is not None:
            raise_log(
                ValueError(
                    "`NBEATSExplainer` does not support probabilistic models (models with a `likelihood`)."
                ),
            )
        if cast(_NBEATSModule, model.model).use_reversible_instance_norm:
            raise_log(
                ValueError(
                    "`NBEATSExplainer` does not support models with `use_reversible_instance_norm=True`."
                ),
            )

    def explain(
        self,
        foreground_series: TimeSeriesLike | None = None,
        foreground_past_covariates: TimeSeriesLike | None = None,
        foreground_future_covariates: TimeSeriesLike | None = None,
        horizons: Sequence[int] | None = None,
        target_components: Sequence[str] | None = None,
    ) -> NBEATSExplainabilityResult:
        """Returns the :class:`NBEATSExplainabilityResult
        <darts.explainability.explainability_result.NBEATSExplainabilityResult>` for all series in
        `foreground_series`. If `foreground_series` is `None`, will use the `background` input from
        `NBEATSExplainer` creation (either the `background` passed to creation, or the series stored in the
        `NBEATSModel` in case it was only trained on a single series).

        For each series and target component, the result contains the forecast contributions of the trend and the
        seasonality stack over the forecast horizon.

        Parameters
        ----------
        foreground_series
            Optionally, one or a sequence of target `TimeSeries` to be explained. Can be multivariate.
            If not provided, the background `TimeSeries` will be explained instead.
        foreground_past_covariates
            Optionally, one or a sequence of past covariates `TimeSeries` if required by the forecasting model.
        foreground_future_covariates
            Must be `None`; `NBEATSModel` does not support future covariates.
        horizons
            This parameter is not used by the `NBEATSExplainer`; the contributions are always returned for the full
            output chunk.
        target_components
            Optionally, a sequence of strings with the target components to explain. By default, all components.

        Returns
        -------
        NBEATSExplainabilityResult
            The explainability result containing the trend and seasonality contributions.

        Examples
        --------
        >>> explainer = NBEATSExplainer(model)
        >>> results = explainer.explain(foreground_series=series)
        >>> trend = results.get_trend()
        >>> seasonality = results.get_seasonality()
        """
        if foreground_future_covariates is not None:
            raise_log(ValueError("`NBEATSModel` does not support future covariates."))
        if horizons is not None:
            logger.warning(
                "`horizons` is not supported by `NBEATSExplainer` and will be ignored."
            )
        super().explain(foreground_series, foreground_past_covariates, None)
        (foreground_series, foreground_past_covariates, *_) = self._process_foreground(
            foreground_series, foreground_past_covariates, None
        )
        if (
            get_series_seq_type(foreground_series) is SeriesType.SEQ
            and len(foreground_series) > self.model.batch_size
        ):
            raise_log(
                ValueError(
                    f"The number of back- or foreground series to explain ({len(foreground_series)}) "
                    f"must be smaller than or equal to the model's batch size ({self.model.batch_size})."
                ),
            )
        _, target_components = self._process_horizons_and_targets(
            None, target_components
        )
        preds = self.model.predict(
            n=self.n,
            series=foreground_series,
            past_covariates=foreground_past_covariates,
        )
        if isinstance(preds, TimeSeries):
            preds = [preds]
        # shape (n series, output_chunk_length, n target components, 1) per stack
        stack_forecasts = [
            sf.cpu().numpy()
            for sf in cast(_NBEATSModule, self.model.model)._stack_forecasts
        ]
        all_components = list(preds[0].components)

        results = []
        for idx, pred in enumerate(preds):
            times = pred.time_index
            expl = {}
            for comp in target_components:
                c_idx = all_components.index(comp)
                values = np.stack(
                    [sf[idx, :, c_idx, 0] for sf in stack_forecasts],
                    axis=1,
                )
                expl[comp] = TimeSeries(
                    values=values, times=times, components=_STACK_NAMES
                )
            results.append(expl)
        return NBEATSExplainabilityResult(
            explained_components=results[0] if len(results) == 1 else results
        )
