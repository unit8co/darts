"""
SHAP Explainer for Torch Models
-------------------------------

A `SHAP <https://github.com/slundberg/shap>`__ explainer for Darts ``TorchForecastingModel`` instances.

:class:`TorchExplainer` computes SHAP values, which measure each input feature's contribution to a prediction
relative to a baseline (average prediction).

Depending on the model and training data, features can include:

- lags (input chunk) of the target series,
- lags (input chunk) of past covariates,
- lags (input chunk and output chunk) of future covariates,
- static covariates (global or component-specific).

.. note::
    Input features except static covariates are named according to the convention:
    ``"{name}_{type_of_cov}_lag{idx}"``, where:

    - ``{name}`` is the component name from the original foreground series (target, past, or future).
    - ``{type_of_cov}`` is the covariates type. It can take 3 different values:
      ``"target"``, ``"pastcov"``,  ``"futcov"``.
    - ``{idx}`` is the lag index.

    Static covariates are named according to the convention: ``"{name}_statcov_target_{comp}"``, where:

    - ``{name}`` is the variable name of the static covariate.
    - ``{comp}`` is the component name of the target series if static covariates are component-specific, or
      ``"global_components"`` if they are global.

.. note::
   SHAP uses a feature-independence assumption. Indirect effects between features are not captured.

:class:`TorchExplainer` provides the following methods for explaining forecasts in batches:

- :func:`explain() <TorchExplainer.explain>` computes SHAP values per forecast horizon and target component.
- :func:`summary_plot() <TorchExplainer.summary_plot>` shows SHAP value distributions by feature.
- :func:`force_plot() <TorchExplainer.force_plot>` shows additive SHAP contributions for one target and horizon.

:class:`TorchExplainer` also provides :func:`explain_single() <TorchExplainer.explain_single>` for explaining
a single forecast (equivalent to calling ``model.predict(n=output_chunk_length)``).

.. note::
    All above methods can use optional foreground data to explain forecasts, with background data as reference.
    If foreground data is not provided, background data is used for both.
"""

from collections.abc import Sequence
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
import shap
import torch
from matplotlib import pyplot as plt

from darts import TimeSeries
from darts.explainability.explainability import _ForecastingModelExplainer
from darts.explainability.explainability_result import (
    SHAPExplainabilityResult,
    SHAPSingleExplainabilityResult,
)
from darts.explainability.utils import process_horizons_and_targets
from darts.logging import get_logger, raise_log
from darts.models.forecasting.pl_forecasting_module import PLForecastingModule
from darts.models.forecasting.rnn_model import CustomRNNModule
from darts.models.forecasting.torch_forecasting_model import TorchForecastingModel
from darts.typing import TimeSeriesLike
from darts.utils.data.torch_datasets.utils import TorchInferenceDatasetOutput
from darts.utils.ts_utils import series2seq

logger = get_logger(__name__)

MIN_BACKGROUND_SAMPLE = 10
MAX_BACKGROUND_SAMPLE = 1000

INPUT_PAST_INDICES = [0, 1, 3]
INPUT_FUTURE_INDICES = [4]
INPUT_STATIC_INDICES = [5]


class _SHAPMethod(Enum):
    KERNEL = 3
    SAMPLING = 4
    PARTITION = 5
    LINEAR = 6
    PERMUTATION = 7
    ADDITIVE = 8
    EXACT = 9


def _available_shap_methods() -> list[str]:
    return [method.name.lower() for method in _SHAPMethod]


class TorchExplainer(_ForecastingModelExplainer):
    def __init__(
        self,
        model: TorchForecastingModel,
        background_series: TimeSeriesLike | None = None,
        background_past_covariates: TimeSeriesLike | None = None,
        background_future_covariates: TimeSeriesLike | None = None,
        background_num_samples: int | None = None,
        batch_size: int | None = None,
        shap_method: str = "kernel",
        **kwargs,
    ):
        """Torch Model Explainer.

        **Definitions**:

        - A background series is a ``TimeSeries`` used to train the SHAP explainer.
        - A foreground series is a ``TimeSeries`` that can be explained by a SHAP explainer after it has been fitted.

        ``TorchExplainer`` only works with torch models, i.e., instances of ``TorchForecastingModel``.
        The number of explained horizons `(t+1, t+2, ...)` cannot be greater than ``output_chunk_length`` of ``model``.

        Parameters
        ----------
        model
            A ``TorchForecastingModel`` to be explained. It must be fitted first.
        background_series
            One or several series to *train* the ``TorchExplainer`` as reference for explanations.
            Consider using a reduced well-chosen background to reduce computation time.
            Optional if ``model`` was fit on a single target series. By default, it is the ``series``
            used at fitting time.
            Mandatory if ``model`` was fit on multiple (list of) target series.
        background_past_covariates
            A past covariates series or list of series that the model needs once fitted.
        background_future_covariates
            A future covariates series or list of series that the model needs once fitted.
        background_num_samples
            Optionally, whether to sample a subset of the original background. Randomly picks
            samples of the constructed training dataset.
            Generally used for faster computation, especially when ``shap_method`` is
            ``"kernel"`` or ``"permutation"``.
        shap_method
            Optionally, the SHAP method to apply. Supported values: ``"kernel"``, ``"sampling"``,
            ``"partition"``, ``"linear"``, ``"permutation"``, ``"additive"``, and ``"exact"``.
            Default: ``"kernel"``.
        **kwargs
            Optionally, additional keyword arguments passed to ``shap_method``.

        Examples
        --------
        >>> from darts.datasets import WineDataset
        >>> from darts.explainability import TorchExplainer
        >>> from darts.models import TiDEModel
        >>> series = WineDataset().load().astype("float32")
        >>> model = TiDEModel(12, 12).fit(series[:36])
        >>> explainer = TorchExplainer(model)
        >>> results = explainer.explain()
        >>> explainer.summary_plot()
        >>> explainer.force_plot()
        """
        # initialize the explainer with sanity checks and background validation
        super().__init__(
            model=model,
            background_series=background_series,
            background_past_covariates=background_past_covariates,
            background_future_covariates=background_future_covariates,
            requires_background=True,
            requires_covariates_encoding=True,
            check_component_names=True,
            test_stationarity=True,
        )

        # validate model type
        if not isinstance(self.model, TorchForecastingModel):
            raise_log(
                ValueError(
                    f"Invalid `model` type: `{type(model)}`. Only models of type `TorchForecastingModel` are supported."
                ),
                logger,
            )

        shap_method_upper = shap_method.upper()
        if shap_method_upper in _SHAPMethod.__members__:
            self.shap_method = _SHAPMethod[shap_method_upper]
        else:
            raise_log(
                ValueError(
                    f"Invalid `shap_method`={shap_method}. Please choose one value among the following: "
                    f"{_available_shap_methods()}."
                )
            )

        if (
            background_num_samples is not None
            and background_num_samples > MAX_BACKGROUND_SAMPLE
        ):
            raise_log(
                ValueError(
                    f"`background_num_samples` must be less than or equal to "
                    f"MAX_BACKGROUND_SAMPLE={MAX_BACKGROUND_SAMPLE}. Got {background_num_samples}."
                ),
                logger,
            )

        self.explainer = _DeepSHAPExplainer(
            model=self.model,
            n=self.n,
            target_components=self.target_components,
            static_covariates_components=self.static_covariates_components,
            past_covariates_components=self.past_covariates_components,
            future_covariates_components=self.future_covariates_components,
            background_series=self.background_series,
            background_past_covariates=self.background_past_covariates,
            background_future_covariates=self.background_future_covariates,
            background_num_samples=background_num_samples,
            shap_method=self.shap_method,
            batch_size=batch_size,
            **kwargs,
        )

    def _process_horizons_and_targets(
        self,
        horizons: int | Sequence[int] | None,
        target_components: str | Sequence[str] | None,
    ) -> tuple[Sequence[int], Sequence[str]]:
        return process_horizons_and_targets(
            horizons=horizons,
            fallback_horizon=self.n,
            target_components=target_components,
            fallback_target_components=self.explainer.target_components_likelihood,
            check_component_names=self.check_component_names,
        )

    def explain(
        self,
        foreground_series: TimeSeriesLike | None = None,
        foreground_past_covariates: TimeSeriesLike | None = None,
        foreground_future_covariates: TimeSeriesLike | None = None,
        horizons: int | Sequence[int] | None = None,
        target_components: Sequence[str] | None = None,
    ) -> SHAPExplainabilityResult:
        """
        Explains foreground time series forecasts and returns a :class:`SHAPExplainabilityResult
        <darts.explainability.explainability_result.SHAPExplainabilityResult>` of SHAP values.

        The results can then be retrieved with method :func:`get_explanation()
        <darts.explainability.explainability_result.SHAPExplainabilityResult.get_explanation>`,
        which returns a multivariate ``TimeSeries`` instance containing the SHAP values for the
        ``(horizon, target_component)`` forecasts at all timestamps forecastable in the foreground series.

        The components of the ``TimeSeries`` correspond to the input features used by the model to produce
        the forecasts. See above for the naming convention.

        Parameters
        ----------
        foreground_series
            Optionally, one or a sequence of target ``TimeSeries`` to be explained. Can be multivariate.
            If not provided, the background ``TimeSeries`` will be explained instead.
        foreground_past_covariates
            Optionally, one or a sequence of past covariates ``TimeSeries`` if required by the forecasting model.
        foreground_future_covariates
            Optionally, one or a sequence of future covariates ``TimeSeries`` if required by the forecasting model.
        horizons
            Optionally, an integer or sequence of integers representing the future time steps to be explained.
            ``1`` corresponds to the first timestamp being forecasted.
            All values must be no greater than ``output_chunk_length`` of the explained forecasting model.
        target_components
            Optionally, a string or sequence of strings with the target components to explain.

        Returns
        -------
        SHAPExplainabilityResult
            The forecast explanations of the specified horizons and target components.

        Examples
        --------
        Say we have a ``TorchForecastingModel`` instance with:

          - 2 target components named ``"T_0"`` and ``"T_1"``,
          - 3 past covariates with default component names ``"P_0"``, ``"P_1"``, and ``"P_2"``,
          - 1 future covariate with default component name ``"F_0"``,
          - ``input_chunk_length=3``,
          - ``output_chunk_length=2``.

        We provide ``foreground_series``, ``foreground_past_covariates``, ``foreground_future_covariates`` (extending
        far enough into the future) each of length 5.

        >>> results = explainer.explain(
        >>>     foreground_series=foreground_series,
        >>>     foreground_past_covariates=foreground_past_covariates,
        >>>     foreground_future_covariates=foreground_future_covariates)

        Calling the method returns a ``SHAPExplainabilityResult`` object containing the SHAP values, feature values,
        and raw ``shap.Explanation`` objects for each horizon and target component. They can be accessed with:

        >>> # Get SHAP values for forecasting "T_1" at horizon 1 as a `TimeSeries`
        >>> output = results.get_explanation(horizon=1, component="T_1")
        >>> # Get feature values used for forecasting as a `TimeSeries`
        >>> feature_values = results.get_feature_values(horizon=1, component="T_1")
        >>> # Get the raw `shap.Explanation` object for further processing
        >>> shap_objects = results.get_shap_explanation_object(horizon=1, component="T_1")

        For SHAP and feature values, the components of the returned ``TimeSeries`` correspond to different lags of the
        target and covariates (see convention above). In our example, the component names would be:

             - T_0_target_lag-1
             - T_0_target_lag-2
             - T_0_target_lag-3
             - T_1_target_lag-1
             - T_1_target_lag-2
             - T_1_target_lag-3
             - P_0_pastcov_lag-1
             - P_0_pastcov_lag-2
             - P_0_pastcov_lag-3
             - P_1_pastcov_lag-1
             - P_1_pastcov_lag-2
             - P_1_pastcov_lag-3
             - P_2_pastcov_lag-1
             - P_2_pastcov_lag-2
             - P_2_pastcov_lag-3
             - F_0_futcov_lag0
             - F_0_futcov_lag1

        Each series has length 3, as the model can explain 5-3+1 forecasts (timestamp indices 4, 5, and 6).
        """
        fallback = foreground_series is None
        (
            foreground_series,
            foreground_past_covariates,
            foreground_future_covariates,
            _,
            _,
            _,
            _,
        ) = self._process_foreground(
            foreground_series,
            foreground_past_covariates,
            foreground_future_covariates,
        )
        horizons, target_names = self._process_horizons_and_targets(
            horizons,
            target_components,
        )

        shap_values_list = []
        feature_values_list = []
        shap_explanation_object_list = []
        for idx, foreground_ts in enumerate(foreground_series):
            foreground_past_cov_ts = None
            foreground_future_cov_ts = None

            if foreground_past_covariates:
                foreground_past_cov_ts = foreground_past_covariates[idx]

            if foreground_future_covariates:
                foreground_future_cov_ts = foreground_future_covariates[idx]

            foreground_X, _, prediction_times = self.explainer.create_shap_array(
                foreground_ts,
                foreground_past_cov_ts,
                foreground_future_cov_ts,
                train=fallback,
            )

            shap_ = self.explainer.shap_explanations(
                foreground_X, horizons, target_names
            )

            shap_values_dict = {}
            feature_values_dict = {}
            shap_explanation_object_dict = {}
            for h in horizons:
                shap_values_dict_single_h = {}
                feature_values_dict_single_h = {}
                shap_explanation_object_dict_single_h = {}
                for t in target_names:
                    shap_values_dict_single_h[t] = TimeSeries(
                        times=prediction_times,
                        values=shap_[h][t].values,
                        components=shap_[h][t].feature_names,
                        copy=False,
                    )
                    feature_values_dict_single_h[t] = TimeSeries(
                        times=prediction_times,
                        values=shap_[h][t].data,
                        components=shap_[h][t].feature_names,
                        copy=False,
                    )
                    shap_explanation_object_dict_single_h[t] = shap_[h][t]
                shap_values_dict[h] = shap_values_dict_single_h
                feature_values_dict[h] = feature_values_dict_single_h
                shap_explanation_object_dict[h] = shap_explanation_object_dict_single_h

            shap_values_list.append(shap_values_dict)
            feature_values_list.append(feature_values_dict)
            shap_explanation_object_list.append(shap_explanation_object_dict)

        if len(shap_values_list) == 1:
            shap_values_list = shap_values_list[0]
            feature_values_list = feature_values_list[0]
            shap_explanation_object_list = shap_explanation_object_list[0]

        return SHAPExplainabilityResult(
            shap_values_list, feature_values_list, shap_explanation_object_list
        )

    def explain_single(
        self,
        foreground_series: TimeSeries | None = None,
        foreground_past_covariates: TimeSeries | None = None,
        foreground_future_covariates: TimeSeries | None = None,
        target_components: Sequence[str] | None = None,
    ):
        """
        Explains a foreground time series forecast starting from one last forecastable timestamp and returns a
        :class:`SHAPSingleExplainabilityResult
        <darts.explainability.explainability_result.SHAPSingleExplainabilityResult>` of SHAP values.

        The results can then be retrieved with method :func:`get_explanation()
        <darts.explainability.explainability_result.SHAPSingleExplainabilityResult.get_explanation>`,
        which returns a multivariate ``TimeSeries`` instance containing the SHAP values for ``target_component``
        starting from the last forecastable timestamp.

        The components of the ``TimeSeries`` correspond to the input features used by the model to produce
        the forecast. See above for the naming convention.

        Parameters
        ----------
        foreground_series
            Optionally, one or a sequence of target ``TimeSeries`` to be explained. Can be multivariate.
            If not provided, the background ``TimeSeries`` will be explained instead.
        foreground_past_covariates
            Optionally, one or a sequence of past covariates ``TimeSeries`` if required by the forecasting model.
        foreground_future_covariates
            Optionally, one or a sequence of future covariates ``TimeSeries`` if required by the forecasting model.
        target_components
            Optionally, a string or sequence of strings with the target components to explain.

        Returns
        -------
        SHAPSingleExplainabilityResult
            The forecast explanations of the specified target components for the single forecasted timestamp.

        Examples
        --------
        Say we have a ``TorchForecastingModel`` instance with:

          - 2 target components named ``"T_0"`` and ``"T_1"``,
          - 3 past covariates with default component names ``"P_0"``, ``"P_1"``, and ``"P_2"``,
          - 1 future covariate with default component name ``"F_0"``,
          - ``input_chunk_length=3``,
          - ``output_chunk_length=2``.

        We provide ``foreground_series``, ``foreground_past_covariates``, ``foreground_future_covariates`` (extending
        far enough into the future) each of length 5.

        >>> results = explainer.explain_single(
        >>>     foreground_series=foreground_series,
        >>>     foreground_past_covariates=foreground_past_covariates,
        >>>     foreground_future_covariates=foreground_future_covariates)

        Calling the method returns a ``SHAPSingleExplainabilityResult`` object containing the SHAP values,
        feature values, and raw ``shap.Explanation`` objects for each target component at the single forecasted
        timestamp (timestamp index 6 in our example, as it is the last forecastable).

        >>> # Get SHAP values for forecasting "T_1" as a `TimeSeries`
        >>> output = results.get_explanation(component="T_1")
        >>> # Get feature values used for forecasting as a `TimeSeries`
        >>> feature_values = results.get_feature_values(component="T_1")
        >>> # Get the raw `shap.Explanation` object for further processing
        >>> shap_objects = results.get_shap_explanation_object(component="T_1")

        For SHAP and feature values, the components of the returned ``TimeSeries`` correspond to different lags of the
        target and covariates (see convention above). In our example, the component names would be:

             - T_0_target_lag-1
             - T_0_target_lag-2
             - T_0_target_lag-3
             - T_1_target_lag-1
             - T_1_target_lag-2
             - T_1_target_lag-3
             - P_0_pastcov_lag-1
             - P_0_pastcov_lag-2
             - P_0_pastcov_lag-3
             - P_1_pastcov_lag-1
             - P_1_pastcov_lag-2
             - P_1_pastcov_lag-3
             - P_2_pastcov_lag-1
             - P_2_pastcov_lag-2
             - P_2_pastcov_lag-3
             - F_0_futcov_lag0
             - F_0_futcov_lag1

        The SHAP value ``TimeSeries`` has length ``output_chunk_length=2``, as the model predicts that many timestamps
        in the future. The feature value ``TimeSeries`` has length 1, as it corresponds to the single forecasted
        timestamp explained.

        .. note::
            The single forecast explained by this method should be equivalent to the one obtained by calling
            ``model.predict(n=output_chunk_length)`` when foreground data is provided. However, the "equivalent"
            forecast is temporally backshifted by ``output_chunk_length`` when the model uses future covariates
            AND both foreground and background data are not provided. In this case, the explainer would use training
            data as reference, whose future covariates were trimmed to match the target series during training.
            As a result, the forecast explained would be backshifted to the last timestamp when future covariates are
            known.
        """
        fallback = foreground_series is None
        (
            foreground_series_,
            foreground_past_covariates_,
            foreground_future_covariates_,
            _,
            _,
            _,
            _,
        ) = self._process_foreground(
            foreground_series,
            foreground_past_covariates,
            foreground_future_covariates,
        )
        _, target_names = self._process_horizons_and_targets(None, target_components)

        foreground_X, schemas, prediction_times = self.explainer.create_shap_array(
            foreground_series_,
            foreground_past_covariates_,
            foreground_future_covariates_,
            train=fallback,
        )

        # explain only the last forecasted timestamp
        foreground_X = foreground_X[-1:]
        schema = schemas[-1]
        prediction_time = prediction_times[-1]

        shap_ = self.explainer.shap_explanations_single(foreground_X, target_names)

        shap_values_dict = {}
        feature_values_dict = {}
        shap_explanation_object_dict = {}
        for t in target_names:
            shap_values_dict[t] = TimeSeries(
                times=pd.date_range(
                    start=prediction_time,
                    freq=schema["time_freq"],
                    periods=shap_[t].values.shape[0],
                ),
                values=shap_[t].values,
                components=shap_[t].feature_names,
            )
            feature_values_dict[t] = TimeSeries(
                times=pd.date_range(
                    start=prediction_time,
                    freq=schema["time_freq"],
                    periods=1,
                ),
                values=shap_[t].data[:1],
                components=shap_[t].feature_names,
            )
            shap_explanation_object_dict[t] = shap_[t]

        return SHAPSingleExplainabilityResult(
            shap_values_dict,
            feature_values_dict,
            shap_explanation_object_dict,
        )

    def summary_plot(
        self,
        foreground_series: TimeSeriesLike | None = None,
        foreground_past_covariates: TimeSeriesLike | None = None,
        foreground_future_covariates: TimeSeriesLike | None = None,
        horizons: int | Sequence[int] | None = None,
        target_components: str | Sequence[str] | None = None,
        num_samples: int | None = None,
        plot_type: str | None = "dot",
        **kwargs,
    ) -> dict[int, dict[str, shap.Explanation]]:
        """
        Display a SHAP "Summary Plot" for each horizon and each component dimension of the target.

        On each summary plot, SHAP values of each input feature are plotted with dots (``plot_type="dot"``,
        each dot corresponds to a forecasted timestamp), a bar (``plot_type="bar"``), or a violin
        (``plot_type="violin"``). The input features are sorted by importance, defined as the mean absolute SHAP value.

        Parameters
        ----------
        foreground_series
            Optionally, one or a sequence of target ``TimeSeries`` to be explained. Can be multivariate.
            If not provided, the background ``TimeSeries`` will be explained instead.
        foreground_past_covariates
            Optionally, one or a sequence of past covariates ``TimeSeries`` if required by the forecasting model.
        foreground_future_covariates
            Optionally, one or a sequence of future covariates ``TimeSeries`` if required by the forecasting model.
        horizons
            Optionally, an integer or sequence of integers representing which points/steps in the future to explain,
            starting from the first prediction step at 1. Each horizon must be no greater than ``output_chunk_length``
            of the explained forecasting model. Default: ``None``, which means that all horizons will be plotted.
        target_components
            Optionally, a string or sequence of strings with the target components to explain.
            Default: ``None``, which means that all target components will be plotted.
        num_samples
            Optionally, an integer for sampling the foreground series for the sake of performance.
        plot_type
            Optionally, specify which of the SHAP library plot type to use. Can be one of ``"dot"``, ``"bar"``,
            ``"violin"``.

        Returns
        -------
        dict[int, dict[str, shap.Explanation]]
            A nested dictionary ``{horizon : {component : shap.Explanation}}`` containing the raw Explanation objects
            for all the horizons and components.
        """
        (
            foreground_series_,
            foreground_past_covariates_,
            foreground_future_covariates_,
            _,
            _,
            _,
            _,
        ) = self._process_foreground(
            foreground_series,
            foreground_past_covariates,
            foreground_future_covariates,
        )
        horizons, target_components = self._process_horizons_and_targets(
            horizons, target_components
        )

        foreground_X, _, _ = self.explainer.create_shap_array(
            foreground_series_,
            foreground_past_covariates_,
            foreground_future_covariates_,
            n_samples=num_samples,
            train=foreground_series is None,
        )

        shaps_ = self.explainer.shap_explanations(
            foreground_X, horizons, target_components
        )

        for t in target_components:
            for h in horizons:
                plt.title(
                    f"Target: `{t}` - Horizon: t+{h + self.model.output_chunk_shift}"
                )
                shap.summary_plot(
                    shaps_[h][t],
                    foreground_X,
                    plot_type=plot_type,
                    **kwargs,
                )
        return shaps_

    def force_plot(
        self,
        foreground_series: TimeSeries | None = None,
        foreground_past_covariates: TimeSeries | None = None,
        foreground_future_covariates: TimeSeries | None = None,
        horizon: int | None = 1,
        target_component: str | None = None,
        **kwargs,
    ):
        """
        Display a SHAP "Force Plot" for one target and one horizon.

        It shows SHAP values of all input features with an additive force layout for each forecastable timestamp
        in the foreground series. At each timestamp, SHAP values of all features and the base value would sum up
        to the model prediction.

        .. note::
            Once the plot is displayed, select **"original sample ordering"** to observe the forecasted timestamps
            chronologically.

        Parameters
        ----------
        foreground_series
            Optionally, the target series to explain. Can be multivariate. Default: ``None``, which means that the
            background series will be used as foreground.
        foreground_past_covariates
            Optionally, a past covariate series if required by the forecasting model.
        foreground_future_covariates
            Optionally, a future covariate series if required by the forecasting model.
        horizon
            Optionally, an integer for the point/step in the future to explain, starting from the first prediction
            step at 1. Must not be larger than ``output_chunk_length`` of the model.
        target_component
            Optionally, the target component to plot. If the target series is multivariate, the target component
            must be specified.
        **kwargs
            Optionally, additional keyword arguments passed to ``shap.force_plot()``.
        """
        if (
            target_component is None
            and len(self.explainer.target_components_likelihood) > 1
        ):
            raise_log(
                ValueError(
                    f"`target_component` is required when the model has more than one component. "
                    f"Please select a component from {self.explainer.target_components_likelihood}."
                ),
                logger,
            )

        if target_component is None:
            target_component = self.explainer.target_components_likelihood[0]

        (
            foreground_series_,
            foreground_past_covariates_,
            foreground_future_covariates_,
            _,
            _,
            _,
            _,
        ) = self._process_foreground(
            foreground_series,
            foreground_past_covariates,
            foreground_future_covariates,
        )
        horizons, target_components = self._process_horizons_and_targets(
            horizon,
            target_component,
        )
        horizon, target_component = horizons[0], target_components[0]

        foreground_X, _, _ = self.explainer.create_shap_array(
            foreground_series_,
            foreground_past_covariates_,
            foreground_future_covariates_,
            train=foreground_series is None,
        )

        shap_ = self.explainer.shap_explanations(
            foreground_X, [horizon], [target_component]
        )

        return shap.force_plot(
            base_value=shap_[horizon][target_component],
            features=foreground_X,
            out_names=target_component,
            **kwargs,
        )


class _DeepSHAPExplainer:
    # TODO: add docstring
    n_targets: int

    def __init__(
        self,
        model: TorchForecastingModel,
        n: int,
        target_components: Sequence[str],
        static_covariates_components: Sequence[str] | None,
        past_covariates_components: Sequence[str] | None,
        future_covariates_components: Sequence[str] | None,
        background_series: Sequence[TimeSeries],
        background_past_covariates: Sequence[TimeSeries] | None,
        background_future_covariates: Sequence[TimeSeries] | None,
        background_num_samples: int | None = None,
        shap_method: _SHAPMethod = _SHAPMethod.LINEAR,
        batch_size: int | None = None,
        **kwargs,
    ):
        """
        Helper class to wrap the SHAP explainer and its interaction with the torch forecasting model.
        It is initialized with the model and the background data, and provides methods to create SHAP arrays and
        compute SHAP explanations for given foreground data.
        """
        self.model = model

        self.target_components = target_components
        self.static_covariates_components = static_covariates_components
        self.past_covariates_components = past_covariates_components
        self.future_covariates_components = future_covariates_components

        if self.model.likelihood is not None:
            self.target_components_likelihood = self.model.likelihood.component_names(
                components=self.target_components
            )
            logger.warning(
                f"The explained model is probabilistic and the SHAP explanations will be computed for the likelihood "
                f"parameters of the target components, which includes the following components: "
                f"{self.target_components_likelihood}. Adjust the `target_components` argument accordingly."
            )
        else:
            self.target_components_likelihood = self.target_components

        self.n = n
        self.background_series = background_series
        self.background_past_covariates = background_past_covariates
        self.background_future_covariates = background_future_covariates

        self.input_chunk_length = model.input_chunk_length
        self.output_chunk_length = model.output_chunk_length or 1
        self.output_chunk_shift = model.output_chunk_shift

        self.background_X, _, _ = self.create_shap_array(
            series=self.background_series,
            past_covariates=self.background_past_covariates,
            future_covariates=self.background_future_covariates,
            n_samples=background_num_samples,
            train=True,
        )

        self._setup_func_wrapper(
            model.model,
            batch_size=batch_size or model.batch_size,
        )
        self._build_feature_names()

        self.explainer = self._build_explainer(
            self._func_wrapper,
            self.background_X,
            shap_method,
            **kwargs,
        )

    def _setup_func_wrapper(
        self,
        model: PLForecastingModule,
        batch_size: int,
    ):
        """
        Sets up the parameters for the function wrapper that will be passed to the SHAP explainer.
        See :func:`_func_wrapper()` for details on the wrapper function.
        """
        self.pl_module = model
        self.n_past_covs = (
            self.background_past_covariates[0].n_components
            if self.background_past_covariates is not None
            else 0
        )
        self.n_future_covs = (
            self.background_future_covariates[0].n_components
            if self.background_future_covariates is not None
            else 0
        )
        static_covs = self.background_series[0].static_covariates_values(copy=False)
        self.n_static_covs = static_covs.shape[1] if static_covs is not None else 0

        self.n_targets = model.n_targets
        self.n_targets_likelihood = len(self.target_components_likelihood)
        self.n_variables = self.n_targets + self.n_past_covs + self.n_future_covs

        self.past_slice = slice(0, self.input_chunk_length * self.n_variables)
        self.future_slice = slice(
            self.past_slice.stop,
            self.past_slice.stop + self.output_chunk_length * self.n_future_covs,
        )
        self.static_slice = slice(self.future_slice.stop, None)

        self.batch_size = batch_size

    def _build_feature_names(self):
        """
        Builds the feature names for the SHAP explanations based on the input features used by the
        torch forecasting model. See above for the naming convention.
        """
        self.feature_names = []
        for i in range(self.input_chunk_length):
            lag = self.input_chunk_length - i
            for t in self.target_components:
                self.feature_names.append(f"{t}_target_lag-{lag}")
            if self.past_covariates_components is not None:
                for c in self.past_covariates_components:
                    self.feature_names.append(f"{c}_pastcov_lag-{lag}")
            if self.future_covariates_components is not None:
                for c in self.future_covariates_components:
                    self.feature_names.append(f"{c}_futcov_lag-{lag}")

        for i in range(self.output_chunk_length):
            lag = i + self.output_chunk_shift
            if self.future_covariates_components is not None:
                for c in self.future_covariates_components:
                    self.feature_names.append(f"{c}_futcov_lag{lag}")

        if self.model.uses_static_covariates:
            static_covs = self.background_series[0].static_covariates
            if static_covs is not None:
                # static covariate names
                names = static_covs.columns.tolist()
                # target components that the static covariates reference to
                comps = static_covs.index.tolist()
                self.feature_names += [
                    f"{name}_statcov_target_{comp}" for name in names for comp in comps
                ]

    @torch.inference_mode()
    def _func_wrapper(self, x_np: np.ndarray) -> np.ndarray:
        """
        Wrapper function to adapt the SHAP explainer to the torch forecasting model. It takes as input a numpy array
        of shape `(num_samples, num_features)` and outputs a numpy array of shape
        `(num_samples, output_chunk_length * n_targets_likelihood)`.

        Internally, it does the following steps:
        1. Reshape the input numpy array into the format expected by the torch forecasting model, separating past
           covariates, future covariates, and static covariates based on the slices defined
           in :func:`_setup_func_wrapper()`.
        2. If the model is an RNN, handle the special case where future covariates are concatenated to
           past covariates with a shift in time dimension.
        3. Pass the reshaped inputs to the model in batches and collect the outputs.
        4. Concatenate the outputs and reshape them into the expected output format for SHAP, which is a 2D array where
           each column corresponds to a target component at a specific horizon.

        Parameters
        ----------
        x_np
            A numpy array of shape `(num_samples, num_features)` containing the input features for SHAP explanations.

        Returns
        -------
        np.ndarray
            A numpy array of shape `(num_samples, output_chunk_length * n_targets_likelihood)` containing the model
            predictions for each target component at each horizon, to be used by the SHAP explainer.
        """
        x = torch.from_numpy(x_np).float()
        num_samples = x.shape[0]

        x_past = x[:, self.past_slice]
        x_past = x_past.reshape(num_samples, self.input_chunk_length, self.n_variables)

        if self.n_future_covs > 0:
            x_future = x[:, self.future_slice]
            x_future = x_future.reshape(
                num_samples, self.output_chunk_length, self.n_future_covs
            )
        else:
            x_future = None

        if self.n_static_covs > 0:
            x_static = x[:, self.static_slice]
            x_static = x_static.reshape(num_samples, -1, self.n_static_covs)
        else:
            x_static = None

        if isinstance(self.pl_module, CustomRNNModule):
            # handle the special case of RNN where future covariates are concatenated to
            # past covariates with a shift in time dimension
            if x_future is not None:
                x_future = torch.cat(
                    [
                        x_past[:, 1:, -self.n_future_covs :],
                        x_future,  # output chunk length is always 1 for RNN
                    ],
                    dim=1,
                )
                x_past = torch.cat(
                    [
                        x_past[:, :, : self.n_targets],
                        x_future,
                    ],
                    dim=2,
                )
                x_future = None

        # set model to eval mode to deactivate dropout layers
        self.pl_module.eval()

        outputs: list[torch.Tensor] = []
        for i in range(0, num_samples, self.batch_size):
            s = slice(i, i + self.batch_size)
            batch_x_past = x_past[s].to(self.pl_module.device)
            batch_x_future = (
                x_future[s].to(self.pl_module.device) if x_future is not None else None
            )
            batch_x_static = (
                x_static[s].to(self.pl_module.device) if x_static is not None else None
            )

            batch_output: torch.Tensor = self.pl_module((
                batch_x_past,
                batch_x_future,
                batch_x_static,
            ))

            if isinstance(self.pl_module, CustomRNNModule):
                # Note: RNN outputs predictions and hidden states
                batch_output = batch_output[0]
                # RNN also outputs predictions for all time steps,
                # but we only need the last one for SHAP explanations
                batch_output = batch_output[:, -1:, :, :]
            else:
                # Note: TCN has a different `first_prediction_index` than 0
                batch_output = batch_output[
                    :, self.pl_module.first_prediction_index :, :
                ]

            outputs.append(batch_output)

        # `output`: (batch, output_chunk_length, n_targets, likelihood_parameters)
        output = torch.cat(outputs, dim=0)
        # flatten the output to shape (batch, output_chunk_length * n_targets_likelihood)
        output = output.flatten(start_dim=1)

        return output.cpu().numpy()

    @staticmethod
    def _build_explainer(
        func,
        background_X: np.ndarray,
        shap_method: _SHAPMethod,
        **kwargs,
    ):
        """
        Builds the SHAP explainer based on the specified SHAP method.

        Parameters
        ----------
        func
            The function wrapper that takes a numpy array of input features and outputs model predictions, to be passed
            to the SHAP explainer.
        background_X
            The background dataset in the form of a numpy array, to be passed to the SHAP explainer.
        shap_method
            The SHAP method to use for explanations. Must be one of the methods available in the SHAP library,
            specified in the enum ``_SHAPMethod``.
        """
        # we define properly the explainer given a shap method
        # Note: DeepExplainer has some compatibility issues with torch models
        if shap_method == _SHAPMethod.PERMUTATION:
            explainer = shap.PermutationExplainer(func, background_X, **kwargs)
        elif shap_method == _SHAPMethod.PARTITION:
            explainer = shap.PermutationExplainer(func, background_X, **kwargs)
        elif shap_method == _SHAPMethod.KERNEL:
            explainer = shap.KernelExplainer(func, background_X, **kwargs)
        elif shap_method == _SHAPMethod.LINEAR:
            explainer = shap.LinearExplainer(func, background_X, **kwargs)
        elif shap_method == _SHAPMethod.ADDITIVE:
            explainer = shap.AdditiveExplainer(func, background_X, **kwargs)
        elif shap_method == _SHAPMethod.EXACT:
            explainer = shap.ExactExplainer(func, background_X, **kwargs)
        else:
            raise_log(
                ValueError(
                    f"Invalid `shap_method`={shap_method}. Please choose one value among the following: "
                    f"{_available_shap_methods()}."
                )
            )

        return explainer

    def shap_explanations(
        self,
        foreground_X: np.ndarray,
        horizons: Sequence[int],
        target_components: Sequence[str],
    ) -> dict[int, dict[str, shap.Explanation]]:
        """
        Computes SHAP explanations for the given foreground data, horizons, and target components.
        It returns a nested dictionary of SHAP Explanation objects for each horizon and target component, where the SHAP
        values are extracted from the raw Explanation object returned by the SHAP explainer and reshaped
        into the expected format for easier accessibility.

        Parameters
        ----------
        foreground_X
            A numpy array of shape `(num_samples, num_features)` containing the input features for SHAP explanations.
        horizons
            A sequence of integers representing which points/steps in the future to explain, starting from the first
            prediction step at 1. Each horizon must be no greater than ``output_chunk_length`` of the explained
            forecasting model.
        target_components
            A sequence of strings with the target components to explain. Each component must be among the target
            components of the explained forecasting model.

        Returns
        -------
        dict[int, dict[str, shap.Explanation]]
            A nested dictionary ``{horizon : {target_component : shap.Explanation}}`` containing the SHAP Explanation
            objects for each horizon and target component, where the SHAP values are extracted and reshaped for
            easier accessibility.
        """
        shap_explanation_tmp: shap.Explanation = self.explainer(foreground_X)
        shap_values: np.ndarray = shap_explanation_tmp.values
        shap_data: np.ndarray = shap_explanation_tmp.data
        shap_base_values: np.ndarray = shap_explanation_tmp.base_values

        # create a nested dictionary {horizon : {target_component : shap.Explanation}}
        # for better accessibility of the explanations
        shap_explanations = {}

        for h in horizons:
            tmp_n = {}
            for t_idx, t in enumerate(self.target_components_likelihood):
                if t not in target_components:
                    continue
                tmp_t = shap.Explanation(
                    shap_values[:, :, self.n_targets_likelihood * (h - 1) + t_idx],
                    data=shap_data,
                    base_values=shap_base_values[
                        :, self.n_targets_likelihood * (h - 1) + t_idx
                    ].ravel(),
                    feature_names=self.feature_names,
                )

                tmp_n[t] = tmp_t
            shap_explanations[h] = tmp_n

        return shap_explanations

    def shap_explanations_single(
        self,
        foreground_X: np.ndarray,
        target_components: Sequence[str],
    ) -> dict[str, shap.Explanation]:
        """
        Similar to :func:`shap_explanations()`, but computes SHAP explanations for only one forecasted timestamp, which
        corresponds to the last forecastable timestamp in the foreground series. The output is a dictionary of SHAP
        Explanation objects for each target component, where the SHAP values are extracted from the raw Explanation
        object returned by the SHAP explainer and reshaped into the expected format for easier accessibility.

        Parameters
        ----------
        foreground_X
            A numpy array of shape `(1, num_features)` containing the input features for SHAP explanations for the
            single forecasted timestamp. Must have only one sample corresponding to the single forecasted timestamp.
        target_components
            A sequence of strings with the target components to explain. Each component must be among the target
            components of the explained forecasting model.

        Returns
        -------
        dict[str, shap.Explanation]
            A dictionary ``{target_component : shap.Explanation}`` containing the SHAP Explanation objects for each
            target component, where the SHAP values are extracted and reshaped for easier accessibility.
        """
        shap_explanation_tmp: shap.Explanation = self.explainer(foreground_X)
        shap_values: np.ndarray = shap_explanation_tmp.values
        shap_data: np.ndarray = shap_explanation_tmp.data
        shap_base_values: np.ndarray = shap_explanation_tmp.base_values

        # create a nested dictionary {target_component : shap.Explanation}
        # for better accessibility of the explanations
        shap_explanations = {}

        horizon = self.output_chunk_length

        for t_idx, t in enumerate(self.target_components_likelihood):
            if t not in target_components:
                continue
            tmp_t = shap.Explanation(
                shap_values[0, :, t_idx :: self.n_targets_likelihood].T,
                data=np.repeat(shap_data, repeats=horizon, axis=0),
                base_values=shap_base_values[
                    0, t_idx :: self.n_targets_likelihood
                ].ravel(),
                feature_names=self.feature_names,
            )

            shap_explanations[t] = tmp_t

        return shap_explanations

    def _create_dataset_bounds(
        self,
        series: Sequence[TimeSeries],
        train: bool,
    ) -> np.ndarray:
        """
        Creates the bounds for the inference dataset based on the input series and whether it is for training or not.
        """
        offset = self.output_chunk_length if train else 0
        bounds = np.array([(self.input_chunk_length, len(s) - offset) for s in series])
        return bounds

    @staticmethod
    def _batch_collate_np(batch: list[tuple], indices: list[int]) -> np.ndarray | None:
        """
        Collates a batch of samples from the inference dataset into a numpy array for SHAP explanations,
        based on the specified indices for past covariates, future covariates, and static covariates.
        It handles the case where some samples in the batch may have None values for certain inputs,
        by skipping those samples when collating.
        """
        data = []
        for index in indices:
            if batch[0][index] is None:
                continue
            data.append(np.stack([sample[index] for sample in batch]))

        if len(data) == 0:
            return None
        else:
            data = np.concatenate(data, axis=2)
            return data

    def create_shap_array(
        self,
        series: TimeSeriesLike,
        past_covariates: TimeSeriesLike | None,
        future_covariates: TimeSeriesLike | None,
        n_samples: int | None = None,
        train: bool = False,
    ) -> tuple[np.ndarray, list[dict[str, Any]], pd.Index]:
        """
        Creates the SHAP array for the given input series and covariates, by following the logic of the torch
        forecasting model's inference dataset and prediction step. It returns the SHAP array, the schemas of the
        samples, and the prediction times corresponding to each sample in the SHAP array.

        Parameters
        ----------
        series
            A sequence of target series to be explained. Can be a single TimeSeries or a sequence of TimeSeries.
        past_covariates
            Optionally, a sequence of past covariate series if required by the forecasting model. Can be a single
            TimeSeries or a sequence of TimeSeries. Must be provided if the model uses past covariates.
        future_covariates
            Optionally, a sequence of future covariate series if required by the forecasting model. Can be a single
            TimeSeries or a sequence of TimeSeries. Must be provided if the model uses future covariates.
        n_samples
            Optionally, an integer for sampling the dataset for the sake of performance. If ``train=True``,
            the samples will be randomly drawn from the dataset. If ``train=False``, the last ``n_samples`` samples
            will be taken from the dataset. Default: ``None``, which means that all samples in the dataset will be used.
        train
            A boolean indicating whether the SHAP array is being created for training (background) data or for
            foreground data. This affects how the dataset is sampled and how the bounds are created. Default: ``False``.
        """
        # convert to sequence of TimeSeries if not already
        series_: Sequence[TimeSeries] = series2seq(series)
        past_covariates_: Sequence[TimeSeries] | None = series2seq(past_covariates)
        future_covariates_: Sequence[TimeSeries] | None = series2seq(future_covariates)

        # create inference dataset
        dataset = self.model._build_inference_dataset(
            n=self.n,
            series=series_,
            past_covariates=past_covariates_,
            future_covariates=future_covariates_,
            stride=1,
            bounds=self._create_dataset_bounds(series_, train=train),
        )

        # sample from dataset if required
        n_samples = n_samples or len(dataset)
        if train:
            if len(dataset) < MIN_BACKGROUND_SAMPLE:
                raise_log(
                    ValueError(
                        f"Background series must contain at least {MIN_BACKGROUND_SAMPLE} samples to create a "
                        f"valid background. Got background dataset length={len(dataset)}."
                    ),
                    logger,
                )
            if n_samples > len(dataset):
                raise_log(
                    ValueError(
                        f"`background_num_samples` must be less than or equal to the number of samples in the dataset. "
                        f"Got `background_num_samples={n_samples}` but dataset length={len(dataset)}."
                    ),
                    logger,
                )
            if n_samples > MAX_BACKGROUND_SAMPLE:
                logger.warning(
                    f"Background series contains more than MIN_BACKGROUND_SAMPLE={MAX_BACKGROUND_SAMPLE} samples. "
                    f"Sampling {MAX_BACKGROUND_SAMPLE} samples to create the background for SHAP explanations."
                )
                n_samples = MAX_BACKGROUND_SAMPLE
        else:
            if n_samples > len(dataset):
                raise_log(
                    ValueError(
                        f"`n_samples` must be less than or equal to the number of samples in the dataset. "
                        f"Got `n_samples={n_samples}` but dataset length={len(dataset)}."
                    ),
                    logger,
                )

        # follow the logic of `TorchForecastingModel.predict_from_dataset()`
        # to collect samples and collate them into a sample tuple
        # collect batch of samples from the end of the dataset
        batch: list[TorchInferenceDatasetOutput] = []
        if train:
            if n_samples < len(dataset):
                # randomly sample from the dataset if in training mode
                indices = np.random.choice(len(dataset), size=n_samples, replace=False)
            else:
                indices = range(len(dataset))
        else:
            indices = range(len(dataset) - n_samples, len(dataset))
        for i in indices:
            batch.append(dataset[i])

        # follow the logic of `PLForecastingModule.predict_step()`
        # to convert to 1D tensor
        # - past_target
        # - past_covariates
        # - future_past_covariates
        # - historic_future_covariates
        # - future_covariates
        # - static_covariates
        input_past = self._batch_collate_np(batch, INPUT_PAST_INDICES)
        input_future = self._batch_collate_np(batch, INPUT_FUTURE_INDICES)
        input_static = self._batch_collate_np(batch, INPUT_STATIC_INDICES)
        schemas = [c[-2] for c in batch]
        prediction_times = pd.Index([c[-1] for c in batch])

        shap_array = np.concatenate(
            [
                array.reshape(array.shape[0], -1)
                for array in [input_past, input_future, input_static]
                if array is not None
            ],
            axis=-1,
        )

        return shap_array, schemas, prediction_times
