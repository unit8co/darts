"""
SHAP Explainer for SKLearn Models
---------------------------------

A `SHAP <https://github.com/slundberg/shap>`__ explainer for Darts ``SKLearnModel`` instances.

For detailed examples and tutorials, see:

* `Explainability of Forecasting Models
  <https://unit8co.github.io/darts/examples/29-Explainability-examples.html>`__.

:class:`SKLearnExplainer` computes SHAP values, which measure each input feature's contribution to a prediction
relative to a baseline (average prediction).

Depending on the model and training data, features can include:

- lags of the target series,
- lags of past covariates,
- lags of future covariates,
- static covariates (global or component-specific).

.. note::
    Input features except static covariates are named according to the convention:
    ``"{name}_{type_of_cov}_lag{idx}"``, where:

    - ``{name}`` is the component name from the original foreground series (target, past covariates, or future
      covariates).
    - ``{type_of_cov}`` is the covariates type. It can take 3 different values:
      ``"target"``, ``"pastcov"``,  ``"futcov"``.
    - ``{idx}`` is the lag index.

    Static covariates are named according to the convention: ``"{name}_statcov_target_{comp}"``, where:

    - ``{name}`` is the variable name of the static covariate.
    - ``{comp}`` is the component name of the target series if static covariates are component-specific, or
      ``"global_components"`` if they are global.

.. note::
   SHAP uses a feature-independence assumption. Indirect effects between features are not captured.

:class:`SKLearnExplainer` provides the following methods for explaining multiple forecasts in batches:

- :func:`explain() <SKLearnExplainer.explain>` computes SHAP values per forecast horizon and target component.
- :func:`summary_plot() <SKLearnExplainer.summary_plot>` shows SHAP value distributions by feature.
- :func:`force_plot() <SKLearnExplainer.force_plot>` shows additive SHAP contributions for one target component and
  horizon.

:class:`SKLearnExplainer` also provides :func:`explain_single() <SKLearnExplainer.explain_single>` for explaining
a single forecast (equivalent to calling ``model.predict(n=output_chunk_length)``).

.. note::
    All above methods can use optional foreground data to explain forecasts, with background data as reference.
    If foreground data is not provided, background data is used for both.
"""

from collections.abc import Sequence
from enum import Enum
from typing import Any, NewType

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.multioutput import MultiOutputRegressor

from darts import TimeSeries
from darts.explainability.explainability import _ForecastingModelExplainer
from darts.explainability.explainability_result import (
    SHAPExplainabilityResult,
    SHAPSingleExplainabilityResult,
)
from darts.logging import get_logger, raise_log
from darts.models.forecasting.sklearn_model import SKLearnModel
from darts.typing import TimeSeriesLike
from darts.utils.data.tabularization import create_lagged_prediction_data
from darts.utils.utils import generate_index

logger = get_logger(__name__)

MIN_BACKGROUND_SAMPLE = 10


class _SHAPMethod(Enum):
    TREE = 0
    DEEP = 2
    KERNEL = 3
    PARTITION = 5
    LINEAR = 6
    PERMUTATION = 7
    ADDITIVE = 8


SHAPMethod = NewType("SHAPMethod", _SHAPMethod)


class SKLearnExplainer(_ForecastingModelExplainer):
    def __init__(
        self,
        model: SKLearnModel,
        background_series: TimeSeriesLike | None = None,
        background_past_covariates: TimeSeriesLike | None = None,
        background_future_covariates: TimeSeriesLike | None = None,
        background_num_samples: int | None = None,
        shap_method: str | None = None,
        **kwargs,
    ):
        """SKLearn Model Explainer.

        **Definitions**:

        - A background series is a ``TimeSeries`` used to train the SHAP explainer.
        - A foreground series is a ``TimeSeries`` that can be explained by a SHAP explainer after it has been fitted.

        Currently, ``SKLearnExplainer`` only works with ``SKLearnModel`` forecasting models.
        The number of explained horizons `(t+1, t+2, ...)` cannot be greater than ``output_chunk_length`` of ``model``.

        Parameters
        ----------
        model
            A ``SKLearnModel`` to be explained. It must be fitted first.
        background_series
            One or several series to *train* the ``SKLearnExplainer`` as reference for explanations.
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
            Optionally, the SHAP method to apply. By default, an attempt is made
            to select the most appropriate method based on a pre-defined set of known models
            internal mapping. Supported values: ``"tree"``, ``"kernel"``, ``"partition"``,
            ``"linear"``, ``"permutation"``, and ``"additive"``.
        **kwargs
            Optionally, additional keyword arguments passed to ``shap_method``.

        Examples
        --------
        >>> from darts.datasets import AirPassengersDataset
        >>> from darts.explainability import SKLearnExplainer
        >>> from darts.models import LinearRegressionModel
        >>> series = AirPassengersDataset().load()
        >>> model = LinearRegressionModel(lags=12).fit(series[:-36])
        >>> explainer = SKLearnExplainer(model)
        >>> results = explainer.explain()
        >>> explainer.summary_plot()
        >>> explainer.force_plot()
        """
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

        if not isinstance(self.model, SKLearnModel):
            raise_log(
                ValueError(
                    "Invalid `model` type. Currently, only models of type `SKLearnModel` are supported."
                ),
                logger,
            )

        if not self.model.multi_models:
            raise_log(
                ValueError(
                    "Invalid `multi_models` value `False`. Currently, "
                    "SKLearnExplainer only supports SKLearnModels "
                    "with `multi_models=True`."
                ),
                logger,
            )

        if self.model.supports_probabilistic_prediction:
            logger.warning(
                "The model is probabilistic, but num_samples=1 will be used for explainability."
            )

        if shap_method is not None:
            shap_method = shap_method.upper()
            if shap_method in _SHAPMethod.__members__:
                self.shap_method = _SHAPMethod[shap_method]
            else:
                raise_log(
                    ValueError(
                        f"Invalid `shap_method`. Please choose one value among the following:"
                        f" {[e.name.lower() for e in _SHAPMethod]}."
                    )
                )
        else:
            self.shap_method = None

        self.explainers = _RegressionSHAPExplainers(
            model=self.model,
            n=self.n,
            target_components=self.target_components,
            past_covariates_components=self.past_covariates_components,
            future_covariates_components=self.future_covariates_components,
            background_series=self.background_series,
            background_past_covariates=self.background_past_covariates,
            background_future_covariates=self.background_future_covariates,
            shap_method=self.shap_method,
            background_num_samples=background_num_samples,
            **kwargs,
        )

    def explain(
        self,
        foreground_series: TimeSeriesLike | None = None,
        foreground_past_covariates: TimeSeriesLike | None = None,
        foreground_future_covariates: TimeSeriesLike | None = None,
        horizons: int | Sequence[int] | None = None,
        target_components: Sequence[str] | None = None,
        **kwargs,
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
        **kwargs
            Other keyword arguments to be passed to the SHAP explainer.

        Returns
        -------
        SHAPExplainabilityResult
            The forecast explanations of the specified horizons and target components.

        Examples
        --------
        Say we have a ``SKLearnModel`` instance with:

          - 2 target components named ``"T_0"`` and ``"T_1"``,
          - 3 past covariates with default component names ``"P_0"``, ``"P_1"``, and ``"P_2"``,
          - 1 future covariate with default component name ``"F_0"``,
          - ``output_chunk_length=2``,
          - ``lags = 3``, ``lags_past_covariates=[-1, -3]``, and ``lags_future_covariates = [0]``.

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
             - P_0_pastcov_lag-3
             - P_1_pastcov_lag-1
             - P_1_pastcov_lag-3
             - P_2_pastcov_lag-1
             - P_2_pastcov_lag-3
             - F_0_futcov_lag0

        Each series has length 3, as the model can explain 5-3+1 forecasts (timestamp indices 4, 5, and 6).
        """
        super().explain(
            foreground_series, foreground_past_covariates, foreground_future_covariates
        )
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

            foreground_X = self.explainers._create_regression_model_shap_X(
                foreground_ts,
                foreground_past_cov_ts,
                foreground_future_cov_ts,
                train=False,
            )

            shap_ = self.explainers.shap_explanations(
                foreground_X, horizons, target_names, **kwargs
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
                        times=shap_[h][t].time_index,
                        values=shap_[h][t].values,
                        components=shap_[h][t].feature_names,
                        copy=False,
                    )
                    feature_values_dict_single_h[t] = TimeSeries(
                        times=shap_[h][t].time_index,
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
        **kwargs,
    ) -> SHAPSingleExplainabilityResult:
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
        **kwargs
            Other keyword arguments to be passed to the SHAP explainer.

        Returns
        -------
        SHAPSingleExplainabilityResult
            The forecast explanations of the specified target components for the single forecasted timestamp.

        Examples
        --------
        Say we have a ``SKLearnModel`` instance with:

          - 2 target components named ``"T_0"`` and ``"T_1"``,
          - 3 past covariates with default component names ``"P_0"``, ``"P_1"``, and ``"P_2"``,
          - 1 future covariate with default component name ``"F_0"``,
          - ``output_chunk_length=2``,
          - ``lags = 3``, ``lags_past_covariates=[-1, -3]``, and ``lags_future_covariates = [0]``.

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

        foreground_X = self.explainers._create_regression_model_shap_X(
            foreground_series_,
            foreground_past_covariates_,
            foreground_future_covariates_,
            train=False,
        )

        # explain only the last forecasted timestamp
        foreground_X = foreground_X[-1:]

        shap_ = self.explainers.shap_explanations_single(
            foreground_X, target_names, **kwargs
        )

        freq = foreground_series_[0].freq
        prediction_time = (
            foreground_X.index[-1] + (self.model.output_chunk_shift) * freq
        )

        shap_values_dict = {}
        feature_values_dict = {}
        shap_explanation_object_dict = {}
        for t in target_names:
            shap_values_dict[t] = TimeSeries(
                times=generate_index(
                    start=prediction_time,
                    freq=freq,
                    length=self.n,
                ),
                values=shap_[t].values,
                components=shap_[t].feature_names,
            )
            feature_values_dict[t] = TimeSeries(
                times=generate_index(
                    start=prediction_time,
                    freq=freq,
                    length=1,
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
        plot_kwargs: dict[str, Any] | None = None,
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
            Default: ``None``, which means that the background series will be used as foreground.
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
        plot_kwargs
            Optionally, a dictionary of keyword arguments to be passed to ``shap.summary_plot()``.
        **kwargs
            Other keyword arguments to be passed to the SHAP explainer.

        Returns
        -------
        dict[int, dict[str, shap.Explanation]]
            A nested dictionary ``{horizon : {component : shap.Explanation}}`` containing the raw Explanation objects
            for all the horizons and components.
        """
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
        horizons, target_components = self._process_horizons_and_targets(
            horizons, target_components
        )

        foreground_X = self.explainers._create_regression_model_shap_X(
            foreground_series,
            foreground_past_covariates,
            foreground_future_covariates,
            n_samples=num_samples,
            train=False,
        )

        shaps_ = self.explainers.shap_explanations(
            foreground_X, horizons, target_components, **kwargs
        )

        for t in target_components:
            for h in horizons:
                plt.title(
                    "Target: `{}` - Horizon: {}".format(
                        t, "t+" + str(h + self.model.output_chunk_shift)
                    )
                )
                shap.summary_plot(
                    shaps_[h][t],
                    foreground_X,
                    plot_type=plot_type,
                    **(plot_kwargs or {}),
                )
        return shaps_

    def force_plot(
        self,
        foreground_series: TimeSeries | None = None,
        foreground_past_covariates: TimeSeries | None = None,
        foreground_future_covariates: TimeSeries | None = None,
        horizon: int | None = 1,
        target_component: str | None = None,
        plot_kwargs: dict[str, Any] | None = None,
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
        plot_kwargs
            Optionally, a dictionary of keyword arguments to be passed to ``shap.force_plot()``.
        **kwargs
            Other keyword arguments to be passed to the SHAP explainer.
        """
        if target_component is None and len(self.target_components) > 1:
            raise_log(
                ValueError(
                    f"The `target_component` parameter is required when the model has more than one component. "
                    f"Please select a component from {self.target_components}."
                ),
                logger,
            )

        if target_component is None:
            target_component = self.target_components[0]

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
        horizons, target_components = self._process_horizons_and_targets(
            horizon,
            target_component,
        )
        horizon, target_component = horizons[0], target_components[0]

        foreground_X = self.explainers._create_regression_model_shap_X(
            foreground_series, foreground_past_covariates, foreground_future_covariates
        )

        shap_ = self.explainers.shap_explanations(
            foreground_X, [horizon], [target_component], **kwargs
        )

        return shap.force_plot(
            base_value=shap_[horizon][target_component],
            features=foreground_X,
            out_names=target_component,
            **(plot_kwargs or {}),
        )


class _RegressionSHAPExplainers:
    """
    Helper Class to wrap the different cases encountered with SHAP different explainers, multivariates,
    horizon etc.
    Aim to provide SHAP values for any type of SKLearnModel. Manage the MultioutputRegressor cases.
    For darts SKLearnModel only.
    """

    default_sklearn_shap_explainers = {
        # Gradient boosting models
        "LGBMRegressor": _SHAPMethod.TREE,
        "CatBoostRegressor": _SHAPMethod.TREE,
        "XGBRegressor": _SHAPMethod.TREE,
        "GradientBoostingRegressor": _SHAPMethod.TREE,
        "HistGradientBoostingRegressor": _SHAPMethod.TREE,
        # Tree models
        "DecisionTreeRegressor": _SHAPMethod.TREE,
        "ExtraTreeRegressor": _SHAPMethod.TREE,
        "ExtraTreesRegressor": _SHAPMethod.TREE,
        "RandomForestRegressor": _SHAPMethod.TREE,
        # Ensemble model
        "AdaBoostRegressor": _SHAPMethod.PERMUTATION,
        "BaggingRegressor": _SHAPMethod.PERMUTATION,
        "RidgeCV": _SHAPMethod.PERMUTATION,
        "Ridge": _SHAPMethod.PERMUTATION,
        # Linear models
        "LinearRegression": _SHAPMethod.LINEAR,
        "ARDRegression": _SHAPMethod.LINEAR,
        "MultiTaskElasticNet": _SHAPMethod.LINEAR,
        "MultiTaskElasticNetCV": _SHAPMethod.LINEAR,
        "MultiTaskLasso": _SHAPMethod.LINEAR,
        "MultiTaskLassoCV": _SHAPMethod.LINEAR,
        "PassiveAggressiveRegressor": _SHAPMethod.LINEAR,
        "PoissonRegressor": _SHAPMethod.LINEAR,
        "QuantileRegressor": _SHAPMethod.LINEAR,
        "RANSACRegressor": _SHAPMethod.LINEAR,
        "GammaRegressor": _SHAPMethod.LINEAR,
        "HuberRegressor": _SHAPMethod.LINEAR,
        "BayesianRidge": _SHAPMethod.LINEAR,
        "SGDRegressor": _SHAPMethod.LINEAR,
        "TheilSenRegressor": _SHAPMethod.LINEAR,
        "TweedieRegressor": _SHAPMethod.LINEAR,
        # Gaussian process
        "GaussianProcessRegressor": _SHAPMethod.PERMUTATION,
        # neighbors
        "KNeighborsRegressor": _SHAPMethod.PERMUTATION,
        "RadiusNeighborsRegressor": _SHAPMethod.PERMUTATION,
        # Neural network
        "MLPRegressor": _SHAPMethod.PERMUTATION,
    }

    def __init__(
        self,
        model: SKLearnModel,
        n: int,
        target_components: Sequence[str],
        past_covariates_components: Sequence[str],
        future_covariates_components: Sequence[str],
        background_series: Sequence[TimeSeries],
        background_past_covariates: Sequence[TimeSeries],
        background_future_covariates: Sequence[TimeSeries],
        shap_method: _SHAPMethod,
        background_num_samples: int | None = None,
        **kwargs,
    ):
        self.model = model
        self.target_dim = self.model.input_dim["target"]
        self.is_multioutputregressor = isinstance(
            self.model.model, MultiOutputRegressor
        )

        self.target_components = target_components
        self.past_covariates_components = past_covariates_components
        self.future_covariates_components = future_covariates_components

        self.n = n
        self.shap_method = shap_method
        self.background_series = background_series
        self.background_past_covariates = background_past_covariates
        self.background_future_covariates = background_future_covariates

        self.single_output = False
        if self.n == 1 and self.target_dim == 1:
            self.single_output = True

        self.background_X = self._create_regression_model_shap_X(
            self.background_series,
            self.background_past_covariates,
            self.background_future_covariates,
            background_num_samples,
            train=True,
        )

        if self.is_multioutputregressor:
            self.explainers = {}
            for i in range(self.n):
                self.explainers[i] = {}
                for j in range(self.target_dim):
                    self.explainers[i][j] = self._build_explainer_sklearn(
                        self.model.get_estimator(horizon=i, target_dim=j),
                        self.background_X,
                        self.shap_method,
                        **kwargs,
                    )
        else:
            self.explainers = self._build_explainer_sklearn(
                self.model.model, self.background_X, self.shap_method, **kwargs
            )

    def shap_explanations(
        self,
        foreground_X: pd.DataFrame,
        horizons: Sequence[int] | None = None,
        target_components: Sequence[str] | None = None,
        **kwargs,
    ) -> dict[int, dict[str, shap.Explanation]]:
        """
        Return a dictionary of dictionaries of shap.Explanation instances:
        - the first dimension corresponds to the n forecasts ahead we want to explain (Horizon).
        - the second dimension corresponds to each component of the target time series.
        Parameters
        ----------
        foreground_X
            the Dataframe of lags features specific of darts SKLearnModel.
        horizons
            Optionally, a list of integers representing which points/steps in the future we want to explain,
            starting from the first prediction step at 1. Currently, only forecasting models are supported which
            provide an `output_chunk_length` parameter. `horizons` must not be larger than `output_chunk_length`.
        target_components
            Optionally, a list of strings with the target components we want to explain.
        **kwargs
            Other keyword arguments to be passed to the SHAP explainer.

        """

        # create a unified dictionary between multiOutputRegressor estimators and
        # native multiOutput estimators
        shap_explanations = {}
        if self.is_multioutputregressor:
            for h in horizons:
                tmp_n = {}
                for t_idx, t in enumerate(self.target_components):
                    if t not in target_components:
                        continue
                    explainer = self.explainers[h - 1][t_idx](foreground_X, **kwargs)
                    explainer.base_values = explainer.base_values.ravel()
                    explainer.time_index = foreground_X.index
                    tmp_n[t] = explainer
                shap_explanations[h] = tmp_n
        else:
            # the native multioutput forces us to recompute all horizons and targets
            shap_explanation_tmp = self.explainers(foreground_X, **kwargs)
            for h in horizons:
                tmp_n = {}
                for t_idx, t in enumerate(self.target_components):
                    if t not in target_components:
                        continue
                    if not self.single_output:
                        tmp_t = shap.Explanation(
                            shap_explanation_tmp.values[
                                :, :, self.target_dim * (h - 1) + t_idx
                            ]
                        )
                        tmp_t.data = shap_explanation_tmp.data
                        tmp_t.base_values = shap_explanation_tmp.base_values[
                            :, self.target_dim * (h - 1) + t_idx
                        ].ravel()
                    else:
                        tmp_t = shap_explanation_tmp
                        tmp_t.base_values = shap_explanation_tmp.base_values.ravel()

                    tmp_t.feature_names = shap_explanation_tmp.feature_names
                    tmp_t.time_index = foreground_X.index
                    tmp_n[t] = tmp_t
                shap_explanations[h] = tmp_n

        return shap_explanations

    def shap_explanations_single(
        self,
        foreground_X: pd.DataFrame,
        target_components: Sequence[str],
        **kwargs,
    ) -> dict[str, shap.Explanation]:
        """
        Return a dictionary of dictionaries of shap.Explanation instances:
        - the first dimension corresponds to the n forecasts ahead we want to explain (Horizon).
        - the second dimension corresponds to each component of the target time series.
        Parameters
        ----------
        foreground_X
            the Dataframe of lags features specific of darts SKLearnModel.
        target_components
            A list of strings with the target components we want to explain.
        **kwargs
            Other keyword arguments to be passed to the SHAP explainer.
        """
        # create a unified dictionary {target_component : shap.Explanation}
        # between multiOutputRegressor estimators and native multiOutput estimators
        shap_explanations = {}
        if isinstance(self.explainers, dict):
            for t_idx, t in enumerate(self.target_components):
                if t not in target_components:
                    continue
                shap_values_list, shap_data_list, base_values_list = [], [], []
                feature_names = None
                for h in range(1, self.n + 1):
                    sub_explanation = self.explainers[h - 1][t_idx](
                        foreground_X, **kwargs
                    )
                    shap_values_list.append(sub_explanation.values.ravel())
                    shap_data_list.append(sub_explanation.data.ravel())
                    base_values_list.append(sub_explanation.base_values.ravel())
                    if feature_names is None:
                        feature_names = sub_explanation.feature_names
                shap_values = np.array(shap_values_list)
                shap_data = np.array(shap_data_list)
                base_values = np.array(base_values_list).ravel()
                shap_explanations[t] = shap.Explanation(
                    values=shap_values,
                    data=shap_data,
                    base_values=base_values,
                    feature_names=feature_names,
                )
        else:
            # the native multioutput forces us to recompute all horizons and targets
            shap_explanation_tmp = self.explainers(foreground_X, **kwargs)
            shap_values: np.ndarray = shap_explanation_tmp.values
            shap_data: np.ndarray = shap_explanation_tmp.data
            base_values: np.ndarray = shap_explanation_tmp.base_values
            feature_names = shap_explanation_tmp.feature_names
            for t_idx, t in enumerate(self.target_components):
                if t not in target_components:
                    continue
                if not self.single_output:
                    tmp_t = shap.Explanation(
                        values=shap_values[0, :, t_idx :: self.target_dim].T,
                        data=np.repeat(shap_data, self.n, axis=0),
                        base_values=base_values[:, t_idx :: self.target_dim].ravel(),
                        feature_names=feature_names,
                    )
                else:
                    tmp_t = shap.Explanation(
                        values=shap_values.reshape(1, -1),
                        data=shap_data,
                        base_values=base_values.ravel(),
                        feature_names=feature_names,
                    )
                shap_explanations[t] = tmp_t

        return shap_explanations

    def _build_explainer_sklearn(
        self,
        model_sklearn,
        background_X: pd.DataFrame,
        shap_method: SHAPMethod | None = None,
        **kwargs,
    ):
        model_name = type(model_sklearn).__name__

        # no shap methods - we need to take the default one
        if shap_method is None:
            if model_name in self.default_sklearn_shap_explainers:
                shap_method = self.default_sklearn_shap_explainers[model_name]
            else:
                shap_method = _SHAPMethod.KERNEL

        # we define properly the explainer given a shap method
        if shap_method == _SHAPMethod.TREE:
            if kwargs.get("feature_perturbation") == "interventional":
                explainer = shap.TreeExplainer(model_sklearn, background_X, **kwargs)
            else:
                explainer = shap.TreeExplainer(model_sklearn, **kwargs)
        elif shap_method == _SHAPMethod.PERMUTATION:
            explainer = shap.PermutationExplainer(
                model_sklearn.predict, background_X, **kwargs
            )
        elif shap_method == _SHAPMethod.PARTITION:
            explainer = shap.PartitionExplainer(
                model_sklearn.predict, background_X, **kwargs
            )
        elif shap_method == _SHAPMethod.KERNEL:
            explainer = shap.KernelExplainer(
                model_sklearn.predict, background_X, keep_index=True, **kwargs
            )
        elif shap_method == _SHAPMethod.LINEAR:
            explainer = shap.LinearExplainer(model_sklearn, background_X, **kwargs)
        elif shap_method == _SHAPMethod.ADDITIVE:
            explainer = shap.AdditiveExplainer(model_sklearn, background_X, **kwargs)

        logger.info("The SHAP method used is of type: " + str(type(explainer)))

        return explainer

    def _create_regression_model_shap_X(
        self,
        target_series: TimeSeriesLike | None,
        past_covariates: TimeSeriesLike | None,
        future_covariates: TimeSeriesLike | None,
        n_samples: int | None = None,
        train: bool = False,
    ) -> pd.DataFrame:
        """
        Creates the SHAP format input for regression models.
        The output is a pandas DataFrame representing all lags of different covariates, and with adequate
        column names in order to map feature / SHAP values.
        It uses create_lagged_data also used in SKLearnModel to build the tabular dataset.

        """

        lags_list = self.model._get_lags("target")
        lags_past_covariates_list = self.model._get_lags("past")
        lags_future_covariates_list = self.model._get_lags("future")

        X, indexes = create_lagged_prediction_data(
            target_series=target_series if lags_list else None,
            past_covariates=past_covariates if lags_past_covariates_list else None,
            future_covariates=(
                future_covariates if lags_future_covariates_list else None
            ),
            lags=lags_list,
            lags_past_covariates=lags_past_covariates_list if past_covariates else None,
            lags_future_covariates=(
                lags_future_covariates_list if future_covariates else None
            ),
            uses_static_covariates=self.model.uses_static_covariates,
            last_static_covariates_shape=self.model._static_covariates_shape,
        )
        # Remove sample axis:
        X = X[:, :, 0]

        if train:
            X = pd.DataFrame(X)
            if len(X) <= MIN_BACKGROUND_SAMPLE:
                raise_log(
                    ValueError(
                        "The number of samples in the background dataset is too small to compute SHAP values."
                    )
                )
        else:
            X = pd.DataFrame(X, index=indexes[0])

        if n_samples:
            X = shap.utils.sample(X, n_samples)

        # rename output columns to the matching lagged features names
        X = X.rename(
            columns={
                name: self.model.lagged_feature_names[idx]
                for idx, name in enumerate(X.columns.to_list())
            }
        )
        return X
