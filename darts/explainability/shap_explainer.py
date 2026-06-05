"""
SHAP Explainer for SKLearn and Torch Models
-------------------------------------------

A `SHAP <https://github.com/slundberg/shap>`__ explainer for Darts' ``SKLearnModel`` and ``TorchForecastingModel``
instances.

For detailed examples and tutorials, see:

* `Explainability of Forecasting Models
  <https://unit8co.github.io/darts/examples/28-Explainability-examples.html>`__.

:class:`ShapExplainer` computes SHAP values, which measure each input feature's contribution to a prediction
relative to a baseline (average prediction).

Depending on the model and training data, features can include:

- lags of the target series (input chunk for torch models)
- lags of past covariates  (input chunk for torch models)
- lags of future covariates (input and output chunk for torch models)
- static covariates (global or component-specific)

.. note::
    All input features except static covariates are named according to the convention
    ``"{name}_{type_of_cov}_lag{idx}"``, where:

    - ``{name}`` is the component name from the original foreground series (target, past covariates, or future
      covariates).
    - ``{type_of_cov}`` is the covariates type. It can take 3 different values:
      ``"target"``, ``"pastcov"``,  ``"futcov"``.
    - ``{idx}`` is the lag index, where ``0`` represents the position of the first predicted step.

    Static covariates are named according to the convention: ``"{name}_statcov_target_{comp}"``, where:

    - ``{name}`` is the variable name of the static covariate.
    - ``{comp}`` is the component name of the target series if static covariates are component-specific, or
      ``"global_components"`` if they are global.

.. note::
   SHAP uses a feature-independence assumption. Indirect effects between features are not captured.

:class:`ShapExplainer` provides the following methods for explaining multiple forecasts in batches:

- :func:`explain() <ShapExplainer.explain>` computes SHAP values per forecast horizon and target component.
- :func:`summary_plot() <ShapExplainer.summary_plot>` shows SHAP value distributions by feature.
- :func:`force_plot() <ShapExplainer.force_plot>` shows additive SHAP contributions for one target component and
  horizon.

:class:`ShapExplainer` also provides :func:`explain_single() <ShapExplainer.explain_single>` for explaining
a single forecast (equivalent to calling ``model.predict(n=output_chunk_length)``).

.. note::
    All above methods can use optional foreground data to explain forecasts, with background data as reference.
    If foreground data is not provided, background data is used for both.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import shap

from darts import TimeSeries
from darts.explainability.explainability import _ForecastingModelExplainer
from darts.explainability.explainability_result import (
    ShapExplainabilityResult,
    ShapSingleExplainabilityResult,
)
from darts.logging import get_logger, raise_log
from darts.models.forecasting.sklearn_model import SKLearnModel
from darts.typing import TimeSeriesLike
from darts.utils.utils import generate_index

if TYPE_CHECKING:
    from darts.models.forecasting.torch_forecasting_model import TorchForecastingModel

logger = get_logger(__name__)


class ShapExplainer(_ForecastingModelExplainer):
    def __init__(
        self,
        model: SKLearnModel | TorchForecastingModel,
        background_series: TimeSeriesLike | None = None,
        background_past_covariates: TimeSeriesLike | None = None,
        background_future_covariates: TimeSeriesLike | None = None,
        background_num_samples: int | None = None,
        shap_method: str | None = None,
        batch_size: int | None = None,
        test_stationarity: bool = False,
        **kwargs,
    ):
        """SHAP Explainer for SKLearn and Torch Models.

        **Definitions**:

        - A background series is a ``TimeSeries`` used to train the SHAP explainer.
        - A foreground series is a ``TimeSeries`` that can be explained by a SHAP explainer after it has been fitted.

        The number of explained horizons `(t+1, t+2, ...)` cannot be greater than ``output_chunk_length`` of ``model``.

        Parameters
        ----------
        model
            The ``SKLearnModel`` or ``TorchForecastingModel`` to be explained. It must be fitted first.
        background_series
            One or several series to *train* the ``ShapExplainer`` as reference for explanations. Consider using a
            reduced well-chosen background to reduce computation time. Optional if ``model`` was fit on a single target
            series. By default, it is the ``series`` used at fitting time. Mandatory if ``model`` was fit on multiple
            (list of) target series.
        background_past_covariates
            A past covariates series or list of series that the model needs once fitted.
        background_future_covariates
            A future covariates series or list of series that the model needs once fitted.
        background_num_samples
            Optionally, whether to sample a subset of the original background. Randomly picks samples of the
            constructed training dataset. Generally used for faster computation, especially when ``shap_method`` is
            ``"kernel"`` or ``"permutation"``.
        shap_method
            Optionally, the SHAP method to apply. By default, an attempt is made to select the most appropriate method
            based on a pre-defined set of known models internal mapping. Supported values for ``SKLearnModel``:
            ``["tree", "kernel", "partition", "linear", "permutation", "additive"]``. Supported values
            ``TorchForecastingModel``: ``["kernel", "partition", "sampling", "permutation"]``.
        batch_size
            Optionally, the batch size to use when ``model`` is a ``TorchForecastingModel``. Increasing the batch size
            can significantly reduce computation time.
        test_stationarity
            Whether to perform stationarity checks and raise a warning if not all `background_series` are stationary.
        **kwargs
            Optionally, additional keyword arguments passed to ``shap_method``.

        Examples
        --------

        For ``SKLearnModel``:
        >>> from darts.datasets import AirPassengersDataset
        >>> from darts.explainability import ShapExplainer
        >>> from darts.models import LinearRegressionModel
        >>> series = AirPassengersDataset().load().astype("float32")[:-36]
        >>> model = LinearRegressionModel(lags=12, output_chunk_length=1).fit(series)
        >>> explainer = ShapExplainer(model)
        >>> result = explainer.explain()
        >>> explainer.summary_plot()
        >>> explainer.force_plot()

        For ``TorchForecastingModel`` (extending the previous example):
        >>> from darts.models import TiDEModel
        >>> model = TiDEModel(input_chunk_length=12, output_chunk_length=1).fit(series)
        >>> explainer = ShapExplainer(model, batch_size=2048)
        >>> result = explainer.explain()
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
            test_stationarity=test_stationarity,
        )

        if isinstance(self.model, SKLearnModel):
            from darts.explainability.shap.sklearn_explainer import SKLearnShapExplainer

            explainer_cls = SKLearnShapExplainer
        else:
            # lazily import torch dependencies
            from darts.explainability.shap.torch_explainer import TorchShapExplainer
            from darts.models.forecasting.torch_forecasting_model import (
                TorchForecastingModel,
            )

            if isinstance(self.model, TorchForecastingModel):
                explainer_cls = TorchShapExplainer
            else:
                raise_log(
                    ValueError(
                        f"Invalid `model` type: `{type(self.model)}`. Only models of type "
                        f"`SKLearnModel` or `TorchForecastingModel` are supported."
                    ),
                    logger,
                )

        self.explainer = explainer_cls(
            model=self.model,
            n=self.n,
            target_components=self.target_components,
            past_covariates_components=self.past_covariates_components,
            future_covariates_components=self.future_covariates_components,
            static_covariates_components=self.static_covariates_components,
            background_series=self.background_series,
            background_past_covariates=self.background_past_covariates,
            background_future_covariates=self.background_future_covariates,
            background_num_samples=background_num_samples,
            shap_method=shap_method,
            batch_size=batch_size,
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
    ) -> ShapExplainabilityResult:
        """
        Explains all possible foreground series forecasts (or background, if foreground is not provided) and returns
        a :class:`ShapExplainabilityResult <darts.explainability.explainability_result.ShapExplainabilityResult>` of
        SHAP values.

        The results can then be retrieved with method :func:`get_explanation()
        <darts.explainability.explainability_result.ShapExplainabilityResult.get_explanation>`,
        which returns a multivariate ``TimeSeries`` instance containing the SHAP values for the
        ``(horizon, target_component)`` forecasts at all timestamps forecastable in the foreground series.

        The components of the ``TimeSeries`` correspond to the input features used by the model to produce
        the forecasts. See above for the naming convention.

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
            Optionally, an integer or sequence of integers representing the future time steps to be explained.
            ``1`` corresponds to the first timestamp being forecasted.
            All values must be no greater than ``output_chunk_length`` of the explained forecasting model.
        target_components
            Optionally, a string or sequence of strings with the target components to explain.
        **kwargs
            Other keyword arguments to be passed to the SHAP explainer.

        Returns
        -------
        ShapExplainabilityResult
            The forecast explanations of the specified horizons and target components.

        Examples
        --------
        Say we have a ``SKLearnModel`` instance with:

            - 1 target component named ``"Y"``,
            - 1 future covariate named ``"month"``,
            - ``lags = 2``, and ``lags_future_covariates = [-1, 0]``.

        Let's explain the background series that the model was trained on:

        >>> from darts.datasets import AusBeerDataset
        >>> from darts.explainability import ShapExplainer
        >>> from darts.models import LinearRegressionModel
        >>> from darts.utils.timeseries_generation import datetime_attribute_timeseries as dta
        >>>
        >>> # load a target series and create future covariates holding the calendar month values
        >>> series = AusBeerDataset().load()
        >>> fc = dta(series, attribute="month", add_length=12)
        >>>
        >>> # create and fit a model
        >>> model = LinearRegressionModel(lags=2, lags_future_covariates=[-1, 0])
        >>> model.fit(series, future_covariates=fc)
        >>>
        >>> # create an explainer; requires background series if the model was trained on multiple series
        >>> explainer = ShapExplainer(model)
        >>> # explain the background series (or foreground if passed to `explain()`)
        >>> result = explainer.explain()
        >>>
        >>> # get explanations for a specific horizon (and optional `component` for multivariate models)
        >>> # the feature SHAP values for all possible forecast start points
        >>> result.get_explanation(horizon=1)
                    Y_target_lag-2  Y_target_lag-1  month_futcov_lag-1  month_futcov_lag0
        1956-07-01      -56.332566     -106.927156          -24.253184          33.064478
        1956-10-01      -88.545937      -99.569541           13.642416          84.727725
        1957-01-01      -82.194005      -57.000488           51.538016         -70.262016
        1957-04-01      -45.443539      -81.175506          -62.148784         -18.598769
        1957-07-01      -66.314174      -99.043998          -24.253184          33.064478
        ...                    ...             ...                 ...                ...
        2007-10-01      -11.415330      -11.803715           13.642416          84.727725
        2008-01-01       -6.424526       29.714250           51.538016         -70.262016
        2008-04-01       29.418521        1.860425          -62.148784         -18.598769
        2008-07-01        5.371920      -13.905891          -24.253184          33.064478
        2008-10-01       -8.239364       -3.395013           13.642416          84.727725

        shape: (210, 4, 1), freq: QS-OCT, size: 6.56 KB

        The explanation has length 210, containing the feature SHAP values for all possible forecast start points
        over the background series.

        Now, let's get the feature values that were used as model input to forecast the series:

        >>> result.get_feature_values(horizon=1)
                    Y_target_lag-2  Y_target_lag-1  month_futcov_lag-1  month_futcov_lag0
        1956-07-01           284.0           213.0                 3.0                6.0
        1956-10-01           213.0           227.0                 6.0                9.0
        1957-01-01           227.0           308.0                 9.0                0.0
        1957-04-01           308.0           262.0                 0.0                3.0
        1957-07-01           262.0           228.0                 3.0                6.0
        ...                    ...             ...                 ...                ...
        2007-10-01           383.0           394.0                 6.0                9.0
        2008-01-01           394.0           473.0                 9.0                0.0
        2008-04-01           473.0           420.0                 0.0                3.0
        2008-07-01           420.0           390.0                 3.0                6.0
        2008-10-01           390.0           410.0                 6.0                9.0

        shape: (210, 4, 1), freq: QS-OCT, size: 6.56 KB

        And also, we can get the raw `shap.Explanation` object for further processing:

        >>> shap_object = result.get_shap_explanation_object(horizon=1)
        """
        input_type = "foreground" if foreground_series is not None else "background"
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

            foreground_arr, foreground_times = self.explainer.create_shap_input(
                series=foreground_ts,
                past_covariates=foreground_past_cov_ts,
                future_covariates=foreground_future_cov_ts,
                n_samples=None,
                input_type=input_type,
            )

            shap_ = self.explainer.shap_explanations(
                foreground_arr=foreground_arr,
                foreground_times=foreground_times,
                horizons=horizons,
                target_components=target_names,
                **kwargs,
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

        return ShapExplainabilityResult(
            explained_forecasts=shap_values_list,
            feature_values=feature_values_list,
            shap_explanation_object=shap_explanation_object_list,
        )

    def explain_single(
        self,
        foreground_series: TimeSeries | None = None,
        foreground_past_covariates: TimeSeries | None = None,
        foreground_future_covariates: TimeSeries | None = None,
        target_components: Sequence[str] | None = None,
        **kwargs,
    ) -> ShapSingleExplainabilityResult:
        """
        Explains the last forecast of a foreground series (or background, if foreground is not provided) and
        returns a :class:`ShapSingleExplainabilityResult
        <darts.explainability.explainability_result.ShapSingleExplainabilityResult>` of SHAP values.

        The results can then be retrieved with method :func:`get_explanation()
        <darts.explainability.explainability_result.ShapSingleExplainabilityResult.get_explanation>`,
        which returns a multivariate ``TimeSeries`` instance containing the SHAP values for ``target_component``
        starting from the last forecastable timestamp.

        The components of the ``TimeSeries`` correspond to the input features used by the model to produce
        the forecast. See above for the naming convention.

        .. note::
            The forecast explained by this method is equivalent to the one obtained by calling
            ``model.predict(n=output_chunk_length, series=series, ...)`` where ``series`` is either
            ``foreground_series`` or ``background_series`` depending on what was used when calling ``explain_single()``.

        Parameters
        ----------
        foreground_series
            Optionally, one or a sequence of target ``TimeSeries`` to be explained. Can be multivariate.
            Default: ``None``, which means that the background series will be used as foreground.
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
        ShapSingleExplainabilityResult
            The forecast explanations of the specified target components for the single forecasted timestamp.

        Examples
        --------
        Say we have a ``SKLearnModel`` instance with:

            - 1 target component named ``"Y"``,
            - 1 future covariate named ``"month"``,
            - ``lags = 2``, and ``lags_future_covariates = [-1, 0]``.

        Let's explain the background series that the model was trained on:

        >>> from darts.datasets import AusBeerDataset
        >>> from darts.explainability import ShapExplainer
        >>> from darts.models import LinearRegressionModel
        >>> from darts.utils.timeseries_generation import datetime_attribute_timeseries as dta
        >>>
        >>> # load a target series and create future covariates holding the calendar month values
        >>> series = AusBeerDataset().load()
        >>> fc = dta(series, attribute="month", add_length=12)
        >>>
        >>> # create and fit a model
        >>> model = LinearRegressionModel(lags=2, lags_future_covariates=[-1, 0])
        >>> model.fit(series, future_covariates=fc)
        >>>
        >>> # create an explainer; requires background series if the model was trained on multiple series
        >>> explainer = ShapExplainer(model)
        >>> # explain the background forecast (or foreground forecast if passed to `explain_single()`)
        >>> result = explainer.explain_single()
        >>>
        >>> # get explanations for that forecast (and optional component for multivariate models)
        >>> # the feature SHAP values for that forecast
        >>> result.get_explanation()
                    Y_target_lag-2  Y_target_lag-1  month_futcov_lag-1  month_futcov_lag0
        2008-10-01       -8.239364       -3.395013           13.642416          84.727725

        shape: (1, 4, 1), freq: QS-OCT, size: 6.56 KB

        The explanation has length 210, containing the feature SHAP values for all possible forecast start points
        over the background series.

        Now, let's get the feature values that were used as model input to forecast the series:

        >>> result.get_feature_values()
                    Y_target_lag-2  Y_target_lag-1  month_futcov_lag-1  month_futcov_lag0
        2008-10-01           390.0           410.0                 6.0                9.0

        shape: (1, 4, 1), freq: QS-OCT, size: 6.56 KB

        And also, we can get the raw `shap.Explanation` object for further processing:

        >>> shap_object = result.get_shap_explanation_object()
        """
        input_type = "foreground" if foreground_series is not None else "background"
        (
            foreground_series_,
            foreground_past_covariates_,
            foreground_future_covariates_,
            _,
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

        foreground_arr, foreground_times = self.explainer.create_shap_input(
            series=foreground_series_,
            past_covariates=foreground_past_covariates_,
            future_covariates=foreground_future_covariates_,
            n_samples=None,
            input_type=input_type,
        )

        # explain only the last forecasted timestamp
        shap_ = self.explainer.shap_explanations_single(
            foreground_arr=foreground_arr,
            foreground_times=foreground_times,
            target_components=target_names,
            **kwargs,
        )

        freq = foreground_series_[0].freq
        shap_values_dict = {}
        feature_values_dict = {}
        shap_explanation_object_dict = {}
        for t in target_names:
            shap_values_dict[t] = TimeSeries(
                times=generate_index(
                    start=shap_[t].time_index[0],
                    freq=freq,
                    length=shap_[t].values.shape[0],
                ),
                values=shap_[t].values,
                components=shap_[t].feature_names,
            )
            feature_values_dict[t] = TimeSeries(
                times=generate_index(
                    start=shap_[t].time_index[0],
                    freq=freq,
                    length=1,
                ),
                values=shap_[t].data[:1],
                components=shap_[t].feature_names,
            )
            shap_explanation_object_dict[t] = shap_[t]

        return ShapSingleExplainabilityResult(
            explained_components=shap_values_dict,
            feature_values=feature_values_dict,
            shap_explanation_object=shap_explanation_object_dict,
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
        input_type = "foreground" if foreground_series is not None else "background"
        (
            foreground_series_,
            foreground_past_covariates_,
            foreground_future_covariates_,
            _,
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

        foreground_arr, foreground_times = self.explainer.create_shap_input(
            series=foreground_series_,
            past_covariates=foreground_past_covariates_,
            future_covariates=foreground_future_covariates_,
            n_samples=num_samples,
            input_type=input_type,
        )

        shaps_ = self.explainer.shap_explanations(
            foreground_arr=foreground_arr,
            foreground_times=foreground_times,
            horizons=horizons,
            target_components=target_components,
            **kwargs,
        )

        for t in target_components:
            for h in horizons:
                plt.title(
                    f"Target: `{t}` - Horizon: t+{h + self.model.output_chunk_shift}"
                )
                shap.summary_plot(
                    shap_values=shaps_[h][t],
                    features=foreground_arr,
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
        input_type = "foreground" if foreground_series is not None else "background"
        if target_component is None and len(self.target_components_likelihood) > 1:
            raise_log(
                ValueError(
                    f"The `target_component` parameter is required when the model has more than one component. "
                    f"Please select a component from {self.target_components_likelihood}."
                ),
                logger,
            )

        if target_component is None:
            target_component = self.target_components_likelihood[0]

        (
            foreground_series_,
            foreground_past_covariates_,
            foreground_future_covariates_,
            _,
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

        foreground_arr, foreground_times = self.explainer.create_shap_input(
            series=foreground_series_,
            past_covariates=foreground_past_covariates_,
            future_covariates=foreground_future_covariates_,
            n_samples=None,
            input_type=input_type,
        )

        shap_ = self.explainer.shap_explanations(
            foreground_arr=foreground_arr,
            foreground_times=foreground_times,
            horizons=[horizon],
            target_components=[target_component],
            **kwargs,
        )

        return shap.force_plot(
            base_value=shap_[horizon][target_component],
            features=foreground_arr,
            out_names=target_component,
            **(plot_kwargs or {}),
        )
