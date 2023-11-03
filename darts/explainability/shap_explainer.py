"""
Shap Explainer for RegressionModels
------------------------------------
A `shap explainer <https://github.com/slundberg/shap>`_ specifically for time series
forecasting models.

This class is (currently) limited to Darts' `RegressionModel` instances of forecasting models. It uses shap values to
provide "explanations" of each input features. The input features are the different past lags (of the target and/or
past covariates), as well as potential future lags of future covariates used as inputs by the forecasting model to
produce its forecasts. Furthermore, in the case of multivariate series, the features contain each dimension of
each of the (lagged) series.

.. note::
   This explainer is subject to the usual features independence assumption used to compute shap values.
   This means that it does not capture potential indirect influence that some lags
   may have on the target by influencing other lags.

- :func:`explain() <ShapExplainer.explain>` generates the explanations for a given foreground series (or
  background series, if foreground is not provided).
- :func:`summary_plot() <ShapExplainer.summary_plot>` displays a shap plot summary for each horizon and each
  component dimension of the target series.
- :func:`force_plot_from_ts() <ShapExplainer.force_plot_from_ts>` displays a shap force_plot for one target
  and one horizon, for a given target series. It displays shap values of each lag/covariate with an additive force
   layout.
"""

from enum import Enum
from typing import Dict, NewType, Optional, Sequence, Union

import matplotlib.pyplot as plt
import pandas as pd
import shap
from sklearn.multioutput import MultiOutputRegressor

from darts import TimeSeries
from darts.explainability.explainability import _ForecastingModelExplainer
from darts.explainability.explainability_result import ShapExplainabilityResult
from darts.logging import get_logger, raise_if, raise_log
from darts.models.forecasting.regression_model import RegressionModel
from darts.utils.data.tabularization import create_lagged_prediction_data

logger = get_logger(__name__)

MIN_BACKGROUND_SAMPLE = 10


class _ShapMethod(Enum):
    TREE = 0
    GRADIENT = 1
    DEEP = 2
    KERNEL = 3
    SAMPLING = 4
    PARTITION = 5
    LINEAR = 6
    PERMUTATION = 7
    ADDITIVE = 8


ShapMethod = NewType("ShapMethod", _ShapMethod)


class ShapExplainer(_ForecastingModelExplainer):
    model: RegressionModel

    def __init__(
        self,
        model: RegressionModel,
        background_series: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        background_past_covariates: Optional[
            Union[TimeSeries, Sequence[TimeSeries]]
        ] = None,
        background_future_covariates: Optional[
            Union[TimeSeries, Sequence[TimeSeries]]
        ] = None,
        background_num_samples: Optional[int] = None,
        shap_method: Optional[str] = None,
        **kwargs,
    ):
        """ShapExplainer

        **Definitions**

        - A background series is a `TimeSeries` used to train the shap explainer.
        - A foreground series is a `TimeSeries` that can be explained by a shap explainer after it has been fitted.

        Currently, `ShapExplainer` only works with `RegressionModel` forecasting models.
        The number of explained horizons (t+1, t+2, ...) can be at most equal to `output_chunk_length` of `model`.

        Parameters
        ----------
        model
            A `RegressionModel` to be explained. It must be fitted first.
        background_series
            One or several series to *train* the `ShapExplainer` along with any foreground series.
            Consider using a reduced well-chosen background to reduce computation time.
            Optional if `model` was fit on a single target series. By default, it is the `series` used at fitting time.
            Mandatory if `model` was fit on multiple (list of) target series.
        background_past_covariates
            A past covariates series or list of series that the model needs once fitted.
        background_future_covariates
            A future covariates series or list of series that the model needs once fitted.
        background_num_samples
            Optionally, whether to sample a subset of the original background. Randomly picks
            `background_num_samples` training samples of the constructed training dataset
            (using ``shap.utils.sample()``).
            Generally used for faster computation, especially when `shap_method` is
            ``"kernel"`` or ``"permutation"``.
        shap_method
            Optionally, the shap method to apply. By default, an attempt is made
            to select the most appropriate method based on a pre-defined set of known models.
            internal mapping. Supported values : ``"permutation", "partition", "tree", "kernel", "sampling", "linear",
            "deep", "gradient", "additive"``.
        **kwargs
            Optionally, additional keyword arguments passed to `shap_method`.
        Examples
        --------
        >>> from darts.datasets import AirPassengersDataset
        >>> from darts.explainability.shap_explainer import ShapExplainer
        >>> from darts.models import LinearRegressionModel
        >>> series = AirPassengersDataset().load()
        >>> model = LinearRegressionModel(lags=12)
        >>> model.fit(series[:-36])
        >>> shap_explain = ShapExplainer(model)
        >>> results = shap_explain.explain()
        >>> shap_explain.summary_plot()
        >>> shap_explain.force_plot_from_ts()
        """

        # TODO
        # - Optional De-trend  if the timeseries is not stationary.
        # There would be
        # 1) a stationarity test and
        # 2) a de-trend methodology for the target. It can be for
        # example target - moving_average(input_chunk_length).

        if not issubclass(type(model), RegressionModel):
            raise_log(
                ValueError(
                    "Invalid `model` type. Currently, only models of type `RegressionModel` are supported."
                ),
                logger,
            )

        if not model.multi_models:
            raise_log(
                ValueError(
                    "Invalid `multi_models` value `False`. Currently, "
                    "ShapExplainer only supports RegressionModels "
                    "with `multi_models=True`."
                ),
                logger,
            )

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

        if model._is_probabilistic:
            logger.warning(
                "The model is probabilistic, but num_samples=1 will be used for explainability."
            )

        if shap_method is not None:
            shap_method = shap_method.upper()
            if shap_method in _ShapMethod.__members__:
                self.shap_method = _ShapMethod[shap_method]
            else:
                raise_log(
                    ValueError(
                        "Invalid `shap_method`. Please choose one value among the following: ['partition', 'tree', "
                        "'kernel', 'sampling', 'linear', 'deep', 'gradient', 'additive']."
                    )
                )
        else:
            self.shap_method = None

        self.explainers = _RegressionShapExplainers(
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
        foreground_series: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        foreground_past_covariates: Optional[
            Union[TimeSeries, Sequence[TimeSeries]]
        ] = None,
        foreground_future_covariates: Optional[
            Union[TimeSeries, Sequence[TimeSeries]]
        ] = None,
        horizons: Optional[Sequence[int]] = None,
        target_components: Optional[Sequence[str]] = None,
    ) -> ShapExplainabilityResult:
        """
        Explains a foreground time series and returns a :class:`ShapExplainabilityResult
        <darts.explainability.explainability_result.ShapExplainabilityResult>`.
        The results can be retrieved with method :func:`get_explanation()
        <darts.explainability.explainability_result.ShapExplainabilityResult.get_explanation>`.
        The result is a multivariate `TimeSeries` instance containing the 'explanation'
        for the (horizon, target_component) forecast at any timestamp forecastable corresponding to
        the foreground `TimeSeries` input.

        The component name convention of this multivariate `TimeSeries` is:
        ``"{name}_{type_of_cov}_lag_{idx}"``, where:

        - ``{name}`` is the component name from the original foreground series (target, past, or future).
        - ``{type_of_cov}`` is the covariates type. It can take 3 different values:
          ``"target"``, ``"past_cov"`` or ``"future_cov"``.
        - ``{idx}`` is the lag index.

        Parameters
        ----------
        foreground_series
            Optionally, one or a sequence of target `TimeSeries` to be explained. Can be multivariate.
            If not provided, the background `TimeSeries` will be explained instead.
        foreground_past_covariates
            Optionally, one or a sequence of past covariates `TimeSeries` if required by the forecasting model.
        foreground_future_covariates
            Optionally, one or a sequence of future covariates `TimeSeries` if required by the forecasting model.
        horizons
            Optionally, an integer or sequence of integers representing the future time steps to be explained.
            `1` corresponds to the first timestamp being forecasted.
            All values must be `<=output_chunk_length` of the explained forecasting model.
        target_components
            Optionally, a string or sequence of strings with the target components to explain.

        Returns
        -------
        ShapExplainabilityResult
            The forecast explanations

        Examples
        --------
        Say we have a model with 2 target components named ``"T_0"`` and ``"T_1"``,
        3 past covariates with default component names ``"0"``, ``"1"``, and ``"2"``,
        and one future covariate with default component name ``"0"``.
        Also, ``horizons = [1, 2]``.
        The model is a regression model, with ``lags = 3``, ``lags_past_covariates=[-1, -3]``,
        ``lags_future_covariates = [0]``.

        We provide `foreground_series`, `foreground_past_covariates`, `foreground_future_covariates` each of length 5.

        >>> explain_results = explainer.explain(
        >>>     foreground_series=foreground_series,
        >>>     foreground_past_covariates=foreground_past_covariates,
        >>>     foreground_future_covariates=foreground_future_covariates,
        >>>     horizons=[1, 2],
        >>>     target_names=["T_0", "T_1"])
        >>> output = explain_results.get_explanation(horizon=1, component="T_1")
        >>> feature_values = explain_results.get_feature_values(horizon=1, component="T_1")
        >>> shap_objects = explain_results.get_shap_explanation_objects(horizon=1, component="T_1")

        Then the method returns a multivariate TimeSeries containing the *explanations* of
        the `ShapExplainer`, with the following component names:

             - T_0_target_lag-1
             - T_0_target_lag-2
             - T_0_target_lag-3
             - T_1_target_lag-1
             - T_1_target_lag-2
             - T_1_target_lag-3
             - 0_past_cov_lag-1
             - 0_past_cov_lag-3
             - 1_past_cov_lag-1
             - 1_past_cov_lag-3
             - 2_past_cov_lag-1
             - 2_past_cov_lag-3
             - 0_fut_cov_lag_0

        This series has length 3, as the model can explain 5-3+1 forecasts
        (timestamp indexes 4, 5, and 6)
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
                    shap_values_dict_single_h[t] = TimeSeries.from_times_and_values(
                        shap_[h][t].time_index,
                        shap_[h][t].values,
                        columns=shap_[h][t].feature_names,
                    )
                    feature_values_dict_single_h[t] = TimeSeries.from_times_and_values(
                        shap_[h][t].time_index,
                        shap_[h][t].data,
                        columns=shap_[h][t].feature_names,
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
            shap_values_list, feature_values_list, shap_explanation_object_list
        )

    def summary_plot(
        self,
        horizons: Optional[Union[int, Sequence[int]]] = None,
        target_components: Optional[Union[str, Sequence[str]]] = None,
        num_samples: Optional[int] = None,
        plot_type: Optional[str] = "dot",
        **kwargs,
    ) -> Dict[int, Dict[str, shap.Explanation]]:
        """
        Display a shap plot summary for each horizon and each component dimension of the target.
        This method reuses the initial background data as foreground (potentially sampled) to give a general importance
        plot for each feature.
        If no target names and/or no horizons are provided, all summary plots are produced.

        Parameters
        ----------
        horizons
            Optionally, an integer or sequence of integers representing which points/steps in the future to explain,
            starting from the first prediction step at 1. `horizons` must `<=output_chunk_length` of the forecasting
            model.
        target_components
            Optionally, a string or sequence of strings with the target components to explain.
        num_samples
            Optionally, an integer for sampling the foreground series (based on the background),
            for the sake of performance.
        plot_type
            Optionally, specify which of the shap library plot type to use. Can be one of ``'dot', 'bar', 'violin'``.

        Returns
        -------
        shaps_
            A nested dictionary {horizon : {component : shap.Explaination}} containing the raw Explanations for all
            the horizons and components.
        """

        horizons, target_components = self._process_horizons_and_targets(
            horizons, target_components
        )

        if num_samples:
            foreground_X_sampled = shap.utils.sample(
                self.explainers.background_X, num_samples
            )
        else:
            foreground_X_sampled = self.explainers.background_X

        shaps_ = self.explainers.shap_explanations(
            foreground_X_sampled, horizons, target_components
        )

        for t in target_components:
            for h in horizons:
                plt.title("Target: `{}` - Horizon: {}".format(t, "t+" + str(h)))
                shap.summary_plot(
                    shaps_[h][t],
                    foreground_X_sampled,
                    plot_type=plot_type,
                    **kwargs,
                )
        return shaps_

    def force_plot_from_ts(
        self,
        foreground_series: Optional[TimeSeries] = None,
        foreground_past_covariates: Optional[TimeSeries] = None,
        foreground_future_covariates: Optional[TimeSeries] = None,
        horizon: Optional[int] = 1,
        target_component: Optional[str] = None,
        **kwargs,
    ):
        """
        Display a shap force_plot for one target and one horizon, for a given foreground_series.
        It displays shap values of each lag/covariate with an additive force layout.

        Once the plot is displayed, select "original sample ordering"
        to observe the time series chronologically.

        Parameters
        ----------
        foreground_series
            Optionally, the target series to explain. Can be multivariate. If `None`, will use the `background_series`.
        foreground_past_covariates
            Optionally, a past covariate series if required by the forecasting model. If `None`, will use the
            `background_past_covariates`.
        foreground_future_covariates
            Optionally, a future covariate series if required by the forecasting model. If `None`, will use the
            `background_future_covariates`.
        horizon
            Optionally, an integer for the point/step in the future to explain, starting from the first prediction
            step at 1. `horizons` must not be larger than `output_chunk_length`.
        target_component
            Optionally, the target component to plot. If the target series is multivariate, the target component
            must be specified.
        **kwargs
            Optionally, additional keyword arguments passed to `shap.force_plot()`.
        """

        raise_if(
            target_component is None and len(self.target_components) > 1,
            "The component parameter is required when the model has more than one component.",
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
            foreground_X, [horizon], [target_component]
        )

        return shap.force_plot(
            base_value=shap_[horizon][target_component],
            features=foreground_X,
            out_names=target_component,
            **kwargs,
        )


class _RegressionShapExplainers:
    """
    Helper Class to wrap the different cases encountered with shap different explainers, multivariates,
    horizon etc.
    Aim to provide shap values for any type of RegressionModel. Manage the MultioutputRegressor cases.
    For darts RegressionModel only.
    """

    default_sklearn_shap_explainers = {
        # Gradient boosting models
        "LGBMRegressor": _ShapMethod.TREE,
        "CatBoostRegressor": _ShapMethod.TREE,
        "XGBRegressor": _ShapMethod.TREE,
        "GradientBoostingRegressor": _ShapMethod.TREE,
        # Tree models
        "DecisionTreeRegressor": _ShapMethod.TREE,
        "ExtraTreeRegressor": _ShapMethod.TREE,
        # Ensemble model
        "AdaBoostRegressor": _ShapMethod.PERMUTATION,
        "BaggingRegressor": _ShapMethod.PERMUTATION,
        "ExtraTreesRegressor": _ShapMethod.PERMUTATION,
        "HistGradientBoostingRegressor": _ShapMethod.PERMUTATION,
        "RandomForestRegressor": _ShapMethod.PERMUTATION,
        "RidgeCV": _ShapMethod.PERMUTATION,
        "Ridge": _ShapMethod.PERMUTATION,
        # Linear models
        "LinearRegression": _ShapMethod.LINEAR,
        "ARDRegression": _ShapMethod.LINEAR,
        "MultiTaskElasticNet": _ShapMethod.LINEAR,
        "MultiTaskElasticNetCV": _ShapMethod.LINEAR,
        "MultiTaskLasso": _ShapMethod.LINEAR,
        "MultiTaskLassoCV": _ShapMethod.LINEAR,
        "PassiveAggressiveRegressor": _ShapMethod.LINEAR,
        "PoissonRegressor": _ShapMethod.LINEAR,
        "QuantileRegressor": _ShapMethod.LINEAR,
        "RANSACRegressor": _ShapMethod.LINEAR,
        "GammaRegressor": _ShapMethod.LINEAR,
        "HuberRegressor": _ShapMethod.LINEAR,
        "BayesianRidge": _ShapMethod.LINEAR,
        "SGDRegressor": _ShapMethod.LINEAR,
        "TheilSenRegressor": _ShapMethod.LINEAR,
        "TweedieRegressor": _ShapMethod.LINEAR,
        # Gaussian process
        "GaussianProcessRegressor": _ShapMethod.PERMUTATION,
        # neighbors
        "KNeighborsRegressor": _ShapMethod.PERMUTATION,
        "RadiusNeighborsRegressor": _ShapMethod.PERMUTATION,
        # Neural network
        "MLPRegressor": _ShapMethod.PERMUTATION,
    }

    def __init__(
        self,
        model: RegressionModel,
        n: int,
        target_components: Sequence[str],
        past_covariates_components: Sequence[str],
        future_covariates_components: Sequence[str],
        background_series: Sequence[TimeSeries],
        background_past_covariates: Sequence[TimeSeries],
        background_future_covariates: Sequence[TimeSeries],
        shap_method: _ShapMethod,
        background_num_samples: Optional[int] = None,
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
                        self.model.get_multioutput_estimator(horizon=i, target_dim=j),
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
        horizons: Optional[Sequence[int]] = None,
        target_components: Optional[Sequence[str]] = None,
    ) -> Dict[int, Dict[str, shap.Explanation]]:

        """
        Return a dictionary of dictionaries of shap.Explanation instances:
        - the first dimension corresponds to the n forecasts ahead we want to explain (Horizon).
        - the second dimension corresponds to each component of the target time series.
        Parameters
        ----------
        foreground_X
            the Dataframe of lags features specific of darts RegressionModel.
        horizons
            Optionally, a list of integers representing which points/steps in the future we want to explain,
            starting from the first prediction step at 1. Currently, only forecasting models are supported which
            provide an `output_chunk_length` parameter. `horizons` must not be larger than `output_chunk_length`.
        target_components
            Optionally, a list of strings with the target components we want to explain.

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
                    explainer = self.explainers[h - 1][t_idx](foreground_X)
                    explainer.base_values = explainer.base_values.ravel()
                    explainer.time_index = foreground_X.index
                    tmp_n[t] = explainer
                shap_explanations[h] = tmp_n
        else:
            # the native multioutput forces us to recompute all horizons and targets
            shap_explanation_tmp = self.explainers(foreground_X)
            for h in horizons:
                tmp_n = {}
                for t_idx, t in enumerate(target_components):
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

    def _build_explainer_sklearn(
        self,
        model_sklearn,
        background_X: pd.DataFrame,
        shap_method: Optional[ShapMethod] = None,
        **kwargs,
    ):

        model_name = type(model_sklearn).__name__

        # no shap methods - we need to take the default one
        if shap_method is None:
            if model_name in self.default_sklearn_shap_explainers:
                shap_method = self.default_sklearn_shap_explainers[model_name]
            else:
                shap_method = _ShapMethod.KERNEL

        # we define properly the explainer given a shap method
        if shap_method == _ShapMethod.TREE:
            if kwargs.get("feature_perturbation") == "interventional":
                explainer = shap.TreeExplainer(model_sklearn, background_X, **kwargs)
            else:
                explainer = shap.TreeExplainer(model_sklearn, **kwargs)
        elif shap_method == _ShapMethod.PERMUTATION:
            explainer = shap.PermutationExplainer(
                model_sklearn.predict, background_X, **kwargs
            )
        elif shap_method == _ShapMethod.PARTITION:
            explainer = shap.PermutationExplainer(
                model_sklearn.predict, background_X, **kwargs
            )
        elif shap_method == _ShapMethod.KERNEL:
            explainer = shap.KernelExplainer(
                model_sklearn.predict, background_X, keep_index=True, **kwargs
            )
        elif shap_method == _ShapMethod.LINEAR:
            explainer = shap.LinearExplainer(model_sklearn, background_X, **kwargs)
        elif shap_method == _ShapMethod.DEEP:
            explainer = shap.LinearExplainer(model_sklearn, background_X, **kwargs)
        elif shap_method == _ShapMethod.ADDITIVE:
            explainer = shap.AdditiveExplainer(model_sklearn, background_X, **kwargs)
        else:
            raise ValueError(
                "shap_method must be one of the following: "
                + ", ".join([e.value for e in _ShapMethod])
            )

        logger.info("The shap method used is of type: " + str(type(explainer)))

        return explainer

    def _create_regression_model_shap_X(
        self,
        target_series: Optional[Union[TimeSeries, Sequence[TimeSeries]]],
        past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]],
        future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]],
        n_samples: Optional[int] = None,
        train: bool = False,
    ) -> pd.DataFrame:
        """
        Creates the shap format input for regression models.
        The output is a pandas DataFrame representing all lags of different covariates, and with adequate
        column names in order to map feature / shap values.
        It uses create_lagged_data also used in RegressionModel to build the tabular dataset.

        """

        lags_list = self.model._get_lags("target")
        lags_past_covariates_list = self.model._get_lags("past")
        lags_future_covariates_list = self.model._get_lags("future")

        X, indexes = create_lagged_prediction_data(
            target_series=target_series if lags_list else None,
            past_covariates=past_covariates if lags_past_covariates_list else None,
            future_covariates=future_covariates
            if lags_future_covariates_list
            else None,
            lags=lags_list,
            lags_past_covariates=lags_past_covariates_list if past_covariates else None,
            lags_future_covariates=lags_future_covariates_list
            if future_covariates
            else None,
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
                        "The number of samples in the background dataset is too small to compute shap values."
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
