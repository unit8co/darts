"""
Shap Explainer for RegressionModels
------------------------------------
A `shap explainer <https://github.com/slundberg/shap>`_ specifically for time series
forecasting models.

This class is (currently) limited to Darts' `RegressionModel` instances of forecasting models.
It uses shap values to provide "explanations" of each input features.
The input features are the different past lags (of the target and/or past covariates),
as well as potential future lags of future covariates used as inputs by the forecasting
model to produce its forecasts.
Furthermore, in the case of multivariate series, the features contain each dimension of
each of the (lagged) series.

.. note::
   This explainer is subject to the usual features independence assumption used to compute shap values.
   This means that it does not capture potential indirect influence that some lags
   may have on the target by influencing other lags.

"""

from enum import Enum
from typing import Dict, NewType, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import pandas as pd
import shap
from numpy import integer
from sklearn.multioutput import MultiOutputRegressor

from darts import TimeSeries
from darts.explainability.explainability import (
    ExplainabilityResult,
    ForecastingModelExplainer,
)
from darts.logging import get_logger, raise_if, raise_log
from darts.models.forecasting.regression_model import RegressionModel
from darts.utils.data.tabularization import _create_lagged_data
from darts.utils.utils import series2seq

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


class ShapExplainer(ForecastingModelExplainer):
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

        Currently, ShapExplainer only works with `RegressionModel` forecasting models.
        The number of explained horizons (t+1, t+2, ...) can be at most equal to `output_chunk_length` of `model`.

        Parameters
        ----------
        model
            A `ForecastingModel` to be explained. It must be fitted first.
        background_series
            One or several series to *train* the `ForecastingModelExplainer` along with any foreground series.
            Consider using a reduced well-chosen backgroundto to reduce computation time.
                - optional if `model` was fit on a single target series. By default,
                  it is the `series` used at fitting time.

                - mandatory if `model` was fit on multiple (list of) target series.
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
        >>> from darts.explainability.shap_explainer import ShapExplainer
        >>> from darts.models import LinearRegressionModel
        >>> series = AirPassengersDataset().load()
        >>> model = LinearRegressionModel(lags=12)
        >>> model.fit(series[:-36])
        >>> shap_explain = ShapExplainer(model)
        >>> shap_explain.summary_plot()
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

        super().__init__(
            model,
            background_series,
            background_past_covariates,
            background_future_covariates,
        )

        # As we only use RegressionModel, we fix the forecast n step ahead we want to explain as
        # output_chunk_length
        self.n = self.model.output_chunk_length

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
    ) -> ExplainabilityResult:
        super().explain(
            foreground_series, foreground_past_covariates, foreground_future_covariates
        )

        if foreground_series is None:
            foreground_series = self.background_series
            foreground_past_covariates = self.background_past_covariates
            foreground_future_covariates = self.background_future_covariates
        else:
            foreground_series = series2seq(foreground_series)
            foreground_past_covariates = series2seq(foreground_past_covariates)
            foreground_future_covariates = series2seq(foreground_future_covariates)

            if self.model.encoders.encoding_available:
                (
                    foreground_past_covariates,
                    foreground_future_covariates,
                ) = self.model.generate_fit_encodings(
                    series=foreground_series,
                    past_covariates=foreground_past_covariates,
                    future_covariates=foreground_future_covariates,
                )

        horizons, target_names = self._check_horizons_and_targets(
            horizons, target_components
        )

        shap_values_list = []

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
            for h in horizons:
                tmp = {}
                for t in target_names:
                    tmp[t] = TimeSeries.from_times_and_values(
                        shap_[h][t].time_index,
                        shap_[h][t].values,
                        columns=shap_[h][t].feature_names,
                    )
                shap_values_dict[h] = tmp

            shap_values_list.append(shap_values_dict)

        if len(shap_values_list) == 1:
            shap_values_list = shap_values_list[0]

        return ExplainabilityResult(shap_values_list)

    def summary_plot(
        self,
        horizons: Optional[Sequence[int]] = None,
        target_components: Optional[Sequence[str]] = None,
        num_samples: Optional[int] = None,
        plot_type: Optional[str] = "dot",
        **kwargs,
    ):
        """
        Display a shap plot summary for each horizon and each component dimension of the target.
        This method reuses the initial background data as foreground (potentially sampled) to give a general importance
        plot for each feature.
        If no target names and/or no horizons are provided, all summary plots are produced.

        Parameters
        ----------
        horizons
            Optionally, a list of integers representing which points/steps in the future to explain,
            starting from the first prediction step at 1. `horizons` must not be larger than
            `output_chunk_length`.
        target_components
            Optionally, a list of strings with the target components to be explained.
        num_samples
            Optionally, an integer for sampling the foreground series (based on the backgound),
            for the sake of performance.
        plot_type
            Optionally, specify which of the propres shap library plot type to use. Can be one of
            ``'dot', 'bar', 'violin'``.

        """

        horizons, target_components = self._check_horizons_and_targets(
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

    def force_plot_from_ts(
        self,
        foreground_series: TimeSeries = None,
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
            The target series to explain. Can be multivariate.
        foreground_past_covariates
            Optionally, a past covariate series if required by the forecasting model.
        foreground_future_covariates
            Optionally, a future covariate series if required by the forecasting model.
        horizon
            Optionally, an integer for the point/step in the future to explain,
            starting from the first
            prediction step at 1. `horizons` must not be larger than `output_chunk_length`.
            by default, horizon = 1.
        target_component
            Optionally, the target component to plot. If the target series is multivariate,
            the target component must be specified.
        **kwargs
            Optionally, additional keyword arguments passed to `shap.force_plot()`.
        """

        raise_if(
            target_component is None and len(self.target_components) > 1,
            "The component parameter is required when the model has more than one component.",
        )

        if target_component is None:
            target_component = self.target_components[0]

        horizon, target_component = self._check_horizons_and_targets(
            horizon, target_component
        )

        horizon, target_component = horizon[0], target_component[0]

        if self.model.encoders.encoding_available:
            (
                foreground_past_covariates,
                foreground_future_covariates,
            ) = self.model.generate_fit_encodings(
                series=foreground_series,
                past_covariates=foreground_past_covariates,
                future_covariates=foreground_future_covariates,
            )

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

    def _check_horizons_and_targets(
        self,
        horizons: Union[int, Sequence[int]],
        target_components: Union[str, Sequence[str]],
    ) -> Tuple[Sequence[int], Sequence[str]]:

        if target_components is not None:
            if isinstance(target_components, str):
                target_components = [target_components]
            raise_if(
                any(
                    [
                        target_name not in self.target_components
                        for target_name in target_components
                    ]
                ),
                "One of the target names doesn't exist. Please review your target_names input",
            )
        else:
            target_components = self.target_components

        if horizons is not None:
            if isinstance(horizons, int):
                horizons = [horizons]

            raise_if(max(horizons) > self.n, "One of the horizons is too large.")
            raise_if(min(horizons) < 1, "One of the horizons is too small.")
        else:
            horizons = range(1, self.n + 1)

        return horizons, target_components


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
        shap_method: ShapMethod,
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
        foreground_X,
        horizons: Optional[Sequence[int]] = None,
        target_components: Optional[Sequence[str]] = None,
    ) -> Dict[integer, Dict[str, shap.Explanation]]:

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
        target_names
            Optionally, a list of strings with the target components we want to explain.

        """

        # create a unified dictionary between multiOutputRegressor estimators and
        # native multiOutput estimators
        shap_explanations = {}
        if self.is_multioutputregressor:

            for h in horizons:
                tmp_n = {}
                for t_idx, t in enumerate(target_components):
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
                    if not self.single_output:
                        tmp_t = shap.Explanation(
                            shap_explanation_tmp.values[
                                :, :, self.target_dim * (h - 1) + t_idx
                            ]
                        )
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
        target_series,
        past_covariates,
        future_covariates,
        n_samples=None,
        train=False,
    ) -> pd.DataFrame:
        """
        Creates the shap format input for regression models.
        The output is a pandas DataFrame representing all lags of different covariates, and with adequate
        column names in order to map feature / shap values.
        It uses create_lagged_data also used in RegressionModel to build the tabular dataset.

        """

        lags_list = self.model.lags.get("target")
        lags_past_covariates_list = self.model.lags.get("past")
        lags_future_covariates_list = self.model.lags.get("future")

        X, _, indexes = _create_lagged_data(
            target_series,
            self.n,
            past_covariates,
            future_covariates,
            lags_list,
            lags_past_covariates_list,
            lags_future_covariates_list,
            is_training=False,
        )

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

        # We keep the creation order of the different lags/features in create_lagged_data
        lags_names_list = []
        if lags_list:
            for lag in lags_list:
                for t_name in self.target_components:
                    lags_names_list.append(t_name + "_target_lag" + str(lag))
        if lags_past_covariates_list:
            for lag in lags_past_covariates_list:
                for t_name in self.past_covariates_components:
                    lags_names_list.append(t_name + "_past_cov_lag" + str(lag))
        if lags_future_covariates_list:
            for lag in lags_future_covariates_list:
                for t_name in self.future_covariates_components:
                    lags_names_list.append(t_name + "_fut_cov_lag" + str(lag))

        X = X.rename(
            columns={
                name: lags_names_list[idx]
                for idx, name in enumerate(X.columns.to_list())
            }
        )

        return X
