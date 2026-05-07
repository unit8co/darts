from collections.abc import Sequence
from enum import Enum
from typing import NewType

import numpy as np
import pandas as pd
import shap
from sklearn.multioutput import MultiOutputRegressor

from darts import TimeSeries
from darts.logging import get_logger, raise_log
from darts.models.forecasting.sklearn_model import SKLearnModel
from darts.typing import TimeSeriesLike
from darts.utils.data.tabularization import create_lagged_prediction_data

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
