import numpy as np
import pandas as pd
import shap
from sklearn.multioutput import MultiOutputRegressor

from darts.explainability.shap.base_explainer import BaseShapExplainer, SHAPMethod
from darts.logging import get_logger, raise_log
from darts.models.forecasting.sklearn_model import SKLearnModel
from darts.typing import TimeSeriesLike
from darts.utils.data.tabularization import create_lagged_prediction_data
from darts.utils.multioutput import MultiOutputMixin

logger = get_logger(__name__)

MIN_BACKGROUND_SAMPLE = 10
MAX_BACKGROUND_SAMPLE = 1000


class SKLearnShapExplainer(BaseShapExplainer):
    model: SKLearnModel
    default_sklearn_shap_explainers: dict[str, SHAPMethod] = {
        # Gradient boosting models
        "LGBMRegressor": SHAPMethod.TREE,
        "CatBoostRegressor": SHAPMethod.TREE,
        "XGBRegressor": SHAPMethod.TREE,
        "GradientBoostingRegressor": SHAPMethod.TREE,
        "HistGradientBoostingRegressor": SHAPMethod.TREE,
        # Tree models
        "DecisionTreeRegressor": SHAPMethod.TREE,
        "ExtraTreeRegressor": SHAPMethod.TREE,
        "ExtraTreesRegressor": SHAPMethod.TREE,
        "RandomForestRegressor": SHAPMethod.TREE,
        # Ensemble model
        "AdaBoostRegressor": SHAPMethod.PERMUTATION,
        "BaggingRegressor": SHAPMethod.PERMUTATION,
        "RidgeCV": SHAPMethod.PERMUTATION,
        "Ridge": SHAPMethod.PERMUTATION,
        # Linear models
        "LinearRegression": SHAPMethod.LINEAR,
        "ARDRegression": SHAPMethod.LINEAR,
        "MultiTaskElasticNet": SHAPMethod.LINEAR,
        "MultiTaskElasticNetCV": SHAPMethod.LINEAR,
        "MultiTaskLasso": SHAPMethod.LINEAR,
        "MultiTaskLassoCV": SHAPMethod.LINEAR,
        "PassiveAggressiveRegressor": SHAPMethod.LINEAR,
        "PoissonRegressor": SHAPMethod.LINEAR,
        "QuantileRegressor": SHAPMethod.LINEAR,
        "RANSACRegressor": SHAPMethod.LINEAR,
        "GammaRegressor": SHAPMethod.LINEAR,
        "HuberRegressor": SHAPMethod.LINEAR,
        "BayesianRidge": SHAPMethod.LINEAR,
        "SGDRegressor": SHAPMethod.LINEAR,
        "TheilSenRegressor": SHAPMethod.LINEAR,
        "TweedieRegressor": SHAPMethod.LINEAR,
        # Gaussian process
        "GaussianProcessRegressor": SHAPMethod.PERMUTATION,
        # neighbors
        "KNeighborsRegressor": SHAPMethod.PERMUTATION,
        "RadiusNeighborsRegressor": SHAPMethod.PERMUTATION,
        # Neural network
        "MLPRegressor": SHAPMethod.PERMUTATION,
    }

    def _build_explainer(
        self,
        model: SKLearnModel,
        background_arr: np.ndarray,
        shap_method: SHAPMethod,
        **kwargs,
    ) -> shap.Explainer | dict[int, dict[int, shap.Explainer]]:
        if not isinstance(self.model.model, MultiOutputRegressor):
            return self._build_explainer_sklearn(
                model_sklearn=model.model,
                background_arr=background_arr,
                shap_method=shap_method,
                **kwargs,
            )

        explainers = {}
        for i in range(self.n):
            explainers[i] = {}
            for j in range(self.n_targets_likelihood):
                explainers[i][j] = self._build_explainer_sklearn(
                    model_sklearn=model.get_estimator(horizon=i, target_dim=j),
                    background_arr=background_arr,
                    shap_method=shap_method,
                    **kwargs,
                )
        return explainers

    def _build_explainer_sklearn(
        self,
        model_sklearn,
        background_arr: np.ndarray,
        shap_method: SHAPMethod,
        **kwargs,
    ) -> shap.Explainer:
        # we define properly the explainer given a shap method
        if shap_method == SHAPMethod.TREE:
            if kwargs.get("feature_perturbation") == "interventional":
                explainer = shap.TreeExplainer(model_sklearn, background_arr, **kwargs)
            else:
                explainer = shap.TreeExplainer(model_sklearn, **kwargs)
        elif shap_method == SHAPMethod.PERMUTATION:
            explainer = shap.PermutationExplainer(
                model_sklearn.predict, background_arr, **kwargs
            )
        elif shap_method == SHAPMethod.PARTITION:
            explainer = shap.PartitionExplainer(
                model_sklearn.predict, background_arr, **kwargs
            )
        elif shap_method == SHAPMethod.KERNEL:
            explainer = shap.KernelExplainer(
                model_sklearn.predict, background_arr, keep_index=True, **kwargs
            )
        elif shap_method == SHAPMethod.LINEAR:
            explainer = shap.LinearExplainer(model_sklearn, background_arr, **kwargs)
        elif shap_method == SHAPMethod.ADDITIVE:
            explainer = shap.AdditiveExplainer(model_sklearn, background_arr, **kwargs)
        else:
            raise_log(ValueError(f"Unknown SHAP method {shap_method}"), logger=logger)

        logger.info("The SHAP method used is of type: " + str(type(explainer)))
        return explainer

    def create_shap_array(
        self,
        series: TimeSeriesLike,
        past_covariates: TimeSeriesLike | None,
        future_covariates: TimeSeriesLike | None,
        n_samples: int | None = None,
        input_type: str = "background",
    ) -> tuple[np.ndarray, pd.Index]:
        lags_list = self.model._get_lags("target")
        lags_past_covariates_list = self.model._get_lags("past")
        lags_future_covariates_list = self.model._get_lags("future")

        X, indexes = create_lagged_prediction_data(
            target_series=series if lags_list else None,
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

        if input_type == "background" and len(X) <= MIN_BACKGROUND_SAMPLE:
            raise_log(
                ValueError(
                    "The number of samples in the background dataset is too small to compute SHAP values."
                )
            )

        index_complete = indexes[0]
        for index_i in indexes[1:]:
            index_complete = index_complete.append(index_i)

        if n_samples:
            X = shap.utils.sample(X, n_samples)

        return X, index_complete

    def _build_feature_names(self) -> list[str]:
        return self.model.lagged_feature_names

    @property
    def _supported_shap_methods(self) -> set[SHAPMethod]:
        return {
            SHAPMethod.TREE,
            SHAPMethod.DEEP,
            SHAPMethod.KERNEL,
            SHAPMethod.SAMPLING,
            SHAPMethod.PARTITION,
            SHAPMethod.LINEAR,
            SHAPMethod.PERMUTATION,
            SHAPMethod.ADDITIVE,
        }

    def _get_default_shap_method(self, model: SKLearnModel) -> SHAPMethod:
        if isinstance(model.model, MultiOutputMixin):
            sklearn_model = model.get_estimator(horizon=0, target_dim=0)
        else:
            sklearn_model = model.model

        model_name = type(sklearn_model).__name__
        if model_name in self.default_sklearn_shap_explainers:
            shap_method = self.default_sklearn_shap_explainers[model_name]
        else:
            shap_method = SHAPMethod.KERNEL
        return shap_method

    def _validate_model(self, model: SKLearnModel) -> None:
        if not isinstance(model, SKLearnModel):
            raise_log(
                ValueError(
                    f"Invalid `model` type: `{type(model)}`. Only models of type "
                    f"`SKLearnModel` are supported."
                ),
                logger,
            )

        if not model.multi_models:
            raise_log(
                ValueError(
                    "Invalid `multi_models` value `False`. Currently, "
                    "ShapExplainer only supports SKLearnModels "
                    "with `multi_models=True`."
                ),
                logger,
            )

        if model.supports_probabilistic_prediction:
            logger.warning(
                "The model is probabilistic, but num_samples=1 will be used for explainability."
            )
