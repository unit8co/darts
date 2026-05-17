from abc import ABC, abstractmethod
from collections.abc import Sequence
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
import shap

from darts import TimeSeries
from darts.logging import get_logger, raise_log
from darts.models.forecasting.forecasting_model import ForecastingModel
from darts.typing import TimeIndex, TimeSeriesLike

logger = get_logger(__name__)

MIN_BACKGROUND_SAMPLE = 10
MAX_BACKGROUND_SAMPLE = 1000


class SHAPMethod(Enum):
    TREE = 0
    DEEP = 2
    KERNEL = 3
    SAMPLING = 4
    PARTITION = 5
    LINEAR = 6
    PERMUTATION = 7
    ADDITIVE = 8


class BaseShapExplainer(ABC):
    def __init__(
        self,
        model: ForecastingModel,
        n: int,
        target_components: Sequence[str],
        past_covariates_components: Sequence[str] | None,
        future_covariates_components: Sequence[str] | None,
        static_covariates_components: Sequence[str] | None,
        background_series: Sequence[TimeSeries],
        background_past_covariates: Sequence[TimeSeries] | None,
        background_future_covariates: Sequence[TimeSeries] | None,
        background_num_samples: int | None,
        shap_method: str | None,
        batch_size: int | None = None,
        **kwargs,
    ):
        """
        Helper Class to wrap the different cases encountered with SHAP different explainers, multivariates,
        horizon etc.
        Aim to provide SHAP values for any type of SKLearnModel. Manage the MultioutputRegressor cases.
        For darts SKLearnModel only.
        """
        self._validate_model(model)

        if isinstance(shap_method, str):
            shap_method_upper = shap_method.upper()
            if shap_method_upper in {sm.name for sm in self._supported_shap_methods}:
                self.shap_method = SHAPMethod[shap_method_upper]
            else:
                raise_log(
                    ValueError(
                        f"Invalid `shap_method='{shap_method}'`. Expected one of the following:"
                        f" {[e.name.lower() for e in self._supported_shap_methods]}."
                    )
                )
        else:
            self.shap_method = self._get_default_shap_method(model)

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

        self.model = model

        likelihood = model.likelihood
        if likelihood is not None:
            target_components_likelihood = likelihood.component_names(
                components=target_components
            )
            logger.warning(
                f"The explained model is probabilistic and the SHAP explanations will be computed for the likelihood "
                f"parameters of the target components, which includes the following components: "
                f"{target_components_likelihood}. Adjust the `target_components` argument accordingly."
            )
        else:
            target_components_likelihood = target_components
        self.target_components_likelihood = target_components_likelihood
        self.target_components = target_components
        self.past_covariates_components = past_covariates_components
        self.future_covariates_components = future_covariates_components
        self.static_covariates_components = static_covariates_components

        self.n_targets = len(target_components)
        self.n_targets_likelihood = len(target_components_likelihood)
        self.n_past_covs = (
            len(past_covariates_components)
            if past_covariates_components is not None
            else 0
        )
        self.n_future_covs = (
            len(future_covariates_components)
            if future_covariates_components is not None
            else 0
        )
        self.n_static_covs = (
            len(static_covariates_components)
            if static_covariates_components is not None
            else 0
        )
        self.n_variables = self.n_targets + self.n_past_covs + self.n_future_covs

        self.n = n
        self.output_chunk_length = model.output_chunk_length or 1
        self.output_chunk_shift = model.output_chunk_shift
        self.batch_size: int = batch_size or getattr(model, "batch_size", 0)
        self.single_output = self.n == 1 and self.n_targets_likelihood == 1

        self.background_series = background_series
        self.background_past_covariates = background_past_covariates
        self.background_future_covariates = background_future_covariates
        self.feature_names = self._build_feature_names()

        self.background_X, _, _ = self.create_shap_array(
            series=self.background_series,
            past_covariates=self.background_past_covariates,
            future_covariates=self.background_future_covariates,
            n_samples=background_num_samples,
            input_type="background",
        )

        self.explainer = self._build_explainer(
            model=self.model,
            background_X=self.background_X,
            shap_method=self.shap_method,
            **kwargs,
        )

    def shap_explanations(
        self,
        foreground_X: pd.DataFrame,
        horizons: Sequence[int],
        target_components: Sequence[str],
        **kwargs,
    ) -> dict[int, dict[str, shap.Explanation]]:
        """
        Computes SHAP explanations for the given foreground data, horizons, and target components.
        It returns a nested dictionary of SHAP Explanation objects for each horizon and target component:

        - first dimension: each step in the forecast horizon.
        - second dimension: each component of the target time series.

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
        **kwargs
            Additional keyword arguments to be passed to the SHAP explainer when calling it for explanations.
             This can include parameters for sampling or approximation methods used by some SHAP explainers.

        Returns
        -------
        dict[int, dict[str, shap.Explanation]]
            A nested dictionary ``{horizon : {target_component : shap.Explanation}}`` containing the SHAP Explanation
            objects for each horizon and target component, where the SHAP values are extracted and reshaped for
            easier accessibility.
        """
        # create a nested dictionary {horizon : {target_component : shap.Explanation}}
        # for better accessibility of the explanations
        explanations = {}
        if isinstance(self.explainer, dict):
            for h in horizons:
                tmp_n = {}
                for t_idx, t in enumerate(self.target_components_likelihood):
                    if t not in target_components:
                        continue
                    explanation = self.explainer[h - 1][t_idx](foreground_X, **kwargs)
                    explanation.base_values = explanation.base_values.ravel()
                    explanation.time_index = foreground_X.index
                    tmp_n[t] = explanation
                explanations[h] = tmp_n
            return explanations

        # the native multioutput forces us to recompute all horizons and targets
        explanation: shap.Explanation = self.explainer(foreground_X, **kwargs)
        base_values: np.ndarray = explanation.base_values
        if base_values.ndim == 1:
            # for unknown reasons, some SHAP explainers (`shap.SamplingExplainer`) returns 1D base values, which
            # need to be reshaped and repeated to match the expected shape for accessibility
            base_values = base_values[np.newaxis, :]
            base_values = np.repeat(
                base_values, repeats=explanation.values.shape[0], axis=0
            )

        for h in horizons:
            tmp_n = {}
            for t_idx, t in enumerate(self.target_components_likelihood):
                if t not in target_components:
                    continue

                if not self.single_output:
                    tmp_t = shap.Explanation(
                        values=explanation.values[
                            :, :, self.n_targets_likelihood * (h - 1) + t_idx
                        ],
                        data=explanation.data,
                        base_values=base_values[
                            :, self.n_targets_likelihood * (h - 1) + t_idx
                        ].ravel(),
                        feature_names=self.feature_names,
                    )
                else:
                    tmp_t = explanation
                    tmp_t.base_values = base_values.ravel()

                # TODO: for torch we should infer the index somehow
                if hasattr(foreground_X, "index"):
                    tmp_t.time_index = foreground_X.index

                tmp_n[t] = tmp_t
            explanations[h] = tmp_n

        return explanations

    def shap_explanations_single(
        self,
        foreground_X: pd.DataFrame,
        target_components: Sequence[str],
        **kwargs,
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
        **kwargs
            Additional keyword arguments to be passed to the SHAP explainer when calling it for explanations.
            This can include parameters for sampling or approximation methods used by some SHAP explainers.

        Returns
        -------
        dict[str, shap.Explanation]
            A dictionary ``{target_component : shap.Explanation}`` containing the SHAP Explanation objects for each
            target component, where the SHAP values are extracted and reshaped for easier accessibility.
        """
        # create a nested dictionary {target_component : shap.Explanation}
        # for better accessibility of the explanations
        explanations = {}
        if isinstance(self.explainer, dict):
            for t_idx, t in enumerate(self.target_components):
                if t not in target_components:
                    continue
                values_list, data_list, base_values_list = [], [], []
                feature_names = None
                for h in range(1, self.n + 1):
                    explanation = self.explainer[h - 1][t_idx](foreground_X, **kwargs)
                    values_list.append(explanation.values.ravel())
                    data_list.append(explanation.data.ravel())
                    base_values_list.append(explanation.base_values.ravel())
                    if feature_names is None:
                        feature_names = explanation.feature_names
                values = np.array(values_list)
                data = np.array(data_list)
                base_values = np.array(base_values_list).ravel()
                explanations[t] = shap.Explanation(
                    values=values,
                    data=data,
                    base_values=base_values,
                    feature_names=feature_names,
                )
            return explanations

        # the native multioutput forces us to recompute all horizons and targets
        explanation: shap.Explanation = self.explainer(foreground_X, **kwargs)
        base_values: np.ndarray = explanation.base_values
        if base_values.ndim == 1:
            # for unknown reasons, some SHAP explainers (`shap.SamplingExplainer`) returns 1D base values, which
            # need to be reshaped and repeated to match the expected shape for accessibility
            base_values = base_values[np.newaxis, :]
            base_values = np.repeat(
                base_values, repeats=explanation.values.shape[0], axis=0
            )

        for t_idx, t in enumerate(self.target_components_likelihood):
            if t not in target_components:
                continue

            if not self.single_output:
                tmp_t = shap.Explanation(
                    values=explanation.values[
                        0, :, t_idx :: self.n_targets_likelihood
                    ].T,
                    data=np.repeat(explanation.data, repeats=self.n, axis=0),
                    base_values=base_values[
                        :, t_idx :: self.n_targets_likelihood
                    ].ravel(),
                    feature_names=self.feature_names,
                )
            else:
                tmp_t = shap.Explanation(
                    values=explanation.values.reshape(1, -1),
                    data=explanation.data,
                    base_values=base_values.ravel(),
                    feature_names=self.feature_names,
                )
            explanations[t] = tmp_t

            # # TODO: for torch, we might need:
            # #  tmp_t = shap.Explanation(
            # #      values=explanation.values[0, :, t_idx :: self.n_targets_likelihood].T,
            # #      data=np.repeat(explanation.data, repeats=self.n, axis=0),
            # #      base_values=base_values[0, t_idx :: self.n_targets_likelihood].ravel(),
            # #      feature_names=self.feature_names,
            # #  )
        return explanations

    @abstractmethod
    def _build_feature_names(self) -> list[str]:
        """
        Builds the feature names for the SHAP explanations based on the input features used by the
        forecasting model. See above for the naming convention.
        """

    @staticmethod
    @abstractmethod
    def _build_explainer(
        model: ForecastingModel,
        background_X: Any,
        shap_method: SHAPMethod,
        **kwargs,
    ) -> Any:
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
            specified in the enum ``SHAPMethod``.
        **kwargs
            Additional keyword arguments to be passed to the SHAP explainer constructor.
        """

    @abstractmethod
    def create_shap_array(
        self,
        series: TimeSeriesLike,
        past_covariates: TimeSeriesLike | None,
        future_covariates: TimeSeriesLike | None,
        n_samples: int | None = None,
        input_type: str = "background",
    ) -> tuple[
        np.ndarray | pd.DataFrame, list[dict[str, Any]] | None, TimeIndex | None
    ]:
        """
        Creates the SHAP array for the given input series and covariates, by following the logic of the model's
        prediction / inference dataset and prediction step. It returns the SHAP array, the schemas of the
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
        input_type
            A string indicating whether the SHAP array is being created for the background or foreground data. This
            affects how the dataset is sampled and how the bounds are created. Default: ``background``.
        """

    @property
    @abstractmethod
    def _supported_shap_methods(self) -> set[SHAPMethod]:
        """Specifies the supported SHAP methods."""

    @abstractmethod
    def _get_default_shap_method(self, model) -> SHAPMethod:
        """Return the default SHAP method."""

    @abstractmethod
    def _validate_model(self, model: ForecastingModel) -> None:
        """Validates the model."""
