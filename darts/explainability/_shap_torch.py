from collections.abc import Sequence
from enum import Enum

import numpy as np
import shap

from darts.logging import get_logger

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
    PERMUTATION = 7


def _available_shap_methods() -> list[str]:
    return [method.name.lower() for method in _SHAPMethod]


class _DeepSHAPExplainer:
    n_targets: int

    def shap_explanations(
        self,
        foreground_X: np.ndarray,
        horizons: Sequence[int],
        target_components: Sequence[str],
        **kwargs,
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
        shap_explanation_tmp: shap.Explanation = self.explainer(foreground_X, **kwargs)
        shap_values: np.ndarray = shap_explanation_tmp.values
        shap_data: np.ndarray = shap_explanation_tmp.data
        shap_base_values: np.ndarray = shap_explanation_tmp.base_values
        if shap_base_values.ndim == 1:
            # for unknown reasons, some SHAP explainers (`shap.SamplingExplainer`) returns 1D base values, which
            # need to be reshaped and repeated to match the expected shape for accessibility
            shap_base_values = shap_base_values[np.newaxis, :]
            shap_base_values = np.repeat(
                shap_base_values, repeats=shap_values.shape[0], axis=0
            )

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
        shap_explanation_tmp: shap.Explanation = self.explainer(foreground_X, **kwargs)
        shap_values: np.ndarray = shap_explanation_tmp.values
        shap_data: np.ndarray = shap_explanation_tmp.data
        shap_base_values: np.ndarray = shap_explanation_tmp.base_values
        if shap_base_values.ndim == 1:
            # for unknown reasons, some SHAP explainers (`shap.SamplingExplainer`) returns 1D base values, which
            # need to be reshaped and repeated to match the expected shape for accessibility
            shap_base_values = shap_base_values[np.newaxis, :]
            shap_base_values = np.repeat(
                shap_base_values, repeats=shap_values.shape[0], axis=0
            )

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
