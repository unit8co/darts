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
from darts.typing import TimeSeriesLike

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


class DartsShapExplanation(shap.Explanation):
    def __init__(self, *args, time_index: pd.Index, **kwargs):
        super().__init__(*args, **kwargs)
        self.time_index = time_index


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
        self.n_output_features = self.output_chunk_length * self.n_targets_likelihood
        self.single_output = self.n == 1 and self.n_targets_likelihood == 1

        self.background_series = background_series
        self.background_past_covariates = background_past_covariates
        self.background_future_covariates = background_future_covariates
        self.feature_names = self._build_feature_names()

        self.background_arr, _ = self.create_shap_input(
            series=self.background_series,
            past_covariates=self.background_past_covariates,
            future_covariates=self.background_future_covariates,
            n_samples=background_num_samples,
            input_type="background",
        )

        self.explainer = self._build_explainer(
            model=self.model,
            background_arr=self.background_arr,
            shap_method=self.shap_method,
            **kwargs,
        )

    def shap_explanations(
        self,
        foreground_arr: np.ndarray,
        foreground_times: pd.Index,
        horizons: Sequence[int],
        target_components: Sequence[str],
        **kwargs,
    ) -> dict[int, dict[str, DartsShapExplanation]]:
        """
        Computes SHAP explanations for the given foreground data, horizons, and target components.
        It returns a nested dictionary of SHAP Explanation objects for each horizon and target component:

        - first dimension: each step in the forecast horizon.
        - second dimension: each component of the target time series.

        Parameters
        ----------
        foreground_arr
            A numpy array of shape `(num_samples, num_features)` containing the input features for SHAP explanations.
        foreground_times
            The prediction times corresponding to each sample in ``foreground_arr``.
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
        dict[int, dict[str, DartsShapExplanation]]
            A nested dictionary ``{horizon : {target_component : DartsShapExplanation}}`` containing the SHAP
            Explanation objects for each horizon and target component, where the SHAP values are extracted and
            reshaped for easier accessibility.
        """
        # for models without native multioutput support (only for ``SKLearnModel`` if wrapped by MultiOutputMixin):
        # create a nested dictionary {horizon : {target_component : shap.Explanation}} for better accessibility of
        # the explanations
        explanations = {}
        if isinstance(self.explainer, dict):
            for h in horizons:
                tmp_n = {}
                for t_idx, t in enumerate(self.target_components_likelihood):
                    if t not in target_components:
                        continue
                    explanation = self.explainer[h - 1][t_idx](foreground_arr, **kwargs)
                    explanation = DartsShapExplanation(
                        values=explanation.values,
                        data=explanation.data,
                        base_values=explanation.base_values.ravel(),
                        feature_names=self.feature_names,
                        time_index=foreground_times,
                    )
                    tmp_n[t] = explanation
                explanations[h] = tmp_n
            return explanations

        # for models with native multioutput support:
        # the native multioutput forces us to recompute all horizons and targets; can be either
        # ``TorchForecastingModel`` or ``SKLearnModel`` with native multioutput
        explanation: shap.Explanation = self.explainer(foreground_arr, **kwargs)

        # bring arrays into expected shapes:
        # `values`: (E, F, C * OCL)
        # `base_values`: (E, C * OCL)
        # E: number of forecast examples
        # F: number of lagged model input features
        # C: number of target components (including likelihood parameters)
        # OCL: output chunk length
        values = explanation.values
        base_values = explanation.base_values

        if self.single_output:
            if values.shape == foreground_arr.shape:
                values = values[:, :, np.newaxis]
            if base_values.shape == foreground_arr.shape[:1]:
                base_values = base_values[:, np.newaxis]
        if base_values.shape == (self.n_output_features,):
            # some SHAP explainers (e.g. shap.SamplingExplainer) return 1D base values
            base_values = np.repeat(
                base_values[np.newaxis, :], repeats=values.shape[0], axis=0
            )

        for h in horizons:
            tmp_n = {}
            for t_idx, t in enumerate(self.target_components_likelihood):
                if t not in target_components:
                    continue

                tmp_t = DartsShapExplanation(
                    values=values[:, :, self.n_targets_likelihood * (h - 1) + t_idx],
                    data=explanation.data,
                    base_values=base_values[
                        :, self.n_targets_likelihood * (h - 1) + t_idx
                    ].ravel(),
                    feature_names=self.feature_names,
                    time_index=foreground_times,
                )
                tmp_n[t] = tmp_t
            explanations[h] = tmp_n

        return explanations

    def shap_explanations_single(
        self,
        foreground_arr: np.ndarray,
        foreground_times: pd.Index,
        target_components: Sequence[str],
        **kwargs,
    ) -> dict[str, DartsShapExplanation]:
        """
        Similar to :func:`shap_explanations()`, but computes SHAP explanations for only one forecasted timestamp, which
        corresponds to the last forecastable timestamp in the foreground series. The output is a dictionary of SHAP
        Explanation objects for each target component, where the SHAP values are extracted from the raw Explanation
        object returned by the SHAP explainer and reshaped into the expected format for easier accessibility.

        Parameters
        ----------
        foreground_arr
            A numpy array of shape `(num_samples, num_features)` containing the input features for SHAP explanations.
            Only the last sample (forecast) will be explained.
        foreground_times
            The prediction times corresponding to each sample in ``foreground_arr``. Only the last prediction time will
            be explained.
        target_components
            A sequence of strings with the target components to explain. Each component must be among the target
            components of the explained forecasting model.
        **kwargs
            Additional keyword arguments to be passed to the SHAP explainer when calling it for explanations.
            This can include parameters for sampling or approximation methods used by some SHAP explainers.

        Returns
        -------
        dict[str, DartsShapExplanation]
            A dictionary ``{target_component : DartsShapExplanation}`` containing the SHAP Explanation objects for each
            target component, where the SHAP values are extracted and reshaped for easier accessibility.
        """
        # explain only the last forecast
        foreground_arr = foreground_arr[-1:]
        foreground_times = foreground_times[-1:]

        horizons = list(range(1, self.n + 1))
        explanations = self.shap_explanations(
            foreground_arr,
            foreground_times,
            horizons,
            target_components,
            **kwargs,
        )

        result = {}
        for t in target_components:
            if t not in explanations[horizons[0]]:
                continue
            horizon_expls = [explanations[h][t] for h in horizons]
            result[t] = DartsShapExplanation(
                values=np.concatenate(
                    [np.atleast_2d(e.values) for e in horizon_expls], axis=0
                ),
                data=np.concatenate(
                    [np.atleast_2d(e.data) for e in horizon_expls], axis=0
                ),
                base_values=np.concatenate([
                    np.atleast_1d(e.base_values) for e in horizon_expls
                ]),
                feature_names=horizon_expls[0].feature_names,
                time_index=horizon_expls[0].time_index,
            )
        return result

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
        background_arr: np.ndarray,
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
        background_arr
            The background dataset in the form of a numpy array, to be passed to the SHAP explainer.
        shap_method
            The SHAP method to use for explanations. Must be one of the methods available in the SHAP library,
            specified in the enum ``SHAPMethod``.
        **kwargs
            Additional keyword arguments to be passed to the SHAP explainer constructor.
        """

    @abstractmethod
    def create_shap_input(
        self,
        series: TimeSeriesLike,
        past_covariates: TimeSeriesLike | None,
        future_covariates: TimeSeriesLike | None,
        n_samples: int | None = None,
        input_type: str = "background",
    ) -> tuple[np.ndarray, pd.Index]:
        """
        Creates the SHAP input for the given series and covariates, by following the logic of the model's prediction /
        inference dataset and prediction step. It returns the SHAP array (lagged features values) and the prediction
        times corresponding to each sample in the SHAP array.

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
