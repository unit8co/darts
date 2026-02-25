from collections.abc import Sequence
from enum import Enum

import numpy as np
import pandas as pd
import shap
import torch

from darts import TimeSeries
from darts.explainability.explainability import _ForecastingModelExplainer
from darts.explainability.explainability_result import ShapExplainabilityResult
from darts.logging import get_logger, raise_log
from darts.models.forecasting.pl_forecasting_module import PLForecastingModule
from darts.models.forecasting.torch_forecasting_model import TorchForecastingModel
from darts.typing import TimeSeriesLike
from darts.utils.data.torch_datasets.utils import TorchInferenceDatasetOutput
from darts.utils.ts_utils import series2seq

logger = get_logger(__name__)

MIN_BACKGROUND_SAMPLE = 10
MAX_BACKGROUND_SAMPLE = 1000


class _ShapMethod(Enum):
    GRADIENT = 1
    KERNEL = 3
    SAMPLING = 4
    PARTITION = 5
    LINEAR = 6
    PERMUTATION = 7
    ADDITIVE = 8


class TorchExplainer(_ForecastingModelExplainer):
    model: TorchForecastingModel

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
        # validate model type
        if not issubclass(type(model), TorchForecastingModel):
            raise_log(
                ValueError(
                    f"Invalid `model` type: `{type(model)}`. Only models of type `TorchForecastingModel` are supported."
                ),
                logger,
            )

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

        # TODO: add support for probabilistic models
        if model.likelihood is not None:
            raise_log(
                ValueError(
                    "Explainer does not currently support probabilistic models. "
                    "Please use a `TorchForecastingModel` with `likelihood=None`."
                ),
                logger,
            )

        shap_method_upper = shap_method.upper()
        if shap_method_upper in _ShapMethod.__members__:
            self.shap_method = _ShapMethod[shap_method_upper]
        else:
            raise_log(
                ValueError(
                    f"Invalid `shap_method`={shap_method}. Please choose one value among the following: "
                    f"['partition', 'tree', 'kernel', 'sampling', 'linear', 'gradient', 'additive']."
                )
            )

        if (
            background_num_samples is not None
            and background_num_samples > MAX_BACKGROUND_SAMPLE
        ):
            raise_log(
                ValueError(
                    f"`background_num_samples` must be less than or equal to {MAX_BACKGROUND_SAMPLE}. "
                    f"Got {background_num_samples}."
                ),
                logger,
            )

        self.explainer = _DeepShapExplainer(
            model=self.model,
            n=self.n,
            target_components=self.target_components,
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

    def explain(
        self,
        foreground_series: TimeSeriesLike | None = None,
        foreground_past_covariates: TimeSeriesLike | None = None,
        foreground_future_covariates: TimeSeriesLike | None = None,
        horizons: Sequence[int] | None = None,
        target_components: Sequence[str] | None = None,
    ) -> ShapExplainabilityResult:
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

            foreground_X, prediction_times = self.explainer._create_shap_tensor(
                foreground_ts,
                foreground_past_cov_ts,
                foreground_future_cov_ts,
                train=False,
            )
            print(f"Foreground X shape: {foreground_X.shape}")
            print(f"Prediction times: {prediction_times}")

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

        return ShapExplainabilityResult(
            shap_values_list, feature_values_list, shap_explanation_object_list
        )


class _DeepShapExplainer:
    n_targets: int

    def __init__(
        self,
        model: TorchForecastingModel,
        n: int,
        target_components: Sequence[str],
        past_covariates_components: Sequence[str] | None,
        future_covariates_components: Sequence[str] | None,
        background_series: Sequence[TimeSeries],
        background_past_covariates: Sequence[TimeSeries] | None,
        background_future_covariates: Sequence[TimeSeries] | None,
        background_num_samples: int | None = None,
        shap_method: _ShapMethod = _ShapMethod.LINEAR,
        batch_size: int | None = None,
        **kwargs,
    ):
        self.model = model

        self.target_components = target_components
        self.past_covariates_components = past_covariates_components
        self.future_covariates_components = future_covariates_components

        self.n = n
        self.background_series = background_series
        self.background_past_covariates = background_past_covariates
        self.background_future_covariates = background_future_covariates

        # TODO: support RNNModel with special handling of tensor shapes
        self.background_X, _ = self._create_shap_tensor(
            self.background_series,
            self.background_past_covariates,
            self.background_future_covariates,
            background_num_samples,
            train=True,
        )
        print(f"Background X shape: {self.background_X.shape}")

        self._build_func_wrapper(
            model.model,
            batch_size=batch_size or model.batch_size,
        )

        # TODO: support static covariates with special handling
        if model._uses_static_covariates:
            raise_log(
                NotImplementedError(
                    "Explainer does not currently support models with static covariates. "
                    "Please use a `TorchForecastingModel` without static covariates."
                ),
                logger,
            )

        self.explainer = self._build_explainer(
            self._func_wrapper,
            self.background_X,
            shap_method,
            **kwargs,
        )
        # shap_ = self.explainer(self.background_X[:3])
        # print(f"Initial shap_ values shape: {shap_.values.shape}")
        # print(f"Initial shap_ data shape: {shap_.data.shape}")
        # print(f"Initial shap_ base_values shape: {shap_.base_values.shape}")

    def _build_func_wrapper(
        self,
        model: PLForecastingModule,
        batch_size: int,
    ):
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

        self.input_chunk_length = model.input_chunk_length
        self.output_chunk_length = model.output_chunk_length or 1
        self.n_targets = model.n_targets
        self.n_variables = self.n_targets + self.n_past_covs + self.n_future_covs

        self.past_slice = slice(0, self.input_chunk_length * self.n_variables)
        self.future_slice = slice(
            self.past_slice.stop,
            self.past_slice.stop + self.output_chunk_length * self.n_future_covs,
        )

        self.batch_size = batch_size

    @torch.inference_mode()
    def _func_wrapper(self, x_np: np.ndarray) -> np.ndarray:
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

        x_static = None

        outputs: list[torch.Tensor] = []
        for i in range(0, num_samples, self.batch_size):
            s = slice(i, i + self.batch_size)
            batch_x_past = x_past[s]
            batch_x_future = x_future[s] if x_future is not None else None
            batch_x_static = x_static[s] if x_static is not None else None

            batch_output: torch.Tensor = self.pl_module((
                batch_x_past,
                batch_x_future,
                batch_x_static,
            ))
            # remove last dimension of likelihood parameters
            batch_output = batch_output.flatten(start_dim=1)
            outputs.append(batch_output)

        output = torch.cat(outputs, dim=0)

        return output.cpu().numpy()

    @staticmethod
    def _build_explainer(
        func,
        background_tensor: torch.Tensor,
        shap_method: _ShapMethod,
        **kwargs,
    ):
        background_X = background_tensor.cpu().numpy()
        # we define properly the explainer given a shap method
        if shap_method == _ShapMethod.PERMUTATION:
            explainer = shap.PermutationExplainer(func, background_X, **kwargs)
        elif shap_method == _ShapMethod.PARTITION:
            explainer = shap.PermutationExplainer(func, background_X, **kwargs)
        elif shap_method == _ShapMethod.KERNEL:
            explainer = shap.KernelExplainer(func, background_X, **kwargs)
        elif shap_method == _ShapMethod.LINEAR:
            explainer = shap.LinearExplainer(func, background_X, **kwargs)
        # DeepExplainer has some compatibility issues with torch models
        # elif shap_method == _ShapMethod.DEEP:
        #     explainer = shap.LinearExplainer(model, background_X, **kwargs)
        elif shap_method == _ShapMethod.ADDITIVE:
            explainer = shap.AdditiveExplainer(func, background_X, **kwargs)
        else:
            raise_log(
                ValueError(
                    f"Invalid `shap_method`={shap_method}. Please choose one value among the following: "
                    f"['partition', 'tree', 'kernel', 'sampling', 'linear', 'gradient', 'additive']."
                )
            )

        return explainer

    def shap_explanations(
        self,
        foreground_tensor: torch.Tensor,
        horizons: Sequence[int],
        target_components: Sequence[str],
    ) -> dict[int, dict[str, shap.Explanation]]:
        # create a unified dictionary between multiOutputRegressor estimators and
        # native multiOutput estimators
        shap_explanations = {}
        # the native multioutput forces us to recompute all horizons and targets
        foreground_X = foreground_tensor.cpu().numpy()
        shap_explanation_tmp: shap.Explanation = self.explainer(foreground_X)
        shap_values: np.ndarray = shap_explanation_tmp.values
        shap_data: np.ndarray = shap_explanation_tmp.data
        shap_base_values: np.ndarray = shap_explanation_tmp.base_values
        feature_names: list[str] = shap_explanation_tmp.feature_names
        print(f"shap_explanation_tmp.values.shape: {shap_explanation_tmp.values.shape}")
        print(f"shap_explanation_tmp.data.shape: {shap_explanation_tmp.data.shape}")
        print(
            f"shap_explanation_tmp.base_values.shape: {shap_explanation_tmp.base_values.shape}"
        )
        for h in horizons:
            tmp_n = {}
            for t_idx, t in enumerate(self.target_components):
                if t not in target_components:
                    continue
                tmp_t = shap.Explanation(
                    shap_values[:, :, self.n_targets * (h - 1) + t_idx],
                    data=shap_data,
                    base_values=shap_base_values[
                        :, self.n_targets * (h - 1) + t_idx
                    ].ravel(),
                    feature_names=feature_names,
                )

                tmp_n[t] = tmp_t
            shap_explanations[h] = tmp_n

        return shap_explanations

    def _create_dataset_bounds(
        self,
        series: Sequence[TimeSeries],
    ) -> np.ndarray:
        input_chunk_length = self.model.input_chunk_length
        bounds = np.array([(input_chunk_length, len(s)) for s in series])
        return bounds

    def _create_shap_tensor(
        self,
        series: TimeSeriesLike,
        past_covariates: TimeSeriesLike | None,
        future_covariates: TimeSeriesLike | None,
        n_samples: int | None = None,
        train: bool = False,
    ) -> tuple[torch.Tensor, pd.Index]:
        series: Sequence[TimeSeries] = series2seq(series)
        past_covariates: Sequence[TimeSeries] | None = series2seq(past_covariates)
        future_covariates: Sequence[TimeSeries] | None = series2seq(future_covariates)

        # create inference dataset
        if True:
            stride, bounds = 1, self._create_dataset_bounds(series)
        else:
            stride, bounds = 0, None
        dataset = self.model._build_inference_dataset(
            n=self.n,
            series=series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            stride=stride,
            bounds=bounds,
        )

        # sample from dataset if required
        if not train:
            n_samples = len(dataset)
        else:
            if len(dataset) < MIN_BACKGROUND_SAMPLE:
                raise_log(
                    ValueError(
                        f"Dataset must contain at least {MIN_BACKGROUND_SAMPLE} samples to create a valid background. "
                        f"Got dataset length={len(dataset)}."
                    ),
                    logger,
                )
            n_samples = n_samples or len(dataset)
            if n_samples > len(dataset):
                raise_log(
                    ValueError(
                        f"`background_num_samples` must be less than or equal to the number of samples in the dataset. "
                        f"Got `background_num_samples={n_samples}` but dataset length={len(dataset)}."
                    ),
                    logger,
                )

        # follow the logic of `TorchForecastingModel.predict_from_dataset()`
        # to collect samples and collate them into a sample tuple
        # collect batch of samples from the end of the dataset
        batch: list[TorchInferenceDatasetOutput] = []
        if train:
            # randomly sample from the dataset if in training mode
            indices = np.random.choice(len(dataset), size=n_samples, replace=False)
        else:
            indices = range(len(dataset) - n_samples, len(dataset))
        for i in indices:
            batch.append(dataset[i])

        # collate batch and convert to tuple of tensors & metadata
        batch_aggregated = self.model._batch_collate_fn(batch)

        # follow the logic of `PLForecastingModule.predict_step()`
        # to convert to 1D tensor
        # remove last two elements (metadata) and the remaining first six elements are:
        # - past_target
        # - past_covariates
        # - future_past_covariates
        # - historic_future_covariates
        # - future_covariates
        # - static_covariates
        input_data_tuple = batch_aggregated[:-2]
        prediction_times = batch_aggregated[-1]
        prediction_times = pd.Index(prediction_times)

        # `PLForecastingModule._get_batch_prediction()`
        (
            past_target,
            past_covariates_,
            future_past_covariates,
            historic_future_covariates,
            future_covariates_,
            static_covariates,
        ) = input_data_tuple

        input_past, input_future, input_static = self.model.model._process_input_batch((
            past_target,
            past_covariates_,
            historic_future_covariates,
            future_covariates_,
            static_covariates,
        ))
        self.static_shape = input_static.shape if input_static is not None else None

        shap_tensor = torch.cat(
            [
                tensor.flatten(start_dim=1)
                for tensor in [input_past, input_future, input_static]
                if tensor is not None
            ],
            dim=-1,
        )

        return shap_tensor, prediction_times
