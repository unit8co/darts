from collections.abc import Sequence
from enum import Enum

import numpy as np
import pandas as pd
import shap
import torch
from matplotlib import pyplot as plt

from darts import TimeSeries
from darts.explainability.explainability import _ForecastingModelExplainer
from darts.explainability.explainability_result import ShapExplainabilityResult
from darts.logging import get_logger, raise_log
from darts.models.forecasting.pl_forecasting_module import PLForecastingModule
from darts.models.forecasting.rnn_model import CustomRNNModule
from darts.models.forecasting.torch_forecasting_model import TorchForecastingModel
from darts.typing import TimeSeriesLike
from darts.utils.data.torch_datasets.utils import TorchInferenceDatasetOutput
from darts.utils.ts_utils import series2seq

logger = get_logger(__name__)

MIN_BACKGROUND_SAMPLE = 10
MAX_BACKGROUND_SAMPLE = 1000

INPUT_PAST_INDICES = [0, 1, 3]
INPUT_FUTURE_INDICES = [4]
INPUT_STATIC_INDICES = [5]


class _ShapMethod(Enum):
    KERNEL = 3
    SAMPLING = 4
    PARTITION = 5
    LINEAR = 6
    PERMUTATION = 7
    ADDITIVE = 8
    EXACT = 9


def _available_shap_methods() -> list[str]:
    return [method.name.lower() for method in _ShapMethod]


class TorchExplainer(_ForecastingModelExplainer):
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
                    f"{_available_shap_methods()}."
                )
            )

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

        self.explainer = _DeepShapExplainer(
            model=self.model,
            n=self.n,
            target_components=self.target_components,
            static_covariates_components=self.static_covariates_components,
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
        fallback = foreground_series is None
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

            foreground_X, prediction_times = self.explainer.create_shap_array(
                foreground_ts,
                foreground_past_cov_ts,
                foreground_future_cov_ts,
                train=fallback,
            )

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

    def summary_plot(
        self,
        horizons: int | Sequence[int] | None = None,
        target_components: str | Sequence[str] | None = None,
        num_samples: int | None = None,
        plot_type: str | None = "dot",
        **kwargs,
    ) -> dict[int, dict[str, shap.Explanation]]:
        horizons, target_components = self._process_horizons_and_targets(
            horizons, target_components
        )

        if num_samples:
            n_background_samples = self.explainer.background_X.shape[0]
            if num_samples > n_background_samples:
                raise_log(
                    ValueError(
                        f"`num_samples` must be less than or equal to the number of samples in the background. "
                        f"Got `num_samples={num_samples}` but background samples={n_background_samples}."
                    )
                )
            foreground_X_sampled = shap.utils.sample(
                self.explainer.background_X, num_samples
            )
        else:
            foreground_X_sampled = self.explainer.background_X

        shaps_ = self.explainer.shap_explanations(
            foreground_X_sampled, horizons, target_components
        )

        for t in target_components:
            for h in horizons:
                plt.title(
                    f"Target: `{t}` - Horizon: t+{h + self.model.output_chunk_shift}"
                )
                shap.summary_plot(
                    shaps_[h][t],
                    foreground_X_sampled,
                    plot_type=plot_type,
                    **kwargs,
                )
        return shaps_

    def force_plot_from_ts(
        self,
        foreground_series: TimeSeries | None = None,
        foreground_past_covariates: TimeSeries | None = None,
        foreground_future_covariates: TimeSeries | None = None,
        horizon: int | None = 1,
        target_component: str | None = None,
        **kwargs,
    ):
        if target_component is None and len(self.target_components) > 1:
            raise_log(
                ValueError(
                    f"`target_component` is required when the model has more than one component. "
                    f"Please select a component from {self.target_components}."
                ),
                logger,
            )

        if target_component is None:
            target_component = self.target_components[0]

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
        horizons, target_components = self._process_horizons_and_targets(
            horizon,
            target_component,
        )
        horizon, target_component = horizons[0], target_components[0]

        foreground_X, _ = self.explainer.create_shap_array(
            foreground_series_,
            foreground_past_covariates_,
            foreground_future_covariates_,
            train=foreground_series is None,
        )

        shap_ = self.explainer.shap_explanations(
            foreground_X, [horizon], [target_component]
        )

        return shap.force_plot(
            base_value=shap_[horizon][target_component],
            features=foreground_X,
            out_names=target_component,
            **kwargs,
        )


class _DeepShapExplainer:
    n_targets: int

    def __init__(
        self,
        model: TorchForecastingModel,
        n: int,
        target_components: Sequence[str],
        static_covariates_components: Sequence[str] | None,
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
        self.static_covariates_components = static_covariates_components
        self.past_covariates_components = past_covariates_components
        self.future_covariates_components = future_covariates_components

        self.n = n
        self.background_series = background_series
        self.background_past_covariates = background_past_covariates
        self.background_future_covariates = background_future_covariates

        self.input_chunk_length = model.input_chunk_length
        self.output_chunk_length = model.output_chunk_length or 1
        self.output_chunk_shift = model.output_chunk_shift

        self.background_X, _ = self.create_shap_array(
            series=self.background_series,
            past_covariates=self.background_past_covariates,
            future_covariates=self.background_future_covariates,
            n_samples=background_num_samples,
            train=True,
        )

        self._setup_func_wrapper(
            model.model,
            batch_size=batch_size or model.batch_size,
        )
        self._build_feature_names()

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

    def _setup_func_wrapper(
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
        static_covs = self.background_series[0].static_covariates_values(copy=False)
        self.n_static_covs = static_covs.shape[1] if static_covs is not None else 0

        self.n_targets = model.n_targets
        self.n_variables = self.n_targets + self.n_past_covs + self.n_future_covs

        self.past_slice = slice(0, self.input_chunk_length * self.n_variables)
        self.future_slice = slice(
            self.past_slice.stop,
            self.past_slice.stop + self.output_chunk_length * self.n_future_covs,
        )
        self.static_slice = slice(self.future_slice.stop, None)

        self.batch_size = batch_size

    def _build_feature_names(self):
        self.feature_names = []
        for i in range(self.input_chunk_length):
            lag = self.input_chunk_length - i
            for t in self.target_components:
                self.feature_names.append(f"{t}_target_lag-{lag}")
            if self.past_covariates_components is not None:
                for c in self.past_covariates_components:
                    self.feature_names.append(f"{c}_pastcov_lag-{lag}")
            if self.future_covariates_components is not None:
                for c in self.future_covariates_components:
                    self.feature_names.append(f"{c}_futcov_lag-{lag}")

        for i in range(self.output_chunk_length):
            lag = i + self.output_chunk_shift
            if self.future_covariates_components is not None:
                for c in self.future_covariates_components:
                    self.feature_names.append(f"{c}_futcov_lag{lag}")

        if self.model.uses_static_covariates:
            static_covs = self.background_series[0].static_covariates
            if static_covs is not None:
                # static covariate names
                names = static_covs.columns.tolist()
                # target components that the static covariates reference to
                comps = static_covs.index.tolist()
                self.feature_names += [
                    f"{name}_statcov_target_{comp}" for name in names for comp in comps
                ]

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

        if self.n_static_covs > 0:
            x_static = x[:, self.static_slice]
            x_static = x_static.reshape(num_samples, -1, self.n_static_covs)
        else:
            x_static = None

        if isinstance(self.pl_module, CustomRNNModule):
            # handle the special case of RNN where future covariates are concatenated to
            # past covariates with a shift in time dimension
            if x_future is not None:
                x_future = torch.cat(
                    [
                        x_past[:, 1:, -self.n_future_covs :],
                        x_future,  # output chunk length is always 1 for RNN
                    ],
                    dim=1,
                )
                x_past = torch.cat(
                    [
                        x_past[:, :, : self.n_targets],
                        x_future,
                    ],
                    dim=2,
                )
                x_future = None

        # set model to eval mode to deactivate dropout layers
        self.pl_module.eval()

        outputs: list[torch.Tensor] = []
        for i in range(0, num_samples, self.batch_size):
            s = slice(i, i + self.batch_size)
            batch_x_past = x_past[s].to(self.pl_module.device)
            batch_x_future = (
                x_future[s].to(self.pl_module.device) if x_future is not None else None
            )
            batch_x_static = (
                x_static[s].to(self.pl_module.device) if x_static is not None else None
            )

            batch_output: torch.Tensor = self.pl_module((
                batch_x_past,
                batch_x_future,
                batch_x_static,
            ))

            if isinstance(self.pl_module, CustomRNNModule):
                # Note: RNN outputs predictions and hidden states
                batch_output = batch_output[0]
                # RNN also outputs predictions for all time steps,
                # but we only need the last one for SHAP explanations
                batch_output = batch_output[:, -1:, :, :]
            else:
                # Note: TCN has a different `first_prediction_index` than 0
                batch_output = batch_output[
                    :, self.pl_module.first_prediction_index :, :
                ]

            outputs.append(batch_output)

        output = torch.cat(outputs, dim=0)
        # remove last dimension of likelihood parameters
        output = output.flatten(start_dim=1)

        return output.cpu().numpy()

    @staticmethod
    def _build_explainer(
        func,
        background_X: np.ndarray,
        shap_method: _ShapMethod,
        **kwargs,
    ):
        # we define properly the explainer given a shap method
        # Note: DeepExplainer has some compatibility issues with torch models
        if shap_method == _ShapMethod.PERMUTATION:
            explainer = shap.PermutationExplainer(func, background_X, **kwargs)
        elif shap_method == _ShapMethod.PARTITION:
            explainer = shap.PermutationExplainer(func, background_X, **kwargs)
        elif shap_method == _ShapMethod.KERNEL:
            explainer = shap.KernelExplainer(func, background_X, **kwargs)
        elif shap_method == _ShapMethod.LINEAR:
            explainer = shap.LinearExplainer(func, background_X, **kwargs)
        elif shap_method == _ShapMethod.ADDITIVE:
            explainer = shap.AdditiveExplainer(func, background_X, **kwargs)
        elif shap_method == _ShapMethod.EXACT:
            explainer = shap.ExactExplainer(func, background_X, **kwargs)
        else:
            raise_log(
                ValueError(
                    f"Invalid `shap_method`={shap_method}. Please choose one value among the following: "
                    f"{_available_shap_methods()}."
                )
            )

        return explainer

    def shap_explanations(
        self,
        foreground_X: np.ndarray,
        horizons: Sequence[int],
        target_components: Sequence[str],
    ) -> dict[int, dict[str, shap.Explanation]]:
        shap_explanation_tmp: shap.Explanation = self.explainer(foreground_X)
        shap_values: np.ndarray = shap_explanation_tmp.values
        shap_data: np.ndarray = shap_explanation_tmp.data
        shap_base_values: np.ndarray = shap_explanation_tmp.base_values

        # create a unified dictionary between multiOutputRegressor estimators and
        # native multiOutput estimators
        shap_explanations = {}

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
                    feature_names=self.feature_names,
                )

                tmp_n[t] = tmp_t
            shap_explanations[h] = tmp_n

        return shap_explanations

    def _create_dataset_bounds(
        self,
        series: Sequence[TimeSeries],
        train: bool,
    ) -> np.ndarray:
        offset = self.output_chunk_length if train else 0
        bounds = np.array([(self.input_chunk_length, len(s) - offset) for s in series])
        return bounds

    @staticmethod
    def _batch_collate_np(batch: list[tuple], indices: list[int]) -> np.ndarray | None:
        data = []
        for index in indices:
            if batch[0][index] is None:
                continue
            data.append(np.stack([sample[index] for sample in batch]))

        if len(data) == 0:
            return None
        else:
            data = np.concatenate(data, axis=2)
            return data

    def create_shap_array(
        self,
        series: TimeSeriesLike,
        past_covariates: TimeSeriesLike | None,
        future_covariates: TimeSeriesLike | None,
        n_samples: int | None = None,
        train: bool = False,
    ) -> tuple[np.ndarray, pd.Index]:
        # convert to sequence of TimeSeries if not already
        series_: Sequence[TimeSeries] = series2seq(series)
        past_covariates_: Sequence[TimeSeries] | None = series2seq(past_covariates)
        future_covariates_: Sequence[TimeSeries] | None = series2seq(future_covariates)

        # create inference dataset
        dataset = self.model._build_inference_dataset(
            n=self.n,
            series=series_,
            past_covariates=past_covariates_,
            future_covariates=future_covariates_,
            stride=1,
            bounds=self._create_dataset_bounds(series_, train=train),
        )

        # sample from dataset if required
        if not train:
            n_samples = len(dataset)
        else:
            if len(dataset) < MIN_BACKGROUND_SAMPLE:
                raise_log(
                    ValueError(
                        f"Background series must contain at least {MIN_BACKGROUND_SAMPLE} samples to create a "
                        f"valid background. Got background dataset length={len(dataset)}."
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
            if n_samples > MAX_BACKGROUND_SAMPLE:
                logger.warning(
                    f"Background series contains more than MIN_BACKGROUND_SAMPLE={MAX_BACKGROUND_SAMPLE} samples. "
                    f"Sampling {MAX_BACKGROUND_SAMPLE} samples to create the background for SHAP explanations."
                )
                n_samples = MAX_BACKGROUND_SAMPLE

        # follow the logic of `TorchForecastingModel.predict_from_dataset()`
        # to collect samples and collate them into a sample tuple
        # collect batch of samples from the end of the dataset
        batch: list[TorchInferenceDatasetOutput] = []
        if train:
            if n_samples < len(dataset):
                # randomly sample from the dataset if in training mode
                indices = np.random.choice(len(dataset), size=n_samples, replace=False)
            else:
                indices = range(len(dataset))
        else:
            indices = range(len(dataset) - n_samples, len(dataset))
        for i in indices:
            batch.append(dataset[i])

        # follow the logic of `PLForecastingModule.predict_step()`
        # to convert to 1D tensor
        # - past_target
        # - past_covariates
        # - future_past_covariates
        # - historic_future_covariates
        # - future_covariates
        # - static_covariates
        input_past = self._batch_collate_np(batch, INPUT_PAST_INDICES)
        input_future = self._batch_collate_np(batch, INPUT_FUTURE_INDICES)
        input_static = self._batch_collate_np(batch, INPUT_STATIC_INDICES)
        prediction_times = pd.Index([c[-1] for c in batch])

        shap_array = np.concatenate(
            [
                array.reshape(array.shape[0], -1)
                for array in [input_past, input_future, input_static]
                if array is not None
            ],
            axis=-1,
        )

        return shap_array, prediction_times
