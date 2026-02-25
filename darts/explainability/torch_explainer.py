from collections.abc import Sequence

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

logger = get_logger(__name__)

MIN_BACKGROUND_SAMPLE = 10
MAX_BACKGROUND_SAMPLE = 1000


class TorchExplainer(_ForecastingModelExplainer):
    model: TorchForecastingModel

    def __init__(
        self,
        model: TorchForecastingModel,
        background_series: TimeSeriesLike | None = None,
        background_past_covariates: TimeSeriesLike | None = None,
        background_future_covariates: TimeSeriesLike | None = None,
        background_num_samples: int | None = None,
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
        pass


class _DeepShapExplainer:
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
        self.background_X = self._create_shap_tensor(
            self.background_series,
            self.background_past_covariates,
            self.background_future_covariates,
            background_num_samples,
            train=True,
        )

        n_past_covs = (
            self.background_past_covariates[0].n_components
            if self.background_past_covariates is not None
            else 0
        )
        n_future_covs = (
            self.background_future_covariates[0].n_components
            if self.background_future_covariates is not None
            else 0
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
        self.model_wrapper = _TorchModelWrapper(
            self.model.model, n_past_covs=n_past_covs, n_future_covs=n_future_covs
        )

        self.explainer = shap.DeepExplainer(self.model_wrapper, self.background_X)

    def shap_explanations(
        self,
        foreground_X: pd.DataFrame,
        horizons: Sequence[int] | None = None,
        target_components: Sequence[str] | None = None,
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

    def _create_dataset_bounds(
        self,
        series: Sequence[TimeSeries],
    ) -> np.ndarray:
        input_chunk_length = self.model.input_chunk_length
        bounds = np.array([(input_chunk_length, len(s)) for s in series])
        return bounds

    def _create_shap_tensor(
        self,
        series: Sequence[TimeSeries],
        past_covariates: Sequence[TimeSeries] | None,
        future_covariates: Sequence[TimeSeries] | None,
        n_samples: int | None = None,
        train: bool = False,
    ) -> torch.Tensor:
        # TODO: convert all inputs to list of `TimeSeries`

        # create inference dataset
        dataset = self.model._build_inference_dataset(
            n=self.n,
            series=series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            stride=1,
            bounds=self._create_dataset_bounds(series),
        )

        # sample from dataset if required
        if not train:
            n_samples = None
        n_samples = n_samples or len(dataset)
        if n_samples > len(dataset):
            raise_log(
                ValueError(
                    f"`n_samples` must be less than or equal to the number of samples in the dataset. "
                    f"Got `n_samples={n_samples}` but dataset length={len(dataset)}."
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

        return shap_tensor


class _TorchModelWrapper(torch.nn.Module):
    def __init__(
        self,
        model: PLForecastingModule,
        n_past_covs: int = 0,
        n_future_covs: int = 0,
    ):
        super().__init__()
        self.model = model
        self.input_chunk_length = model.input_chunk_length
        self.output_chunk_length = model.output_chunk_length or 1
        self.n_targets = model.n_targets
        self.n_past_covs = n_past_covs
        self.n_future_covs = n_future_covs
        self.n_variables = self.n_targets + self.n_past_covs + self.n_future_covs

        self.past_slice = slice(0, self.input_chunk_length * self.n_variables)
        self.future_slice = slice(
            self.past_slice.stop,
            self.past_slice.stop + self.output_chunk_length * self.n_future_covs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]

        x_past = x[:, self.past_slice]
        x_past = x_past.view(batch_size, self.input_chunk_length, self.n_variables)

        if self.n_future_covs > 0:
            x_future = x[:, self.future_slice]
            x_future = x_future.view(
                batch_size, self.output_chunk_length, self.n_future_covs
            )
        else:
            x_future = None

        x_static = None

        output: torch.Tensor = self.model((x_past, x_future, x_static))
        # remove last dimension of likelihood parameters
        output = output.squeeze(dim=-1)

        return output
