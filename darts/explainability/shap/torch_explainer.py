from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import shap
import torch

from darts import TimeSeries
from darts.explainability.shap.base_explainer import BaseShapExplainer, SHAPMethod
from darts.logging import get_logger, raise_log
from darts.models.forecasting.pl_forecasting_module import PLForecastingModule
from darts.models.forecasting.rnn_model import CustomRNNModule
from darts.models.forecasting.torch_forecasting_model import TorchForecastingModel
from darts.typing import TimeIndex, TimeSeriesLike
from darts.utils.data.torch_datasets.utils import TorchInferenceDatasetOutput
from darts.utils.historical_forecasts.optimized_historical_forecasts_torch import (
    _create_dataset_bounds,
)
from darts.utils.ts_utils import series2seq

logger = get_logger(__name__)

MIN_BACKGROUND_SAMPLE = 10
MAX_BACKGROUND_SAMPLE = 1000

INPUT_PAST_INDICES = [0, 1, 3]
INPUT_FUTURE_INDICES = [4]
INPUT_STATIC_INDICES = [5]


class TorchShapExplainer(BaseShapExplainer):
    model: TorchForecastingModel

    def create_shap_array(
        self,
        series: TimeSeriesLike,
        past_covariates: TimeSeriesLike | None,
        future_covariates: TimeSeriesLike | None,
        n_samples: int | None = None,
        input_type: str = "background",
    ) -> tuple[np.ndarray, list[dict[str, Any]], TimeIndex]:
        # convert to sequence of TimeSeries if not already
        series_: Sequence[TimeSeries] = series2seq(series)
        past_covariates_: Sequence[TimeSeries] | None = series2seq(past_covariates)
        future_covariates_: Sequence[TimeSeries] | None = series2seq(future_covariates)

        bounds, _ = _create_dataset_bounds(
            model=self.model,
            series=series_,
            past_covariates=past_covariates_,
            future_covariates=future_covariates_,
            start=None,
            forecast_horizon=self.n,
            stride=1,
            overlap_end=True,
            show_warnings=False,
        )

        # create inference dataset
        dataset = self.model._build_inference_dataset(
            n=self.n,
            series=series_,
            past_covariates=past_covariates_,
            future_covariates=future_covariates_,
            stride=1,
            bounds=bounds,
        )

        # sample from dataset if required
        n_samples = n_samples or len(dataset)
        if input_type == "background":
            if len(dataset) < MIN_BACKGROUND_SAMPLE:
                raise_log(
                    ValueError(
                        f"Background series must contain at least {MIN_BACKGROUND_SAMPLE} samples to create a "
                        f"valid background. Got background dataset length={len(dataset)}."
                    ),
                    logger,
                )
            if n_samples > MAX_BACKGROUND_SAMPLE:
                logger.warning(
                    f"Background series contains more than MAX_BACKGROUND_SAMPLE={MAX_BACKGROUND_SAMPLE} samples. "
                    f"Sampling {MAX_BACKGROUND_SAMPLE} samples to create the background for SHAP explanations."
                )
                n_samples = MAX_BACKGROUND_SAMPLE

        # follow the logic of `TorchForecastingModel.predict_from_dataset()`
        # to collect samples and collate them into a sample tuple
        # collect batch of samples from the end of the dataset
        batch: list[TorchInferenceDatasetOutput] = []
        if n_samples < len(dataset):
            # randomly sample from the dataset if in training mode
            indices = np.random.choice(len(dataset), size=n_samples, replace=False)
        else:
            indices = range(len(dataset))

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
        schemas = [c[-2] for c in batch]
        prediction_times = pd.Index([c[-1] for c in batch])

        shap_array = np.concatenate(
            [
                array.reshape(array.shape[0], -1)
                for array in [input_past, input_future, input_static]
                if array is not None
            ],
            axis=-1,
        )

        return shap_array, schemas, prediction_times

    @staticmethod
    def _batch_collate_np(batch: list[tuple], indices: list[int]) -> np.ndarray | None:
        """
        Collates a batch of samples from the inference dataset into a numpy array for SHAP explanations,
        based on the specified indices for past covariates, future covariates, and static covariates.
        It handles the case where some samples in the batch may have None values for certain inputs,
        by skipping those samples when collating.
        """
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

    def _build_feature_names(self) -> list[str]:
        feature_names = []
        input_chunk_length = self.model.input_chunk_length
        for i in range(input_chunk_length):
            lag = input_chunk_length - i
            for t in self.target_components:
                feature_names.append(f"{t}_target_lag-{lag}")
            if self.past_covariates_components is not None:
                for c in self.past_covariates_components:
                    feature_names.append(f"{c}_pastcov_lag-{lag}")
            if self.future_covariates_components is not None:
                for c in self.future_covariates_components:
                    feature_names.append(f"{c}_futcov_lag-{lag}")

        for i in range(self.output_chunk_length):
            lag = i + self.output_chunk_shift
            if self.future_covariates_components is not None:
                for c in self.future_covariates_components:
                    feature_names.append(f"{c}_futcov_lag{lag}")

        if self.model.uses_static_covariates:
            static_covs = self.background_series[0].static_covariates
            if static_covs is not None:
                # static covariate names
                names = static_covs.columns.tolist()
                # target components that the static covariates reference to
                comps = static_covs.index.tolist()
                feature_names += [
                    f"{name}_statcov_target_{comp}" for name in names for comp in comps
                ]

        return feature_names

    def _build_explainer(
        self,
        model: TorchForecastingModel,
        background_X: tuple[np.ndarray, list[dict[str, Any]], pd.Index],
        shap_method: SHAPMethod,
        **kwargs,
    ) -> shap.Explainer:
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
        # we define properly the explainer given a shap method
        # Note: DeepExplainer has some compatibility issues with torch models
        if shap_method == SHAPMethod.KERNEL:
            explainer_cls = shap.KernelExplainer
        elif shap_method == SHAPMethod.SAMPLING:
            explainer_cls = shap.SamplingExplainer
        elif shap_method == SHAPMethod.PARTITION:
            explainer_cls = shap.PartitionExplainer
        elif shap_method == SHAPMethod.PERMUTATION:
            explainer_cls = shap.PermutationExplainer
        else:
            raise_log(ValueError(f"Unknown SHAP method {shap_method}"))

        return explainer_cls(self._func_wrapper, background_X, **kwargs)

    @torch.inference_mode()
    def _func_wrapper(self, x_np: np.ndarray) -> np.ndarray:
        """
        Wrapper function to adapt the SHAP explainer to the torch forecasting model. It takes as input a numpy array
        of shape `(num_samples, num_features)` and outputs a numpy array of shape
        `(num_samples, output_chunk_length * n_targets_likelihood)`.

        Internally, it does the following steps:
        1. Reshape the input numpy array into the format expected by the torch forecasting model, separating past
           covariates, future covariates, and static covariates based on the slices defined
           in :func:`_setup_func_wrapper()`.
        2. If the model is an RNN, handle the special case where future covariates are concatenated to
           past covariates with a shift in time dimension.
        3. Pass the reshaped inputs to the model in batches and collect the outputs.
        4. Concatenate the outputs and reshape them into the expected output format for SHAP, which is a 2D array where
           each column corresponds to a target component at a specific horizon.

        Parameters
        ----------
        x_np
            A numpy array of shape `(num_samples, num_features)` containing the input features for SHAP explanations.

        Returns
        -------
        np.ndarray
            A numpy array of shape `(num_samples, output_chunk_length * n_targets_likelihood)` containing the model
            predictions for each target component at each horizon, to be used by the SHAP explainer.
        """
        pl_module: PLForecastingModule = self.model.model
        input_chunk_length = self.model.input_chunk_length
        past_slice = slice(0, input_chunk_length * self.n_variables)
        future_slice = slice(
            past_slice.stop,
            past_slice.stop + self.output_chunk_length * self.n_future_covs,
        )
        static_slice = slice(future_slice.stop, None)

        x = torch.from_numpy(x_np).float()
        num_samples = x.shape[0]

        x_past = x[:, past_slice]
        x_past = x_past.reshape(num_samples, input_chunk_length, self.n_variables)

        if self.n_future_covs > 0:
            x_future = x[:, future_slice]
            x_future = x_future.reshape(
                num_samples, self.output_chunk_length, self.n_future_covs
            )
        else:
            x_future = None

        if self.n_static_covs > 0:
            x_static = x[:, static_slice]
            x_static = x_static.reshape(num_samples, -1, self.n_static_covs)
        else:
            x_static = None

        if isinstance(pl_module, CustomRNNModule):
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
        pl_module.eval()

        outputs: list[torch.Tensor] = []
        for i in range(0, num_samples, self.batch_size):
            s = slice(i, i + self.batch_size)
            batch_x_past = x_past[s].to(pl_module.device)
            batch_x_future = (
                x_future[s].to(pl_module.device) if x_future is not None else None
            )
            batch_x_static = (
                x_static[s].to(pl_module.device) if x_static is not None else None
            )

            batch_output: torch.Tensor = pl_module((
                batch_x_past,
                batch_x_future,
                batch_x_static,
            ))

            if isinstance(pl_module, CustomRNNModule):
                # Note: RNN outputs predictions and hidden states
                batch_output = batch_output[0]
                # RNN also outputs predictions for all time steps,
                # but we only need the last one for SHAP explanations
                batch_output = batch_output[:, -1:, :, :]
            else:
                # Note: TCN has a different `first_prediction_index` than 0
                batch_output = batch_output[:, pl_module.first_prediction_index :, :]

            outputs.append(batch_output)

        # `output`: (batch, output_chunk_length, n_targets, likelihood_parameters)
        output = torch.cat(outputs, dim=0)
        # flatten the output to shape (batch, output_chunk_length * n_targets_likelihood)
        output = output.flatten(start_dim=1)

        return output.cpu().numpy()

    @property
    def _supported_shap_methods(self) -> set[SHAPMethod]:
        return {
            SHAPMethod.KERNEL,
            SHAPMethod.SAMPLING,
            SHAPMethod.PARTITION,
            SHAPMethod.PERMUTATION,
        }

    def _get_default_shap_method(self, model) -> SHAPMethod:
        return SHAPMethod.PERMUTATION

    def _validate_model(self, model: TorchForecastingModel) -> None:
        if not isinstance(model, TorchForecastingModel):
            raise_log(
                ValueError(
                    f"Invalid `model` type: `{type(model)}`. Only models of type "
                    f"`TorchForecastingModel` are supported."
                ),
                logger,
            )
