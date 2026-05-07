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
from darts.typing import TimeSeriesLike
from darts.utils.data.torch_datasets.utils import TorchInferenceDatasetOutput
from darts.utils.ts_utils import series2seq

logger = get_logger(__name__)

MIN_BACKGROUND_SAMPLE = 10
MAX_BACKGROUND_SAMPLE = 1000

INPUT_PAST_INDICES = [0, 1, 3]
INPUT_FUTURE_INDICES = [4]
INPUT_STATIC_INDICES = [5]


class TorchShapExplainer(BaseShapExplainer):
    model: TorchForecastingModel

    def __init__(
        self,
        model: TorchForecastingModel,
        n: int,
        target_components: Sequence[str],
        past_covariates_components: Sequence[str] | None,
        future_covariates_components: Sequence[str] | None,
        static_covariates_components: Sequence[str] | None,
        background_series: Sequence[TimeSeries],
        background_past_covariates: Sequence[TimeSeries] | None,
        background_future_covariates: Sequence[TimeSeries] | None,
        shap_method: str | None,
        background_num_samples: int | None = None,
        batch_size: int | None = None,
        **kwargs,
    ):
        self.input_chunk_length = model.input_chunk_length
        super().__init__(
            model=model,
            n=n,
            target_components=target_components,
            past_covariates_components=past_covariates_components,
            future_covariates_components=future_covariates_components,
            static_covariates_components=static_covariates_components,
            background_series=background_series,
            background_past_covariates=background_past_covariates,
            background_future_covariates=background_future_covariates,
            shap_method=shap_method,
            background_num_samples=background_num_samples,
            batch_size=batch_size,
            **kwargs,
        )

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

    def create_shap_array(
        self,
        series: TimeSeriesLike,
        past_covariates: TimeSeriesLike | None,
        future_covariates: TimeSeriesLike | None,
        n_samples: int | None = None,
        train: bool = False,
    ) -> tuple[np.ndarray, list[dict[str, Any]], pd.Index]:
        """
        Creates the SHAP array for the given input series and covariates, by following the logic of the torch
        forecasting model's inference dataset and prediction step. It returns the SHAP array, the schemas of the
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
        train
            A boolean indicating whether the SHAP array is being created for training (background) data or for
            foreground data. This affects how the dataset is sampled and how the bounds are created. Default: ``False``.
        """
        # convert to sequence of TimeSeries if not already
        series_: Sequence[TimeSeries] = series2seq(series)
        past_covariates_: Sequence[TimeSeries] | None = series2seq(past_covariates)
        future_covariates_: Sequence[TimeSeries] | None = series2seq(future_covariates)

        # if future covariates are used, trim the series if the last timestamp of the series where making a forecast
        # is possible is before the end of the series. This is to avoid creating samples in the dataset that would not
        # be able to make a forecast.
        if future_covariates_ is not None:
            for i in range(len(series_)):
                if train:
                    shift = 0
                else:
                    shift = self.output_chunk_length + self.output_chunk_shift
                end_time = (
                    future_covariates_[i].end_time()
                    - shift * future_covariates_[i].freq
                )
                if end_time < series_[i].end_time():
                    series_[i] = series_[i][:end_time]

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
        n_samples = n_samples or len(dataset)
        if train:
            if len(dataset) < MIN_BACKGROUND_SAMPLE:
                raise_log(
                    ValueError(
                        f"Background series must contain at least {MIN_BACKGROUND_SAMPLE} samples to create a "
                        f"valid background. Got background dataset length={len(dataset)}."
                    ),
                    logger,
                )
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
        else:
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
        for i in range(self.input_chunk_length):
            lag = self.input_chunk_length - i
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

        past_slice = slice(0, self.input_chunk_length * self.n_variables)
        future_slice = slice(
            past_slice.stop,
            past_slice.stop + self.output_chunk_length * self.n_future_covs,
        )
        static_slice = slice(future_slice.stop, None)

        x = torch.from_numpy(x_np).float()
        num_samples = x.shape[0]

        x_past = x[:, past_slice]
        x_past = x_past.reshape(num_samples, self.input_chunk_length, self.n_variables)

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

    def _create_dataset_bounds(
        self,
        series: Sequence[TimeSeries],
        train: bool,
    ) -> np.ndarray:
        """
        Creates the bounds for the inference dataset based on the input series and whether it is for training or not.
        """
        offset = self.output_chunk_length if train else 0
        bounds = np.array([(self.input_chunk_length, len(s) - offset) for s in series])
        return bounds

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
