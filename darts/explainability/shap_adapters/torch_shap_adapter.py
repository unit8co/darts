from collections.abc import Sequence

import numpy as np
import pandas as pd
import shap
import torch

from darts import TimeSeries
from darts.explainability.shap_adapters.shap_adapter import (
    MAX_BACKGROUND_SAMPLE,
    MIN_BACKGROUND_SAMPLE,
    ShapAdapter,
    SHAPMethod,
)
from darts.logging import get_logger, raise_log
from darts.models.forecasting.pl_forecasting_module import PLForecastingModule
from darts.models.forecasting.torch_forecasting_model import TorchForecastingModel
from darts.typing import TimeSeriesLike
from darts.utils.data.tabularization import create_lagged_component_names
from darts.utils.data.torch_datasets.utils import TorchInferenceDatasetOutput
from darts.utils.historical_forecasts.optimized_historical_forecasts_torch import (
    _create_dataset_bounds,
)
from darts.utils.ts_utils import series2seq

logger = get_logger(__name__)


class TorchShapAdapter(ShapAdapter):
    model: TorchForecastingModel

    def create_shap_input(
        self,
        series: TimeSeriesLike,
        past_covariates: TimeSeriesLike | None,
        future_covariates: TimeSeriesLike | None,
        n_samples: int | None = None,
        input_type: str = "background",
    ) -> tuple[np.ndarray, pd.Index]:
        # convert to sequence of TimeSeries if not already
        series: Sequence[TimeSeries] = series2seq(series)
        past_covariates: Sequence[TimeSeries] | None = series2seq(past_covariates)
        future_covariates: Sequence[TimeSeries] | None = series2seq(future_covariates)

        bounds, _ = _create_dataset_bounds(
            model=self.model,
            series=series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            start=None,
            forecast_horizon=self.n,
            stride=1,
            overlap_end=True,
            show_warnings=False,
        )

        # create inference dataset
        dataset = self.model._build_inference_dataset(
            n=self.n,
            series=series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
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

        # - Collate each input type separately and arrange in the same feature order as SKLearnModel X array:
        #   - lagged_target | lagged_past_covariates | lagged_future_covariates | static,
        #   where lagged_future_covariates includes both historic (-ICL to -1) and actual future (0 to OCL-1)
        # - `batch` is a list of tuples of (past target, past cov, future past cov, historic future cov, future cov,
        #   static cov, target series schema, pred time)
        # - since `ShapExplainer` never performs auto-regression, we can skip the "future past cov" part
        extract_batch_indices = [0, 1, 3, 4, 5]
        arrays = [
            np.stack([sample[idx] for sample in batch])
            for idx in extract_batch_indices
            if batch[0][idx] is not None
        ]
        shap_array = np.concatenate(
            [array.reshape(array.shape[0], -1) for array in arrays],
            axis=-1,
        )
        prediction_times = pd.Index([c[-1] for c in batch])
        return shap_array, prediction_times

    def _build_feature_names(self) -> list[str]:
        lags_past = [i for i in range(-self.model.input_chunk_length, 0)]
        lags_future = lags_past + [i for i in range(self.model.output_chunk_length)]
        feature_names, _ = create_lagged_component_names(
            target_series=self.background_series,
            past_covariates=self.background_past_covariates,
            future_covariates=self.background_future_covariates,
            lags=lags_past,
            lags_past_covariates=lags_past,
            lags_future_covariates=lags_future,
            output_chunk_length=self.model.output_chunk_length,
            concatenate=False,
            use_static_covariates=self.model.uses_static_covariates,
        )
        return feature_names

    def _build_explainer(
        self,
        model: TorchForecastingModel,
        background_arr: np.ndarray,
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
        background_arr
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
            raise_log(
                ValueError(
                    f"Unknown SHAP method `'{shap_method}'`. Must be one of "
                    f"{[el.name.lower() for el in self._supported_shap_methods]}"
                )
            )

        return explainer_cls(self._func_wrapper, background_arr, **kwargs)

    @torch.inference_mode()
    def _func_wrapper(self, x_np: np.ndarray) -> np.ndarray:
        """
        Wrapper function to adapt the SHAP explainer to the torch forecasting model. It takes as input a numpy array
        of shape `(num_samples, num_features)` and outputs a numpy array of shape
        `(num_samples, output_chunk_length * n_targets_likelihood)`.

        Internally, it does the following steps:
        1. Reshape the input numpy array into the format expected by the torch forecasting model, separating past
           target, past covariates, future covariates, and static covariates based on the slices defined in
           :func:`_setup_func_wrapper()`.
        2. Pass the reshaped inputs to the model in batches and collect the outputs.
        3. Concatenate the outputs and reshape them into the expected output format for SHAP, which is a 2D array where
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
        pl_module.set_predict_parameters(
            n=self.n,
            num_samples=1,
            roll_size=self.n,
            batch_size=self.batch_size,
            predict_likelihood_parameters=self.model.supports_likelihood_parameter_prediction,
            mc_dropout=False,
        )

        input_chunk_length = self.model.input_chunk_length
        output_chunk_length = self.output_chunk_length

        x = torch.from_numpy(x_np).to(dtype=pl_module.dtype, device=pl_module.device)
        num_samples = x.shape[0]

        # The shap array follows the sklearn-style feature ordering:
        #   lagged_target | lagged_past_covariates | lagged_future_covariates | static
        offset = 0

        # extract all required batch elements from the input array
        # past target
        pt_size = input_chunk_length * self.n_targets
        x_pt = x[:, :pt_size].reshape(num_samples, input_chunk_length, -1)
        offset += pt_size

        # past covariates
        if self.n_past_covs > 0:
            pc_size = input_chunk_length * self.n_past_covs
            x_pc = x[:, offset : offset + pc_size].reshape(
                num_samples, input_chunk_length, -1
            )
            offset += pc_size
        else:
            x_pc = None

        # future covariates (historic + future)
        if self.n_future_covs > 0:
            hfc_size = input_chunk_length * self.n_future_covs
            x_hfc = x[:, offset : offset + hfc_size].reshape(
                num_samples, input_chunk_length, -1
            )
            offset += hfc_size

            fc_size = output_chunk_length * self.n_future_covs
            x_fc = x[:, offset : offset + fc_size].reshape(
                num_samples, output_chunk_length, -1
            )
            offset += fc_size
        else:
            x_hfc = None
            x_fc = None

        # static covariates
        if self.n_static_covs > 0:
            x_sc = x[:, offset:].reshape(num_samples, -1, self.n_static_covs)
        else:
            x_sc = None

        # set model to eval mode to deactivate dropout layers
        pl_module.eval()

        outputs = []
        for batch_idx, i in enumerate(range(0, num_samples, self.batch_size)):
            batch_slice = slice(i, i + self.batch_size)
            batch = tuple(
                x_i[batch_slice] if x_i is not None else None
                for x_i in [x_pt, x_pc, None, x_hfc, x_fc, x_sc, None, None]
            )

            # output shape: (num_samples = 1, batch_size, output_chunk_length, n_targets_likelihood)
            output, _, _ = pl_module.predict_step(
                batch=batch,
                batch_idx=batch_idx,
                dataloader_idx=None,
            )
            outputs.append(output)

        # concatenate and reshape to: (n forecasts, output_chunk_length * n_targets_likelihood)
        outputs = torch.cat(outputs, dim=1)[0].flatten(start_dim=1)

        if outputs.dtype == torch.bfloat16:
            outputs = outputs.float()
        return outputs.cpu().numpy()

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
