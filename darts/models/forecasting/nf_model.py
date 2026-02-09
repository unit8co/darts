"""
NeuralForecastModel
------------------
"""

from typing import Optional, TypedDict

import torch
from neuralforecast.common._base_model import BaseModel
from neuralforecast.losses.pytorch import BasePointLoss

from darts.logging import get_logger, raise_log
from darts.models.forecasting.pl_forecasting_module import (
    PLForecastingModule,
    io_processor,
)
from darts.models.forecasting.torch_forecasting_model import MixedCovariatesTorchModel
from darts.utils.data.torch_datasets.utils import PLModuleInput, TorchTrainingSample
from darts.utils.likelihood_models.torch import TorchLikelihood

logger = get_logger(__name__)


"""

Throughout this file, we use the following notation for tensor shapes:

    SYMBOL: Darts / NeuralForecast definition
    ------------------------------------------------
    B: batch size / number of windows
    L: input chunk length / input window length
    H: output chunk length / horizon
    C: target components / number of series
    X: past covariate components / historical exogenous variables
    F: future covariate components / future exogenous variables
    S: static covariate components / static exogenous variables (per target component)
    N: likelihood parameters

In NeuralForecast, `BaseModel.forward()` takes a single argument which is a dictionary
containing all inputs. See `BaseModel._parse_windows()` and `BaseModel.training_step()`
to see how these inputs are being built and used.

We thus define the expected keys and their types below:
"""


class _WindowBatch(TypedDict):
    insample_y: torch.Tensor
    insample_mask: torch.Tensor
    # outsample_y: Optional[torch.Tensor]
    # outsample_mask: Optional[torch.Tensor]
    hist_exog: Optional[torch.Tensor]
    futr_exog: Optional[torch.Tensor]
    stat_exog: Optional[torch.Tensor]


IGNORED_NF_MODEL_PARAM_NAMES = {
    "loss",
    "valid_loss",
    "learning_rate",
    "max_steps",
    "val_check_steps",
    "batch_size",
    "valid_batch_size",
    "windows_batch_size",
    "inference_windows_batch_size",
    "start_padding_enabled",
    "training_data_availability_threshold",
    "n_series",
    "n_samples",
    "h_train",
    "inference_input_size",
    "step_size",
    "num_lr_decays",
    "early_stop_patience_steps",
    "scaler_type",
    "futr_exog_list",  # prepared by Darts
    "hist_exog_list",  # prepared by Darts
    "stat_exog_list",  # prepared by Darts
    "exclude_insample_y",  # TODO: check if this should be ignored
    "drop_last_loader",
    "random_seed",  # TODO: check if this should be ignored
    "alias",  # TODO: check if this should be ignored
    "optimizer",
    "optimizer_kwargs",
    "lr_scheduler",
    "lr_scheduler_kwargs",
    "dataloader_kwargs",
}


# TODO: implement pseudo loss class to pass to `nf_model.loss`
class _PseudoLoss(BasePointLoss):
    def __init__(self, likelihood: Optional[TorchLikelihood]):
        n_likelihood_params = likelihood.num_parameters if likelihood is not None else 1
        super().__init__(outputsize_multiplier=n_likelihood_params)


class _NFModel(BaseModel):
    """This serves as a protocol for expected NeuralForecast BaseModel API."""

    def forward(self, window_batch: _WindowBatch) -> torch.Tensor: ...


class _PLForecastingModule(PLForecastingModule):
    def __init__(
        self,
        nf_model: _NFModel,
        n_past_covs: int,
        n_future_covs: int,
        is_multivariate: bool,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.nf = nf_model
        self.is_multivariate = is_multivariate
        self.past_slice = (
            slice(self.n_targets, self.n_targets + n_past_covs)
            if n_past_covs > 0
            else None
        )
        self.future_slice = (
            slice(
                self.n_targets + n_past_covs,
                self.n_targets + n_past_covs + n_future_covs,
            )
            if n_future_covs > 0
            else None
        )

    @io_processor
    def forward(self, x_in: PLModuleInput):
        # unpack inputs
        # `x_past`: (B, L, C + X + F)
        # `x_future`: (B, H, F)
        # `x_static`: (B, C, S)
        x_past, x_future, x_static = x_in

        # build window_batch dict expected by `nf.forward()`
        # Expected shapes in the univariate case (C=1):
        # - `insample_y`: (B, L, C)
        # - `insample_mask`: (B, L)
        # - `hist_exog`: (B, L, X) or None
        # - `futr_exog`: (B, L + H, F) or None
        # - `stat_exog`: (B, C * S) or None
        # Expected shapes in the multivariate case (C >= 1):
        # - `insample_y`: (B, L, C)
        # - `insample_mask`: (B, L)
        # - `hist_exog`: (B, X, L, C) or None
        # - `futr_exog`: (B, F, L + H, C) or None
        # - `stat_exog`: (C, S) or None

        insample_y = x_past[:, :, : self.n_targets]
        insample_mask = torch.ones_like(x_past[:, :, 0])
        hist_exog, futr_exog, stat_exog = None, None, None

        # process past covariates if supported and provided
        if self.past_slice is not None:
            # `hist_exog`: (B, L, X)
            hist_exog = x_past[:, :, self.past_slice]
            if self.is_multivariate:
                # -> (B, X, L, 1)
                hist_exog = hist_exog.transpose(1, 2).unsqueeze(-1)
                # -> (B, X, L, C)
                hist_exog = hist_exog.repeat(1, 1, 1, self.n_targets)

        # process future covariates if supported and provided
        if x_future is not None:
            # `futr_exog`: (B, L + H, F)
            futr_exog = torch.cat([x_past[:, :, self.future_slice], x_future], dim=1)
            if self.is_multivariate:
                # -> (B, F, L + H, 1)
                futr_exog = futr_exog.transpose(1, 2).unsqueeze(-1)
                # -> (B, F, L + H, C)
                futr_exog = futr_exog.repeat(1, 1, 1, self.n_targets)

        # process static covariates if supported and provided
        if x_static is not None:
            if self.is_multivariate:
                # `stat_exog`: (B, C, S) -> (C, S)
                # For multivariate models, NeuralForecast expects `stat_exog` to be of
                # shape (C, S) and shared across the batch dimension,
                # but Darts provides them in shape (B, C, S).
                # Here, we assume that static covariates are the same across each sample
                # in the batch and simply take the first sample's static covariates.
                stat_exog = x_static[0]
            else:
                # `stat_exog`: (B, C * S) [C=1]
                stat_exog = x_static.squeeze(1)

        window_batch = _WindowBatch(
            insample_y=insample_y,
            insample_mask=insample_mask,
            hist_exog=hist_exog,
            futr_exog=futr_exog,
            stat_exog=stat_exog,
        )

        # forward pass through NeuralForecast model
        # `y_pred`: (B, H, C * N)
        y_pred: torch.Tensor = self.nf(window_batch)
        # -> (B, H, C, N)
        y_pred = y_pred.unflatten(-1, (self.n_targets, -1))

        return y_pred


class NeuralForecastModel(MixedCovariatesTorchModel):
    """NeuralForecastModel is a wrapper for NeuralForecast models to be used in Darts.

    Parameters
    ----------
    model
        An instance of a NeuralForecast model (inheriting from `NeuralForecast.BaseModel`).
    quantiles
        Optionally, produce quantile predictions at `quantiles` levels when performing probabilistic forecasting
        with `num_samples > 1` or `predict_likelihood_parameters=True`.
    random_state
        Controls the randomness for reproducible forecasting.
    """

    def __init__(
        self,
        model: BaseModel,
        use_static_covariates: bool = False,
        **kwargs,
    ):
        super().__init__(**self._extract_torch_model_params(**self.model_params))

        # extract pytorch lightning module kwargs
        self.pl_module_params = self._extract_pl_module_params(**self.model_params)
        # assign input/output chunk lengths
        self.pl_module_params["input_chunk_length"] = model.input_size
        self.pl_module_params["output_chunk_length"] = model.h
        # NeuralForecast models do not use output_chunk_shift
        self.pl_module_params["output_chunk_shift"] = 0

        self.nf_model_class = model.__class__
        self.nf_model_params = dict(model.hparams)
        self._validate_nf_model_params()

        if self.nf_model_class.RECURRENT:
            raise_log(
                NotImplementedError(
                    "Recurrent NeuralForecast models are currently not supported."
                ),
                logger,
            )
        if self.supports_multivariate and use_static_covariates:
            logger.warning(
                "Multivariate NeuralForecast models require static covariates to be the same "
                "across time series, but may be different across target components. "
                "If you have multiple time series, setting `use_static_covariates=True` "
                "will use the static covariates of the first sample in each batch, instead of "
                "providing different static covariates per time series."
            )

        # consider static covariates if supported by `nf_model_class`
        self._considers_static_covariates = use_static_covariates

    def _validate_nf_model_params(self) -> None:
        ignored_params_in_use = IGNORED_NF_MODEL_PARAM_NAMES.intersection(
            self.nf_model_params.keys()
        )
        # remove ignored params
        if len(ignored_params_in_use) > 0:
            logger.info(
                f"The following NeuralForecast model parameters will be ignored "
                f"as they are either managed by Darts or not relevant: {ignored_params_in_use}"
            )
            for param in ignored_params_in_use:
                self.nf_model_params.pop(param)

    def _create_model(self, train_sample: TorchTrainingSample) -> PLForecastingModule:
        # unpack train sample
        # `past_target`: (L, C)
        # `past_covariates`: (L, X)
        # `historic_future_covariates`: (L, F)
        # `future_covariates`: (H, F)
        # `static_covariates`: (C, S)
        (
            past_target,
            past_covariates,
            historic_future_covariates,
            future_covariates,
            static_covariates,
            future_target,
        ) = train_sample

        # TODO: sanity checks on covariate support of the nf_model_class

        # validate number of target components
        n_targets = future_target.shape[1]
        if n_targets != 1 and not self.supports_multivariate:
            raise_log(
                ValueError(
                    f"The provided {self.nf_model_class.__name__} is a univariate model "
                    f"but the target has {n_targets} component(s)."
                ),
                logger,
            )

        # create pseudo *_exog_list inputs expected by NeuralForecast
        def build_exog_list(prefix: str, n_components: int) -> list[str]:
            return [f"{prefix}_{i}" for i in range(n_components)]

        futr_exog_list, hist_exog_list, stat_exog_list = None, None, None
        n_past_covs, n_future_covs, n_stat_covs = 0, 0, 0
        if future_covariates is not None:
            n_future_covs = future_covariates.shape[1]
            futr_exog_list = build_exog_list("futr_exog", n_future_covs)
        if past_covariates is not None:
            n_past_covs = past_covariates.shape[1]
            hist_exog_list = build_exog_list("hist_exog", n_past_covs)
        if static_covariates is not None:
            n_stat_covs = static_covariates.shape[1]
            stat_exog_list = build_exog_list("stat_exog", n_stat_covs)

        # set loss to pseudo loss with correct number of likelihood parameters
        loss = _PseudoLoss(self.likelihood)

        # initialize nf_model instance
        nf_model = self.nf_model_class(
            **self.nf_model_params,
            loss=loss,
            n_series=n_targets,
            futr_exog_list=futr_exog_list,
            hist_exog_list=hist_exog_list,
            stat_exog_list=stat_exog_list,
        )

        pl_module_params = self.pl_module_params or {}
        return _PLForecastingModule(
            nf_model=nf_model,  # pyright: ignore[reportArgumentType]
            n_past_covs=n_past_covs,
            n_future_covs=n_future_covs,
            is_multivariate=self.supports_multivariate,
            **pl_module_params,
        )

    @property
    def supports_multivariate(self) -> bool:
        return self.nf_model_class.MULTIVARIATE

    @property
    def supports_past_covariates(self) -> bool:
        return self.nf_model_class.EXOGENOUS_HIST

    @property
    def supports_future_covariates(self) -> bool:
        return self.nf_model_class.EXOGENOUS_FUTR

    @property
    def supports_static_covariates(self) -> bool:
        return self.nf_model_class.EXOGENOUS_STAT
