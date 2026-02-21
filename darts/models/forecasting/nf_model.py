"""
NeuralForecastModel
-------------------
"""

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
- `insample_y`: (B, L, C), historical target values in the input window.
- `insample_mask`: (B, L), binary mask indicating available target values in `insample_y`.
- `hist_exog`: (B, L, X) for univariate models or (B, X, L, C) for multivariate models,
    historic exogenous variables in the input window (if any).
- `futr_exog`: (B, L + H, F) for univariate models or (B, F, L + H, C) for multivariate models,
    future exogenous variables in the input window (if any).
- `stat_exog`: (B, C * S) for univariate models or (C, S) for multivariate models,
    static exogenous variables (if any). For multivariate models, NeuralForecast expects `stat_exog`
    to be shared across the batch dimension, but may be different across target components.
    For univariate models, static exogenous variables can be different across time series.
"""
import inspect
from typing import TypedDict

import torch
from neuralforecast.common._base_model import BaseModel
from neuralforecast.losses.pytorch import BasePointLoss
from numpy.random import RandomState

from darts.logging import get_logger, raise_log
from darts.models.forecasting.pl_forecasting_module import (
    PLForecastingModule,
    io_processor,
)
from darts.models.forecasting.torch_forecasting_model import MixedCovariatesTorchModel
from darts.utils.data.torch_datasets.utils import PLModuleInput, TorchTrainingSample
from darts.utils.likelihood_models.torch import TorchLikelihood
from darts.utils.utils import MAX_NUMPY_SEED_VALUE

logger = get_logger(__name__)


class _WindowBatch(TypedDict):
    insample_y: torch.Tensor
    insample_mask: torch.Tensor
    hist_exog: torch.Tensor | None
    futr_exog: torch.Tensor | None
    stat_exog: torch.Tensor | None


_NF_MODEL_IGNORED_PARAMS = {
    "input_size",
    "h",
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
    "exclude_insample_y",
    "drop_last_loader",
    "random_seed",
    "alias",
    "optimizer",
    "optimizer_kwargs",
    "lr_scheduler",
    "lr_scheduler_kwargs",
    "dataloader_kwargs",
    "trainer_kwargs",
}
_NF_MODEL_RINORM_PARAMS = {
    "use_norm",
    "revin",
}


def _build_exog_list(prefix: str, n_components: int) -> list[str]:
    """Utility function to create pseudo *_exog_list inputs expected by NeuralForecast"""
    return [f"{prefix}_{i}" for i in range(n_components)]


class _PseudoLoss(BasePointLoss):
    """A pseudo loss class to create a compatible interface for NeuralForecast models that expect a loss function to
    be specified, but the actual loss will be managed by Darts' `TorchForecastingModel`.

    The key attribute here is `outputsize_multiplier`, which is set to the number of parameters required by the
    likelihood model (e.g., 2 for Gaussian likelihood with mean and variance) or 1 if no likelihood is specified.
    This allows the NeuralForecast base model to output the correct number of parameters for probabilistic forecasting.

    """

    def __init__(self, likelihood: TorchLikelihood | None):
        n_likelihood_params = likelihood.num_parameters if likelihood is not None else 1
        super().__init__(outputsize_multiplier=n_likelihood_params)


class _NeuralForecastModule(PLForecastingModule):
    def __init__(
        self,
        nf_model_class: type[BaseModel],
        nf_model_params: dict,
        n_past_covs: int,
        n_future_covs: int,
        n_stat_covs: int,
        **kwargs,
    ):
        """PyTorch Lightning module that wraps around the NeuralForecast model and
        implements the :func:`forward()` API for Darts' ``PLForecastingModule``.

        Parameters
        ----------
        nf_model_class
            The class of the NeuralForecast model to be used. It should inherit from `BaseModel` and implement the
            expected `forward()` method.
        nf_model_params
            A dictionary of parameters to initialize the `nf_model_class`.
        n_past_covs
            Number of past covariate components (X).
        n_future_covs
            Number of future covariate components (F).
        n_stat_covs
            Number of static covariate components (S) per target component.
        **kwargs
            all parameters required for :class:`darts.models.forecasting.pl_forecasting_module.PLForecastingModule`
            base class.
        """
        super().__init__(**kwargs)

        futr_exog_list, hist_exog_list, stat_exog_list = None, None, None
        if n_future_covs > 0:
            futr_exog_list = _build_exog_list("futr_exog", n_future_covs)
        if n_past_covs > 0:
            hist_exog_list = _build_exog_list("hist_exog", n_past_covs)
        if n_stat_covs > 0:
            stat_exog_list = _build_exog_list("stat_exog", n_stat_covs)

        # set loss to pseudo loss with correct number of likelihood parameters
        loss = _PseudoLoss(self.likelihood)

        self.nf = nf_model_class(
            **nf_model_params,
            loss=loss,
            n_series=self.n_targets,
            futr_exog_list=futr_exog_list,
            hist_exog_list=hist_exog_list,
            stat_exog_list=stat_exog_list,
        )
        self.is_multivariate_base = self.nf.MULTIVARIATE
        # whether to convert the NF base model from univariate to multivariate by:
        # 1) Folding the target components into the batch dimension, and
        # 2) Repeating the past, future, and static covariates across the folded target components accordingly.
        self.converts_to_multivariate = (
            not self.is_multivariate_base
        ) and self.n_targets > 1

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
        """PyTorch-native forward pass.

        Parameters
        ----------
        x_in
            comes as tuple `(x_past, x_future, x_static)` where `x_past` is the input/past chunk, `x_future`
            is the output/future chunk, and `x_static` is the static covariates.
            Input dimensions are `(n_samples, n_time_steps, n_variables)` for `x_past` and `x_future`,
            and `(n_samples, n_targets, n_static_covariates)` for `x_static`.

        Returns
        -------
        torch.Tensor
            the output tensor in the shape of `(n_samples, n_time_steps, n_targets, n_likelihood_params)`,
            where `n_likelihood_params` is the number of parameters required by the likelihood model
            (e.g., 2 for Gaussian likelihood with mean and variance) or 1 if no likelihood is specified.
        """
        # unpack inputs
        # `x_past`: (B, L, C + X + F)
        # `x_future`: (B, H, F)
        # `x_static`: (B, C, S) or (B, 1, S)
        x_past, x_future, x_static = x_in

        # build window_batch dict expected by `nf.forward()`
        # Expected shapes in the univariate case (C=1):
        # - `insample_y`: (B, L, C)
        # - `insample_mask`: (B, L)
        # - `hist_exog`: (B, L, X) or None
        # - `futr_exog`: (B, L + H, F) or None
        # - `stat_exog`: (B, S) or None
        # Expected shapes in the multivariate case (C >= 1):
        # - `insample_y`: (B, L, C)
        # - `insample_mask`: (B, L)
        # - `hist_exog`: (B, X, L, C) or None
        # - `futr_exog`: (B, F, L + H, C) or None
        # - `stat_exog`: (C, S) or None

        insample_y = x_past[:, :, : self.n_targets]
        insample_mask = torch.ones_like(x_past[:, :, 0])
        hist_exog, futr_exog, stat_exog = None, None, None

        if self.converts_to_multivariate:
            # For univariate base models with C > 1 target components,
            # we fold the target components into the batch dimension.
            # The new batch size becomes B * C, and the number of target components becomes 1.
            # `insample_y`: (B, L, C) -> (B, C, L)
            insample_y = insample_y.transpose(1, 2)
            # -> (B * C, L, 1)
            insample_y = insample_y.reshape(-1, self.input_chunk_length, 1)
            # `insample_mask`: (B, L) -> (B * C, L)
            insample_mask = insample_mask.repeat_interleave(self.n_targets, dim=0)

        # process past covariates if supported and provided
        if self.past_slice is not None:
            # `hist_exog`: (B, L, X)
            hist_exog = x_past[:, :, self.past_slice]
            if self.is_multivariate_base:
                # -> (B, X, L, 1)
                hist_exog = hist_exog.transpose(1, 2).unsqueeze(-1)
                # -> (B, X, L, C)
                hist_exog = hist_exog.repeat(1, 1, 1, self.n_targets)
            elif self.converts_to_multivariate:
                # For univariate base models with C > 1 target components,
                # we repeat the past covariates across the folded target components.
                # -> (B * C, L, X)
                hist_exog = hist_exog.repeat_interleave(self.n_targets, dim=0)

        # process future covariates if supported and provided
        if x_future is not None and self.future_slice is not None:
            # `futr_exog`: (B, L + H, F)
            futr_exog = torch.cat([x_past[:, :, self.future_slice], x_future], dim=1)
            if self.is_multivariate_base:
                # -> (B, F, L + H, 1)
                futr_exog = futr_exog.transpose(1, 2).unsqueeze(-1)
                # -> (B, F, L + H, C)
                futr_exog = futr_exog.repeat(1, 1, 1, self.n_targets)
            elif self.converts_to_multivariate:
                # For univariate base models with C > 1 target components,
                # we repeat the future covariates across the folded target components.
                # -> (B * C, L + H, F)
                futr_exog = futr_exog.repeat_interleave(self.n_targets, dim=0)

        # process static covariates if supported and provided
        if x_static is not None:
            if self.is_multivariate_base:
                # `stat_exog`: (B, C, S) or (B, 1, S) -> (C, S)
                # For multivariate models, NeuralForecast expects `stat_exog` to be of
                # shape (C, S) and shared across the batch dimension,
                # but Darts provides them in shape (B, C, S) or (B, 1, S).
                # Here, we assume that static covariates are the same across each sample
                # in the batch and simply take the first sample's static covariates.
                stat_exog = x_static[0].expand(self.n_targets, -1)
            elif x_static.shape[1] == 1:
                # For univariate base models, regardless of the number of target components,
                # if static covariates are provided in shape (B, 1, S)--i.e.,
                # 1) they are the global static covariates shared across all target components, or
                # 2) they are the static covariates for ONE target component--
                # we repeat them across the target components to create the shape expected by NeuralForecast.
                # `stat_exog`: (B, C, S)
                stat_exog = x_static.repeat(1, self.n_targets, 1)
                # -> (B * C, S)
                stat_exog = stat_exog.flatten(start_dim=0, end_dim=1)
            else:
                # For other cases in univariate base models, we fold the target components into the batch dimension
                # together with the static covariates.
                # `stat_exog`: (B * C, S)
                stat_exog = x_static.flatten(start_dim=0, end_dim=1)

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
        if self.converts_to_multivariate:
            # (B * C, H, N) -> (B, C, H, N)
            y_pred = y_pred.unflatten(0, (-1, self.n_targets))
            # -> (B, H, C, N)
            y_pred = y_pred.transpose(1, 2)
        else:
            # -> (B, H, C, N)
            y_pred = y_pred.unflatten(-1, (self.n_targets, -1))

        return y_pred


class NeuralForecastModel(MixedCovariatesTorchModel):
    def __init__(
        self,
        input_chunk_length: int,
        output_chunk_length: int,
        output_chunk_shift: int = 0,
        model: str | type[BaseModel] = "TiDE",
        model_kwargs: dict | None = None,
        use_static_covariates: bool = True,
        **kwargs,
    ):
        """NeuralForecast Model.

        Can be used to fit any `NeuralForecast` univariate or multivariate base model.
        For a list of available base models,
        see `NeuralForecast package <https://nixtlaverse.nixtla.io/neuralforecast/docs/capabilities/overview.html>`__.

        This converts the `NeuralForecast` base model into a ``TorchForecastingModel`` and enables full Darts
        functionality, such as covariate support, probabilistic forecasting, optimized backtesting, etc.
        See the `Torch Forecasting Models User Guide
        <https://unit8co.github.io/darts/userguide/torch_forecasting_models.html>`__ for details and usage examples.

        The general setup looks like this:

        - Simply set ``model`` to any `NeuralForecast` base model name or class, e.g., ``model="TiDE"`` or
          ``model=TiDE``.

        - To configure the model's architectural parameters, pass them in ``model_kwargs`` as a dictionary,
          e.g., ``model_kwargs={"hidden_size": 64}`` for ``TiDE``.

        - Darts will take care of automatically setting other non-architectural parameters for you. These parameters
          **will be ignored** in ``model_kwargs``:

          - Input and output parameters: ``input_size``, ``h``, and ``n_series``. These are automatically set
            to match the ``input_chunk_length``, ``output_chunk_length``, and number of target series (components),
            respectively.

          - Covariate parameters: ``futr_exog_list``, ``hist_exog_list``, and ``stat_exog_list``. These are
            inferred directly from the input time series passed to :func:`fit()`.

          - Training and PyTorch (Lightning)-related setup: ``loss``, ``learning_rate``, ``max_steps``, etc.
            are all handled by Darts. You can specify these parameters directly. See the parameter description
            below for all available options.

        Our ``NeuralForecastModel`` has the following support, depending on the provided `NeuralForecast` base model:

        - **Univariate forecasting**: Supported by all base models. Simply pass a univariate time series as
          ``series`` to :func:`fit()` and :func:`predict()`.

        - **Multivariate forecasting**: Supported by all base models. Simply pass a multivariate time series as
          ``series`` to :func:`fit()` and :func:`predict()`.

          - For univariate base models, multivariate forecasting is achieved by folding the target components into the
            batch dimension and repeating the covariates for each target component accordingly. This translates to
            global training and forecasting on multiple univariate series.

        - **Multiple time series**: Supported by all base models. Simply pass a sequence of uni- or multivariate
          time series as ``series`` to :func:`fit()` and :func:`predict()`.

        - **Past / future / static covariates**: Supported only if the base model supports exogenous historical /
          future / static variables, respectively. Simply pass your time series as ``past_covariates`` and /
          or ``future_covariates`` to :func:`fit()` and :func:`predict()`. For static covariates, set
          ``use_static_covariates=True`` at model creation.

          - For multivariate base models, `NeuralForecast` requires static covariates to be the same across time
            series, but may be different across target components. See the warning below for recommendations.

        - **Probabilistic forecasting**: Supported by all base models. Simply set ``likelihood`` to a
          :meth:`TorchLikelihood <darts.utils.likelihood_models.torch.TorchLikelihood>` instance to be used for
          probabilistic forecasting.

        - **Loss function**: Supported by all base models. Simply set ``loss_fn`` to a PyTorch loss function (default
          is ``torch.nn.MSELoss()``).

        Parameters
        ----------
        input_chunk_length
            Number of time steps in the past to take as a model input (per chunk). Applies to the target
            series, and past and/or future covariates (if the model supports it).
        output_chunk_length
            Number of time steps predicted at once (per chunk) by the internal model. Also, the number of future values
            from future covariates to use as a model input (if the model supports future covariates). It is not the same
            as forecast horizon `n` used in `predict()`, which is the desired number of prediction points generated
            using either a one-shot- or autoregressive forecast. Setting `n <= output_chunk_length` prevents
            auto-regression. This is useful when the covariates don't extend far enough into the future, or to prohibit
            the model from using future values of past and / or future covariates for prediction (depending on the
            model's covariate support).
        output_chunk_shift
            Optionally, the number of steps to shift the start of the output chunk into the future (relative to the
            input chunk end). This will create a gap between the input and output. If the model supports
            `future_covariates`, the future values are extracted from the shifted output chunk. Predictions will start
            `output_chunk_shift` steps after the end of the target `series`. If `output_chunk_shift` is set, the model
            cannot generate autoregressive predictions (`n > output_chunk_length`). Default: ``0``.
        model
            Name or class of the NeuralForecast base model to be used from ``neuralforecast.models``, e.g., ``"TiDE"``
            or ``TiDE``. See all `NeuralForecast models
            <https://nixtlaverse.nixtla.io/neuralforecast/docs/capabilities/overview.html>`__ here.
        model_kwargs
            A dictionary of architectural parameters to initialize the `NeuralForecast` base model. The expected
            parameters depend on the base model used. Read the general description above for more info on which
            parameters are relevant. See the NeuralForecast base model documentation for details. Default: ``None``.
        use_static_covariates
            Whether to consider static covariates if supported by the base model. Default: ``True``.
            See **Static covariates** section above for details and caveats.
        **kwargs
            Optional arguments to initialize the ``pytorch_lightning.Module``, ``pytorch_lightning.Trainer``, and
            Darts' :class:`TorchForecastingModel`.

        loss_fn
            PyTorch loss function used for training.
            This parameter will be ignored for probabilistic models if the ``likelihood`` parameter is specified.
            Default: ``torch.nn.MSELoss()``.
        likelihood
            One of Darts' :meth:`TorchLikelihood <darts.utils.likelihood_models.torch.TorchLikelihood>` models to be
            used for probabilistic forecasts. Default: ``None``.
        torch_metrics
            A torch metric or a ``MetricCollection`` used for evaluation. A full list of available metrics can be found
            at https://torchmetrics.readthedocs.io/en/latest/. Default: ``None``.
        optimizer_cls
            The PyTorch optimizer class to be used. Default: ``torch.optim.Adam``.
        optimizer_kwargs
            Optionally, some keyword arguments for the PyTorch optimizer (e.g., ``{'lr': 1e-3}``
            for specifying a learning rate). Otherwise, the default values of the selected ``optimizer_cls``
            will be used. Default: ``None``.
        lr_scheduler_cls
            Optionally, the PyTorch learning rate scheduler class to be used. Specifying ``None`` corresponds
            to using a constant learning rate. Default: ``None``.
        lr_scheduler_kwargs
            Optionally, some keyword arguments for the PyTorch learning rate scheduler. Default: ``None``.
        use_reversible_instance_norm
            Whether to use reversible instance normalization `RINorm` against distribution shift as shown in [2]_.
            It is only applied to the features of the target series and not the covariates.
        batch_size
            Number of time series (input and output sequences) used in each training pass. Default: ``32``.
        n_epochs
            Number of epochs over which to train the model. Default: ``100``.
        model_name
            Name of the model. Used for creating checkpoints and saving tensorboard data. If not specified,
            defaults to the following string ``"YYYY-mm-dd_HH_MM_SS_torch_model_run_PID"``, where the initial part
            of the name is formatted with the local date and time, while PID is the processed ID (preventing models
            spawned at the same time by different processes to share the same model_name). E.g.,
            ``"2021-06-14_09_53_32_torch_model_run_44607"``.
        work_dir
            Path of the working directory, where to save checkpoints and Tensorboard summaries.
            Default: current working directory.
        log_tensorboard
            If set, use Tensorboard to log the different parameters. The logs will be located in:
            ``"{work_dir}/darts_logs/{model_name}/logs/"``. Default: ``False``.
        nr_epochs_val_period
            Number of epochs to wait before evaluating the validation loss (if a validation
            ``TimeSeries`` is passed to the :func:`fit()` method). Default: ``1``.
        force_reset
            If set to ``True``, any previously-existing model with the same name will be reset (all checkpoints will
            be discarded). Default: ``False``.
        save_checkpoints
            Whether to automatically save the untrained model and checkpoints from training.
            To load the model from checkpoint, call :func:`MyModelClass.load_from_checkpoint()`, where
            :class:`MyModelClass` is the :class:`TorchForecastingModel` class that was used (such as :class:`TFTModel`,
            :class:`NBEATSModel`, etc.). If set to ``False``, the model can still be manually saved using
            :func:`save()` and loaded using :func:`load()`. Default: ``False``.
        add_encoders
            A large number of past and future covariates can be automatically generated with `add_encoders`.
            This can be done by adding multiple pre-defined index encoders and/or custom user-made functions that
            will be used as index encoders. Additionally, a transformer such as Darts' :class:`Scaler` can be added to
            transform the generated covariates. This happens all under one hood and only needs to be specified at
            model creation.
            Read :meth:`SequentialEncoder <darts.dataprocessing.encoders.SequentialEncoder>` to find out more about
            ``add_encoders``. Default: ``None``. An example showing some of ``add_encoders`` features:

            .. highlight:: python
            .. code-block:: python

                def encode_year(idx):
                    return (idx.year - 1950) / 50

                add_encoders={
                    'cyclic': {'future': ['month']},
                    'datetime_attribute': {'future': ['hour', 'dayofweek']},
                    'position': {'past': ['relative'], 'future': ['relative']},
                    'custom': {'past': [encode_year]},
                    'transformer': Scaler(),
                    'tz': 'CET'
                }
            ..
        random_state
            Controls the randomness of the weights initialization and reproducible forecasting.
        pl_trainer_kwargs
            By default :class:`TorchForecastingModel` creates a PyTorch Lightning Trainer with several useful presets
            that performs the training, validation and prediction processes. These presets include automatic
            checkpointing, tensorboard logging, setting the torch device and more.
            With ``pl_trainer_kwargs`` you can add additional kwargs to instantiate the PyTorch Lightning trainer
            object. Check the `PL Trainer documentation
            <https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html>`__ for more information about the
            supported kwargs. Default: ``None``.
            Running on GPU(s) is also possible using ``pl_trainer_kwargs`` by specifying keys ``"accelerator",
            "devices", and "auto_select_gpus"``. Some examples for setting the devices inside the ``pl_trainer_kwargs``
            dict:

            - ``{"accelerator": "cpu"}`` for CPU,
            - ``{"accelerator": "gpu", "devices": [i]}`` to use only GPU ``i`` (``i`` must be an integer),
            - ``{"accelerator": "gpu", "devices": -1, "auto_select_gpus": True}`` to use all available GPUS.

            For more info, see here:
            https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#trainer-flags , and
            https://pytorch-lightning.readthedocs.io/en/stable/accelerators/gpu_basic.html#train-on-multiple-gpus

            With parameter ``"callbacks"`` you can add custom or PyTorch-Lightning built-in callbacks to Darts'
            :class:`TorchForecastingModel`. Below is an example for adding EarlyStopping to the training process.
            The model will stop training early if the validation loss `val_loss` does not improve beyond
            specifications. For more information on callbacks, visit:
            `PyTorch Lightning Callbacks
            <https://pytorch-lightning.readthedocs.io/en/stable/extensions/callbacks.html>`__

            .. highlight:: python
            .. code-block:: python

                from pytorch_lightning.callbacks.early_stopping import EarlyStopping

                # stop training when validation loss does not decrease more than 0.05 (`min_delta`) over
                # a period of 5 epochs (`patience`)
                my_stopper = EarlyStopping(
                    monitor="val_loss",
                    patience=5,
                    min_delta=0.05,
                    mode='min',
                )

                pl_trainer_kwargs={"callbacks": [my_stopper]}
            ..

            Note that you can also use a custom PyTorch Lightning Trainer for training and prediction with optional
            parameter ``trainer`` in :func:`fit()` and :func:`predict()`.
        show_warnings
            whether to show warnings raised from PyTorch Lightning. Useful to detect potential issues of
            your forecasting use case. Default: ``False``.

        References
        ----------
        .. [1] Nixtla's `NeuralForecast Package
                <https://nixtlaverse.nixtla.io/neuralforecast/docs/>`__.
        .. [2] T. Kim et al. "Reversible Instance Normalization for Accurate Time-Series Forecasting against
                Distribution Shift", https://openreview.net/forum?id=cGDAkQo1C0p

        Examples
        --------
        >>> from darts.datasets import WeatherDataset
        >>> from darts.models import NeuralForecastModel
        >>> # load the dataset
        >>> series = WeatherDataset().load().astype("float32")
        >>> # predicting temperatures
        >>> target = series['T (degC)'][:100]
        >>> # optionally, use future atmospheric pressure (pretending this component is a forecast)
        >>> future_cov = series['p (mbar)'][:106]
        >>> # create a NeuralForecastModel with TiDE as the base model
        >>> model = NeuralForecastModel(7, 6, model="TiDE", n_epochs=20)
        >>> # fit and predict
        >>> model.fit(target, future_covariates=future_cov)
        >>> pred = model.predict(6)
        >>> print(pred.values())
        [[0.00738985]
         [1.5933118 ]
         [3.8275871 ]
         [2.1793625 ]
         [3.6311367 ]
         [4.0539136 ]]

        .. note::
            The following `NeuralForecast` models are not supported:
              - Core ``NeuralForecast`` class which is a combination of multiple base models.
              - Automatic models like ``AutoInformer`` and ``AutoMLP`` which are not base models.
              - ``HINT`` model which is not a base model.
              - Recurrent base models like ``GRU`` and ``LSTM``. Many are, however, natively implemented
                as :class:`RNNModel <darts.models.forecasting.rnn_model.RNNModel>` in Darts.
        .. warning::
            For compatibility, when static covariates are enabled for a multivariate base model, Darts will use the
            static covariates of the first sample in each batch as the static covariates for the entire batch.
            This may cause issues if you have multiple time series with different static covariates.
            Please consider setting ``use_static_covariates=False`` to disable support or setting ``batch_size=1`` to
            ensure that each batch only contains one time series.
        """
        super().__init__(**self._extract_torch_model_params(**self.model_params))

        # extract pytorch lightning module kwargs
        self.pl_module_params = self._extract_pl_module_params(**self.model_params)

        # import and validate the NeuralForecast base model class
        if isinstance(model, str):
            self.nf_model_class = self._import_nf_model_class(model)
            self._validate_nf_model_class(name=f"neuralforecast.models.{model}")
        else:
            self.nf_model_class = model
            self._validate_nf_model_class()

        # extract, validate, and update the NeuralForecast base model parameters
        self.nf_model_params = model_kwargs.copy() if model_kwargs else {}
        self._validate_nf_model_params(
            use_reversible_instance_norm=self.pl_module_params.get(
                "use_reversible_instance_norm", False
            ),
        )
        self._update_nf_model_params(
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
        )

        # recurrent models are not supported due to incompatibable tensor shapes
        if self.nf_model_class.RECURRENT:
            raise_log(
                NotImplementedError(
                    "Recurrent NeuralForecast models are currently not supported."
                ),
                logger,
            )

        # warn if static covariates are enabled for multivariate base models
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

    @staticmethod
    def _import_nf_model_class(model: str) -> type[BaseModel]:
        import neuralforecast.models

        try:
            model_class = getattr(neuralforecast.models, model)
        except AttributeError:
            raise_log(
                ValueError(
                    f"Could not find a NeuralForecast model class named `{model}` in `neuralforecast.models`."
                ),
                logger,
            )
        return model_class

    def _validate_nf_model_class(self, name: str = "model") -> None:
        if not issubclass(self.nf_model_class, BaseModel):
            raise_log(
                ValueError(
                    f"`{name}` must be a NeuralForecast base model class,  but got {type(self.nf_model_class)}."
                ),
                logger,
            )

    def _validate_nf_model_params(
        self,
        use_reversible_instance_norm: bool,
    ) -> None:
        # check all provided parameters are valid parameters for the nf_model_class
        signature = inspect.signature(self.nf_model_class.__init__)
        valid_param_names = set(signature.parameters.keys())
        valid_param_names.discard("self")
        invalid_params = set(self.nf_model_params.keys()) - valid_param_names
        if len(invalid_params) > 0:
            raise_log(
                ValueError(
                    f"The following parameters are not valid for the provided NeuralForecast model "
                    f"{self.nf_model_class.__name__} and should be removed from `model_kwargs`: {invalid_params}"
                ),
                logger,
            )

        # remove ignored params
        ignored_params_in_use = _NF_MODEL_IGNORED_PARAMS.intersection(
            self.nf_model_params.keys()
        )
        if len(ignored_params_in_use) > 0:
            logger.info(
                f"The following NeuralForecast model parameters will be ignored "
                f"as they are either managed by Darts or not relevant: {ignored_params_in_use}"
            )
            for param in ignored_params_in_use:
                self.nf_model_params.pop(param)

        # warn if RINorm is enabled for NF model while `use_reversible_instance_norm` is enabled for the PL module
        if use_reversible_instance_norm:
            self._check_rinorm_compatibility(signature)

    def _check_rinorm_compatibility(
        self,
        signature: inspect.Signature,
    ) -> None:
        for rinorm_name in _NF_MODEL_RINORM_PARAMS:
            rinorm_param = signature.parameters.get(rinorm_name)
            if rinorm_param is None:
                continue
            if self.nf_model_params.get(rinorm_name, rinorm_param.default):
                logger.warning(
                    f"NeuralForecast model's `{rinorm_name}=True` may be incompatible with "
                    f"`PLForecastingModule`'s `use_reversible_instance_norm=True` since they "
                    f"both apply reversible instance normalization to the target series. "
                    f"If you experience issues, please consider setting one of them to `False`."
                )
            return

        # Models like `RMoK` has RINorm always enabled and
        # can only be inferred from the presence of `revin_affine` parameter
        rinorm_param = signature.parameters.get("revin_affine")
        if rinorm_param is not None:
            logger.warning(
                "NeuralForecast model has reversible instance normalization enabled and "
                "may be incompatible with `PLForecastingModule`'s `use_reversible_instance_norm=True`. "
                "If you experience issues, please consider setting `use_reversible_instance_norm=False`."
            )

    def _update_nf_model_params(
        self,
        input_chunk_length: int,
        output_chunk_length: int,
    ) -> None:
        # set `input_size` and `h` to match the `input_chunk_length` and `output_chunk_length` of the PL module
        self.nf_model_params["input_size"] = input_chunk_length
        self.nf_model_params["h"] = output_chunk_length

        # set random seed for reproducibility
        random_instance: RandomState = getattr(self, "_random_instance")
        self.nf_model_params["random_seed"] = random_instance.randint(
            0, MAX_NUMPY_SEED_VALUE
        )

    def _create_model(self, train_sample: TorchTrainingSample) -> PLForecastingModule:
        # unpack train sample
        # `past_target`: (L, C)
        # `past_covariates`: (L, X)
        # `historic_future_covariates`: (L, F)
        # `future_covariates`: (H, F)
        # `static_covariates`: (C, S) or (1, S)
        (
            past_target,
            past_covariates,
            historic_future_covariates,
            future_covariates,
            static_covariates,
            future_target,
        ) = train_sample

        n_past_covs, n_future_covs, n_stat_covs = 0, 0, 0
        if future_covariates is not None:
            n_future_covs = future_covariates.shape[1]
        if past_covariates is not None:
            n_past_covs = past_covariates.shape[1]
        if static_covariates is not None:
            n_stat_covs = static_covariates.shape[1]

        pl_module_params = self.pl_module_params or {}
        return _NeuralForecastModule(
            nf_model_class=self.nf_model_class,
            nf_model_params=self.nf_model_params,
            n_past_covs=n_past_covs,
            n_future_covs=n_future_covs,
            n_stat_covs=n_stat_covs,
            **pl_module_params,
        )

    @property
    def supports_past_covariates(self) -> bool:
        return self.nf_model_class.EXOGENOUS_HIST

    @property
    def supports_future_covariates(self) -> bool:
        return self.nf_model_class.EXOGENOUS_FUTR

    @property
    def supports_static_covariates(self) -> bool:
        return self.nf_model_class.EXOGENOUS_STAT
