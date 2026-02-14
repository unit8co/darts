"""
TimesFM 2.5
-----------

TimesFM 2.5 can be used the same way as other foundation models (e.g. Chronos2), with the exception
that it does not support any type of covariates.

For detailed examples and tutorials, check out the Chronos2 notebook:

* `Chronos-2 Foundation Model Examples
  <https://unit8co.github.io/darts/examples/25-Chronos-2-examples.html>`__
"""

import os
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from darts.logging import get_logger, raise_log
from darts.models.components.huggingface_connector import HuggingFaceConnector
from darts.models.components.timesfm2p5_submodels import (
    _ResidualBlock,
    _ResidualBlockConfig,
    _revin,
    _StackedTransformersConfig,
    _Transformer,
    _TransformerConfig,
    _update_running_stats,
)
from darts.models.forecasting.foundation_model import FoundationModel
from darts.models.forecasting.pl_forecasting_module import (
    PLForecastingModule,
    io_processor,
)
from darts.utils.data.torch_datasets.utils import PLModuleInput, TorchTrainingSample
from darts.utils.likelihood_models import QuantileRegression

logger = get_logger(__name__)


@dataclass(frozen=True)
class _TimesFM2p5_200M_Definition:
    """Framework-agnostic config of TimesFM 2.5."""

    context_limit = 16384
    input_patch_len: int = 32
    output_patch_len: int = 128
    quantiles: list[float] = field(
        default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    )
    tokenizer: _ResidualBlockConfig = _ResidualBlockConfig(
        input_dims=64,
        hidden_dims=1280,
        output_dims=1280,
        use_bias=True,
        activation="swish",
    )
    stacked_transformers: _StackedTransformersConfig = _StackedTransformersConfig(
        num_layers=20,
        transformer=_TransformerConfig(
            model_dims=1280,
            hidden_dims=1280,
            num_heads=16,
            attention_norm="rms",
            feedforward_norm="rms",
            qk_norm="rms",
            use_bias=False,
            use_rotary_position_embeddings=True,
            ff_activation="swish",
            fuse_qkv=True,
        ),
    )
    output_projection_point: _ResidualBlockConfig = _ResidualBlockConfig(
        input_dims=1280,
        hidden_dims=1280,
        output_dims=1280,
        use_bias=False,
        activation="swish",
    )
    output_projection_quantiles: _ResidualBlockConfig = _ResidualBlockConfig(
        input_dims=1280,
        hidden_dims=1280,
        output_dims=10240,
        use_bias=False,
        activation="swish",
    )


class _TimesFM2p5Module(PLForecastingModule):
    config = _TimesFM2p5_200M_Definition()

    def __init__(
        self,
        **kwargs,
    ):
        """PyTorch module implementing the TimesFM 2.5 model, ported from
        `google-research/timesfm <https://github.com/google-research/timesfm/>`_ and
        adapted for Darts :class:`PLForecastingModule` interface.

        Parameters
        ----------
        **kwargs
            all parameters required for :class:`darts.models.forecasting.pl_forecasting_module.PLForecastingModule`
            base class.
        """

        super().__init__(**kwargs)

        # default model parameters (config.json is ignored)
        self.input_patch_len = self.config.input_patch_len  # 32
        self.output_patch_len = self.config.output_patch_len  # 128
        self.num_layers = self.config.stacked_transformers.num_layers  # 20
        # see below `user_quantile_indices` for explanation of +1
        self.num_quantiles_plus_one = len(self.config.quantiles) + 1  # 10

        # padding length for input target series to make its length a multiple of
        # input_patch_len (32).
        self.pad_len = -self.input_chunk_length % self.input_patch_len

        # define model submodules
        self.tokenizer = _ResidualBlock(self.config.tokenizer)
        self.stacked_xf = nn.ModuleList([
            _Transformer(self.config.stacked_transformers.transformer)
            for _ in range(self.num_layers)
        ])
        self.output_projection_point = _ResidualBlock(
            self.config.output_projection_point
        )
        self.output_projection_quantiles = _ResidualBlock(
            self.config.output_projection_quantiles
        )

        self.future_slice = slice(
            self.output_chunk_shift,
            self.output_chunk_shift + (self.output_chunk_length or 0),
        )

        # gather indices of user-specified quantiles
        user_quantiles: list[float] = (
            self.likelihood.quantiles
            if isinstance(self.likelihood, QuantileRegression)
            else [0.5]
        )
        # The original quantile outputs contain mean + quantiles (0.1 to 0.9),
        # but the mean is not being used even in deterministic setting.
        # Instead, the median (0.5 quantile) is used as the deterministic output.
        self.user_quantile_indices = [
            self.config.quantiles.index(q) + 1 for q in user_quantiles
        ]

    def _forward(
        self,
        inputs: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        """Original forward pass of the TimesFM 2.5 model.

        Parameters
        ----------
        inputs
            Input tensor of shape (batch_size, num_input_patches, input_patch_len).
        masks
            Mask tensor of shape (batch_size, num_input_patches, input_patch_len),
            where True indicates a missing value.

        Returns
        -------
        torch.Tensor
            Quantile predictions of shape `(batch_size, output_patch_len * num_quantiles_plus_one)`.
            The last dimension contains the (unused) mean followed by nine quantile predictions (0.1 to 0.9).
        """
        # See comments in `forward()` for explanation of dimension notations.
        # `inputs`, `masks`: (B * C, Q, I)
        # `tokenizer_inputs`: (B * C, Q, I * 2)
        tokenizer_inputs = torch.cat([inputs, masks.to(inputs.dtype)], dim=-1)

        # tokenization
        # `output_embeddings`: (B * C, Q, D)
        output_embeddings = self.tokenizer(tokenizer_inputs)

        # stacked transformer layers
        for _, layer in enumerate(self.stacked_xf):
            # -> (B * C, Q, D)
            output_embeddings = layer(output_embeddings, masks[..., -1])

        # use only the last patch embeddings
        # `last_embeddings`: (B * C, D)
        last_embeddings = output_embeddings[:, -1, :]

        # output projections
        # `output_ts`: (B * C, O * W)
        output_ts = self.output_projection_point(last_embeddings)
        # output_quantile_spread = self.output_projection_quantiles(last_embeddings)

        return output_ts

    # TODO: fine-tuning support
    @io_processor
    def forward(self, x_in: PLModuleInput, *args, **kwargs) -> Any:
        """TimesFM 2.5 model forward pass.

        Parameters
        ----------
        x_in
            comes as tuple `(x_past, x_future, x_static)` where `x_past` is the input/past chunk and `x_future`
            is the output/future chunk. Input dimensions are `(n_samples, n_time_steps, n_variables)`

        Returns
        -------
        torch.Tensor
            the output tensor in the shape of `(n_samples, n_time_steps, n_targets, n_quantiles)` for
            probabilistic forecasts, or `(n_samples, n_time_steps, n_targets, 1)` for
            deterministic forecasts (median only).
        """
        # B: batch size
        # L: input chunk length
        # T: output chunk length
        # I = 32: input patch length
        # O = 128: output patch length
        # P: minimum left-pad length such that (P+L) is divisible by I
        # Z = P + L: padded input chunk length
        # Q = Z / I: patches for the input chunk
        # W = 10: quantiles + 1 (mean + 9 quantiles)
        # C: target components
        # D: hidden dimensions
        # N: likelihood quantiles (user-specified)

        # `x_past`: (B, L, C)
        x_past, _, _ = x_in

        # TimesFM 2.5 is a univariate model and its inputs does not have a variable dimension,
        # so here we reshape `x_past` to (B * C, L)
        x_past = x_past.permute(0, 2, 1).reshape(-1, self.input_chunk_length)

        # We assume there are no missing values in x_past, so strip_leading_nans() and
        # linear_interpolation() are not needed here.

        # left-pad x_past with NaNs to make its length a multiple of input_patch_len (32)
        # `x_past` -> (B * C, Z)
        if self.pad_len > 0:
            x_past = F.pad(x_past, (self.pad_len, 0), value=float("nan"))

        # create mask for x_past
        # `x_mask`: (B * C, Z)
        mask = torch.isnan(x_past)

        # divide x_past and mask into patches of size input_patch_len (32)
        # -> (B * C, Q, I)
        patched_x_past = x_past.unfold(1, self.input_patch_len, self.input_patch_len)
        patched_mask = mask.unfold(1, self.input_patch_len, self.input_patch_len)
        # determine batch size and number of input patches after patching
        batch_comp_size, num_input_patches, _ = patched_x_past.shape

        # running stats of mean (mu) and stddev (sigma) for each input patch
        # `n`, `mu`, `sigma`: (B * C,)
        n = torch.zeros(batch_comp_size, device=patched_x_past.device)
        mu = torch.zeros(batch_comp_size, device=patched_x_past.device)
        sigma = torch.zeros(batch_comp_size, device=patched_x_past.device)
        patch_mu = []
        patch_sigma = []
        for i in range(num_input_patches):
            (n, mu, sigma), _ = _update_running_stats(
                n, mu, sigma, patched_x_past[:, i], patched_mask[:, i]
            )
            patch_mu.append(mu)
            patch_sigma.append(sigma)
        # `context_mu`, `context_sigma`: (B * C, Q)
        context_mu = torch.stack(patch_mu, dim=1)
        context_sigma = torch.stack(patch_sigma, dim=1)

        # normalize inputs and apply mask
        # `normed_inputs`: (B * C, Q, I)
        normed_inputs = _revin(patched_x_past, context_mu, context_sigma, reverse=False)
        normed_inputs = torch.where(patched_mask, 0.0, normed_inputs)

        # forward pass
        # `normed_outputs`: (B * C, O * W)
        normed_outputs = self._forward(normed_inputs, patched_mask)

        # inverse normalization
        # `renormed_outputs`: (B * C, O * W)
        renormed_outputs = _revin(normed_outputs, mu, sigma, reverse=True)

        # -> (B, C, O, W)
        renormed_outputs = torch.reshape(
            renormed_outputs,
            (-1, self.n_targets, self.output_patch_len, self.num_quantiles_plus_one),
        )
        # -> (B, O, C, W)
        renormed_outputs = renormed_outputs.permute(0, 2, 1, 3)

        # truncate to output_chunk_length
        # -> (B, T, C, W)
        renormed_outputs = renormed_outputs[:, self.future_slice, :, :]

        # select only user-specified quantiles or median if deterministic
        # -> (B, T, C, N)
        renormed_outputs = renormed_outputs[:, :, :, self.user_quantile_indices]

        return renormed_outputs


class TimesFM2p5Model(FoundationModel):
    # Fine-tuning is turned off for now pending proper fine-tuning support
    # and configuration.
    _allows_finetuning = False

    def __init__(
        self,
        input_chunk_length: int,
        output_chunk_length: int,
        output_chunk_shift: int = 0,
        likelihood: QuantileRegression | None = None,
        hub_model_name: str = "google/timesfm-2.5-200m-pytorch",
        hub_model_revision: str | None = "1d952420fba87f3c6dee4f240de0f1a0fbc790e3",
        local_dir: str | os.PathLike | None = None,
        **kwargs,
    ):
        """TimesFM 2.5 Model for zero-shot forecasting.

        This is an implementation of Google's TimesFM 2.5 model, ported from
        `google-research/timesfm <https://github.com/google-research/timesfm>`_ with adaptations to use the Darts API.
        It is an updated version of the original TimesFM model [1]_, [2]_, with a larger context length (16,384 vs 512)
        and better predictive accuracy.

        This model supports either univariate or multivariate time series, but does not support covariates.
        For multivariate time series, the model is applied independently to each component.

        Using this model will automatically download and cache the pre-trained model from HuggingFace Hub
        (`google/timesfm-2.5-200m-pytorch <https://huggingface.co/google/timesfm-2.5-200m-pytorch/tree/main>`_).
        Alternatively, you can specify a local directory containing the model config and weights using the ``local_dir``
        parameter.

        By default, this model is deterministic and outputs only the median (0.5 quantile). To enable probabilistic
        forecasts, pass a :class:`~darts.utils.likelihood_models.torch.QuantileRegression` instance to the
        ``likelihood`` parameter. The quantiles used must be a subset of those used during TimesFM 2.5 pre-training, see
        below for details. It is recommended to call :func:`predict()` with ``predict_likelihood_parameters=True``
        or ``num_samples >> 1`` to get meaningful results.

        Parameters
        ----------
        input_chunk_length
            Number of time steps in the past to take as a model input (per chunk). Applies to the target
            series, and past and/or future covariates (if the model supports it).
            For TimesFM 2.5, `input_chunk_length + output_chunk_length + output_chunk_shift` must be less than or equal
            to 16,384.
        output_chunk_length
            Number of time steps predicted at once (per chunk) by the internal model. Also, the number of future values
            from future covariates to use as a model input (if the model supports future covariates). It is not the same
            as forecast horizon `n` used in `predict()`, which is the desired number of prediction points generated
            using either a one-shot- or autoregressive forecast. Setting `n <= output_chunk_length` prevents
            auto-regression. This is useful when the covariates don't extend far enough into the future, or to prohibit
            the model from using future values of past and / or future covariates for prediction (depending on the
            model's covariate support).
            For TimesFM 2.5, `output_chunk_length + output_chunk_shift` must be less than or equal to 128.
        output_chunk_shift
            Optionally, the number of steps to shift the start of the output chunk into the future (relative to the
            input chunk end). This will create a gap between the input and output. If the model supports
            `future_covariates`, the future values are extracted from the shifted output chunk. Predictions will start
            `output_chunk_shift` steps after the end of the target `series`. If `output_chunk_shift` is set, the model
            cannot generate autoregressive predictions (`n > output_chunk_length`).
        likelihood
            The likelihood model to be used for probabilistic forecasts. Must be ``None`` or an instance of
            :class:`~darts.utils.likelihood_models.torch.QuantileRegression`. If using ``QuantileRegression``,
            the quantiles must be a subset of those used during TimesFM 2.5 pre-training:
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9].
            Default: ``None``, which will make TimesFM 2.5 deterministic (median quantile only).
        hub_model_name
            The model ID on HuggingFace Hub. Default: ``"google/timesfm-2.5-200m-pytorch"``.
        hub_model_revision
            The model version to use. This can be a branch name, tag name, or commit hash. Default is
            ``1d952420fba87f3c6dee4f240de0f1a0fbc790e3``, which will use the October 2, 2025 release of TimesFM 2.5.
        local_dir
            Optional local directory to load the pre-downloaded model. If specified and the directory is empty, the
            model will be downloaded from HuggingFace Hub and saved to this directory. Default is ``None``, which will
            use a cache directory managed by ``huggingface_hub`` instead. Note that this is different from the
            ``work_dir`` parameter used for saving model checkpoints during fine-tuning.
        **kwargs
            Optional arguments to initialize the pytorch_lightning.Module, pytorch_lightning.Trainer, and
            Darts' :class:`TorchForecastingModel`.

        loss_fn
            PyTorch loss function used for fine-tuning a deterministic TimesFM 2.5 model. Ignored for probabilistic
            TimesFM 2.5 when ``likelihood`` is specified. Default: ``nn.MSELoss()``.
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
            Whether to use reversible instance normalization `RINorm` against distribution shift as shown in [3]_.
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
        .. [1] A. Das, W. Kong, R. Sen, Y. Zhou. "A decoder-only foundation model for time-series forecasting", 2025.
                arXiv https://arxiv.org/abs/2310.10688.
        .. [2] "A decoder-only foundation model for time-series forecasting", 2024. Google Research.
                https://research.google/blog/a-decoder-only-foundation-model-for-time-series-forecasting/
        .. [3] T. Kim et al. "Reversible Instance Normalization for Accurate Time-Series Forecasting against
                Distribution Shift", https://openreview.net/forum?id=cGDAkQo1C0p

        Examples
        --------
        >>> from darts.datasets import WeatherDataset
        >>> from darts.models import TimesFM2p5Model
        >>> # load data in float32 format (macOS issues with float64 and PyTorch)
        >>> series = WeatherDataset().load().astype("float32")
        >>> # predicting atmospheric pressure
        >>> target = series['p (mbar)'][:100]
        >>> # by default, TimesFM2p5Model is deterministic; to enable probabilistic forecasts,
        >>> # set likelihood to QuantileRegression and use a subset of the pre-trained quantiles
        >>> model = TimesFM2p5Model(
        >>>     input_chunk_length=6,
        >>>     output_chunk_length=6,
        >>> )
        >>> # calling fit is still mandatory to ensure consistent number of components; however,
        >>> # TimesFM2p5Model is training-free and the model weights are not updated
        >>> model.fit(target)
        >>> # when TimesFM2p5Model is probabilistic, set ``predict_likelihood_parameters=True``
        >>> # or ``num_samples>>1`` to get meaningful results
        >>> pred = model.predict(6)
        >>> print(pred.all_values())
        [[[1005.7797 ]]
        [[1005.78766]]
        [[1005.7985 ]]
        [[1005.7852 ]]
        [[1005.7882 ]]
        [[1005.79565]]]

        .. note::
            TimesFM 2.5 does not support covariates natively. The source implementation uses `Xreg` to fit a ridge
            regression between covariates and the target series (or forecast residuals) as a pre/post-processing step.
            You can implement a similar approach externally in Darts.
            See `Issue #2976 <https://github.com/unit8co/darts/issues/2976#issuecomment-3691415141>`_ for details.
        .. note::
            Fine-tuning of TimesFM 2.5 is not supported at the moment.
        .. note::
            TimesFM 2.5 is licensed under the `Apache-2.0 License <https://github.com/google-research/timesfm/blob/master/LICENSE>`_,
            Copyright 2025 Google LLC. By using this model, you agree to the terms and conditions of the license.
        .. warning::
            Due to differences in probabilistic sampling methods, zero-shot forecasts obtained here would differ from
            those obtained using the original implementation when prediction horizon `n` is larger than 128.
        """
        hf_connector = HuggingFaceConnector(
            model_name=hub_model_name,
            model_revision=hub_model_revision,
            local_dir=local_dir,
        )

        # As per the original implementation, the model config is ignored and default
        # parameters are used instead.
        config = _TimesFM2p5_200M_Definition()

        # validate `input_chunk_length` against model's maximum context_length
        context_length = config.context_limit
        if (
            input_chunk_length + output_chunk_length + output_chunk_shift
            > context_length
        ):
            raise_log(
                ValueError(
                    f"`input_chunk_length` {input_chunk_length} plus `output_chunk_length` {output_chunk_length} "
                    f"plus `output_chunk_shift` {output_chunk_shift} cannot be greater than model's maximum "
                    f"context_length {context_length}"
                ),
                logger,
            )

        # validate `output_chunk_length` and `output_chunk_shift` against model's output limits
        prediction_length = config.output_patch_len
        if output_chunk_length + output_chunk_shift > prediction_length:
            raise_log(
                ValueError(
                    f"`output_chunk_length` {output_chunk_length} plus `output_chunk_shift` {output_chunk_shift} "
                    f"cannot be greater than model's maximum prediction length {prediction_length}"
                ),
                logger,
            )

        quantiles = config.quantiles
        # by default (`likelihood=None`), model is deterministic
        # otherwise, only QuantileRegression likelihood is supported and quantiles must be
        # a subset of the pre-trained quantiles
        if likelihood is not None:
            if not isinstance(likelihood, QuantileRegression):
                raise_log(
                    ValueError(
                        f"Only QuantileRegression likelihood is supported for TimesFM 2.5 in Darts. "
                        f"Got {type(likelihood)}."
                    ),
                    logger,
                )
            user_quantiles: list[float] = likelihood.quantiles
            if not set(user_quantiles).issubset(quantiles):
                raise_log(
                    ValueError(
                        f"The quantiles for QuantileRegression likelihood {user_quantiles} "
                        f"must be a subset of TimesFM 2.5 quantiles {quantiles}."
                    ),
                    logger,
                )

        self.hf_connector = hf_connector
        super().__init__(enable_finetuning=False, **kwargs)

    def _create_model(self, train_sample: TorchTrainingSample) -> PLForecastingModule:
        pl_module_params = self.pl_module_params or {}
        return self.hf_connector.load_model(
            module_class=_TimesFM2p5Module,
            pl_module_params=pl_module_params,
        )

    @property
    def supports_past_covariates(self) -> bool:
        return False

    @property
    def supports_future_covariates(self) -> bool:
        return False
