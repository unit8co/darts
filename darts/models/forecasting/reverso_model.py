"""
Reverso
-------

Reverso is a highly parameter efficient model that achieves comparable performance with models 100x its size.

A combination of long convolutions and DeltaNet sequence mixing modules are used.

Reverso can be used the same way as other foundation models (e.g. Chronos2, TimesFM 2.5), with the exception
that it does not yet support any type of covariates or probabilistic forecasts.

For detailed examples and tutorials, check out the Chronos2 notebook:

* `Chronos-2 Foundation Model Examples
  <https://unit8co.github.io/darts/examples/25-Chronos-2-examples.html>`__
* `Fine-Tuning Examples
  <https://unit8co.github.io/darts/examples/27-Torch-and-Foundation-Model-Fine-Tuning-examples.html>`__
"""

import os
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from darts.logging import get_logger, raise_log
from darts.models.components.huggingface_connector import HuggingFaceConnector
from darts.models.components.reverso_submodels import (
    _AttentionBlock,
    _CNNBlock,
    _MLPBlock,
    _PositionalEmbedding,
)
from darts.models.forecasting.foundation_model import FoundationModel
from darts.models.forecasting.pl_forecasting_module import (
    PLForecastingModule,
    io_processor,
)
from darts.utils.data.torch_datasets.utils import PLModuleInput, TorchTrainingSample

logger = get_logger(__name__)


class _ReversoModule(PLForecastingModule):
    def __init__(
        self,
        seq_len: int = 2048,
        input_token_len: int = 2048,
        output_token_len: int = 48,
        d_model: int = 64,
        d_intermediate: int = 256,
        output_bottleneck_dim: int = 48,
        expand_v: float = 1.0,
        state_weaving: int | bool = False,
        gating_kernel_size: int = 3,
        main_module: str = "conv,attn,conv,attn",
        use_norm: int | bool = True,
        learn_bias: int | bool = True,
        use_output_pe: int | bool = False,
        **kwargs,
    ):
        """PyTorch module implementing the Reverso model, ported from
        `shinfxh/reverso <https://github.com/shinfxh/reverso>`_ and
        adapted for Darts :class:`PLForecastingModule` interface.

        Parameters
        ----------
        seq_len
            Context window length.
        input_token_len
            Input sequence length (must equal seq_len).
        output_token_len
            Number of time steps predicted per forward pass.
        d_model
            Model embedding dimension.
        d_intermediate
            MLP hidden dimension.
        output_bottleneck_dim
            Bottleneck dimension in the decoder head.
        expand_v
            Value dimension expansion factor for DeltaNet.
        state_weaving
            Whether to use state weaving in intermediate attention blocks.
        gating_kernel_size
            Kernel size for gating convolutions.
        main_module
            Comma-separated layer types, e.g. "conv,attn,conv,attn".
        use_norm
            Whether to apply min-max normalization to inputs.
        learn_bias
            Whether to use bias in the decoder head linear layer.
        use_output_pe
            Whether to use positional embeddings in the decoder head.
        **kwargs
            All parameters required for :class:`PLForecastingModule` base class.
        """
        kwargs.pop("enable_finetuning", False)
        super().__init__(**kwargs)

        self.seq_len = seq_len
        self.input_token_len = input_token_len
        self.output_token_len = output_token_len
        self.d_model = d_model
        self.use_norm = bool(use_norm)
        self.use_output_pe = bool(use_output_pe)

        # embedding
        self.embedding = nn.Linear(1, d_model, bias=False)

        # build encoder layers
        state_weaving = bool(state_weaving)
        module_list = [m.strip() for m in main_module.split(",")]
        e_layers = len(module_list)

        layers = []
        for i, layer_type in enumerate(module_list):
            if layer_type == "conv":
                layers.append(_CNNBlock(d_model, seq_len, gating_kernel_size))
            elif layer_type == "attn":
                is_intermediate = (i > 0) and (i < e_layers - 1)
                layers.append(
                    _AttentionBlock(d_model, expand_v, state_weaving, is_intermediate)
                )
            else:
                raise_log(
                    ValueError(f"Invalid layer type: {layer_type}"),
                    logger,
                )
            layers.append(_MLPBlock(d_model, d_model, d_intermediate))
        self.layers = nn.Sequential(*layers)

        # decoder head
        self.head = nn.Linear(
            input_token_len, output_bottleneck_dim, bias=bool(learn_bias)
        )
        self.simple_q_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, 1)

        # optional positional embedding for decoder head (used by full Reverso model)
        if self.use_output_pe:
            pe_max_len = seq_len + output_token_len
            self.output_position_embedding = _PositionalEmbedding(
                d_model, max_len=pe_max_len
            )
            self.post_pe_q_proj = nn.Linear(d_model, d_model)

        # slice for output_chunk_shift / output_chunk_length
        self.future_slice = slice(
            self.output_chunk_shift,
            self.output_chunk_shift + (self.output_chunk_length or 0),
        )

    def _reverso_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Core Reverso forward pass.

        Parameters
        ----------
        x
            Input tensor of shape (batch, seq_len, 1).

        Returns
        -------
        torch.Tensor
            Predictions of shape (batch, output_token_len, 1).
        """
        # min-max normalization
        if self.use_norm:
            x_min = x.min(1, keepdim=True)[0].detach()
            x_max = x.max(1, keepdim=True)[0].detach()
            x_range = torch.clamp(x_max - x_min, min=1e-5).detach()
            x = (x - x_min) / x_range
            means = x_min
            stdev = x_range

        # embedding: (B, seq_len, 1) -> (B, d_model, seq_len)
        x = self.embedding(x).transpose(1, 2)

        # encoder layers
        dec_out = self.layers(x)

        # decoder head
        temp_out = self.head(dec_out).permute(0, 2, 1)
        q = self.simple_q_proj(temp_out)

        dec_out_perm = dec_out.permute(0, 2, 1)

        if self.use_output_pe:
            full_hidden = torch.cat([dec_out_perm, q], dim=1)
            full_hidden = full_hidden + self.output_position_embedding(full_hidden)
            dec_out_pe = full_hidden[:, : dec_out_perm.shape[1], :]
            q = self.post_pe_q_proj(full_hidden[:, dec_out_perm.shape[1] :, :])
            k = self.key_proj(dec_out_pe)
            v = self.value_proj(dec_out_pe)
        else:
            k = self.key_proj(dec_out_perm)
            v = self.value_proj(dec_out_perm)

        attn = F.scaled_dot_product_attention(q, k, v)
        dec_out = self.out_proj(attn)

        # inverse normalization
        if self.use_norm:
            dec_out = dec_out * stdev + means

        return dec_out

    @io_processor
    def forward(self, x_in: PLModuleInput, *args, **kwargs) -> Any:
        """Reverso model forward pass.

        Parameters
        ----------
        x_in
            Comes as tuple ``(x_past, x_future, x_static)`` where ``x_past`` is the
            input/past chunk. Input dimensions are ``(n_samples, n_time_steps, n_variables)``.

        Returns
        -------
        torch.Tensor
            The output tensor of shape ``(n_samples, n_time_steps, n_targets, 1)``
            for deterministic forecasts.
        """
        # B: batch size, L: input chunk length, C: target components
        x_past, _, _ = x_in
        B, L, C = x_past.shape

        # channel independence: (B, L, C) -> (B*C, L)
        x = x_past.permute(0, 2, 1).reshape(-1, L)

        # left-pad with per-series first value to seq_len
        if L < self.seq_len:
            first_val = x[:, :1]  # (B*C, 1)
            x = torch.cat(
                [first_val.expand(-1, self.seq_len - L), x], dim=1
            )  # (B*C, seq_len)

        # (B*C, seq_len) -> (B*C, seq_len, 1)
        x = x.unsqueeze(-1)

        # core forward pass -> (B*C, output_token_len, 1)
        out = self._reverso_forward(x)

        # reshape back: (B*C, T, 1) -> (B, C, T, 1) -> (B, T, C, 1)
        out = out.reshape(B, C, self.output_token_len, 1)
        out = out.permute(0, 2, 1, 3)

        # truncate to output_chunk_length with output_chunk_shift
        out = out[:, self.future_slice, :, :]

        return out


class ReversoModel(FoundationModel):
    def __init__(
        self,
        input_chunk_length: int,
        output_chunk_length: int,
        output_chunk_shift: int = 0,
        hub_model_name: str = "shinfxh/reverso-small",
        hub_model_revision: str | None = None,
        local_dir: str | os.PathLike | None = None,
        **kwargs,
    ):
        """Reverso Model for zero-shot forecasting.

        This is an implementation of the Reverso model, ported from
        `shinfxh/reverso <https://github.com/shinfxh/reverso>`_ with adaptations to use the Darts API.
        Reverso is an efficient time-series foundation model combining long convolutions with
        DeltaNet (delta-rule linear attention) layers. With approximately 3 million parameters,
        it achieves performance parity with foundation models over 100x its size.

        This model supports either univariate or multivariate time series, but does not support covariates
        or probabilistic forecasts. For multivariate time series, the model is applied independently to each
        component.

        Using this model will automatically download and cache the pre-trained model from HuggingFace Hub.
        Alternatively, you can specify a local directory containing the model config and weights using the
        ``local_dir`` parameter.

        .. tip::
            You can perform full or partial fine-tuning of the model by setting the ``enable_finetuning`` parameter.
            Read more in the parameter description below and in the `Fine-Tuning Examples
            <https://unit8co.github.io/darts/examples/27-Torch-and-Foundation-Model-Fine-Tuning-examples.html>`__.

        Parameters
        ----------
        input_chunk_length
            Number of time steps in the past to take as a model input (per chunk). Applies to the target
            series. For Reverso, ``input_chunk_length`` must be less than or equal to the model's context
            length (2048 for all Reverso variants).
        output_chunk_length
            Number of time steps predicted at once (per chunk) by the internal model. It is not the same
            as forecast horizon ``n`` used in ``predict()``, which is the desired number of prediction points
            generated using either a one-shot- or autoregressive forecast. Setting ``n <= output_chunk_length``
            prevents auto-regression.
            For Reverso, ``output_chunk_length + output_chunk_shift`` must be less than or equal to the
            model's output token length (48 for all Reverso variants).
        output_chunk_shift
            Optionally, the number of steps to shift the start of the output chunk into the future (relative to the
            input chunk end). This will create a gap between the input and output. Predictions will start
            ``output_chunk_shift`` steps after the end of the target ``series``. If ``output_chunk_shift`` is set,
            the model cannot generate autoregressive predictions (``n > output_chunk_length``).
        hub_model_name
            The model ID on HuggingFace Hub. Default: ``"shinfxh/reverso-small"``.
        hub_model_revision
            The model version to use. This can be a branch name, tag name, or commit hash.
        local_dir
            Optional local directory to load the pre-downloaded model. If specified and the directory is empty, the
            model will be downloaded from HuggingFace Hub and saved to this directory. Default is ``None``, which will
            use a cache directory managed by ``huggingface_hub`` instead. Note that this is different from the
            ``work_dir`` parameter used for saving model checkpoints during fine-tuning.
        **kwargs
            Optional arguments to initialize the pytorch_lightning.Module, pytorch_lightning.Trainer, and
            Darts' :class:`TorchForecastingModel`.

        loss_fn
            PyTorch loss function used for fine-tuning. Default: ``nn.MSELoss()``.
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
        enable_finetuning
            Enables model fine-tuning. Only effective if not ``None``.
            If a bool, specifies whether to perform full fine-tuning / training (all parameters are updated) or keep
            all parameters frozen. If a dict, specifies which parameters to fine-tune. Must only contain one key-value
            record. Can be used to:

            - Unfreeze specific parameters, while keeping everything else frozen:
              ``{"unfreeze": ["param.name.patterns.*"]}``
            - Freeze specific parameters, while keeping everything else unfrozen:
              ``{"freeze": ["param.name.patterns.*"]}``

            Default: ``None``.

        References
        ----------
        .. [1] X. Fu, Y. Li, G. Papaioannou, Y. Kim. "Reverso: Efficient Time Series Foundation Models for
                Zero-shot Forecasting", 2026. arXiv https://arxiv.org/abs/2602.17634.

        Examples
        --------
        >>> from darts.datasets import WeatherDataset
        >>> from darts.models import ReversoModel
        >>> # load data in float32 format (macOS issues with float64 and PyTorch)
        >>> series = WeatherDataset().load().astype("float32")
        >>> # predicting atmospheric pressure
        >>> target = series['p (mbar)'][:200]
        >>> model = ReversoModel(
        >>>     input_chunk_length=96,
        >>>     output_chunk_length=48,
        >>> )
        >>> # calling fit is still mandatory to ensure consistent number of components; however,
        >>> # ReversoModel is training-free and the model weights are not updated
        >>> model.fit(target)
        >>> pred = model.predict(48)

        .. note::
            Reverso is licensed under the `MIT License <https://github.com/shinfxh/reverso/blob/main/LICENSE>`_,
            Copyright (c) 2026 Xinghong Fu, Yanhong Li, Georgios Papaioannou, Yoon Kim.
            By using this model, you agree to the terms and conditions of the license.
        .. note::
            Reverso does not support covariates natively. For multivariate time series, each component
            is forecasted independently.
        .. warning::
            CPU inference is significantly slower than GPU due to the use of torch Conv instead of
            flashfft and sequential delta-rule computation instead of fla implementation. GPU is
            recommended for production use. See https://github.com/shinfxh/reverso/.
        """
        hf_connector = HuggingFaceConnector(
            model_name=hub_model_name,
            model_revision=hub_model_revision,
            local_dir=local_dir,
        )

        # load model config for validation
        config = hf_connector.load_config()

        # validate input_chunk_length against model's context length
        context_length = config["seq_len"]
        if input_chunk_length > context_length:
            raise_log(
                ValueError(
                    f"`input_chunk_length` {input_chunk_length} cannot be greater than "
                    f"model's context length {context_length}"
                ),
                logger,
            )

        # validate output_chunk_length + output_chunk_shift against model's output length
        prediction_length = config["output_token_len"]
        if output_chunk_length + output_chunk_shift > prediction_length:
            raise_log(
                ValueError(
                    f"`output_chunk_length` {output_chunk_length} plus `output_chunk_shift` "
                    f"{output_chunk_shift} cannot be greater than model's maximum prediction "
                    f"length {prediction_length}"
                ),
                logger,
            )

        self.hf_connector = hf_connector
        super().__init__(**kwargs)

    def _create_model(self, train_sample: TorchTrainingSample) -> PLForecastingModule:
        pl_module_params = self.pl_module_params or {}
        return self.hf_connector.load_model(
            module_class=_ReversoModule,
            pl_module_params=pl_module_params,
        )

    @property
    def supports_past_covariates(self) -> bool:
        return False

    @property
    def supports_future_covariates(self) -> bool:
        return False
