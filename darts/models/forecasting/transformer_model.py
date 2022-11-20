"""
Transformer Model
-----------------
"""

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from darts.logging import get_logger, raise_if, raise_if_not, raise_log
from darts.models.components import glu_variants, layer_norm_variants
from darts.models.components.glu_variants import GLU_FFN
from darts.models.components.transformer import (
    CustomFeedForwardDecoderLayer,
    CustomFeedForwardEncoderLayer,
)
from darts.models.forecasting.pl_forecasting_module import PLPastCovariatesModule
from darts.models.forecasting.torch_forecasting_model import PastCovariatesTorchModel

logger = get_logger(__name__)


BUILT_IN = ["relu", "gelu"]
FFN = GLU_FFN + BUILT_IN


def _generate_coder(
    d_model,
    dim_ff,
    dropout,
    nhead,
    num_layers,
    norm_layer,
    coder_cls,
    layer_cls,
    ffn_cls,
):
    """Generates an Encoder or Decoder with one of Darts' Feed-forward Network variants.
    Parameters
    ----------
    coder_cls
        Either `torch.nn.TransformerEncoder` or `...TransformerDecoder`
    layer_cls
        Either `darts.models.components.transformer.CustomFeedForwardEncoderLayer`,
        `...CustomFeedForwardDecoderLayer`, `nn.TransformerEncoderLayer`, or `nn.TransformerDecoderLayer`.
    ffn_cls
        One of Darts' Position-wise Feed-Forward Network variants `from darts.models.components.glu_variants`
    """

    ffn = (
        dict(ffn=ffn_cls(d_model=d_model, d_ff=dim_ff, dropout=dropout))
        if ffn_cls
        else dict()
    )
    layer = layer_cls(
        **ffn,
        dropout=dropout,
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_ff,
    )
    return coder_cls(
        layer,
        num_layers=num_layers,
        norm=norm_layer(d_model),
    )


# This implementation of positional encoding is taken from the PyTorch documentation:
# https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class _PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        """An implementation of positional encoding as described in 'Attention is All you Need' by Vaswani et al. (2017)

        Parameters
        ----------
        d_model
            The number of expected features in the transformer encoder/decoder inputs.
            Last dimension of the input.
        dropout
            Fraction of neurons affected by Dropout (default=0.1).
        max_len
            The dimensionality of the computed positional encoding array.
            Only its first "input_size" elements will be considered in the output.

        Inputs
        ------
        x of shape `(batch_size, input_size, d_model)`
            Tensor containing the embedded time series.

        Outputs
        -------
        y of shape `(batch_size, input_size, d_model)`
            Tensor containing the embedded time series enhanced with positional encoding.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class _TransformerModule(PLPastCovariatesModule):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        nr_params: int,
        d_model: int,
        nhead: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float,
        activation: str,
        norm_type: Union[str, nn.Module, None] = None,
        custom_encoder: Optional[nn.Module] = None,
        custom_decoder: Optional[nn.Module] = None,
        **kwargs,
    ):
        """PyTorch module implementing a Transformer to be used in `TransformerModel`.

        PyTorch module implementing a simple encoder-decoder transformer architecture.

        Parameters
        ----------
        input_size
            The dimensionality of the TimeSeries instances that will be fed to the the fit and predict functions.
        output_size
            The dimensionality of the output time series.
        nr_params
            The number of parameters of the likelihood (or 1 if no likelihood is used).
        d_model
            The number of expected features in the transformer encoder/decoder inputs.
        nhead
            The number of heads in the multiheadattention model.
        num_encoder_layers
            The number of encoder layers in the encoder.
        num_decoder_layers
            The number of decoder layers in the decoder.
        dim_feedforward
            The dimension of the feedforward network model.
        dropout
            Fraction of neurons affected by Dropout.
        activation
            The activation function of encoder/decoder intermediate layer.
        norm_type: str | nn.Module | None
            The type of LayerNorm variant to use.
        custom_encoder
            A custom transformer encoder provided by the user (default=None).
        custom_decoder
            A custom transformer decoder provided by the user (default=None).
        **kwargs
            All parameters required for :class:`darts.model.forecasting_models.PLForecastingModule` base class.

        Inputs
        ------
        x of shape `(batch_size, input_chunk_length, input_size)`
            Tensor containing the features of the input sequence.

        Outputs
        -------
        y of shape `(batch_size, output_chunk_length, target_size, nr_params)`
            Tensor containing the prediction at the last time step of the sequence.
        """

        super().__init__(**kwargs)

        self.input_size = input_size
        self.target_size = output_size
        self.nr_params = nr_params
        self.target_length = self.output_chunk_length

        self.encoder = nn.Linear(input_size, d_model)
        self.positional_encoding = _PositionalEncoding(
            d_model, dropout, self.input_chunk_length
        )

        if isinstance(norm_type, str):
            try:
                self.layer_norm = getattr(layer_norm_variants, norm_type)
            except AttributeError:
                raise_log(
                    AttributeError("please provide a valid layer norm type"),
                )
        else:
            self.layer_norm = norm_type

        raise_if_not(activation in FFN, f"'{activation}' is not in {FFN}")
        if activation in GLU_FFN:
            raise_if(
                custom_encoder is not None or custom_decoder is not None,
                "Cannot use `custom_encoder` or `custom_decoder` along with an `activation` from "
                f"{GLU_FFN}",
                logger=logger,
            )
            # use glu variant feed-forward layers
            ffn_cls = getattr(glu_variants, activation)

            # custom feed-forward layers have activation built-in. reset activation
            activation = None

            custom_encoder = _generate_coder(
                d_model,
                dim_feedforward,
                dropout,
                nhead,
                num_encoder_layers,
                self.layer_norm if self.layer_norm else nn.LayerNorm,
                coder_cls=nn.TransformerEncoder,
                layer_cls=CustomFeedForwardEncoderLayer,
                ffn_cls=ffn_cls,
            )

            custom_decoder = _generate_coder(
                d_model,
                dim_feedforward,
                dropout,
                nhead,
                num_decoder_layers,
                self.layer_norm if self.layer_norm else nn.LayerNorm,
                coder_cls=nn.TransformerDecoder,
                layer_cls=CustomFeedForwardDecoderLayer,
                ffn_cls=ffn_cls,
            )

        # if layer norm set and no GLU variant is used
        if self.layer_norm and custom_decoder is None:
            custom_encoder = _generate_coder(
                d_model,
                dim_feedforward,
                dropout,
                nhead,
                num_encoder_layers,
                self.layer_norm,
                coder_cls=nn.TransformerEncoder,
                layer_cls=nn.TransformerEncoderLayer,
                ffn_cls=None,
            )

            custom_decoder = _generate_coder(
                d_model,
                dim_feedforward,
                dropout,
                nhead,
                num_decoder_layers,
                self.layer_norm,
                coder_cls=nn.TransformerDecoder,
                layer_cls=nn.TransformerDecoderLayer,
                ffn_cls=None,
            )

        # Defining the Transformer module
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            custom_encoder=custom_encoder,
            custom_decoder=custom_decoder,
        )

        self.decoder = nn.Linear(
            d_model, self.output_chunk_length * self.target_size * self.nr_params
        )

    def _create_transformer_inputs(self, data):
        # '_TimeSeriesSequentialDataset' stores time series in the
        # (batch_size, input_chunk_length, input_size) format. PyTorch's nn.Transformer
        # module needs it the (input_chunk_length, batch_size, input_size) format.
        # Therefore, the first two dimensions need to be swapped.
        src = data.permute(1, 0, 2)
        tgt = src[-1:, :, :]

        return src, tgt

    def forward(self, x_in: Tuple):
        data, _ = x_in
        # Here we create 'src' and 'tgt', the inputs for the encoder and decoder
        # side of the Transformer architecture
        src, tgt = self._create_transformer_inputs(data)

        # "math.sqrt(self.input_size)" is a normalization factor
        # see section 3.2.1 in 'Attention is All you Need' by Vaswani et al. (2017)
        src = self.encoder(src) * math.sqrt(self.input_size)
        src = self.positional_encoding(src)

        tgt = self.encoder(tgt) * math.sqrt(self.input_size)
        tgt = self.positional_encoding(tgt)

        x = self.transformer(src=src, tgt=tgt)
        out = self.decoder(x)

        # Here we change the data format
        # from (1, batch_size, output_chunk_length * output_size)
        # to (batch_size, output_chunk_length, output_size, nr_params)
        predictions = out[0, :, :]
        predictions = predictions.view(
            -1, self.target_length, self.target_size, self.nr_params
        )

        return predictions


class TransformerModel(PastCovariatesTorchModel):
    def __init__(
        self,
        input_chunk_length: int,
        output_chunk_length: int,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        activation: str = "relu",
        norm_type: Union[str, nn.Module, None] = None,
        custom_encoder: Optional[nn.Module] = None,
        custom_decoder: Optional[nn.Module] = None,
        **kwargs,
    ):

        """Transformer model

        Transformer is a state-of-the-art deep learning model introduced in 2017. It is an encoder-decoder
        architecture whose core feature is the 'multi-head attention' mechanism, which is able to
        draw intra-dependencies within the input vector and within the output vector ('self-attention')
        as well as inter-dependencies between input and output vectors ('encoder-decoder attention').
        The multi-head attention mechanism is highly parallelizable, which makes the transformer architecture
        very suitable to be trained with GPUs.

        The transformer architecture implemented here is based on [1]_.

        This model supports past covariates (known for `input_chunk_length` points before prediction time).

        Parameters
        ----------
        input_chunk_length
            Number of time steps to be input to the forecasting module.
        output_chunk_length
            Number of time steps to be output by the forecasting module.
        d_model
            The number of expected features in the transformer encoder/decoder inputs (default=64).
        nhead
            The number of heads in the multi-head attention mechanism (default=4).
        num_encoder_layers
            The number of encoder layers in the encoder (default=3).
        num_decoder_layers
            The number of decoder layers in the decoder (default=3).
        dim_feedforward
            The dimension of the feedforward network model (default=512).
        dropout
            Fraction of neurons affected by Dropout (default=0.1).
        activation
            The activation function of encoder/decoder intermediate layer, (default='relu').
            can be one of the glu variant's FeedForward Network (FFN)[2]. A feedforward network is a
            fully-connected layer with an activation. The glu variant's FeedForward Network are a series
            of FFNs designed to work better with Transformer based models. ["GLU", "Bilinear", "ReGLU", "GEGLU",
            "SwiGLU", "ReLU", "GELU"] or one the pytorch internal activations ["relu", "gelu"]
        norm_type: str | nn.Module
            The type of LayerNorm variant to use.  Default: ``None``. Available options are
            ["LayerNorm", "RMSNorm", "LayerNormNoBias"], or provide a custom nn.Module.
        custom_encoder
            A custom user-provided encoder module for the transformer (default=None).
        custom_decoder
            A custom user-provided decoder module for the transformer (default=None).
        **kwargs
            Optional arguments to initialize the pytorch_lightning.Module, pytorch_lightning.Trainer, and
            Darts' :class:`TorchForecastingModel`.

        loss_fn
            PyTorch loss function used for training.
            This parameter will be ignored for probabilistic models if the ``likelihood`` parameter is specified.
            Default: ``torch.nn.MSELoss()``.
        likelihood
            One of Darts' :meth:`Likelihood <darts.utils.likelihood_models.Likelihood>` models to be used for
            probabilistic forecasts. Default: ``None``.
        torch_metrics
            A torch metric or a ``MetricCollection`` used for evaluation. A full list of available metrics can be found
            at https://torchmetrics.readthedocs.io/en/latest/. Default: ``None``.
        optimizer_cls
            The PyTorch optimizer class to be used. Default: ``torch.optim.Adam``.
        optimizer_kwargs
            Optionally, some keyword arguments for the PyTorch optimizer (e.g., ``{'lr': 1e-3}``
            for specifying a learning rate). Otherwise the default values of the selected ``optimizer_cls``
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
            defaults to the following string ``"YYYY-mm-dd_HH:MM:SS_torch_model_run_PID"``, where the initial part
            of the name is formatted with the local date and time, while PID is the processed ID (preventing models
            spawned at the same time by different processes to share the same model_name). E.g.,
            ``"2021-06-14_09:53:32_torch_model_run_44607"``.
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
            Whether or not to automatically save the untrained model and checkpoints from training.
            To load the model from checkpoint, call :func:`MyModelClass.load_from_checkpoint()`, where
            :class:`MyModelClass` is the :class:`TorchForecastingModel` class that was used (such as :class:`TFTModel`,
            :class:`NBEATSModel`, etc.). If set to ``False``, the model can still be manually saved using
            :func:`save_model()` and loaded using :func:`load_model()`. Default: ``False``.
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

                add_encoders={
                    'cyclic': {'future': ['month']},
                    'datetime_attribute': {'future': ['hour', 'dayofweek']},
                    'position': {'past': ['relative'], 'future': ['relative']},
                    'custom': {'past': [lambda idx: (idx.year - 1950) / 50]},
                    'transformer': Scaler()
                }
            ..
        random_state
            Control the randomness of the weights initialization. Check this
            `link <https://scikit-learn.org/stable/glossary.html#term-random_state>`_ for more details.
            Default: ``None``.
        pl_trainer_kwargs
            By default :class:`TorchForecastingModel` creates a PyTorch Lightning Trainer with several useful presets
            that performs the training, validation and prediction processes. These presets include automatic
            checkpointing, tensorboard logging, setting the torch device and more.
            With ``pl_trainer_kwargs`` you can add additional kwargs to instantiate the PyTorch Lightning trainer
            object. Check the `PL Trainer documentation
            <https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html>`_ for more information about the
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
            <https://pytorch-lightning.readthedocs.io/en/stable/extensions/callbacks.html>`_

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
        .. [1] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser,
        and Illia Polosukhin, "Attention Is All You Need", 2017. In Advances in Neural Information Processing Systems,
        pages 6000-6010. https://arxiv.org/abs/1706.03762.
        ..[2] Shazeer, Noam, "GLU Variants Improve Transformer", 2020. arVix https://arxiv.org/abs/2002.05202.

        Notes
        -----
        Disclaimer:
        This current implementation is fully functional and can already produce some good predictions. However,
        it is still limited in how it uses the Transformer architecture because the `tgt` input of
        `torch.nn.Transformer` is not utlized to its full extent. Currently, we simply pass the last value of the
        `src` input to `tgt`. To get closer to the way the Transformer is usually used in language models, we
        should allow the model to consume its own output as part of the `tgt` argument, such that when predicting
        sequences of values, the input to the `tgt` argument would grow as outputs of the transformer model would be
        added to it. Of course, the training of the model would have to be adapted accordingly.
        """
        super().__init__(**self._extract_torch_model_params(**self.model_params))

        # extract pytorch lightning module kwargs
        self.pl_module_params = self._extract_pl_module_params(**self.model_params)

        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation
        self.norm_type = norm_type
        self.custom_encoder = custom_encoder
        self.custom_decoder = custom_decoder

    def _create_model(self, train_sample: Tuple[torch.Tensor]) -> torch.nn.Module:
        # samples are made of (past_target, past_covariates, future_target)
        input_dim = train_sample[0].shape[1] + (
            train_sample[1].shape[1] if train_sample[1] is not None else 0
        )
        output_dim = train_sample[-1].shape[1]
        nr_params = 1 if self.likelihood is None else self.likelihood.num_parameters

        return _TransformerModule(
            input_size=input_dim,
            output_size=output_dim,
            nr_params=nr_params,
            d_model=self.d_model,
            nhead=self.nhead,
            num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            activation=self.activation,
            norm_type=self.norm_type,
            custom_encoder=self.custom_encoder,
            custom_decoder=self.custom_decoder,
            **self.pl_module_params,
        )

    @staticmethod
    def _supports_static_covariates() -> bool:
        return False
