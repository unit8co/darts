"""
Query Selector
------
https://github.com/moraieu/query-selector
"""

import math
from functools import partial

import torch
from torch import nn

from darts.logging import raise_if_not
from darts.models.components import glu_variants
from darts.models.components.glu_variants import GLU_FFN

# from darts.models.components.power_norm import MaskPowerNorm
# from darts.models.components import layer_norm_variants
from darts.models.forecasting.pl_forecasting_module import PLPastCovariatesModule
from darts.models.forecasting.torch_forecasting_model import PastCovariatesTorchModel
from darts.utils.torch import MonteCarloDropout


def a_norm(Q, K):
    m = torch.matmul(Q, K.transpose(2, 1))
    m /= torch.sqrt(torch.tensor(Q.shape[-1]).float())
    return torch.softmax(m, -1)


def attention(Q, K, V):
    a = a_norm(Q, K)  # (batch_size, dim_attn, seq_length)
    return torch.matmul(a, V)  # (batch_size, seq_length, seq_length)


class QuerySelector(nn.Module):
    def __init__(self, fraction=0.33):
        super().__init__()
        self.fraction = fraction

    def forward(self, queries, keys, values):
        B, L_Q, D = queries.shape
        _, L_K, _ = keys.shape
        l_Q = int((1.0 - self.fraction) * L_Q)
        K_reduce = torch.mean(keys.topk(l_Q, dim=1).values, dim=1).unsqueeze(1)
        sqk = torch.matmul(K_reduce, queries.transpose(1, 2))
        indices = sqk.topk(l_Q, dim=-1).indices.squeeze(1)
        Q_sample = queries[torch.arange(B)[:, None], indices, :]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_sample, keys.transpose(-2, -1))
        attn = torch.softmax(Q_K / math.sqrt(D), dim=-1)
        mean_values = values.mean(dim=-2)
        result = mean_values.unsqueeze(-2).expand(B, L_Q, mean_values.shape[-1]).clone()
        result[torch.arange(B)[:, None], indices, :] = torch.matmul(
            attn, values
        ).type_as(result)
        return result, None


class InferenceModuleList(torch.nn.ModuleList):
    def inference(self):
        for mod in self.modules():
            if mod != self:
                mod.inference()


class AttentionBlock(nn.Module):
    def __init__(self, dim_val, dim_attn, debug=False, attn_type="full"):
        super().__init__()
        self.value = Value(dim_val, dim_val)
        self.key = Key(dim_val, dim_attn)
        self.query = Query(dim_val, dim_attn)
        self.debug = debug
        self.qk_record = None
        self.qkv_record = None
        self.n = 0
        if attn_type == "full":
            self.attentionLayer = None
        elif attn_type.startswith("query_selector"):
            args = {}
            if len(attn_type.split("_")) == 1 or len(attn_type.split("_")) == 3:
                args["fraction"] = float(attn_type.split("_")[-1])
            self.attentionLayer = QuerySelector(**args)
        else:
            raise Exception

    def forward(self, x, kv=None):
        if kv is None:
            if self.attentionLayer:
                qkv = self.attentionLayer(self.query(x), self.key(x), self.value(x))[0]
            else:
                qkv = attention(self.query(x), self.key(x), self.value(x))
            return qkv
        return attention(self.query(x), self.key(kv), self.value(kv))


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, dim_val, dim_attn, n_heads, attn_type):
        super().__init__()
        self.heads = []
        for i in range(n_heads):
            self.heads.append(AttentionBlock(dim_val, dim_attn, attn_type=attn_type))

        self.heads = InferenceModuleList(self.heads)
        self.fc = nn.Linear(n_heads * dim_val, dim_val, bias=False)

    def forward(self, x, kv=None):
        a = []
        for h in self.heads:
            a.append(h(x, kv=kv))

        a = torch.stack(a, dim=-1)  # combine heads
        a = a.flatten(start_dim=2)  # flatten all head outputs

        x = self.fc(a)

        return x

    def record(self):
        for h in self.heads:
            h.record()


class Value(nn.Module):
    def __init__(self, dim_input, dim_val):
        super().__init__()
        self.dim_val = dim_val
        self.fc1 = nn.Linear(dim_input, dim_val, bias=False)

    def forward(self, x):
        return self.fc1(x)


class Key(nn.Module):
    def __init__(self, dim_input, dim_attn):
        super().__init__()
        self.dim_attn = dim_attn
        self.fc1 = nn.Linear(dim_input, dim_attn, bias=False)

    def forward(self, x):
        return self.fc1(x)


class Query(nn.Module):
    def __init__(self, dim_input, dim_attn):
        super().__init__()
        self.dim_attn = dim_attn
        self.fc1 = nn.Linear(dim_input, dim_attn, bias=False)

    def forward(self, x):
        return self.fc1(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = MonteCarloDropout(p=dropout)

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


def FeedForward(d_model, d_ff, dropout=0.1):
    return nn.Sequential(
        nn.Linear(d_model, d_ff),
        nn.Sigmoid(),
        MonteCarloDropout(p=dropout),
        nn.Linear(d_ff, d_model),
        MonteCarloDropout(p=dropout),
    )


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, **kwargs):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, **kwargs)
        self.post_norm = nn.LayerNorm(d_model)

        init_function = partial(init_weights)
        self.ff.apply(init_function)

    def forward(self, x):
        x = self.norm(x)
        return self.post_norm(x + self.ff(x))


def init_weights(module):
    if type(module) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(module.weight)


class EncoderLayer(nn.Module):
    def __init__(
        self,
        dim_val,
        dim_attn,
        feed_forward,
        n_heads=1,
        attn_type="full",
        norm=nn.LayerNorm,
    ):
        super().__init__()
        self.attn = MultiHeadAttentionBlock(
            dim_val, dim_attn, n_heads, attn_type=attn_type
        )

        self.norm1 = norm(dim_val)
        self.norm2 = norm(dim_val)

        self.ffn = feed_forward

    def forward(self, x):
        a = self.attn(x)
        x = self.norm1(x + a)

        a = self.ffn(x)
        x = self.norm2(x + a)

        return x

    def record(self):
        self.attn.record()


class DecoderLayer(nn.Module):
    def __init__(
        self,
        dim_val,
        dim_attn,
        feed_forward,
        n_heads=1,
        attn_type="full",
        norm=nn.LayerNorm,
    ):
        super().__init__()
        self.attn1 = MultiHeadAttentionBlock(
            dim_val, dim_attn, n_heads, attn_type=attn_type
        )
        self.attn2 = MultiHeadAttentionBlock(
            dim_val, dim_attn, n_heads, attn_type=attn_type
        )

        self.norm1 = norm(dim_val)
        self.norm2 = norm(dim_val)
        self.norm3 = norm(dim_val)

        self.ffn = feed_forward

    def forward(self, x, enc):
        a = self.attn1(x)
        x = self.norm1(a + x)

        a = self.attn2(x, kv=enc)
        x = self.norm2(a + x)

        a = self.ffn(x)
        x = self.norm3(x + a)

        return x

    def record(self):
        self.attn1.record()
        self.attn2.record()


class _TransformerModule(PLPastCovariatesModule):
    def __init__(
        self,
        input_dim,
        output_dim,
        nr_params,
        dim_val,
        dim_attn,
        dec_seq_len,
        n_decoder_layers,
        n_encoder_layers,
        enc_attn_type,
        dec_attn_type,
        n_heads,
        dropout,
        feed_forward,
        layer_norm,
        debug=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.nr_params = nr_params
        self.dec_seq_len = dec_seq_len

        self.feed_forward = feed_forward
        if layer_norm == "LayerNorm":
            self.layer_norm = nn.LayerNorm
        # elif layer_norm == "PowerNorm":
        #     self.layer_norm = MaskPowerNorm
        # else:
        #     self.layer_norm = getattr(layer_norm_variants, layer_norm)

        if self.feed_forward == "default":
            self.feed_forward_network_enc = FeedForward
            self.feed_forward_network_dec = FeedForward
        else:
            raise_if_not(
                self.feed_forward in GLU_FFN,
                f"'{self.feed_forward}' is not in {GLU_FFN + ['default']}",
            )
            # use glu variant feedforward layers
            # 4 is a commonly used feedforward multiplier
            self.feed_forward_network_enc = getattr(glu_variants, self.feed_forward)
            self.feed_forward_network_dec = getattr(glu_variants, self.feed_forward)

        self.feed_forward_network_enc = self.feed_forward_network_enc(
            dim_val, dim_val * 4, dropout=0.1
        )
        self.feed_forward_network_dec = self.feed_forward_network_dec(
            dim_val, dim_val * 4, dropout=0.1
        )

        # Initiate encoder and Decoder layers
        self.encs = []
        for i in range(n_encoder_layers):
            self.encs.append(
                EncoderLayer(
                    dim_val,
                    dim_attn,
                    feed_forward=self.feed_forward_network_enc,
                    n_heads=n_heads,
                    attn_type=enc_attn_type,
                    norm=self.layer_norm,
                )
            )
        self.encs = InferenceModuleList(self.encs)
        self.decs = []
        for i in range(n_decoder_layers):
            self.decs.append(
                DecoderLayer(
                    dim_val,
                    dim_attn,
                    feed_forward=self.feed_forward_network_dec,
                    n_heads=n_heads,
                    attn_type=dec_attn_type,
                    norm=self.layer_norm,
                )
            )
        self.decs = InferenceModuleList(self.decs)
        self.pos = PositionalEncoding(dim_val)

        self.enc_dropout = MonteCarloDropout(p=dropout)

        # Dense layers for managing network inputs and outputs
        self.enc_input_fc = nn.Linear(self.input_dim, dim_val)
        self.dec_input_fc = nn.Linear(self.input_dim, dim_val)
        self.out_fc = nn.Linear(
            dec_seq_len * dim_val,
            self.output_chunk_length * self.output_dim * self.nr_params,
        )

        self.debug = debug

    def forward(self, x_in: tuple):
        x, _ = x_in
        # x: [Batch, Input length, Channel]

        # encoder
        e = self.encs[0](self.pos(self.enc_dropout(self.enc_input_fc(x))))

        for enc in self.encs[1:]:
            e = enc(e)
        if self.debug:
            print(f"Encoder output size: {e.shape}")
        # decoder
        decoded = self.dec_input_fc(x[:, -self.dec_seq_len :])

        d = self.decs[0](decoded, e)
        for dec in self.decs[1:]:
            d = dec(d, e)

        # output
        x = self.out_fc(d.flatten(start_dim=1))

        x = torch.reshape(x, (x.shape[0], -1, self.output_dim))
        x = x.view(
            x.shape[0], self.output_chunk_length, self.output_dim, self.nr_params
        )
        return x

    def record(self):
        self.debug = True
        for enc in self.encs:
            enc.record()
        for dec in self.decs:
            dec.record()


class QuerySelectorModel(PastCovariatesTorchModel):
    def __init__(
        self,
        input_chunk_length: int,
        output_chunk_length: int,
        dim_val: int = 32,
        dim_attn: int = 128,
        dec_seq_len: int = 12,
        n_encoder_layers: int = 3,
        n_decoder_layers: int = 3,
        enc_attn_type="query_selector_0.8",
        dec_attn_type="full",
        n_heads: int = 4,
        dropout: float = 0.1,
        feed_forward: str = "default",
        norm_type: str = "LayerNorm",
        **kwargs,
    ):
        """An implementation of the QuerySelector model, as presented in [1]_.



        Parameters
        ----------
        input_chunk_length : int
            The length of the input sequence fed to the model.
        output_chunk_length : int
            The length of the forecast of the model.
        dim_val : int
            hidden dimension size of the value in the attention layer (default=32).
        dim_attn : int
            hidden dimension size of the attention layer (default=128).
        dec_seq_len : int
            the embedding size for the decoder (default=12).
        n_encoder_layers : int
            The number of encoder layers in the encoder (default=3).
        n_decoder_layers : int
            The number of decoder layers in the encoder (default=3).
        enc_attn_type : str
            "full" attention or query selector with a reduction f factor. Example "query_selector_0.4".
            Default to "query_selector_0.8"
        dec_attn_type : str
            "full" attention or query selector with a reduction f factor. Example "query_selector_0.4".
            Default to "full".
        n_heads : int
            number of attention heads (4 is a good default).
        dropout : float
            Fraction of neurons affected by Dropout (default=0.1).
        feed_forward : str
            Set the feedforward network block. default or one of the  glu variant.
            Defaults to `default`.
        norm_type: str
            The type of LayerNorm variant to use.  Default: ``LayerNorm``. Options available are
            ["LayerNorm", "ScaleNorm", "RMSNorm", "PowerNorm"]



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
        torch_device_str
            Optionally, a string indicating the torch device to use. By default, ``torch_device_str`` is ``None``
            which will run on CPU. Set it to ``"cuda"`` to use all available GPUs or ``"cuda:i"`` to only use
            GPU ``i`` (``i`` must be an integer). For example "cuda:0" will use the first GPU only.

            .. deprecated:: v0.17.0
                ``torch_device_str`` has been deprecated in v0.17.0 and will be removed in a future version.
                Instead, specify this with keys ``"accelerator", "gpus", "auto_select_gpus"`` in your
                ``pl_trainer_kwargs`` dict. Some examples for setting the devices inside the ``pl_trainer_kwargs``
                dict:

                - ``{"accelerator": "cpu"}`` for CPU,
                - ``{"accelerator": "gpu", "gpus": [i]}`` to use only GPU ``i`` (``i`` must be an integer),
                - ``{"accelerator": "gpu", "gpus": -1, "auto_select_gpus": True}`` to use all available GPUS.

                For more info, see here:
                https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#trainer-flags , and
                https://pytorch-lightning.readthedocs.io/en/stable/advanced/multi_gpu.html#select-gpu-devices
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
            Read :meth:`SequentialEncoder <darts.utils.data.encoders.SequentialEncoder>` to find out more about
            ``add_encoders``. Default: ``None``. An example showing some of ``add_encoders`` features:

            .. highlight:: python
            .. code-block:: python

                add_encoders={
                    'cyclic': {'future': ['month']},
                    'datetime_attribute': {'future': ['hour', 'dayofweek']},
                    'position': {'past': ['absolute'], 'future': ['relative']},
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
        .. [1] Klimek, Jacek, et al. "Long-term series forecasting with Query Selector--efficient model of
               sparse attention." arXiv preprint arXiv:2107.08687 (2021). https://arxiv.org/abs/2107.08687
        """
        super().__init__(**self._extract_torch_model_params(**self.model_params))

        # extract pytorch lightning module kwargs
        self.pl_module_params = self._extract_pl_module_params(**self.model_params)

        self.dim_val = dim_val
        self.dim_attn = dim_attn
        self.dec_seq_len = dec_seq_len
        self.n_decoder_layers = n_decoder_layers
        self.n_encoder_layers = n_encoder_layers
        self.enc_attn_type = enc_attn_type
        self.dec_attn_type = dec_attn_type
        self.n_heads = n_heads
        self.dropout = dropout
        self.feed_forward = feed_forward
        self.norm_type = norm_type

    def _create_model(self, train_sample: tuple[torch.Tensor]) -> torch.nn.Module:
        # samples are made of (past_target, past_covariates, future_target)
        input_dim = train_sample[0].shape[1] + (
            train_sample[1].shape[1] if train_sample[1] is not None else 0
        )
        output_dim = train_sample[-1].shape[1]
        nr_params = 1 if self.likelihood is None else self.likelihood.num_parameters

        return _TransformerModule(
            input_dim=input_dim,
            output_dim=output_dim,
            nr_params=nr_params,
            dim_val=self.dim_val,
            dim_attn=self.dim_attn,
            dec_seq_len=self.dec_seq_len,
            n_decoder_layers=self.n_decoder_layers,
            n_encoder_layers=self.n_encoder_layers,
            enc_attn_type=self.enc_attn_type,
            dec_attn_type=self.dec_attn_type,
            n_heads=self.n_heads,
            dropout=self.dropout,
            feed_forward=self.feed_forward,
            layer_norm=self.norm_type,
            **self.pl_module_params,
        )

    @staticmethod
    def _supports_static_covariates() -> bool:
        return False
