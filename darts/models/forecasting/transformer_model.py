"""
Transformer Model
-----------------
"""

from numpy.random import RandomState
import math
import torch
import torch.nn as nn
from typing import Optional, Union, Tuple

from darts.utils.likelihood_models import Likelihood
from darts.utils.torch import random_method
from darts.logging import get_logger
from darts.models.forecasting.torch_forecasting_model import (TorchParametricProbabilisticForecastingModel,
                                                              PastCovariatesTorchModel)

logger = get_logger(__name__)


# This implementation of positional encoding is taken from the PyTorch documentation:
# https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class _PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        """ An implementation of positional encoding as described in 'Attention is All you Need' by Vaswani et al. (2017)

        Parameters
        ----------
        d_model
            the number of expected features in the transformer encoder/decoder inputs.
            Last dimension of the input
        dropout
            Fraction of neurons affected by Dropout (default=0.1).
        max_len
            The dimensionality of the computed positional encoding array.
            Only its first "input_size" elements will be considered in the output

        Inputs
        ------
        x of shape `(batch_size, input_size, d_model)`
            Tensor containing the embedded time series

        Outputs
        -------
        y of shape `(batch_size, input_size, d_model)`
            Tensor containing the embedded time series enhanced with positional encoding
        """
        super(_PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class _TransformerModule(nn.Module):
    def __init__(self,
                 input_chunk_length: int,
                 output_chunk_length: int,
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
                 custom_encoder: Optional[nn.Module] = None,
                 custom_decoder: Optional[nn.Module] = None,
                 ):
        """ PyTorch module implementing a Transformer to be used in `TransformerModel`.

        PyTorch module implementing a simple encoder-decoder transformer architecture.

        Parameters
        ----------
        input_size
            The dimensionality of the TimeSeries instances that will be fed to the the fit and predict functions.
        input_chunk_length
            Number of time steps to be input to the forecasting module.
        output_chunk_length
            Number of time steps to be output by the forecasting module.
        output_size
            The dimensionality of the output time series.
        nr_params
            The number of parameters of the likelihood (or 1 if no likelihood is used).
        d_model
            the number of expected features in the transformer encoder/decoder inputs.
        nhead
            the number of heads in the multiheadattention model.
        num_encoder_layers
            the number of encoder layers in the encoder.
        num_decoder_layers
            the number of decoder layers in the decoder.
        dim_feedforward
            the dimension of the feedforward network model.
        dropout
            Fraction of neurons affected by Dropout.
        activation
            the activation function of encoder/decoder intermediate layer, 'relu' or 'gelu'.
        custom_encoder
            a custom transformer encoder provided by the user (default=None)
        custom_decoder
            a custom transformer decoder provided by the user (default=None)

        Inputs
        ------
        x of shape `(batch_size, input_chunk_length, input_size)`
            Tensor containing the features of the input sequence.

        Outputs
        -------
        y of shape `(batch_size, output_chunk_length, target_size, nr_params)`
            Tensor containing the prediction at the last time step of the sequence.
        """

        super(_TransformerModule, self).__init__()

        self.input_size = input_size
        self.target_size = output_size
        self.nr_params = nr_params
        self.target_length = output_chunk_length

        self.encoder = nn.Linear(input_size, d_model)
        self.positional_encoding = _PositionalEncoding(d_model, dropout, input_chunk_length)

        # Defining the Transformer module
        self.transformer = nn.Transformer(d_model=d_model,
                                          nhead=nhead,
                                          num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers,
                                          dim_feedforward=dim_feedforward,
                                          dropout=dropout,
                                          activation=activation,
                                          custom_encoder=custom_encoder,
                                          custom_decoder=custom_decoder)

        self.decoder = nn.Linear(d_model, output_chunk_length * self.target_size * self.nr_params)

    def _create_transformer_inputs(self, data):
        # '_TimeSeriesSequentialDataset' stores time series in the
        # (batch_size, input_chunk_length, input_size) format. PyTorch's nn.Transformer
        # module needs it the (input_chunk_length, batch_size, input_size) format.
        # Therefore, the first two dimensions need to be swapped.
        src = data.permute(1, 0, 2)
        tgt = src[-1:, :, :]

        return src, tgt

    def forward(self, data):
        # Here we create 'src' and 'tgt', the inputs for the encoder and decoder
        # side of the Transformer architecture
        src, tgt = self._create_transformer_inputs(data)

        # "math.sqrt(self.input_size)" is a normalization factor
        # see section 3.2.1 in 'Attention is All you Need' by Vaswani et al. (2017)
        src = self.encoder(src) * math.sqrt(self.input_size)
        src = self.positional_encoding(src)

        tgt = self.encoder(tgt) * math.sqrt(self.input_size)
        tgt = self.positional_encoding(tgt)

        x = self.transformer(src=src,
                             tgt=tgt)
        out = self.decoder(x)

        # Here we change the data format
        # from (1, batch_size, output_chunk_length * output_size)
        # to (batch_size, output_chunk_length, output_size, nr_params)
        predictions = out[0, :, :]
        predictions = predictions.view(-1, self.target_length, self.target_size, self.nr_params)

        return predictions


class TransformerModel(TorchParametricProbabilisticForecastingModel, PastCovariatesTorchModel):
    @random_method
    def __init__(self,
                 input_chunk_length: int,
                 output_chunk_length: int,
                 d_model: int = 64,
                 nhead: int = 4,
                 num_encoder_layers: int = 3,
                 num_decoder_layers: int = 3,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1,
                 activation: str = "relu",
                 custom_encoder: Optional[nn.Module] = None,
                 custom_decoder: Optional[nn.Module] = None,
                 likelihood: Optional[Likelihood] = None,
                 random_state: Optional[Union[int, RandomState]] = None,
                 **kwargs):

        """
        Transformer is a state-of-the-art deep learning model introduced in 2017. It is an encoder-decoder
        architecture whose core feature is the 'multi-head attention' mechanism, which is able to
        draw intra-dependencies within the input vector and within the output vector ('self-attention')
        as well as inter-dependencies between input and output vectors ('encoder-decoder attention').
        The multi-head attention mechanism is highly parallelizable, which makes the transformer architecture
        very suitable to be trained with GPUs.


        The transformer architecture implemented here is based on the paper “Attention Is All You Need”:
        Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and
        Illia Polosukhin. 2017. Attention is all you need. In Advances in Neural Information Processing Systems,
        pages 6000-6010.
        (paper can be found at https://arxiv.org/abs/1706.03762)

        This model supports past covariates (known for `input_chunk_length` points before prediction time).

        Disclaimer:
        This current implementation is fully functional and can already produce some good predictions. However,
        it is still limited in how it uses the Transformer architecture because the `tgt` input of
        `torch.nn.Transformer` is not utlized to its full extent. Currently, we simply pass the last value of the
        `src` input to `tgt`. To get closer to the way the Transformer is usually used in language models, we
        should allow the model to consume its own output as part of the `tgt` argument, such that when predicting
        sequences of values, the input to the `tgt` argument would grow as outputs of the transformer model would be
        added to it. Of course, the training of the model would have to be adapted accordingly.

        Parameters
        ----------
        model
            a custom PyTorch module with the same specifications as
            `darts.models.transformer_model._TransformerModule` (default=None).
        input_chunk_length
            Number of time steps to be input to the forecasting module (default=1).
        output_chunk_length
            Number of time steps to be output by the forecasting module (default=1).
        d_model
            the number of expected features in the transformer encoder/decoder inputs (default=512).
        nhead
            the number of heads in the multiheadattention model (default=8).
        num_encoder_layers
            the number of encoder layers in the encoder (default=6).
        num_decoder_layers
            the number of decoder layers in the decoder (default=6).
        dim_feedforward
            the dimension of the feedforward network model (default=2048).
        dropout
            Fraction of neurons affected by Dropout (default=0.1).
        activation
            the activation function of encoder/decoder intermediate layer, 'relu' or 'gelu' (default='relu').
        custom_encoder
            a custom user-provided encoder module for the transformer (default=None)
        custom_decoder
            a custom user-provided decoder module for the transformer (default=None)
        likelihood
            Optionally, the likelihood model to be used for probabilistic forecasts.
            If no likelihood model is provided, forecasts will be deterministic.
        random_state
            Controls the randomness of the weights initialization. Check this
            `link <https://scikit-learn.org/stable/glossary.html#term-random-state>`_ for more details.

        batch_size
            Number of time series (input and output sequences) used in each training pass.
        n_epochs
            Number of epochs over which to train the model.
        optimizer_cls
            The PyTorch optimizer class to be used (default: `torch.optim.Adam`).
        optimizer_kwargs
            Optionally, some keyword arguments for the PyTorch optimizer (e.g., `{'lr': 1e-3}`
            for specifying a learning rate). Otherwise the default values of the selected `optimizer_cls`
            will be used.
        lr_scheduler_cls
            Optionally, the PyTorch learning rate scheduler class to be used. Specifying `None` corresponds
            to using a constant learning rate.
        lr_scheduler_kwargs
            Optionally, some keyword arguments for the PyTorch optimizer.
        loss_fn
            PyTorch loss function used for training.
            This parameter will be ignored for probabilistic models if the `likelihood` parameter is specified.
            Default: `torch.nn.MSELoss()`.
        model_name
            Name of the model. Used for creating checkpoints and saving tensorboard data. If not specified,
            defaults to the following string "YYYY-mm-dd_HH:MM:SS_torch_model_run_PID", where the initial part of the
            name is formatted with the local date and time, while PID is the processed ID (preventing models spawned at
            the same time by different processes to share the same model_name). E.g.,
            2021-06-14_09:53:32_torch_model_run_44607.
        work_dir
            Path of the working directory, where to save checkpoints and Tensorboard summaries.
            (default: current working directory).
        log_tensorboard
            If set, use Tensorboard to log the different parameters. The logs will be located in:
            `[work_dir]/.darts/runs/`.
        nr_epochs_val_period
            Number of epochs to wait before evaluating the validation loss (if a validation
            `TimeSeries` is passed to the `fit()` method).
        torch_device_str
            Optionally, a string indicating the torch device to use. (default: "cuda:0" if a GPU
            is available, otherwise "cpu")
        force_reset
            If set to `True`, any previously-existing model with the same name will be reset (all checkpoints will
            be discarded).
        save_checkpoints
            Whether or not to automatically save the untrained model and checkpoints from training.
            If set to `False`, the model can still be manually saved using :meth:`save_model()
            <TorchForeCastingModel.save_model()>` and loaded using :meth:`load_model()
            <TorchForeCastingModel.load_model()>`.
        """

        kwargs['input_chunk_length'] = input_chunk_length
        kwargs['output_chunk_length'] = output_chunk_length
        super().__init__(likelihood=likelihood, **kwargs)

        self.input_chunk_length = input_chunk_length
        self.output_chunk_length = output_chunk_length
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation
        self.custom_encoder = custom_encoder
        self.custom_decoder = custom_decoder

    def _create_model(self, train_sample: Tuple[torch.Tensor]) -> torch.nn.Module:
        # samples are made of (past_target, past_covariates, future_target)
        input_dim = train_sample[0].shape[1] + (train_sample[1].shape[1] if train_sample[1] is not None else 0)
        output_dim = train_sample[-1].shape[1]
        nr_params = 1 if self.likelihood is None else self.likelihood.num_parameters

        return _TransformerModule(input_chunk_length=self.input_chunk_length,
                                  output_chunk_length=self.output_chunk_length,
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
                                  custom_encoder=self.custom_encoder,
                                  custom_decoder=self.custom_decoder)

    @random_method
    def _produce_predict_output(self, x):
        if self.likelihood:
            output = self.model(x)
            return self.likelihood.sample(output)
        else:
            return self.model(x).squeeze(dim=-1)

