"""
Transformer
-----------
"""

from numpy.random import RandomState
import math
import torch
import torch.nn as nn
from typing import Optional, Union

from ..utils.torch import random_method
from ..logging import raise_if_not, get_logger
from .torch_forecasting_model import TorchForecastingModel

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
                 input_size: int,
                 input_length: int,
                 output_length: int,
                 output_size: int,
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
        input_length
            Number of time steps to be input to the forecasting module.
        output_length
            Number of time steps to be output by the forecasting module.
        output_size
            The dimensionality of the output time series.
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
        x of shape `(batch_size, input_length, input_size)`
            Tensor containing the features of the input sequence.

        Outputs
        -------
        y of shape `(batch_size, output_length, output_size)`
            Tensor containing the (point) prediction at the last time step of the sequence.
        """

        super(_TransformerModule, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.output_length = output_length

        self.encoder = nn.Linear(input_size, d_model)
        self.positional_encoding = _PositionalEncoding(d_model, dropout, input_length)

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

        self.decoder = nn.Linear(d_model, output_length * output_size)

    def _create_transformer_inputs(self, data):
        # '_TimeSeriesSequentialDataset' stores time series in the
        # (batch_size, input_length, input_size) format. PyTorch's nn.Transformer
        # module needs it the (input_length, batch_size, input_size) format.
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
        # from (1, batch_size, output_length * output_size)
        # to (batch_size, output_length, output_size)
        predictions = out[0, :, :]
        predictions = predictions.view(-1, self.output_length, self.output_size)

        return predictions


class TransformerModel(TorchForecastingModel):
    @random_method
    def __init__(self,
                 model: Optional[nn.Module] = None,
                 input_size: int = 1,
                 input_length: int = 1,
                 output_length: int = 1,
                 output_size: int = 1,
                 d_model: int = 512,
                 nhead: int = 8,
                 num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 activation: str = "relu",
                 custom_encoder: Optional[nn.Module] = None,
                 custom_decoder: Optional[nn.Module] = None,
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
        input_size
            The dimensionality of the TimeSeries that will be fed to the fit and predict functions (default=1).
        input_length
            Number of time steps to be input to the forecasting module (default=1).
        output_size
            The dimensionality of the output time series (default=1).
        output_length
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
        random_state
            Controls the randomness of the weights initialization. Check this
            `link <https://scikit-learn.org/stable/glossary.html#term-random-state>`_ for more details.
        """

        kwargs['output_length'] = output_length
        kwargs['input_size'] = input_size
        kwargs['output_size'] = output_size

        # set self.model
        if model is None:
            self.model = _TransformerModule(input_size=input_size,
                                            input_length=input_length,
                                            output_length=output_length,
                                            output_size=output_size,
                                            d_model=d_model,
                                            nhead=nhead,
                                            num_encoder_layers=num_encoder_layers,
                                            num_decoder_layers=num_decoder_layers,
                                            dim_feedforward=dim_feedforward,
                                            dropout=dropout,
                                            activation=activation,
                                            custom_encoder=custom_encoder,
                                            custom_decoder=custom_decoder)
        else:
            self.model = model
            raise_if_not(isinstance(self.model, nn.Module),
                         '{} is not a valid Transformer model instance.'
                         '\n Please set "model" to "None" or give your own PyTorch nn.Module'.format(
                             model.__class__.__name__),
                         logger)

        super().__init__(**kwargs)
