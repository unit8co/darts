"""
Block Recurrent Neural Networks
-------------------------------
"""

import torch.nn as nn
import torch
from numpy.random import RandomState
from typing import List, Optional, Union

from ..logging import raise_if_not, get_logger
from ..utils.torch import random_method
from .torch_forecasting_model import TorchForecastingModel

logger = get_logger(__name__)


# TODO add batch norm
class _BlockRNNModule(nn.Module):
    def __init__(self,
                 name: str,
                 input_size: int,
                 hidden_dim: int,
                 num_layers: int,
                 output_chunk_length: int = 1,
                 target_size: int = 1,
                 num_layers_out_fc: Optional[List] = None,
                 dropout: float = 0.):

        """ PyTorch module implementing a block RNN to be used in `BlockRNNModel`.

        PyTorch module implementing a simple block RNN with the specified `name` layer.
        This module combines a PyTorch RNN module, together with a fully connected network, which maps the
        last hidden layers to output of the desired size `output_chunk_length` and makes it compatible with
        `BlockRNNModel`s.

        This module uses an RNN to encode the input sequence, and subsequently uses a fully connected
        network as the decoder which takes as input the last hidden state of the encoder RNN.
        The final output of the decoder is a sequence of length `output_chunk_length`. In this sense,
        the `_BlockRNNModule` produces 'blocks' of forecasts at a time (which is different 
        from `_RNNModule` used by the `RNNModel`).

        Parameters
        ----------
        name
            The name of the specific PyTorch RNN module ("RNN", "GRU" or "LSTM").
        input_size
            The dimensionality of the input time series.
        hidden_dim
            The number of features in the hidden state `h` of the RNN module.
        num_layers
            The number of recurrent layers.
        output_chunk_length
            The number of steps to predict in the future.
        target_size
            The dimensionality of the output time series.
        num_layers_out_fc
            A list containing the dimensions of the hidden layers of the fully connected NN.
            This network connects the last hidden layer of the PyTorch RNN module to the output.
        dropout
            The fraction of neurons that are dropped in all-but-last RNN layers.

        Inputs
        ------
        x of shape `(batch_size, input_chunk_length, input_size)`
            Tensor containing the features of the input sequence.

        Outputs
        -------
        y of shape `(batch_size, output_chunk_length, target_size)`
            Tensor containing the (point) prediction at the last time step of the sequence.
        """

        super(_BlockRNNModule, self).__init__()

        # Defining parameters
        self.hidden_dim = hidden_dim
        self.n_layers = num_layers
        self.target_size = target_size
        num_layers_out_fc = [] if num_layers_out_fc is None else num_layers_out_fc
        self.out_len = output_chunk_length
        self.name = name

        # Defining the RNN module
        self.rnn = getattr(nn, name)(input_size, hidden_dim, num_layers, batch_first=True, dropout=dropout)

        # The RNN module is followed by a fully connected layer, which maps the last hidden layer
        # to the output of desired length
        last = hidden_dim
        feats = []
        for feature in num_layers_out_fc + [output_chunk_length * target_size]:
            feats.append(nn.Linear(last, feature))
            last = feature
        self.fc = nn.Sequential(*feats)

    def forward(self, x):
        # data is of size (batch_size, input_chunk_length, input_size)
        batch_size = x.size(0)

        out, hidden = self.rnn(x)

        """ Here, we apply the FC network only on the last output point (at the last time step)
        """
        if self.name == "LSTM":
            hidden = hidden[0]
        predictions = hidden[-1, :, :]
        predictions = self.fc(predictions)
        predictions = predictions.view(batch_size, self.out_len, self.target_size)

        # predictions is of size (batch_size, output_chunk_length, 1)
        return predictions


class BlockRNNModel(TorchForecastingModel):
    @random_method
    def __init__(self,
                 input_chunk_length: int,
                 output_chunk_length: int,
                 model: Union[str, nn.Module] = 'RNN',
                 hidden_size: int = 25,
                 n_rnn_layers: int = 1,
                 hidden_fc_sizes: Optional[List] = None,
                 dropout: float = 0.,
                 random_state: Optional[Union[int, RandomState]] = None,
                 **kwargs):

        """ Block Recurrent Neural Network Model (RNNs).

        This class provides three variants of RNNs:

        * Vanilla RNN

        * LSTM

        * GRU

        This is the 'block' implementation of the RNN model, which produces forecasts by using
        an RNN as an encoder, and a fully connected network as a decoder which produces 'blocks'
        of forecasts as a function of the last hidden state of the encoder.

        Parameters
        ----------
        model
            Either a string specifying the RNN module type ("RNN", "LSTM" or "GRU"),
            or a PyTorch module with the same specifications as
            `darts.models.block_rnn_model._BlockRNNModule`.
        input_chunk_length
            The number of time steps that will be fed to the internal forecasting module
        output_chunk_length
            Number of time steps to be output by the internal forecasting module.
        hidden_size
            Size for feature maps for each hidden RNN layer (:math:`h_n`).
        n_rnn_layers
            Number of layers in the RNN module.
        hidden_fc_sizes
            Sizes of hidden layers connecting the last hidden layer of the RNN module to the output, if any.
        dropout
            Fraction of neurons afected by Dropout.
        random_state
            Control the randomness of the weights initialization. Check this
            `link <https://scikit-learn.org/stable/glossary.html#term-random-state>`_ for more details.
        """

        kwargs['input_chunk_length'] = input_chunk_length
        kwargs['output_chunk_length'] = output_chunk_length
        super().__init__(**kwargs)

        # check we got right model type specified:
        if model not in ['RNN', 'LSTM', 'GRU']:
            raise_if_not(isinstance(model, nn.Module), '{} is not a valid RNN model.\n Please specify "RNN", "LSTM", '
                                                       '"GRU", or give your own PyTorch nn.Module'.format(
                                                        model.__class__.__name__), logger)

        self.input_chunk_length = input_chunk_length
        self.output_chunk_length = output_chunk_length
        self.rnn_type_or_module = model
        self.hidden_fc_sizes = hidden_fc_sizes
        self.hidden_size = hidden_size
        self.n_rnn_layers = n_rnn_layers
        self.dropout = dropout

    def _create_model(self, input_dim: int, output_dim: int) -> torch.nn.Module:
        if self.rnn_type_or_module in ['RNN', 'LSTM', 'GRU']:
            hidden_fc_sizes = [] if self.hidden_fc_sizes is None else self.hidden_fc_sizes
            model = _BlockRNNModule(name=self.rnn_type_or_module,
                                    input_size=input_dim,
                                    target_size=output_dim,
                                    hidden_dim=self.hidden_size,
                                    num_layers=self.n_rnn_layers,
                                    output_chunk_length=self.output_chunk_length,
                                    num_layers_out_fc=hidden_fc_sizes,
                                    dropout=self.dropout)
        else:
            model = self.rnn_type_or_module
        return model
