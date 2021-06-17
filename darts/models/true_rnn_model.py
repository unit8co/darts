"""
Recurrent Neural Networks
-------------------------
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
class _TrueRNNModule(nn.Module):
    def __init__(self,
                 name: str,
                 input_size: int,
                 hidden_dim: int,
                 target_size: int = 1,
                 dropout: float = 0.):

        """ PyTorch module implementing a RNN to be used in `RNNModel`.

        PyTorch module implementing a simple RNN with the specified `name` layer.
        This module combines a PyTorch RNN module, together with a fully connected network, which maps the
        last hidden layers to output of the desired size `output_chunk_length` and makes it compatible with
        `RNNModel`s.

        Parameters
        ----------
        name
            The name of the specific PyTorch RNN module ("RNN", "GRU" or "LSTM").
        input_size
            The dimensionality of the input time series.
        hidden_dim
            The number of features in the hidden state `h` of the RNN module.
        target_size
            The dimensionality of the output time series.
        num_layers_out_fc
            A list containing the dimensions of the hidden layers of the fully connected NN.
            This network connects the last hidden layer of the PyTorch RNN module to the output.
        dropout
            The fraction of neurons that are dropped in all-but-last RNN layers.

        Inputs
        ------
        x of shape `(batch_size, input_length, input_size)`
            Tensor containing the features of the input sequence. The `input_length` is not fixed.

        Outputs
        -------
        y of shape `(batch_size, output_length, output_size)`
            The `output_length` is equal to 1 at prediction time, but equal to the `input_length` during training.
        """

        super(_TrueRNNModule, self).__init__()

        # Defining parameters
        self.target_size = target_size
        self.name = name

        # Defining the RNN module
        self.rnn = getattr(nn, name)(input_size, hidden_dim, 1, batch_first=True, dropout=dropout)

        # The RNN module needs a linear layer V that transforms hidden states into outputs, individually
        self.V = nn.Linear(hidden_dim, target_size)

    def forward(self, x):
        # data is of size (batch_size, input_ength, input_size)
        batch_size = x.size(0)

        # out is of size (batch_size, input_length, hidden_dim)
        out, hidden = self.rnn(x)
        #TODO: confirm shape of out

        """ Here, we apply the V matrix to every hidden state to produce the outputs
        """
        #TODO: for one layer out should be equal to hidden, make sure that's true
        predictions = self.V(out) # TODO: make sure this matrix multiplication is correct

        # predictions is of size (batch_size, input_length, target_size)
        predictions = predictions.view(batch_size, -1, self.target_size)

        # returns outputs for all inputs, only the last one is needed for prediction time
        return predictions


class TrueRNNModel(TorchForecastingModel):
    @random_method
    def __init__(self,
                 model: Union[str, nn.Module] = 'RNN',
                 hidden_dim: int = 25,
                 dropout: float = 0.,
                 random_state: Optional[Union[int, RandomState]] = None,
                 **kwargs):

        """ Recurrent Neural Network Model (RNNs).

        This class provides three variants of RNNs:

        * Vanilla RNN

        * LSTM

        * GRU

        Parameters
        ----------
        model
            Either a string specifying the RNN module type ("RNN", "LSTM" or "GRU"),
            or a PyTorch module with the same specifications as
            `darts.models.rnn_model.RNNModule`.
        hidden_dim
            Size for feature maps for each hidden RNN layer (:math:`h_n`).
        dropout
            Fraction of neurons afected by Dropout.
        random_state
            Control the randomness of the weights initialization. Check this
            `link <https://scikit-learn.org/stable/glossary.html#term-random-state>`_ for more details.
        """

        kwargs['input_chunk_length'] = -1
        kwargs['output_chunk_length'] = 1
        super().__init__(**kwargs)

        # check we got right model type specified:
        if model not in ['RNN', 'LSTM', 'GRU']:
            raise_if_not(isinstance(model, nn.Module), '{} is not a valid RNN model.\n Please specify "RNN", "LSTM", '
                                                       '"GRU", or give your own PyTorch nn.Module'.format(
                                                        model.__class__.__name__), logger)

        self.rnn_type_or_module = model
        self.dropout = dropout
        self.hidden_dim = hidden_dim

    def _create_model(self, input_dim: int, output_dim: int) -> torch.nn.Module:
        if self.rnn_type_or_module in ['RNN', 'LSTM', 'GRU']:
            model = _TrueRNNModule(name=self.rnn_type_or_module,
                               input_size=input_dim,
                               target_size=output_dim,
                               hidden_dim=self.hidden_dim,
                               dropout=self.dropout)
        else:
            model = self.rnn_type_or_module
        return model
