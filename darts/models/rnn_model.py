"""
Recurrent Neural Networks
-------------------------
"""

import torch.nn as nn
from typing import List, Optional, Union

from ..logging import raise_if_not, get_logger
from .torch_forecasting_model import TorchForecastingModel

logger = get_logger(__name__)


# TODO add batch norm
class _RNNModule(nn.Module):
    def __init__(self,
                 name: str,
                 input_size: int,
                 hidden_dim: int,
                 num_layers: int,
                 output_length: int = 1,
                 output_size: int = 1,
                 num_layers_out_fc: Optional[List] = None,
                 dropout: float = 0.):

        """ PyTorch module implementing a RNN to be used in `RNNModel`.

        PyTorch module implementing a simple RNN with the specified `name` layer.
        This module combines a PyTorch RNN module, together with a fully connected network, which maps the
        last hidden layers to output of the desired size `output_length` and makes it compatible with
        `RNNModel`s.

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
        output_length
            The number of steps to predict in the future.
        output_size
            The dimensionality of the output time series.
        num_layers_out_fc
            A list containing the dimensions of the hidden layers of the fully connected NN.
            This network connects the last hidden layer of the PyTorch RNN module to the output.
        dropout
            The fraction of neurons that are dropped in all-but-last RNN layers.

        Inputs
        ------
        x of shape `(batch_size, input_length, input_size)`
            Tensor containing the features of the input sequence.

        Outputs
        -------
        y of shape `(batch_size, out_len, output_size)`
            Tensor containing the (point) prediction at the last time step of the sequence.
        """

        super(_RNNModule, self).__init__()

        # Defining parameters
        self.hidden_dim = hidden_dim
        self.n_layers = num_layers
        self.output_size = output_size
        num_layers_out_fc = [] if num_layers_out_fc is None else num_layers_out_fc
        self.out_len = output_length
        self.name = name

        # Defining the RNN module
        self.rnn = getattr(nn, name)(input_size, hidden_dim, num_layers, batch_first=True, dropout=dropout)

        # The RNN module is followed by a fully connected layer, which maps the last hidden layer
        # to the output of desired length
        last = hidden_dim
        feats = []
        for feature in num_layers_out_fc + [output_length * output_size]:
            feats.append(nn.Linear(last, feature))
            last = feature
        self.fc = nn.Sequential(*feats)

    def forward(self, x):
        # data is of size (batch_size, input_length, input_size)
        batch_size = x.size(0)

        out, hidden = self.rnn(x)

        """ Here, we apply the FC network only on the last output point (at the last time step)
        """
        if self.name == "LSTM":
            hidden = hidden[0]
        predictions = hidden[-1, :, :]
        predictions = self.fc(predictions)
        predictions = predictions.view(batch_size, self.out_len, self.output_size)

        # predictions is of size (batch_size, output_length, 1)
        return predictions


class RNNModel(TorchForecastingModel):

    def __init__(self,
                 model: Union[str, nn.Module] = 'RNN',
                 input_size: int = 1,
                 output_length: int = 1,
                 output_size: int = 1,
                 hidden_size: int = 25,
                 n_rnn_layers: int = 1,
                 hidden_fc_sizes: Optional[List] = None,
                 dropout: float = 0.,
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
        input_size
            The dimensionality of the TimeSeries instances that will be fed to the fit function.
        output_size
            The dimensionality of the output time series.
        output_length
            Number of time steps to be output by the forecasting module.
        hidden_size
            Size for feature maps for each hidden RNN layer (:math:`h_n`).
        n_rnn_layers
            Number of layers in the RNN module.
        hidden_fc_sizes
            Sizes of hidden layers connecting the last hidden layer of the RNN module to the output, if any.
        dropout
            Fraction of neurons afected by Dropout.
        """

        kwargs['output_length'] = output_length
        kwargs['input_size'] = input_size
        kwargs['output_size'] = output_size

        # set self.model
        if model in ['RNN', 'LSTM', 'GRU']:
            hidden_fc_sizes = [] if hidden_fc_sizes is None else hidden_fc_sizes
            self.model = _RNNModule(name=model, input_size=input_size, output_size=output_size, hidden_dim=hidden_size,
                                    num_layers=n_rnn_layers, output_length=output_length,
                                    num_layers_out_fc=hidden_fc_sizes, dropout=dropout)
        else:
            self.model = model
        raise_if_not(isinstance(self.model, nn.Module), '{} is not a valid RNN model.\n Please specify "RNN", "LSTM", '
                     '"GRU", or give your own PyTorch nn.Module'.format(model.__class__.__name__),
                     logger)

        super().__init__(**kwargs)
