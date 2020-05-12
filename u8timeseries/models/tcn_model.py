"""
Temporal Convolutional Network
------------------------------
"""

import numpy as np
import os
import re
from glob import glob
import shutil
import math
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from typing import List, Optional, Dict, Union

from ..timeseries import TimeSeries
from ..custom_logging import raise_if_not, get_logger, raise_log
from .autoregressive_model import AutoRegressiveModel
from ..utils import _build_tqdm_iterator
from .torch_forecasting_model import (
    TorchForecastingModel,
    _TimeSeriesDataset1DShifted,
    _get_checkpoint_folder,
    _get_runs_folder,
    CHECKPOINTS_FOLDER,
    RUNS_FOLDER
)

logger = get_logger(__name__)


class TCNModule(nn.Module):
    def __init__(self,
                 input_size: int,
                 input_length: int,
                 kernel_size: int,
                 num_filters: int,
                 num_layers: Optional[int],
                 dilation_base: int,
                 output_length: int,
                 dropout: float):

        """ PyTorch module implementing a dilated TCN module used in `TCNModel`.


        Parameters
        ----------
        input_size
            The dimensionality of the input time series.
        input_length
            The length of the input time series.
        kernel_size
            The size of every kernel in a convolutional layer.
        num_filters
            The number of filters in a convolutional layer of the TCN.
        num_layers
            The number of convolutional layers.
        dilation_base
            The base of the exponent that will determine the dilation on every level.

        Inputs
        ------
        x of shape `(batch_size, input_length, input_size)`
            Tensor containing the features of the input sequence.

        Outputs
        -------
        y of shape `(batch_size, 1, 1)`
            Tensor containing the point prediciton of the next point in the series after the last entry.
        """

        super(TCNModule, self).__init__()

        # Defining parameters
        self.input_size = input_size
        self.input_length = input_length
        self.n_filters = num_filters
        self.kernel_size = kernel_size
        self.out_len = output_length
        self.dilation_base = dilation_base
        self.dropout = nn.Dropout(p=dropout)

        # If num_layers is not passed, compute number of layers needed for full history coverage
        if (num_layers is None and dilation_base > 1):
            num_layers = math.ceil(math.log((input_length - 1) / (kernel_size - 1), dilation_base))
        else: 
            num_layers = input_length - kernel_size + 1

        # Building TCN module
        self.tcn_layers_list = [
            nn.Conv1d(input_size, num_filters, kernel_size, dilation=1)
        ]
        for i in range(1, num_layers - 1):
            conv1d_layer = nn.Conv1d(num_filters, num_filters, kernel_size, dilation=(dilation_base ** i))
            self.tcn_layers_list.append(conv1d_layer)
        self.tcn_layers_list.append(
            nn.Conv1d(num_filters, input_size, kernel_size, dilation=(dilation_base ** (i + 1)))
        )
        for layer in self.tcn_layers_list:
            nn.init.xavier_uniform(layer.weight)
        self.tcn_layers = nn.ModuleList(self.tcn_layers_list)



    def forward(self, x):
        # data is of size (batch_size, input_length, input_size)
        batch_size = x.size(0)

        x = x.transpose(1, 2)

        for i, conv1d_layer in enumerate(self.tcn_layers_list):
            # pad input
            left_padding = (self.dilation_base ** i) * (self.kernel_size - 1)
            x = F.pad(x, (left_padding, 0))
            # feed input to convolutional layer
            x = conv1d_layer(x)
            # introduce non-linearity
            x = F.relu(x)
            # dropout
            x = self.dropout(x)
        
        x = x.transpose(1, 2)

        x = x.view(batch_size, self.input_length, 1)

        return x


class TCNModel(TorchForecastingModel):

    def __init__(self,
                 input_length: int = 12,
                 kernel_size: int = 3,
                 num_filters: int = 3,
                 num_layers: Optional[int] = None,
                 dilation_base: int = 2,
                 dropout: float = 0.2,
                 **kwargs):

        """ Temporal Convolutional Network Model (TCN).

        Parameters
        ----------
        input_length
            Number of past time steps that are fed to the forecasting module.
        kernel_size
            The size of every kernel in a convolutional layer.
        num_filters
            The number of filters in a convolutional layer of the TCN.
        dilation_base
            The base of the exponent that will determine the dilation on every level.
        num_layers
            The number of convolutional layers.
        """

        raise_if_not(kernel_size < input_length,
                     "The kernel size must be strictly smaller than the input length.", logger)

        self.input_size = 1
        kwargs['input_length'] = input_length

        self.model = TCNModule(input_size=self.input_size, input_length=input_length, 
                               kernel_size=kernel_size, num_filters=num_filters,
                               num_layers=num_layers, dilation_base=dilation_base, 
                               output_length=1, dropout=dropout)

        super().__init__(**kwargs)


    def create_dataset(self, series):
        return _TimeSeriesDataset1DShifted(series, self.input_length, 1)


    def predict(self, n: int) -> TimeSeries:
        super().predict(n)

        scaled_series = self.training_series.values()[-self.seq_len:]
        pred_in = torch.from_numpy(scaled_series).float().view(1, -1, 1).to(self.device)
        test_out = []
        self.model.eval()
        for i in range(n):
            out = self.model(pred_in)
            pred_in = pred_in.roll(-1, 1)
            pred_in[:, -1, :] = out[:, -1]
            test_out.append(out.cpu().detach().numpy()[0, 0])
        test_out = np.stack(test_out)

        return self._build_forecast_series(test_out.squeeze())


   