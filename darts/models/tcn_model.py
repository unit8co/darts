"""
Temporal Convolutional Network
------------------------------
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.random import RandomState
from typing import Optional, Union, Sequence, Tuple
from ..timeseries import TimeSeries
from ..utils.torch import random_method
from ..utils.data import PastCovariatesShiftedDataset
from ..utils.likelihood_models import LikelihoodModel

from ..logging import raise_if_not, get_logger
from .torch_forecasting_model import TorchParametricProbabilisticForecastingModel, PastCovariatesTorchModel

logger = get_logger(__name__)


class _ResidualBlock(nn.Module):

    def __init__(self,
                 num_filters: int,
                 kernel_size: int,
                 dilation_base: int,
                 dropout_fn,
                 weight_norm: bool,
                 nr_blocks_below: int,
                 num_layers: int,
                 input_size: int,
                 target_size: int):
        """ PyTorch module implementing a residual block module used in `_TCNModule`.

        Parameters
        ----------
        num_filters
            The number of filters in a convolutional layer of the TCN.
        kernel_size
            The size of every kernel in a convolutional layer.
        dilation_base
            The base of the exponent that will determine the dilation on every level.
        dropout_fn
            The dropout function to be applied to every convolutional layer.
        weight_norm
            Boolean value indicating whether to use weight normalization.
        nr_blocks_below
            The number of residual blocks before the current one.
        num_layers
            The number of convolutional layers.
        input_size
            The dimensionality of the input time series of the whole network.
        target_size
            The dimensionality of the output time series of the whole network.

        Inputs
        ------
        x of shape `(batch_size, in_dimension, input_chunk_length)`
            Tensor containing the features of the input sequence.
            in_dimension is equal to `input_size` if this is the first residual block,
            in all other cases it is equal to `num_filters`.

        Outputs
        -------
        y of shape `(batch_size, out_dimension, input_chunk_length)`
            Tensor containing the output sequence of the residual block.
            out_dimension is equal to `output_size` if this is the last residual block,
            in all other cases it is equal to `num_filters`.
        """
        super(_ResidualBlock, self).__init__()

        self.dilation_base = dilation_base
        self.kernel_size = kernel_size
        self.dropout_fn = dropout_fn
        self.num_layers = num_layers
        self.nr_blocks_below = nr_blocks_below

        input_dim = input_size if nr_blocks_below == 0 else num_filters
        output_dim = target_size if nr_blocks_below == num_layers - 1 else num_filters
        self.conv1 = nn.Conv1d(input_dim, num_filters, kernel_size, dilation=(dilation_base ** nr_blocks_below))
        self.conv2 = nn.Conv1d(num_filters, output_dim, kernel_size, dilation=(dilation_base ** nr_blocks_below))
        if weight_norm:
            self.conv1, self.conv2 = nn.utils.weight_norm(self.conv1), nn.utils.weight_norm(self.conv2)

        if nr_blocks_below == 0 or nr_blocks_below == num_layers - 1:
            self.conv3 = nn.Conv1d(input_dim, output_dim, 1)

    def forward(self, x):
        residual = x

        # first step
        left_padding = (self.dilation_base ** self.nr_blocks_below) * (self.kernel_size - 1)
        x = F.pad(x, (left_padding, 0))
        x = self.dropout_fn(F.relu(self.conv1(x)))

        # second step
        x = F.pad(x, (left_padding, 0))
        x = self.conv2(x)
        if self.nr_blocks_below < self.num_layers - 1:
            x = F.relu(x)
        x = self.dropout_fn(x)

        # add residual
        if self.nr_blocks_below in {0, self.num_layers - 1}:
            residual = self.conv3(residual)
        x += residual

        return x


class _TCNModule(nn.Module):
    def __init__(self,
                 input_size: int,
                 input_chunk_length: int,
                 kernel_size: int,
                 num_filters: int,
                 num_layers: Optional[int],
                 dilation_base: int,
                 weight_norm: bool,
                 target_size: int,
                 target_length: int,
                 dropout: float):

        """ PyTorch module implementing a dilated TCN module used in `TCNModel`.


        Parameters
        ----------
        input_size
            The dimensionality of the input time series.
        target_size
            The dimensionality of the output time series.
        input_chunk_length
            The length of the input time series.
        target_length
            Number of time steps the torch module will predict into the future at once.
        kernel_size
            The size of every kernel in a convolutional layer.
        num_filters
            The number of filters in a convolutional layer of the TCN.
        num_layers
            The number of convolutional layers.
        weight_norm
            Boolean value indicating whether to use weight normalization.
        dilation_base
            The base of the exponent that will determine the dilation on every level.
        dropout
            The dropout rate for every convolutional layer.

        Inputs
        ------
        x of shape `(batch_size, input_chunk_length, input_size)`
            Tensor containing the features of the input sequence.

        Outputs
        -------
        y of shape `(batch_size, input_chunk_length, 1)`
            Tensor containing the predictions of the next 'output_chunk_length' points in the last
            'output_chunk_length' entries of the tensor. The entries before contain the data points
            leading up to the first prediction, all in chronological order.
        """

        super(_TCNModule, self).__init__()

        # Defining parameters
        self.input_size = input_size
        self.input_chunk_length = input_chunk_length
        self.n_filters = num_filters
        self.kernel_size = kernel_size
        self.target_length = target_length
        self.target_size = target_size
        self.dilation_base = dilation_base
        self.dropout = nn.Dropout(p=dropout)

        # If num_layers is not passed, compute number of layers needed for full history coverage
        if num_layers is None and dilation_base > 1:
            num_layers = math.ceil(math.log((input_chunk_length - 1) * (dilation_base - 1) / (kernel_size - 1) / 2 + 1,
                                            dilation_base))
            logger.info("Number of layers chosen: " + str(num_layers))
        elif num_layers is None:
            num_layers = math.ceil((input_chunk_length - 1) / (kernel_size - 1) / 2)
            logger.info("Number of layers chosen: " + str(num_layers))
        self.num_layers = num_layers

        # Building TCN module
        self.res_blocks_list = []
        for i in range(num_layers):
            res_block = _ResidualBlock(num_filters, kernel_size, dilation_base,
                                       self.dropout, weight_norm, i, num_layers, self.input_size, target_size)
            self.res_blocks_list.append(res_block)
        self.res_blocks = nn.ModuleList(self.res_blocks_list)

    def forward(self, x):
        # data is of size (batch_size, input_chunk_length, input_size)
        batch_size = x.size(0)
        x = x.transpose(1, 2)

        for res_block in self.res_blocks_list:
            x = res_block(x)

        x = x.transpose(1, 2)
        x = x.view(batch_size, self.input_chunk_length, self.target_size)

        return x


class TCNModel(TorchParametricProbabilisticForecastingModel, PastCovariatesTorchModel):
    @random_method
    def __init__(self,
                 input_chunk_length: int,
                 output_chunk_length: int,
                 kernel_size: int = 3,
                 num_filters: int = 3,
                 num_layers: Optional[int] = None,
                 dilation_base: int = 2,
                 weight_norm: bool = False,
                 dropout: float = 0.2,
                 likelihood: Optional[LikelihoodModel] = None,
                 random_state: Optional[Union[int, RandomState]] = None,
                 **kwargs):

        """ Temporal Convolutional Network Model (TCN).

        This is an implementation of a dilated TCN used for forecasting.
        Inspiration: https://arxiv.org/abs/1803.01271

        This model supports past covariates (known for `input_chunk_length` points before prediction time).

        Parameters
        ----------
        input_chunk_length
            Number of past time steps that are fed to the forecasting module.
        output_chunk_length
            Number of time steps the torch module will predict into the future at once.
        kernel_size
            The size of every kernel in a convolutional layer.
        num_filters
            The number of filters in a convolutional layer of the TCN.
        weight_norm
            Boolean value indicating whether to use weight normalization.
        dilation_base
            The base of the exponent that will determine the dilation on every level.
        num_layers
            The number of convolutional layers.
        dropout
            The dropout rate for every convolutional layer.
        likelihood
            Optionally, the likelihood model to be used for probabilistic forecasts.
            If no likelihood model is provided, forecasts will be deterministic.
        random_state
            Control the randomness of the weights initialization. Check this
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
        """

        raise_if_not(kernel_size < input_chunk_length,
                     "The kernel size must be strictly smaller than the input length.", logger)
        raise_if_not(output_chunk_length < input_chunk_length,
                     "The output length must be strictly smaller than the input length", logger)

        kwargs['input_chunk_length'] = input_chunk_length
        kwargs['output_chunk_length'] = output_chunk_length

        super().__init__(likelihood=likelihood, **kwargs)

        self.input_chunk_length = input_chunk_length
        self.output_chunk_length = output_chunk_length
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        self.num_layers = num_layers
        self.dilation_base = dilation_base
        self.dropout = dropout
        self.weight_norm = weight_norm

    def _create_model(self, train_sample: Tuple[torch.Tensor]) -> torch.nn.Module:
        # samples are made of (past_target, past_covariates, future_target)
        input_dim = train_sample[0].shape[1] + (train_sample[1].shape[1] if train_sample[1] is not None else 0)
        output_dim = train_sample[-1].shape[1]

        target_size = (
            self.likelihood._num_parameters * output_dim if self.likelihood is not None else output_dim
        )
        return _TCNModule(input_size=input_dim,
                          input_chunk_length=self.input_chunk_length,
                          target_size=target_size,
                          kernel_size=self.kernel_size,
                          num_filters=self.num_filters,
                          num_layers=self.num_layers,
                          dilation_base=self.dilation_base,
                          target_length=self.output_chunk_length,
                          dropout=self.dropout,
                          weight_norm=self.weight_norm)

    def _build_train_dataset(self,
                             target: Sequence[TimeSeries],
                             past_covariates: Optional[Sequence[TimeSeries]],
                             future_covariates: Optional[Sequence[TimeSeries]]) -> PastCovariatesShiftedDataset:

        return PastCovariatesShiftedDataset(target_series=target,
                                            covariates=past_covariates,
                                            length=self.input_chunk_length,
                                            shift=self.output_chunk_length)
    
    @random_method
    def _produce_predict_output(self, input):
        if self.likelihood:
            output = self.model(input)
            return self.likelihood._sample(output)
        else:
            return self.model(input)

    @property
    def first_prediction_index(self) -> int:
        return -self.output_chunk_length
