"""
N-BEATS
-------
"""

from typing import NewType, Union, List, Optional, Tuple
from enum import Enum
import numpy as np
from numpy.random import RandomState
import torch
import torch.nn as nn

from ..logging import get_logger, raise_log, raise_if_not
from ..utils.torch import random_method
from .torch_forecasting_model import PastCovariatesTorchModel

logger = get_logger(__name__)


class _GType(Enum):
    GENERIC = 1
    TREND = 2
    SEASONALITY = 3


GTypes = NewType('GTypes', _GType)


class _TrendGenerator(nn.Module):

    def __init__(self,
                 expansion_coefficient_dim,
                 output_dim):
        super(_TrendGenerator, self).__init__()
        self.T = nn.Parameter(
            torch.stack([(torch.arange(output_dim) / output_dim)**i for i in range(expansion_coefficient_dim)], 1),
            False)

    def forward(self, x):
        return torch.matmul(x, self.T.T)


class _SeasonalityGenerator(nn.Module):

    def __init__(self,
                 output_dim):
        super(_SeasonalityGenerator, self).__init__()
        half_minus_one = int(output_dim / 2 - 1)
        cos_vectors = [torch.cos(torch.arange(output_dim) * 2 * np.pi * i) for i in range(1, half_minus_one + 1)]
        sin_vectors = [torch.sin(torch.arange(output_dim) * 2 * np.pi * i) for i in range(1, half_minus_one + 1)]
        self.S = nn.Parameter(torch.stack([torch.ones(output_dim)] + cos_vectors + sin_vectors, 1),
                              False)

    def forward(self, x):
        return torch.matmul(x, self.S.T)


class _Block(nn.Module):

    def __init__(self,
                 num_layers: int,
                 layer_width: int,
                 expansion_coefficient_dim: int,
                 input_chunk_length: int,
                 target_length: int,
                 g_type: GTypes):
        """ PyTorch module implementing the basic building block of the N-BEATS architecture.

        Parameters
        ----------
        num_layers
            The number of fully connected layers preceding the final forking layers.
        layer_width
            The number of neurons that make up each fully connected layer.
        expansion_coefficient_dim
            The dimensionality of the waveform generator parameters, also known as expansion coefficients.
            Only used if `generic_architecture` is set to `True`.
        input_chunk_length
            The length of the input sequence fed to the model.
        target_length
            The length of the forecast of the model.
        g_type
            The type of function that is implemented by the waveform generator.

        Inputs
        ------
        x of shape `(batch_size, input_chunk_length)`
            Tensor containing the input sequence.

        Outputs
        -------
        x_hat of shape `(batch_size, input_chunk_length)`
            Tensor containing the 'backcast' of the block, which represents an approximation of `x`
            given the constraints of the functional space determined by `g`.
        y_hat of shape `(batch_size, output_chunk_length)`
            Tensor containing the forward forecast of the block.

        """
        super(_Block, self).__init__()

        self.num_layers = num_layers
        self.layer_width = layer_width
        self.g_type = g_type
        self.relu = nn.ReLU()

        # fully connected stack before fork
        self.linear_layer_stack_list = [nn.Linear(input_chunk_length, layer_width)]
        self.linear_layer_stack_list += [nn.Linear(layer_width, layer_width) for _ in range(num_layers - 1)]
        self.fc_stack = nn.ModuleList(self.linear_layer_stack_list)

        # fully connected layer producing forecast/backcast expansion coeffcients (waveform generator parameters)
        if g_type == _GType.SEASONALITY:
            self.backcast_linear_layer = nn.Linear(layer_width, 2 * int(input_chunk_length / 2 - 1) + 1)
            self.forecast_linear_layer = nn.Linear(layer_width, 2 * int(target_length / 2 - 1) + 1)
        else:
            self.backcast_linear_layer = nn.Linear(layer_width, expansion_coefficient_dim)
            self.forecast_linear_layer = nn.Linear(layer_width, expansion_coefficient_dim)

        # waveform generator functions
        if g_type == _GType.GENERIC:
            self.backcast_g = nn.Linear(expansion_coefficient_dim, input_chunk_length)
            self.forecast_g = nn.Linear(expansion_coefficient_dim, target_length)
        elif g_type == _GType.TREND:
            self.backcast_g = _TrendGenerator(expansion_coefficient_dim, input_chunk_length)
            self.forecast_g = _TrendGenerator(expansion_coefficient_dim, target_length)
        elif g_type == _GType.SEASONALITY:
            self.backcast_g = _SeasonalityGenerator(input_chunk_length)
            self.forecast_g = _SeasonalityGenerator(target_length)
        else:
            raise_log(ValueError("g_type not supported"), logger)

    def forward(self, x):
        # fully connected layer stack
        for layer in self.linear_layer_stack_list:
            x = self.relu(layer(x))

        # forked linear layers producing waveform generator parameters
        theta_backcast = self.backcast_linear_layer(x)
        theta_forecast = self.forecast_linear_layer(x)

        # waveform generator applications
        x_hat = self.backcast_g(theta_backcast)
        y_hat = self.forecast_g(theta_forecast)

        return x_hat, y_hat


class _Stack(nn.Module):

    def __init__(self,
                 num_blocks: int,
                 num_layers: int,
                 layer_width: int,
                 expansion_coefficient_dim: int,
                 input_chunk_length: int,
                 target_length: int,
                 g_type: GTypes,
                 ):
        """ PyTorch module implementing one stack of the N-BEATS architecture that comprises multiple basic blocks.

        Parameters
        ----------
        num_blocks
            The number of blocks making up this stack.
        num_layers
            The number of fully connected layers preceding the final forking layers in each block.
        layer_width
            The number of neurons that make up each fully connected layer in each block.
        expansion_coefficient_dim
            The dimensionality of the waveform generator parameters, also known as expansion coefficients.
            Only used if `generic_architecture` is set to `True`.
        input_chunk_length
            The length of the input sequence fed to the model.
        target_length
            The length of the forecast of the model.
        g_type
            The function that is implemented by the waveform generators in each block.

        Inputs
        ------
        stack_input of shape `(batch_size, input_chunk_length)`
            Tensor containing the input sequence.

        Outputs
        -------
        stack_residual of shape `(batch_size, input_chunk_length)`
            Tensor containing the 'backcast' of the block, which represents an approximation of `x`
            given the constraints of the functional space determined by `g`.
        stack_forecast of shape `(batch_size, output_chunk_length)`
            Tensor containing the forward forecast of the stack.

        """
        super(_Stack, self).__init__()

        self.input_chunk_length = input_chunk_length
        self.target_length = target_length

        if g_type == _GType.GENERIC:
            self.blocks_list = [
                _Block(num_layers, layer_width, expansion_coefficient_dim, input_chunk_length, target_length, g_type)
                for _ in range(num_blocks)
            ]
        else:
            # same block instance is used for weight sharing
            interpretable_block = _Block(num_layers, layer_width, expansion_coefficient_dim,
                                         input_chunk_length, target_length, g_type)
            self.blocks_list = [interpretable_block] * num_blocks

        self.blocks = nn.ModuleList(self.blocks_list)

    def forward(self, x):
        stack_forecast = torch.zeros(x.shape[0], self.target_length, device=x.device)
        for block in self.blocks_list:
            # pass input through block
            x_hat, y_hat = block(x)

            # add block forecast to stack forecast
            stack_forecast = stack_forecast + y_hat

            # subtract backcast from input to produce residual
            x = x - x_hat

        stack_residual = x

        return stack_residual, stack_forecast


class _NBEATSModule(nn.Module):

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 input_chunk_length: int,
                 output_chunk_length: int,
                 generic_architecture: bool,
                 num_stacks: int,
                 num_blocks: int,
                 num_layers: int,
                 layer_widths: List[int],
                 expansion_coefficient_dim: int,
                 trend_polynomial_degree: int
                 ):
        """ PyTorch module implementing the N-BEATS architecture.

        Parameters
        ----------
        input_chunk_length
            The length of the input sequence fed to the model.
        output_chunk_length
            The length of the forecast of the model.
        generic_architecture
            Boolean value indicating whether the generic architecture of N-BEATS is used.
            If not, the interpretable architecture outlined in the paper (consisting of one trend
            and one seasonality stack with appropriate waveform generator functions).
        num_stacks
            The number of stacks that make up the whole model. Only used if `generic_architecture` is set to `True`.
        num_blocks
            The number of blocks making up every stack.
        num_layers
            The number of fully connected layers preceding the final forking layers in each block of every stack.
            Only used if `generic_architecture` is set to `True`.
        layer_widths
            Determines the number of neurons that make up each fully connected layer in each block of every stack.
            If a list is passed, it must have a length equal to `num_stacks` and every entry in that list corresponds
            to the layer width of the corresponding stack. If an integer is passed, every stack will have blocks
            with FC layers of the same width.
        expansion_coefficient_dim
            The dimensionality of the waveform generator parameters, also known as expansion coefficients.
            Only used if `generic_architecture` is set to `True`.
        trend_polynomial_degree
            The degree of the polynomial used as waveform generator in trend stacks. Only used if
            `generic_architecture` is set to `False`.

        Inputs
        ------
        x of shape `(batch_size, input_chunk_length)`
            Tensor containing the input sequence.

        Outputs
        -------
        y of shape `(batch_size, output_chunk_length)`
            Tensor containing the output of the NBEATS module.

        """
        super(_NBEATSModule, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_chunk_length_multi = input_chunk_length*input_dim
        self.output_chunk_length = output_chunk_length
        self.target_length = output_chunk_length*input_dim

        if generic_architecture:
            self.stacks_list = [
                _Stack(num_blocks,
                       num_layers,
                       layer_widths[i],
                       expansion_coefficient_dim,
                       self.input_chunk_length_multi,
                       self.target_length,
                       _GType.GENERIC)
                for i in range(num_stacks)
            ]
        else:
            num_stacks = 2
            trend_stack = _Stack(num_blocks,
                                 num_layers,
                                 layer_widths[0],
                                 trend_polynomial_degree + 1,
                                 self.input_chunk_length_multi,
                                 self.target_length,
                                 _GType.TREND)
            seasonality_stack = _Stack(num_blocks,
                                       num_layers,
                                       layer_widths[1],
                                       -1,
                                       self.input_chunk_length_multi,
                                       self.target_length,
                                       _GType.SEASONALITY)
            self.stacks_list = [trend_stack, seasonality_stack]

        self.stacks = nn.ModuleList(self.stacks_list)

        # setting the last backcast "branch" to be not trainable (without next block/stack, it doesn't need to be
        # backpropagated). Removing this lines would cause logtensorboard to crash, since no gradient is stored
        # on this params (the last block backcast is not part of the final output of the net).
        self.stacks_list[-1].blocks[-1].backcast_linear_layer.requires_grad_(False)
        self.stacks_list[-1].blocks[-1].backcast_g.requires_grad_(False)

    def forward(self, x):

        # if x1, x2,... y1, y2... is one multivariate ts containing x and y, and a1, a2... one covariate ts
        # we reshape into x1, y1, a1, x2, y2, a2... etc
        x = torch.reshape(x, (x.shape[0], self.input_chunk_length_multi, 1))
        # squeeze last dimension (because model is univariate)
        x = x.squeeze(dim=2)

        y = torch.zeros(x.shape[0], self.target_length, device=x.device)
        for stack in self.stacks_list:
            # compute stack output
            stack_residual, stack_forecast = stack(x)

            # add stack forecast to final output
            y = y + stack_forecast

            # set current stack residual as input for next stack
            x = stack_residual

        # in multivariate case, we get a result x1, y1, z1 we want to reshape to original format
        y = y.reshape(y.shape[0], self.output_chunk_length, self.input_dim)

        # if some covariates, we don't want them for the output to be predicted.
        # the covariates are by construction added as extra time series on the right side. So we need to get rid of this
        # right output
        y = y[:, :, :self.output_dim]

        return y


class NBEATSModel(PastCovariatesTorchModel):
    @random_method
    def __init__(self,
                 input_chunk_length: int,
                 output_chunk_length: int,
                 generic_architecture: bool = True,
                 num_stacks: int = 30,
                 num_blocks: int = 1,
                 num_layers: int = 4,
                 layer_widths: Union[int, List[int]] = 256,
                 expansion_coefficient_dim: int = 5,
                 trend_polynomial_degree: int = 2,
                 random_state: Optional[Union[int, RandomState]] = None,
                 **kwargs):
        """ Neural Basis Expansion Analysis Time Series Forecasting (N-BEATS).

        This is an implementation of the N-BEATS architecture, as outlined in this paper:
        https://openreview.net/forum?id=r1ecqn4YwB

        In addition to the univariate version presented in the paper, our implementation also
        supports multivariate series (and covariates) by flattening the model inputs to a 1-D series
        and reshaping the outputs to a tensor of appropriate dimensions.

        This model supports past covariates (known for `input_chunk_length` points before prediction time).

        Parameters
        ----------
        input_chunk_length
            The length of the input sequence fed to the model.
        output_chunk_length
            The length of the forecast of the model.
        generic_architecture
            Boolean value indicating whether the generic architecture of N-BEATS is used.
            If not, the interpretable architecture outlined in the paper (consisting of one trend
            and one seasonality stack with appropriate waveform generator functions).
        num_stacks
            The number of stacks that make up the whole model. Only used if `generic_architecture` is set to `True`.
            The interpretable architecture always uses two stacks - one for trend and one for seasonality.
        num_blocks
            The number of blocks making up every stack.
        num_layers
            The number of fully connected layers preceding the final forking layers in each block of every stack.
            Only used if `generic_architecture` is set to `True`.
        layer_widths
            Determines the number of neurons that make up each fully connected layer in each block of every stack.
            If a list is passed, it must have a length equal to `num_stacks` and every entry in that list corresponds
            to the layer width of the corresponding stack. If an integer is passed, every stack will have blocks
            with FC layers of the same width.
        expansion_coefficient_dim
            The dimensionality of the waveform generator parameters, also known as expansion coefficients.
            Only used if `generic_architecture` is set to `True`.
        trend_polynomial_degree
            The degree of the polynomial used as waveform generator in trend stacks. Only used if
            `generic_architecture` is set to `False`.
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

        kwargs['input_chunk_length'] = input_chunk_length
        kwargs['output_chunk_length'] = output_chunk_length
        super().__init__(**kwargs)

        raise_if_not(isinstance(layer_widths, int) or len(layer_widths) == num_stacks,
                     "Please pass an integer or a list of integers with length `num_stacks`"
                     "as value for the `layer_widths` argument.", logger)

        self.input_chunk_length = input_chunk_length
        self.output_chunk_length = output_chunk_length
        self.generic_architecture = generic_architecture
        self.num_stacks = num_stacks
        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.layer_widths = layer_widths
        self.expansion_coefficient_dim = expansion_coefficient_dim
        self.trend_polynomial_degree = trend_polynomial_degree

        if not generic_architecture:
            self.num_stacks = 2

        if isinstance(layer_widths, int):
            self.layer_widths = [layer_widths] * num_stacks

    def _create_model(self, train_sample: Tuple[torch.Tensor]) -> torch.nn.Module:
        # samples are made of (past_target, past_covariates, future_target)
        input_dim = train_sample[0].shape[1] + (train_sample[1].shape[1] if train_sample[1] is not None else 0)
        output_dim = train_sample[-1].shape[1]

        return _NBEATSModule(
            input_dim=input_dim,
            output_dim=output_dim,
            input_chunk_length=self.input_chunk_length,
            output_chunk_length=self.output_chunk_length,
            generic_architecture=self.generic_architecture,
            num_stacks=self.num_stacks,
            num_blocks=self.num_blocks,
            num_layers=self.num_layers,
            layer_widths=self.layer_widths,
            expansion_coefficient_dim=self.expansion_coefficient_dim,
            trend_polynomial_degree=self.trend_polynomial_degree
        )
