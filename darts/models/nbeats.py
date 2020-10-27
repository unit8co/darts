"""
N-BEATS
-------
"""

from typing import NewType, Optional, Union, List
from enum import Enum
import numpy as np
from numpy.random import RandomState
import torch
import torch.nn as nn

from ..logging import get_logger, raise_log, raise_if_not
from ..utils.torch import random_method
from .torch_forecasting_model import TorchForecastingModel

logger = get_logger(__name__)


class GType(Enum):
    GENERIC = 1
    TREND = 2
    SEASONALITY = 3


GTypes = NewType('GTypes', GType)


class _TrendGenerator(nn.Module):

    def __init__(self,
                 expansion_coefficient_dim,
                 output_dim):
        super(_TrendGenerator, self).__init__()
        self.T = torch.stack([(torch.arange(output_dim) / output_dim)**i for i in range(expansion_coefficient_dim)], 1)

    def forward(self, x):
        return torch.matmul(x, self.T.float().T)


class _SeasonalityGenerator(nn.Module):

    def __init__(self,
                 output_dim):
        super(_SeasonalityGenerator, self).__init__()
        half_minus_one = int(output_dim / 2 - 1)
        cos_vectors = [torch.cos(torch.arange(output_dim) * 2 * np.pi * i) for i in range(1, half_minus_one + 1)]
        sin_vectors = [torch.sin(torch.arange(output_dim) * 2 * np.pi * i) for i in range(1, half_minus_one + 1)]
        self.S = torch.stack([torch.ones(output_dim)] + cos_vectors + sin_vectors, 1)

    def forward(self, x):
        return torch.matmul(x, self.S.float().T)


class _Block(nn.Module):

    def __init__(self,
                 num_layers: int,
                 layer_width: int,
                 expansion_coefficient_dim: int,
                 input_length: int,
                 output_length: int,
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
        input_length
            The length of the input sequence fed to the model.
        output_length
            The length of the forecast of the model.
        g_type
            The type of function that is implemented by the waveform generator.

        Inputs
        ------
        x of shape `(batch_size, input_length)`
            Tensor containing the input sequence.

        Outputs
        -------
        x_hat of shape `(batch_size, input_length)`
            Tensor containing the 'backcast' of the block, which represents an approximation of `x`
            given the constraints of the functional space determined by `g`.
        y_hat of shape `(batch_size, output_length)`
            Tensor containing the forward forecast of the block.

        """
        super(_Block, self).__init__()

        self.num_layers = num_layers
        self.layer_width = layer_width
        self.g_type = g_type
        self.relu = nn.ReLU()

        # fully connected stack before fork
        self.linear_layer_stack_list = [nn.Linear(input_length, layer_width)]
        self.linear_layer_stack_list += [nn.Linear(layer_width, layer_width) for _ in range(num_layers - 1)]
        self.fc_stack = nn.ModuleList(self.linear_layer_stack_list)

        # fully connected layer producing forecast/backcast expansion coeffcients (waveform generator parameters)
        if g_type == GType.SEASONALITY:
            self.backcast_linear_layer = nn.Linear(layer_width, 2 * int(input_length / 2 - 1) + 1)
            self.forecast_linear_layer = nn.Linear(layer_width, 2 * int(output_length / 2 - 1) + 1)
        else:
            self.backcast_linear_layer = nn.Linear(layer_width, expansion_coefficient_dim)
            self.forecast_linear_layer = nn.Linear(layer_width, expansion_coefficient_dim)

        # waveform generator functions
        if g_type == GType.GENERIC:
            self.backcast_g = nn.Linear(expansion_coefficient_dim, input_length)
            self.forecast_g = nn.Linear(expansion_coefficient_dim, output_length)
        elif g_type == GType.TREND:
            self.backcast_g = _TrendGenerator(expansion_coefficient_dim, input_length)
            self.forecast_g = _TrendGenerator(expansion_coefficient_dim, output_length)
        elif g_type == GType.SEASONALITY:
            self.backcast_g = _SeasonalityGenerator(input_length)
            self.forecast_g = _SeasonalityGenerator(output_length)
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
                 input_length: int,
                 output_length: int,
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
        input_length
            The length of the input sequence fed to the model.
        output_length
            The length of the forecast of the model.
        g_type
            The function that is implemented by the waveform generators in each block.

        Inputs
        ------
        stack_input of shape `(batch_size, input_length)`
            Tensor containing the input sequence.

        Outputs
        -------
        stack_residual of shape `(batch_size, input_length)`
            Tensor containing the 'backcast' of the block, which represents an approximation of `x`
            given the constraints of the functional space determined by `g`.
        stack_forecast of shape `(batch_size, output_length)`
            Tensor containing the forward forecast of the stack.

        """
        super(_Stack, self).__init__()

        self.input_length = input_length
        self.output_length = output_length

        if g_type == GType.GENERIC:
            self.blocks_list = [
                _Block(num_layers, layer_width, expansion_coefficient_dim, input_length, output_length, g_type)
                for _ in range(num_blocks)
            ]
        else:
            # same block instance is used for weight sharing
            interpretable_block = _Block(num_layers, layer_width, expansion_coefficient_dim,
                                         input_length, output_length, g_type)
            self.blocks_list = [interpretable_block] * num_blocks

        self.blocks = nn.ModuleList(self.blocks_list)

    def forward(self, x):
        stack_forecast = torch.zeros(x.shape[0], self.output_length)
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
                 generic_architecture: bool,
                 num_stacks: int,
                 num_blocks: int,
                 num_layers: int,
                 layer_widths: int,
                 expansion_coefficient_dim: int,
                 trend_polynomial_degree: int,
                 input_length: int,
                 output_length: int
                 ):
        """ PyTorch module implementing the N-BEATS architecture.

        Parameters
        ----------
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
        input_length
            The length of the input sequence fed to the model.
        output_length
            The length of the forecast of the model.

        Inputs
        ------
        x of shape `(batch_size, input_length)`
            Tensor containing the input sequence.

        Outputs
        -------
        y of shape `(batch_size, output_length)`
            Tensor containing the output of the NBEATS module.

        """
        super(_NBEATSModule, self).__init__()

        self.input_length = input_length
        self.output_length = output_length

        if generic_architecture:
            self.stacks_list = [
                _Stack(num_blocks, num_layers, layer_widths[i], expansion_coefficient_dim,
                       input_length, output_length, GType.GENERIC) for i in range(num_stacks)
            ]
        else:
            num_stacks = 2
            trend_stack = _Stack(num_blocks, num_layers, layer_widths[0], trend_polynomial_degree + 1,
                                 input_length, output_length, GType.TREND)
            seasonality_stack = _Stack(num_blocks, num_layers, layer_widths[1], -1,
                                       input_length, output_length, GType.SEASONALITY)
            self.stacks_list = [trend_stack, seasonality_stack]

        self.stacks = nn.ModuleList(self.stacks_list)

    def forward(self, x):

        # squeeze last dimension (because model is univariate)
        x = x.squeeze(dim=2)

        y = torch.zeros(x.shape[0], self.output_length)
        for stack in self.stacks_list:
            # compute stack output
            stack_residual, stack_forecast = stack(x)

            # add stack forecast to final output
            y = y + stack_forecast

            # set current stack residual as input for next stack
            x = stack_residual

        # unsqueeze last dimension
        y = y.unsqueeze(dim=2)

        return y


class NBEATSModel(TorchForecastingModel):

    @random_method
    def __init__(self,
                 generic_architecture: bool = True,
                 num_stacks: int = 30,
                 num_blocks: int = 1,
                 num_layers: int = 4,
                 layer_widths: Union[int, List[int]] = 256,
                 expansion_coefficient_dim: int = 5,
                 trend_polynomial_degree: int = 2,
                 input_length: int = 12,
                 output_length: int = 1,
                 random_state: Optional[Union[int, RandomState]] = None,
                 **kwargs):
        """ Neural Basis Expansion Analysis Time Series Forecasting (N-BEATS).

        This is an implementation of the N-BEATS architecture as outlined in this paper:
        https://openreview.net/forum?id=r1ecqn4YwB

        Parameters
        ----------
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
        input_length
            The length of the input sequence fed to the model.
        output_length
            The length of the forecast of the model.
        random_state
            Control the randomness of the weights initialization. Check this
            `link <https://scikit-learn.org/stable/glossary.html#term-random-state>`_ for more details.

        """

        if not generic_architecture:
            num_stacks = 2

        raise_if_not(isinstance(layer_widths, int) or len(layer_widths) == num_stacks,
                     "Please pass an integer or a list of integers with length `num_stacks`"
                     "as value for the `layer_widths` argument.", logger)

        if isinstance(layer_widths, int):
            layer_widths = [layer_widths] * num_stacks

        self.model = _NBEATSModule(
            generic_architecture,
            num_stacks,
            num_blocks,
            num_layers,
            layer_widths,
            expansion_coefficient_dim,
            trend_polynomial_degree,
            input_length,
            output_length
        )

        kwargs['input_length'] = input_length
        kwargs['output_length'] = output_length

        super().__init__(**kwargs)

    @random_method
    def fit(self, *args, **kwargs):
        if 'training_series' in kwargs:
            training_series = kwargs['training_series']
        else:
            training_series = args[0]
        training_series._assert_univariate()
        super().fit(*args, **kwargs)
