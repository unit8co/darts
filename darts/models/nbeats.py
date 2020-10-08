"""
N-BEATS
-------
"""

import torch
import torch.nn as nn
from typing import NewType
from enum import Enum


class GType(Enum):
    GENERIC = 1
    TREND = 2
    SEASONALITY = 3


GTypes = NewType('GTypes', GType)


class _Block(nn.Module):

    def __init__(self,
                 num_layers: int,
                 layer_width: int,
                 input_length: int,
                 output_length: int,
                 num_stacks: int,
                 g_type: GTypes):
        """ PyTorch module implementing the basic building block of the N-BEATS architecture.

        Parameters
        ----------
        num_layers
            The number of fully connected layers preceding the final forking layers.
        layer_width
            The number of neurons that make up each fully connected layer.
        input_length
            The length of the input sequence fed to the model.
        output_length
            The length of the forecast of the model.
        num_stacks
            The number of stacks (each comprised of 1 or more `_Block` instances) that make up the whole model.
            This information is needed on the `_Block` level since it dictates the dimensionality of the waveform
            generator parameters.
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

        # fully connected stack before fork
        self.linear_layer_stack_list = [nn.Linear(input_length, layer_width)]
        self.linear_layer_stack_list += [nn.Linear(layer_width, layer_width) for i in range(num_layers - 1)]
        self.fc_stack = nn.ModuleList(self.linear_layer_stack_list)

        # fully connected layer producing backcast waveform generator parameters
        self.backcast_linear_layer = nn.Linear(layer_width, num_stacks)

        # fully connected layer producing forecast waveform generator parameters
        self.forecast_linear_layer = nn.Linear(layer_width, num_stacks)

        if g_type == GType.GENERIC:
            self.backcast_g = nn.Linear(num_stacks, input_length)
            self.forecast_g = nn.Linear(num_stacks, output_length)
        else:
            # TODO: implement seasonality and trend waveform generators
            raise NotImplementedError()

    def forward(self, x):
        # fully connected layer stack
        for layer in self.linear_layer_stack_list:
            x = nn.ReLU(layer(x))

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
                 input_length: int,
                 output_length: int,
                 num_stacks: int,
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
        input_length
            The length of the input sequence fed to the model.
        output_length
            The length of the forecast of the model.
        num_stacks
            The number of stacks that make up the whole model.
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
            Tensor containing the forward forecast of the block.

        """

        self.input_length = input_length
        self.output_length = output_length

        self.blocks_list = [
            _Block(num_layers, layer_width, input_length, output_length, num_stacks, g_type) for i in range(num_stacks)
        ]
        self.blocks = nn.ModuleList(self.linear_layer_stack_list)

    def forward(self, x):
        stack_forecast = torch.zeros(x.shape[0], self.output_length)
        for block in self.blocks_list:
            # pass input through block
            x_hat, y_hat = block(x)

            # add block forecast to stack forecast
            stack_forecast += y_hat

            # subtract backcast from input to produce residual
            x -= x_hat

        stack_residual = x

        return stack_residual, stack_forecast

