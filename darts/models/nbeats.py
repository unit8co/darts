"""
N-BEATS
-------
"""

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
