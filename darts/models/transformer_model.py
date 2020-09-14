"""
Transformer
-------------------------
"""

import numpy as np
from numpy.random import RandomState
import os
import math
import shutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Optional, Any, Union, List

from .. import TimeSeries
from ..utils import _build_tqdm_iterator
from ..utils.torch import random_method
from ..logging import raise_if_not, get_logger
from .torch_forecasting_model import TorchForecastingModel, _TimeSeriesSequentialDataset

CHECKPOINTS_FOLDER = os.path.join('.darts', 'checkpoints')
RUNS_FOLDER = os.path.join('.darts', 'runs')

logger = get_logger(__name__)


def _get_checkpoint_folder(work_dir, model_name):
    return os.path.join(work_dir, CHECKPOINTS_FOLDER, model_name)


def _get_runs_folder(work_dir, model_name):
    return os.path.join(work_dir, RUNS_FOLDER, model_name)


class _TimeSeriesSequentialTransformerDataset(_TimeSeriesSequentialDataset):

    def __init__(self,
                 series: TimeSeries,
                 data_length: int = 1,
                 target_length: int = 1,
                 target_indices: List[int] = [0]):
        """
        A PyTorch Dataset from a multivariate TimeSeries.
        The Dataset iterates a moving window over the time series. The resulting slices contain `(data, target)`,
        where `data` is a sub-sequence of length `data_length` and target is the sub-sequence of length
        `target_length` following it in the time series.
        """

        super().__init__(series,
                         data_length,
                         target_length,
                         target_indices)

    def __getitem__(self, index):
        # TODO: Cast to PyTorch tensors on the right device in advance
        idx = index % (self.len_series - self.data_length - self.target_length + 1)
        data = self.series_values[idx:idx + self.data_length]
        target = self.series_values[idx + self.data_length:idx + self.data_length + self.target_length]
        return torch.from_numpy(data).float(), torch.from_numpy(target).float()


# This implementation of positional encoding is taken from the PyTorch documentation:
# https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):
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
                    The dimensionality of the computed positional encoding array. Only its first "input_size" elements will be
                    considered in the output



                Inputs
                ------
                x of shape `(batch_size, input_size, d_model)`
                    Tensor containing the embedded time series

                Outputs
                -------
                y of shape `(batch_size, input_size, d_model)`
                    Tensor containing the embedded time series enhanced with positional encoding
                """
        super(PositionalEncoding, self).__init__()
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
                 output_length: int,
                 output_size: int,
                 d_model: int,
                 nhead: int,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 dim_feedforward: int,
                 dropout: float,
                 activation: str,
                 custom_encoder: nn.Module = None,
                 custom_decoder: nn.Module = None,
                 ):

        """ PyTorch module implementing a Transformer to be used in `TransformerModel`.

        PyTorch module implementing a simple encoder-decoder transformer architecture.

        Parameters
        ----------
        input_size
            The dimensionality of the TimeSeries instances that will be fed to the fit function.
        output_size
            The dimensionality of the output time series.
        output_length
            Number of time steps to be output by the forecasting module.
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
        self.positional_encoding = PositionalEncoding(d_model, dropout)

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

        self.tgt_mask = self.transformer.generate_square_subsequent_mask(output_length)
        self.decoder = nn.Linear(d_model, output_size)

    # if Training = 'True', we are in training mode:
    # given src=(x_0, ..., x_n) and tgt=(x_n+1, ..., x_n+m), then
    # encoder_src = src and decoder_src = (x_n, x_n+1, .. x_n+m-1)
    # if Training = 'False', then we are in inference mode:
    # given src=(x_0, ... x_n) and tgt=(), then
    # encoder_src = src and decoder_src = (x_n)
    def _create_transformer_inputs(self, src, tgt, training):
        encoder_src = src.permute(1, 0, 2)

        if training:
            decoder_src = torch.cat((encoder_src[-1:, :, :], tgt.permute(1, 0, 2)[:-1, :, :]), dim=0)
            tgt_mask = self.tgt_mask
        else:
            decoder_src = encoder_src[-1:, :, :]
            tgt_mask = None

        return encoder_src, decoder_src, tgt_mask

    def forward(self, src, tgt, training=True):
        encoder_src, decoder_src, tgt_mask = self._create_transformer_inputs(src, tgt, training)

        encoder_src = self.encoder(encoder_src) * math.sqrt(self.input_size)
        encoder_src = self.positional_encoding(encoder_src)

        decoder_src = self.encoder(decoder_src) * math.sqrt(self.input_size)
        decoder_src = self.positional_encoding(decoder_src)

        x = self.transformer(encoder_src, decoder_src, None, tgt_mask)
        x = self.decoder(x)
        x = x.permute(1, 0, 2)

        return x


class TransformerModel(TorchForecastingModel):
    @random_method
    def __init__(self,
                 model: nn.Module = None,
                 input_size: int = 1,
                 output_length: int = 1,
                 output_size: int = 1,
                 d_model: int = 512,
                 nhead: int = 8,
                 num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 activation: str = "relu",
                 custom_encoder: Optional[Any] = None,
                 custom_decoder: Optional[Any] = None,
                 random_state: Optional[Union[int, RandomState]] = None,
                 **kwargs):

        """ Transformer Model.

        Parameters
        ----------
        model
            a custom PyTorch module with the same specifications as
            `darts.models.transformer_model._TransformerModule` (default=None).
        input_size
            The dimensionality of the TimeSeries instances that will be fed to the fit function (default=1).
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
            a custom transformer encoder provided by the user (default=None)
        custom_decoder
            a custom transformer decoder provided by the user (default=None)
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

    def predict(self, n: int,
                use_full_output_length: bool = False,
                input_series: Optional[TimeSeries] = None) -> TimeSeries:

        """ Predicts values for a certain number of time steps after the end of the training series

        In the case of univariate training series, `n` can assume any integer value greater than 0.
        If `use_full_output_length` is set to `False`, the model will perform `n` predictions, where in each iteration
        the first predicted value is kept as output while at the same time being fed into the input for
        the next prediction (the first value of the previous input is discarded). This way, the input sequence
        'rolls over' by 1 step for every prediction in 'n'.
        If `use_full_output_length` is set to `True`, the model will predict not one, but `self.output_length` values
        in every iteration. This means that `ceil(n / self.output_length)` iterations will be required. After
        every iteration the input sequence 'rolls over' by `self.output_length` steps, meaning that the last
        `self.output_length` entries in the input sequence will correspond to the prediction of the previous
        iteration. 'use_full_output_length' has to be set to 'False' when using TransformerModel, because transformer
        architectures used with time series do not support inference with 'self.output_length' > 1.

        In the case of multivariate training series, `n` cannot exceed `self.output_length` and `use_full_output_length`
        has to be set to `True`. In this case, only one iteration of predictions will be performed. Multivariate times
        series are currently not supported with TransformerModel.

        Parameters
        ----------
        n
            The number of time steps after the end of the training time series for which to produce predictions
        use_full_output_length
            Boolean value indicating whether or not the full output sequence of the model prediction should be
            used to produce the output of this function. It has to be set to 'False' when using TransformerModel,
            because transformer architectures used with time series do not support inference
            with 'self.output_length' > 1.
        input_series
            Optionally, the input TimeSeries instance fed to the trained TorchForecastingModel to produce the
            prediction. If it is not passed, the training TimeSeries instance will be used as input.

        Returns
        -------
        TimeSeries
            A time series containing the `n` next points, starting after the end of the training time series
        """
        raise_if_not(not use_full_output_length,
                     "'use_full_output_length' is not supported with 'TransformerModel'. Please set it to 'False'",
                     logger)

        return super().predict(n, use_full_output_length=False, input_series=input_series)

    def _create_dataset(self, series):
        return _TimeSeriesSequentialTransformerDataset(series, self.input_length, self.output_length,
                                                       self.target_indices)

    def _train(self,
               train_loader: DataLoader,
               val_loader: Optional[DataLoader],
               tb_writer: Optional[SummaryWriter],
               verbose: bool) -> None:
        """
        Performs the actual training
        :param train_loader: the training data loader feeding the training data and targets
        :param val_loader: optionally, a validation set loader
        :param tb_writer: optionally, a TensorBoard writer
        """

        best_loss = np.inf

        iterator = _build_tqdm_iterator(range(self.n_epochs), verbose)
        for epoch in iterator:
            epoch = epoch
            total_loss = 0
            total_loss_diff = 0

            for batch_idx, (data, target) in enumerate(train_loader):
                self.model.train()
                data, target = data.to(self.device), target.to(self.device)  # TODO: needed if done in dataset?
                # Important change here for transformer: forward pass has two inputs
                # one for the encoder and one for the decoder
                output = self.model(data, target, training=True)
                loss = self.criterion(output, target[:, :, self.target_indices])
                if self.output_length == 1:
                    loss_of_diff = self.criterion(output[1:] - output[:-1],
                                                  target[1:, :, self.target_indices] - target[:-1, :,
                                                                                       self.target_indices])
                else:
                    loss_of_diff = self.criterion(output[:, 1:] - output[:, :-1],
                                                  target[:, 1:, self.target_indices] - target[:, :-1,
                                                                                       self.target_indices])
                loss = loss + loss_of_diff
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                total_loss_diff += loss_of_diff.item()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            if tb_writer is not None:
                for name, param in self.model.named_parameters():
                    tb_writer.add_histogram(name + '/gradients', param.grad.data.cpu().numpy(), epoch)
                tb_writer.add_scalar("training/loss", total_loss / (batch_idx + 1), epoch)
                tb_writer.add_scalar("training/loss_diff", total_loss_diff / (batch_idx + 1), epoch)
                tb_writer.add_scalar("training/loss_total", (total_loss + total_loss_diff) / (batch_idx + 1), epoch)
                tb_writer.add_scalar("training/learning_rate", self._get_learning_rate(), epoch)

            self._save_model(False, _get_checkpoint_folder(self.work_dir, self.model_name), epoch)

            if epoch % self.nr_epochs_val_period == 0:
                training_loss = (total_loss + total_loss_diff) / (batch_idx + 1)  # TODO: do not use batch_idx
                if val_loader is not None:
                    validation_loss = self._evaluate_validation_loss(val_loader)
                    if tb_writer is not None:
                        tb_writer.add_scalar("validation/loss_total", validation_loss, epoch)

                    if validation_loss < best_loss:
                        best_loss = validation_loss
                        self._save_model(True, _get_checkpoint_folder(self.work_dir, self.model_name), epoch)

                    if verbose:
                        print("Training loss: {:.4f}, validation loss: {:.4f}".
                              format(training_loss, validation_loss), end="\r")
                elif verbose:
                    print("Training loss: {:.4f}".format(training_loss), end="\r")

    def _evaluate_validation_loss(self, val_loader: DataLoader):
        total_loss = 0
        total_loss_of_diff = 0
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_loader):
                data, target = data.to(self.device), target.to(self.device)  # TODO: needed?
                # Important modification here for transformer: forward pass has two inputs
                # one for the encoder and one for the decoder
                output = self.model(data, target, training=True)
                loss = self.criterion(output, target[:, :, self.target_indices])
                if self.output_length == 1:
                    loss_of_diff = self.criterion(output[1:] - output[:-1],
                                                  target[1:, :, self.target_indices] - target[:-1, :,
                                                                                       self.target_indices])
                else:
                    loss_of_diff = self.criterion(output[:, 1:] - output[:, :-1],
                                                  target[:, 1:, self.target_indices] - target[:, :-1,
                                                                                       self.target_indices])
                total_loss += loss.item()
                total_loss_of_diff += loss_of_diff.item()

        validation_loss = (total_loss + total_loss_of_diff) / (batch_idx + 1)
        return validation_loss

    def _prepare_tensorboard_writer(self):
        runs_folder = _get_runs_folder(self.work_dir, self.model_name)
        if self.log_tensorboard:
            if self.from_scratch:
                shutil.rmtree(runs_folder, ignore_errors=True)
                tb_writer = SummaryWriter(runs_folder)
                #two inputs need to be saved
                dummy_encoder_input = torch.empty(self.batch_size, self.input_length, self.input_size).to(self.device)
                dummy_decoder_input = torch.empty(self.batch_size, self.output_length, self.output_size).to(self.device)
                tb_writer.add_graph(self.model, (dummy_encoder_input, dummy_decoder_input))
            else:
                tb_writer = SummaryWriter(runs_folder, purge_step=self.start_epoch)
        else:
            tb_writer = None
        return tb_writer

    def _produce_predict_output_with_single_output_steps(self, pred_in, n):
        test_out = []
        for i in range(n):
            #during inference, we set 'training'=False and 'tgt'=None
            out = self.model(pred_in, None, training=False)
            pred_in = pred_in.roll(-1, 1)
            pred_in[:, -1, 0] = out[:, self.first_prediction_index]
            test_out.append(out.cpu().detach().numpy()[0, self.first_prediction_index])
        return test_out
