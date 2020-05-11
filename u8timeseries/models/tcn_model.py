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
from .rnn_model import _TimeSeriesDataset1DTCN

CHECKPOINTS_FOLDER = os.path.join('.u8ts', 'checkpoints')
RUNS_FOLDER = os.path.join('.u8ts', 'runs')

logger = get_logger(__name__)


def _get_checkpoint_folder(work_dir, model_name):
    return os.path.join(work_dir, CHECKPOINTS_FOLDER, model_name)


def _get_runs_folder(work_dir, model_name):
    return os.path.join(work_dir, RUNS_FOLDER, model_name)


class TCNModule(nn.Module):
    def __init__(self,
                 input_size: int,
                 input_length: int,
                 kernel_size: int,
                 num_filters: int,
                 num_layers: Optional[int],
                 dilation_base: int,
                 output_length: int):

        """ PyTorch module implementing a dilated TCN model used in `RNNModel`.


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

        # If num_layers is not passed, compute number of layers needed for full history coverage
        if (num_layers is None):
            num_layers = math.ceil(math.log((input_length - 1) / (kernel_size - 1), dilation_base))

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

            #TODO: introduce dropout
        
        x = x.transpose(1, 2)

        x = x.view(batch_size, self.input_length, 1)

        return x


class TCNModel(AutoRegressiveModel):
    def __init__(self,
                 input_length: int = 12,
                 kernel_size: int = 3,
                 num_filters: int = 3,
                 num_layers: Optional[int] = None,
                 dilation_base: int = 2,
                 output_length: int = 1,
                 batch_size: int = 32,
                 n_epochs: int = 200,
                 optimizer_cls: torch.optim.Optimizer = torch.optim.Adam,
                 optimizer_kwargs: Dict = None,
                 lr_scheduler_cls: torch.optim.lr_scheduler._LRScheduler = None,
                 lr_scheduler_kwargs: Optional[Dict] = None,
                 loss_fn: nn.modules.loss._Loss = nn.MSELoss(),
                 model_name: str = "TCN_run",  # TODO uid
                 work_dir: str = os.getcwd(),
                 log_tensorboard: bool = False,
                 nr_epochs_val_period: int = 10,
                 torch_device_str: Optional[str] = None
                 ):

        """ Temporal Convolutional Network Model (TCN).

        Parameters
        ----------
        output_length
            Number of time steps to be output by the TCN module.
        input_length
            Number of past time steps that are fed to the TCN module.
        kernel_size
            The size of every kernel in a convolutional layer.
        num_filters
            The number of filters in a convolutional layer of the TCN.
        dilation_base
            The base of the exponent that will determine the dilation on every level.
        num_layers
            The number of convolutional layers.
        batch_size
            Number of time series (input and output sequences) used in each training pass.
        n_epochs
            Number of epochs over which to train the model.
        optimizer_cls
            The PyTorch optimizer class to be used (default: `torch.optim.Adam`).
        optimizer_kwargs
            Optionally, some keyword arguments for the PyTorch optimizer (e.g., `{'lr': 1e-3}`)
            for specifying a learning rate. Otherwise the default values of the selected `optimizer_cls`
            will be used.
        lr_scheduler_cls
            Optionally, the PyTorch learning rate scheduler class to be used. Specifying `None` corresponds
            to using a constant learning rate.
        lr_scheduler_kwargs
            Optionally, some keyword arguments for the PyTorch optimizer.
        loss_fn
            PyTorch loss function used for training (default: `torch.nn.MSELoss()`).
        model_name
            Name of the model. Used for creating the checkpoints and saving tensorboard data.
        work_dir
            Path of the working directory, where to save checkpoints and Tensorboard summaries.
            (default: current working directory).
        log_tensorboard
            If set, use Tensorboard to log the different parameters. The logs will be located in:
            `[work_dir]/.u8timeseries/runs/`.
        nr_epochs_val_period
            Number of epochs to wait before evaluating the validation loss (if a validation
            `TimeSeries` is passed to the `fit()` method).
        torch_device_str
            Optionally, a string indicating the torch device to use. (default: "cuda:0" if a GPU
            is available, otherwise "cpu")
        """

        raise_if_not(kernel_size < input_length,
                     "The kernel size must be strictly smaller than the input length.", logger)

        super().__init__()

        if torch_device_str is None:
            self.device = self._get_best_torch_device()
        else:
            self.device = torch.device(torch_device_str)

        self.input_size = 1  # We support only univariate time series currently
        self.output_length = 1  # This model only predicts the next time step
        self.seq_len = input_length
        self.log_tensorboard = log_tensorboard
        self.nr_epochs_val_period = nr_epochs_val_period

        self.model = TCNModule(self.input_size, input_length, kernel_size, num_filters,
                               num_layers, dilation_base, output_length)

        self.model = self.model.to(self.device)
        self.model_name = model_name
        self.work_dir = work_dir

        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.from_scratch = True  # do we train the model from scratch

        # Define the loss function
        self.criterion = loss_fn

        # The tensorboard writer
        self.tb_writer = None

        # A utility function to create optimizer and lr scheduler from desired classes
        def _create_from_cls_and_kwargs(cls, kws):
            try:
                instance = cls(**kws)
            except (TypeError, ValueError) as e:
                raise_log(ValueError('Error when building the optimizer or learning rate scheduler;'
                                     'please check the provided class and arguments'
                                     '\nclass: {}'
                                     '\narguments (kwargs): {}'
                                     '\nerror:\n{}'.format(cls, kws, e)),
                          logger)
            return instance

        # Create the optimizer and (optionally) the learning rate scheduler
        if optimizer_kwargs is None:
            optimizer_kwargs = {'lr': 1e-3}  # we do not set this as a default argument to avoid mutable values
        optimizer_kwargs['params'] = self.model.parameters()
        self.optimizer = _create_from_cls_and_kwargs(optimizer_cls, optimizer_kwargs)

        if lr_scheduler_cls is not None:
            lr_scheduler_kwargs['optimizer'] = self.optimizer
            self.lr_scheduler = _create_from_cls_and_kwargs(lr_scheduler_cls, lr_scheduler_kwargs)
        else:
            self.lr_scheduler = None  # We won't use a LR scheduler

    def fit(self,
            series: TimeSeries,
            val_series: Optional[TimeSeries] = None,
            verbose: bool = False) -> None:
        """ Fit method for RNNs
        Parameters
        ----------
        series
            The training time series
        val_series
            Optionally, a validation time series, which will be used to compute the validation loss
            throughout training and keep track of the best performing models.
        verbose
            Optionally, whether to print progress.
        """

        super().fit(series)

        if self.from_scratch:
            shutil.rmtree(_get_checkpoint_folder(self.work_dir, self.model_name), ignore_errors=True)

        # Prepare training data:
        dataset = _TimeSeriesDataset1DTCN(series, self.seq_len)
        train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True,
                                  num_workers=0, pin_memory=True, drop_last=True)
        raise_if_not(len(train_loader) > 0,
                     'The provided training time series is too short for obtaining even one training point.',
                     logger)

        # Prepare validation data:
        if val_series is not None:
            val_dataset = _TimeSeriesDataset1DTCN(val_series, self.seq_len)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False,
                                    num_workers=0, pin_memory=True, drop_last=False)
            raise_if_not(len(val_dataset) > 0 and len(val_loader) > 0,
                         'The provided validation time series is too short for this model output length.',
                         logger)
        else:
            val_loader = None

        # Tensorboard
        runs_folder = _get_runs_folder(self.work_dir, self.model_name)
        if self.log_tensorboard:
            if self.from_scratch:
                shutil.rmtree(runs_folder, ignore_errors=True)
                tb_writer = SummaryWriter(runs_folder)
                dummy_input = torch.empty(self.batch_size, self.seq_len, self.input_size).to(self.device)
                tb_writer.add_graph(self.model, dummy_input)
            else:
                tb_writer = SummaryWriter(runs_folder, purge_step=self.start_epoch)
        else:
            tb_writer = None

        self._train(train_loader, val_loader, tb_writer, verbose)

        if tb_writer is not None:
            tb_writer.flush()
            tb_writer.close()

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
                output = self.model(data)
                loss = self.criterion(output, target)
                if self.output_length == 1:
                    loss_of_diff = self.criterion(output[1:] - output[:-1], target[1:] - target[:-1])
                else:
                    loss_of_diff = self.criterion(output[:, 1:] - output[:, :-1], target[:, 1:] - target[:, :-1])
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
                output = self.model(data)
                loss = self.criterion(output, target)
                if self.output_length == 1:
                    loss_of_diff = self.criterion(output[1:] - output[:-1], target[1:] - target[:-1])
                else:
                    loss_of_diff = self.criterion(output[:, 1:] - output[:, :-1], target[:, 1:] - target[:, :-1])
                total_loss += loss.item()
                total_loss_of_diff += loss_of_diff.item()

        validation_loss = (total_loss + total_loss_of_diff) / (batch_idx + 1)
        return validation_loss

    def _save_model(self,
                    is_best: bool,
                    folder: str,
                    epoch: int):
        """
        Saves the whole RNNModel object to a file
        TODO: shall we try to optimize going through torch.save, which uses uses zip?
        :param is_best: whether the model we're currently saving is the best (on validation set)
        :param folder:
        :param epoch:
        :return:
        """

        checklist = glob(os.path.join(folder, "checkpoint_*"))
        checklist = sorted(checklist, key=lambda x: float(re.findall(r'(\d+)', x)[-1]))
        filename = 'checkpoint_{0}.pth.tar'.format(epoch)
        os.makedirs(folder, exist_ok=True)
        filename = os.path.join(folder, filename)

        with open(filename, 'wb') as f:
            pickle.dump(self, f)

        if len(checklist) >= 5:
            # remove older files
            for chkpt in checklist[:-4]:
                os.remove(chkpt)
        if is_best:
            best_name = os.path.join(folder, 'model_best_{0}.pth.tar'.format(epoch))
            shutil.copyfile(filename, best_name)
            checklist = glob(os.path.join(folder, "model_best_*"))
            checklist = sorted(checklist, key=lambda x: float(re.findall(r'(\d+)', x)[-1]))
            if len(checklist) >= 2:
                # remove older files
                for chkpt in checklist[:-1]:
                    os.remove(chkpt)

    @staticmethod
    def load_from_checkpoint(model_name: str,
                             work_dir: str = os.getcwd(),
                             filename: str = None,
                             best: bool = True) -> 'RNNModel':
        """
        Load the model from the given checkpoint.
        if file is not given, will try to restore the most recent checkpoint.
        Parameters
        ----------
        model_name
            The name of the model (used to retrieve the checkpoints folder's name).
        work_dir
            Working directory (containing the checkpoints folder). Defaults to current working directory.
        filename
            The name of the checkpoint file. If not specified, use the most recent one.
        best
            If set, will retrieve the best model (according to validation loss) instead of the most recent one.
        Returns
        -------
        RNNModel
            The corresponding trained `RNNModel`.
        """

        checkpoint_dir = _get_checkpoint_folder(work_dir, model_name)

        # if filename is none, find most recent file in savepath that is a checkpoint
        if filename is None:
            path = os.path.join(checkpoint_dir, "model_best_*" if best else "checkpoint_*")
            checklist = glob(path)
            if len(checklist) == 0:
                raise_log(FileNotFoundError('There is no file matching prefix {} in {}'.format(
                          "model_best_*" if best else "checkpoint_*", checkpoint_dir)),
                          logger)
            filename = max(checklist, key=os.path.getctime)  # latest file TODO: check case where no files match
            filename = os.path.basename(filename)

        full_fname = os.path.join(checkpoint_dir, filename)
        print('loading {}'.format(filename))
        with open(full_fname, 'rb') as f:
            model = pickle.load(f)
        return model

    def _get_best_torch_device(self):
        is_cuda = torch.cuda.is_available()
        if is_cuda:
            return torch.device("cuda:0")
        else:
            return torch.device("cpu")

    def _get_learning_rate(self):
        for p in self.optimizer.param_groups:
            return p['lr']