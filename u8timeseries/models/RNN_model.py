import numpy as np
import os
import re
from glob import glob
import shutil
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import List, Optional, Dict, Union

from ..timeseries import TimeSeries
from ..utils import TimeSeriesDataset1D, build_tqdm_iterator
from ..custom_logging import raise_if_not, get_logger, raise_log
from . import AutoRegressiveModel

CHECKPOINTS_FOLDER = os.path.join('.u8ts', 'checkpoints')
RUNS_FOLDER = os.path.join('.u8ts', 'runs')

logger = get_logger(__name__)


def _get_checkpoint_folder(work_dir, model_name):
    return os.path.join(work_dir, CHECKPOINTS_FOLDER, model_name)


def _get_runs_folder(work_dir, model_name):
    return os.path.join(work_dir, RUNS_FOLDER, model_name)


# TODO add batch norm
class RNNModule(nn.Module):
    def __init__(self,
                 name: str,
                 input_size: int,
                 hidden_dim: int,
                 num_layers: int,
                 output_length: int = 1,
                 num_layers_out_fc: Optional[List] = None,
                 dropout: float = 0.):
        """
        PyTorch `nn.Module` implementing a simple RNN with the specified `name` layer.
        This module combines a PyTorch RNN module, together with a fully connected network, which maps the
        last hidden layers to output of the desired size `output_length`.

        Inputs:
        -------
        * x of shape (batch_size, input_length, input_size): tensor containing the features of the input sequence

        Outputs:
        --------
        * y of shape (batch_size, out_len, 1): tensor containing the (point) prediction at the last time step of the
                                               sequence.

        :param name: The name of the specific PyTorch RNN module ('RNN', 'GRU' or 'LSTM').
        :param input_size: The dimensionality of the input time series.
                           Currently only set to 1 for univariate time series.
        :param output_length: The number of steps to predict in the future.
        :param hidden_dim: The number of features in the hidden state h of the RNN module.
        :param num_layers: The number of recurrent layers.
        :param num_layers_out_fc: A list containing the hidden dimensions of the layers of the fully connected NN.
                                  This network connects the last hidden layer of the PyTorch RNN module to the output.
        :param dropout: The percentage of neurons that are dropped in the non-last RNN layers. Default: 0.
        """

        super(RNNModule, self).__init__()

        # Defining parameters
        self.hidden_dim = hidden_dim
        self.n_layers = num_layers
        num_layers_out_fc = [] if num_layers_out_fc is None else num_layers_out_fc
        self.out_len = output_length
        self.name = name

        # Defining the RNN module
        self.rnn = getattr(nn, name)(input_size, hidden_dim, num_layers, batch_first=True, dropout=dropout)

        # The RNN module is followed by a fully connected layer, which maps the last hidden layer
        # to the output of desired length
        last = hidden_dim
        feats = []
        for feature in num_layers_out_fc + [output_length]:
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
        predictions = predictions.view(batch_size, self.out_len, 1)

        # predictions is of size (batch_size, output_length, 1)
        return predictions


class RNNModel(AutoRegressiveModel):
    def __init__(self,
                 model: Union[str, nn.Module] = 'RNN',
                 output_length: int = 1,
                 input_length: int = 12,
                 hidden_size: int = 25,
                 n_rnn_layers: int = 1,
                 hidden_fc_size: List = None,
                 dropout: float = 0.,
                 batch_size: int = None,
                 n_epochs: int = 800,
                 optimizer_cls: torch.optim.Optimizer = torch.optim.Adam,
                 optimizer_kwargs: Dict = None,
                 lr_scheduler_cls: torch.optim.lr_scheduler._LRScheduler = None,
                 lr_scheduler_kwargs: Optional[Dict] = None,
                 loss_fn: nn.modules.loss._Loss = nn.MSELoss(),
                 model_name: str = "RNN_run",  # TODO uid
                 work_dir: str = os.getcwd(),
                 log_tensorboard: bool = False,
                 nr_epochs_val_period: int = 10,
                 torch_device_str: Optional[str] = None):
        """
        Implementation of different RNNs for forecasting. It is recommended to scale the time series
        (e.g. by using a [u8timeseries.preprocessing.transformer.Transformer] beforehand.

        :param model: Either a string representing the kind of RNN module ('RNN' for vanilla RNN, 'GRU' or 'LSTM'),
                      or custom PyTorch nn.Module instance, with same inputs/outputs as
                      [u8timeseries.models.RNN_models.RNN].
        :param output_length: Number of time steps to predict.
                              Must be consistent with the module output length if a nn.Module is specified.
        :param input_length: number of previous time stamps taken into account by the model.
        :param hidden_size: size for feature maps for each RNN layer (h_n) (unnecessary if module given)
        :param n_rnn_layers: number of RNN layers (unnecessary if module given)
        :param hidden_fc_size: size of hidden layers for the fully connected part (unnecessary if module given)
        :param dropout: percent of neuron dropped in RNN hidden layers (unnecessary if module given)
        :param batch_size: number of time series (input and output sequences) used in each training pass
        :param n_epochs: number of epochs to train the model
        :param optimizer_cls: the type of the PyTorch optimizer (default: torch.optim.Adam)
        :param optimizer_kwargs: keyword arguments for the PyTorch optimizer.
        :param lr_scheduler_cls: optionally, the type of the PyTorch learning rate scheduler (default: None).
        :param lr_scheduler_kwargs: keyword arguments for the PyTorch learning rate scheduler.
        :param loss_fn: PyTorch loss function used for training (default: torch.nn.MSELoss()).
        :param model_name: name of the model. Used for creating/using the checkpoints and tensorboard directories.
        :param work_dir: Path of the working directory, where to save checkpoints and Tensorboard summaries.
                         (default: current working directory).
        :param log_tensorboard: if True, use Tensorboard to log the different parameters. The logs will be located in:
                       `[work_dir]/.u8timeseries/runs/`.
        :param nr_epochs_val_period: Number of epochs to wait before evaluating the validation loss (if a validation
                                     TimeSeries is passed to the fit() method).
        :param torch_device_str: Optionally, a string indicating the torch device to use. (default: "cuda:0" if a GPU
                                 is available, otherwise "cpu").

        # TODO: add init seed
        # TODO: add is_stochastic & reset methods
        # TODO: transparent support for multivariate time series
        """
        super().__init__()

        if torch_device_str is None:
            self.device = self._get_best_torch_device()
        else:
            self.device = torch.device(torch_device_str)

        self.input_size = 1  # We support only univariate time series currently
        self.output_length = output_length
        self.seq_len = input_length
        self.log_tensorboard = log_tensorboard
        self.nr_epochs_val_period = nr_epochs_val_period

        if model in ['RNN', 'LSTM', 'GRU']:
            hidden_fc_size = [] if hidden_fc_size is None else hidden_fc_size
            self.model = RNNModule(name=model, input_size=self.input_size, hidden_dim=hidden_size,
                                   num_layers=n_rnn_layers, output_length=output_length,
                                   num_layers_out_fc=hidden_fc_size, dropout=dropout)
        else:
            self.model = model
        raise_if_not(isinstance(self.model, nn.Module), '{} is not a valid RNN model.\n Please specify "RNN", "LSTM", '
                     '"GRU", or give your own PyTorch nn.Module'.format(model.__class__.__name__),
                     logger)

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
        """
        :param series: The training time series
        :param val_series: Optionally, a validation time series that will
                           be used to compute validation loss throughout training
        """

        super().fit(series)

        if self.from_scratch:
            shutil.rmtree(_get_checkpoint_folder(self.work_dir, self.model_name), ignore_errors=True)

        if self.batch_size is None:
            self.batch_size = len(series) // 10
            print('No batch size set. Using: {}'.format(self.batch_size))

        # Prepare training data:
        dataset = TimeSeriesDataset1D(series, self.seq_len, self.output_length)
        train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True,
                                  num_workers=0, pin_memory=True, drop_last=True)
        raise_if_not(len(train_loader) > 0,
                     'The provided training time series is too short for obtaining even one training point.',
                     logger)

        # Prepare validation data:
        if val_series is not None:
            val_dataset = TimeSeriesDataset1D(val_series, self.seq_len, self.output_length)
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
        """
        :return: A TimeSeries containing the `n` next points, starting after the end of the training time series.
        """

        super().predict(n)

        scaled_series = self.training_series.values()[-self.seq_len:]
        pred_in = torch.from_numpy(scaled_series).float().view(1, -1, 1).to(self.device)
        test_out = []
        self.model.eval()
        for i in range(n):
            out = self.model(pred_in)
            pred_in = pred_in.roll(-1, 1)
            pred_in[:, -1, :] = out[:, 0]
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

        iterator = build_tqdm_iterator(range(self.n_epochs), verbose)
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
        checklist = sorted(checklist, key=lambda x: float(re.findall('(\d+)', x)[-1]))
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
            checklist = sorted(checklist, key=lambda x: float(re.findall('(\d+)', x)[-1]))
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

        :param model_name: the name of the model (used to retrieve the checkpoints folder's name)
        :param work_dir: working directory (containing the checkpoints folder). Defaults to CWD.
        :param filename: the name of the checkpoint file. If None, use the most recent one.
        :param best: if True, will retrieve the best model instead of the most recent one.
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
