from ..timeseries import TimeSeries
from ..utils import TimeSeriesDataset1D
from .. import AutoRegressiveModel

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import os
import re
from glob import glob
import shutil

from tqdm.notebook import tqdm  # add check for different tqdm

from typing import List, Optional, Dict

CHECKPOINTS_FOLDER = os.path.join('.u8timeseries', 'checkpoints')
RUNS_FOLDER = os.path.join('.u8timeseries', 'runs')


# TODO add batch norm
class RNN(nn.Module):
    def __init__(self,
                 name: str,
                 input_size: int,
                 hidden_dim: int,
                 num_layers: int,
                 output_length: int = 1,
                 num_layers_out_fc: Optional[List] = None,
                 dropout: float = 0.,
                 many: bool = False):
        """
        PyTorch nn module implementing a simple RNN with the specified `name` layer.

        :param name: The name of the specific PyTorch RNN layer ('RNN', 'GRU' or 'LSTM').
        :param input_size: The dimensionality of the time series.
        :param output_length: The number of steps to predict in the future.
        :param hidden_dim: The dimensionality (nr of neurons) of the hidden layer.
        :param num_layers: The number of RNN layers.
        :param num_layers_out_fc: A list containing the hidden dimensions of the layers of the fully connected NN.
                                  This network connects the last hidden layer of the RNN module to the output.
        :param dropout: The percentage of neurons that are dropped in the non-last RNN layers.

        :param many: If True, will compare the output of all time-steps instead of only the last one.
                     TODO: clarify
        """
        super(RNN, self).__init__()

        # self.device = torch.device("cpu")

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = num_layers
        num_layers_out_fc = [] if num_layers_out_fc is None else num_layers_out_fc
        self.out_len = output_length
        self.name = name
        self.many = many

        # Defining the RNN module
        self.rnn = getattr(nn, name)(input_size, hidden_dim, num_layers, batch_first=True, dropout=dropout)

        # The RNN module is followed by a fully connected layer, which maps the last hidden layer
        # to the output of desired length
        last = hidden_dim
        feats = []
        for feature in num_layers_out_fc + [output_length]:
            feats.append(nn.Linear(last, feature))
            last = feature
        self.fc = nn.Sequential(*feats)  # nn.Linear(hidden_dim, output_length)

    def forward(self, x, y=None, epoch=0):
        # data is batch_size X input_length X input_size
        batch_size = x.size(0)

        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.rnn(x)

        # Reshaping the outputs such that it can be fit into the fully connected layer
        if self.many:
            predictions = self.fc(out.contiguous().view(-1, self.hidden_dim))
            predictions = predictions.view(batch_size, x.size(1), self.out_len)
        else:
            if self.name == "LSTM":
                hidden = hidden[0]
            predictions = hidden[-1, :, :]
            predictions = self.fc(predictions)
            predictions = predictions.view(batch_size, self.out_len, 1)

        # predictions is of size (batch_size, output_length)
        return predictions


class RNNModel(AutoRegressiveModel):
    def __init__(self,
                 model: [str, nn.Module] = 'RNN',
                 input_size: int = 1,
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
                 exp_name: str = "RNN_run",  # TODO uid
                 vis_tb: bool = False,
                 work_dir: str = os.getcwd(),
                 torch_device_str: Optional[str] = None):
        """
        Implementation of different RNNs for forecasting. This model assumes that the time series has already
        been properly scaled (e.g. by using a [u8timeseries.preprocessing.transformer.Transformer] beforehand.

        :param model: Either a string representing the kind of RNN module ('RNN' for vanilla RNN, 'GRU' or 'LSTM'),
                      or custom PyTorch nn.Module instance.
        :param input_size: The dimensionality of the time series.
                           Must be consistent with the module input size if a nn.Module is specified.
        :param output_length: Number of time steps to predict.
                              Must be consistent with the module output length if a nn.Module is specified.
        :param input_length: number of previous time stamps taken into account
        :param hidden_size: size for feature maps for each RNN layer (h_n) (unnecessary if module given)
        :param n_rnn_layers: number of rnn layers (unnecessary if module given)
        :param hidden_fc_size: size of hidden layers for the fully connected part (unnecessary if module given)
        :param dropout: percent of neuron dropped in RNN hidden layers (unnecessary if module given)
        :param batch_size: number of time series used in each training pass
        :param n_epochs: number of epochs to train the model
        :param loss: pytorch loss used during training (default: torch.nn.MSELoss()).
        :param exp_name: name of the checkpoint and tensorboard directory
        :param vis_tb: if True, use tensorboard to log the different parameters
        :param work_dir: Path of the current working directory, where to save checkpoints and tensorboard summaries
        :param torch_device_str:

        # TODO: add init seed
        # TODO: if mean and/or stdev are wild, print a warning suggesting scaling
        """
        super().__init__()

        if torch_device_str is None:
            self.device = self._get_best_torch_device()
        else:
            self.device = torch.device(torch_device_str)

        self.in_size = input_size
        self.output_length = output_length
        self.seq_len = input_length
        self.vis_tb = vis_tb  # TODO: check if TB is installed here

        if model in ['RNN', 'LSTM', 'GRU']:
            hidden_fc_size = [] if hidden_fc_size is None else hidden_fc_size
            self.model = RNN(name=model, input_size=input_size, hidden_dim=hidden_size,
                             num_layers=n_rnn_layers, output_length=output_length,
                             num_layers_out_fc=hidden_fc_size, dropout=dropout)
        elif isinstance(model, nn.Module):
            self.model = model
        else:
            raise ValueError('{} is not a valid RNN model.\n Please specify' \
                             ' "RNN", "LSTM", "GRU", or give your own PyTorch nn.Module'.
                             format(model.__class__.__name__))
        self.model = self.model.to(self.device)
        self.exp_name = exp_name
        self.cwd = work_dir

        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.from_scratch = True  # do we train the model from scratch

        # Define the loss function
        self.criterion = loss_fn

        # The tensorboard writer
        self.tb_writer = None

        # where to save stuff
        self.checkpoint_folder = os.path.join(self.cwd, CHECKPOINTS_FOLDER, self.exp_name)
        self.runs_folder = os.path.join(self.cwd, RUNS_FOLDER, self.exp_name)

        # A utility function to create optimizer and lr scheduler from desired classes
        def _create_from_cls_and_kwargs(cls, kws):
            try:
                instance = cls(**kws)
            except (TypeError, ValueError) as e:
                raise ValueError('Error when building the optimizer or learning rate scheduler;'
                                 'please check the provided class and arguments'
                                 '\nclass: {}'
                                 '\narguments (kwargs): {}'
                                 '\nerror:\n{}'.format(cls, kws, e))
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
            val_series: Optional[TimeSeries] = None) -> None:
        """
        :param series: The training time series
        :param val_series: Optionally, a validation time series that will
                           be used to compute validation loss throughout training
        :return:

        TODO: also specify number of epochs here?
        """
        # TODO: is it better to have a function to construct a dataset from timeseries? it is, in fact, the class
        # TODO: how to incorporate the scaler? add transform function inside dataset? may be a good idea

        super().fit(series)

        if self.from_scratch:
            shutil.rmtree(self.checkpoint_folder, ignore_errors=True)

        if self.batch_size is None:
            self.batch_size = len(self.dataset)

        # Prepare training data:
        dataset = TimeSeriesDataset1D(series, self.seq_len, self.output_length)
        train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True,
                                  num_workers=0, pin_memory=True, drop_last=True)

        # Prepare validation data:
        if val_series is not None:
            val_dataset = TimeSeriesDataset1D(val_series, self.seq_len, self.output_length)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False,
                                    num_workers=0, pin_memory=True, drop_last=False)
        else:
            val_loader = None

        # Tensorboard
        if self.vis_tb:
            if self.from_scratch:
                shutil.rmtree(self.runs_folder, ignore_errors=True)
                tb_writer = SummaryWriter(self.runs_folder)
                dummy_input = torch.empty(self.batch_size, self.seq_len, self.in_size).to(self.device)
                tb_writer.add_graph(self.model, dummy_input)
            else:
                tb_writer = SummaryWriter(self.runs_folder, purge_step=self.start_epoch)
        else:
            tb_writer = None

        self._train(train_loader, val_loader, tb_writer)

        if tb_writer is not None:
            tb_writer.flush()
            tb_writer.close()

    def predict(self, n: int, use_best: bool = False) -> TimeSeries:
        """
        Produces predictions for n time stamps after the end of the training series
        :param n:
        :param use_best: whether to use the best checkpointed model during training
                        (otherwise will use last checkpoint)
        :return:
        """

        self.load_from_checkpoint(is_best=use_best)

        super().predict(n)

        scaled_series = self.training_series.values()[-self.seq_len:]
        pred_in = torch.from_numpy(scaled_series).float().view(1, -1, 1).to(self.device)
        # try:
        #     pred_in = self.dataset[len(self.dataset)-1+self.out_size][0].unsqueeze(0).to(self.device)
        # except TypeError:
        #     raise AssertionError("Must call set_train_dataset before predict if the model is loaded from checkpoint")
        test_out = []
        self.model.eval()
        for i in range(n):
            out = self.model(pred_in)
            pred_in = pred_in.roll(-1, 1)
            pred_in[:, -1, :] = out[:, 0]
            test_out.append(out.cpu().detach().numpy()[0, 0])
        test_out = np.stack(test_out)

        return self._build_forecast_series(test_out.squeeze())

    # TODO: make this a static method returning a model
    def load_from_checkpoint(self, checkpoint: str = None, file: str = None, is_best: bool = True):
        """
        Load the model from the given checkpoint.
        Warning: all hyper-parameters must be the same
        if file is not given, will try to restore the most recent checkpoint

        :param checkpoint: path where the checkpoints are stored.
        :param file: the name of the checkpoint file. If None, find the most recent one.
        :param is_best: if True, will retrieve the best model instead of the most recent one.
        """
        if self.optimizer is None:
            # TODO: do not require an existing model instance to load (persist everything needed)
            raise AssertionError("optimizer must be set to load the parameters")
        self.from_scratch = False
        if checkpoint is None:
            checkpoint = self.checkpoint_folder
        # if filename is none, find most recent file in savepath that is a checkpoint
        if file is None:
            path = os.path.join(checkpoint, "model_best_*" if is_best else "checkpoint_*")
            checklist = glob(path)
            file = max(checklist, key=os.path.getctime)  # latest file
            file = os.path.basename(file)
        self._load_model(checkpoint, file)
        self._fit_called = True

    def _get_best_torch_device(self):
        is_cuda = torch.cuda.is_available()
        if is_cuda:
            return torch.device("cuda:0")
        else:
            return torch.device("cpu")

    def _get_learning_rate(self):
        for p in self.optimizer.param_groups:
            return p['lr']

    def _train(self,
               train_loader: DataLoader,
               val_loader: Optional[DataLoader],
               tb_writer: Optional[SummaryWriter]) -> None:
        """
        Performs the actual training
        :param train_loader: the training data loader feeding the training data and targets
        :param val_loader: optionally, a validation set loader
        :param tb_writer: optionally, a TensorBoard writer
        :return:
        """

        best_loss = np.inf
        for epoch in tqdm(range(self.n_epochs)):
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

            # TODO: what are these values below?
            # if tb_writer is not None:
            #     for name, param in self.model.named_parameters():
            #         tb_writer.add_histogram(name + '/gradients', param.grad.data.cpu().numpy(), epoch)
            #     tb_writer.add_scalar("loss/training_loss", (total_loss + total_loss_diff) / (batch_idx + 1), epoch)
            #     tb_writer.add_scalar("training/training_loss_diff", total_loss_diff / (batch_idx + 1), epoch)
            #     tb_writer.add_scalar("training/training_loss", total_loss / (batch_idx + 1), epoch)
            #     tb_writer.add_scalar("training/learning_rate", self._get_learning_rate(), epoch)

            self._save_model(False, self.checkpoint_folder, epoch)

            if epoch % 10 == 0:
                training_loss = (total_loss + total_loss_diff) / (batch_idx + 1)  # TODO: do not use batch_idx
                validation_loss = self._evaluate_validation_loss(val_loader, tb_writer)
                print("Training loss: {:.4f}, validation loss: {:.4f}".
                      format(training_loss, validation_loss), end="\r")

                if validation_loss < best_loss:
                    best_loss = validation_loss
                    self._save_model(True, self.checkpoint_folder, epoch)

    def _evaluate_validation_loss(self,
                                  val_loader: DataLoader,
                                  tb_writer: Optional[SummaryWriter]):
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

        # TODO: return this and print only in _train() method
        # if tb_writer is not None:
        #     self.tb_writer.add_scalar("loss/validation_loss", (total_loss + total_loss_of_diff) / (batch_idx + 1), self.epoch)
        #     self.tb_writer.add_scalar("validation/validation_loss_diff", total_loss_of_diff / (batch_idx + 1), self.epoch)
        #     self.tb_writer.add_scalar("validation/validation_loss", total_loss / (batch_idx + 1), self.epoch)

        validation_loss = (total_loss + total_loss_of_diff) / (batch_idx + 1)
        return validation_loss

    def _save_model(self, is_best: bool, save_path: str, epoch: int):
        state = {
            'epoch': epoch + 1,  # state in the loop (why +1 ?)
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': None if self.lr_scheduler is None else self.lr_scheduler.state_dict()
        }
        checklist = glob(os.path.join(save_path, "checkpoint_*"))
        checklist = sorted(checklist, key=lambda x: float(re.findall('(\d+)', x)[-1]))
        filename = 'checkpoint_{0}.pth.tar'.format(epoch)
        os.makedirs(save_path, exist_ok=True)
        filename = os.path.join(save_path, filename)
        torch.save(state, filename)
        if len(checklist) >= 5:
            # remove older files
            for chkpt in checklist[:-4]:
                os.remove(chkpt)
        if is_best:
            best_name = os.path.join(save_path, 'model_best_{0}.pth.tar'.format(epoch))
            shutil.copyfile(filename, best_name)
            checklist = glob(os.path.join(save_path, "model_best_*"))
            checklist = sorted(checklist, key=lambda x: float(re.findall('(\d+)', x)[-1]))
            if len(checklist) >= 2:
                # remove older files
                for chkpt in checklist[:-1]:
                    os.remove(chkpt)

    def _load_model(self, save_path: str, filename: str):
        if os.path.isfile(os.path.join(save_path, filename)):
            checkpoint = torch.load(os.path.join(save_path, filename), map_location=self.device)
            self.start_epoch = checkpoint['epoch']
            if checkpoint['scheduler'] is not None:
                self.lr_scheduler.load_state_dict(checkpoint['scheduler'])
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(os.path.join(save_path, filename)))

    # def plot_result_train(self):
    #     self._plot_result(self.dataset)
    #
    # def test_series(self, tseries: TimeSeries):
    #     if issubclass(type(tseries), torch.utils.data.Dataset):
    #         self._plot_result(tseries)
    #         return
    #     if type(tseries) is not list:
    #         tseries = [tseries]
    #     test_dataset = TimeSeriesDataset1D(tseries, self.seq_len, self.out_size, scaler=self.scaler)
    #     test_dataset.transform()
    #     self._plot_result(test_dataset)

    # def _plot_result(self, dataset):
    #     # TODO: this is in fact a mix of backtesting (on dataset) an plotting
    #     # TODO: this functionality should be written somewhere else
    #
    #     targets = []
    #     predictions = []
    #     data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False,
    #                              num_workers=0, pin_memory=True, drop_last=False)
    #     for batch_idx, (data, target) in enumerate(data_loader):
    #         self.model.eval()
    #         data, target = data.to(self.device), target.to(self.device)
    #         output = self.model(data)
    #         targets.append(target.cpu().numpy())
    #         predictions.append(output.cpu().detach().numpy())
    #     targets = np.vstack(targets)
    #     predictions = np.vstack(predictions)
    #
    #     # TODO: avoid accessing dataset properties here; these should be properties of the present model
    #     label_per_serie = dataset.len_series - dataset.data_length - dataset.target_length + 1
    #     index_p = np.arange(label_per_serie)
    #     # print(label_per_serie)
    #     true_labels_p = targets[:label_per_serie, 0:1, 0]
    #     predictions_p = predictions[:label_per_serie, 0:1, 0]
    #
    #     # index_p = np.stack([np.arange(self.out_size) + i for i in range(dataset.__len__())])
    #     # index_p = index_p.reshape(-1, self.out_size)
    #     true_labels_p = self.scaler.inverse_transform(true_labels_p)
    #     predictions_p = self.scaler.inverse_transform(predictions_p)
    #     if self.out_size != 1:
    #     #     index_p = index_p[:, 0]  # .T
    #         true_labels_p = true_labels_p[:, 0]  # .T
    #         predictions_p = predictions_p[:, 0]  # .T
    #
    #     plt.plot(index_p, true_labels_p, label="true labels")
    #     # if self.out_size != 1:
    #     #     plt.figure()
    #     plt.plot(index_p, predictions_p, label="predictions")
    #     # if self.out_size == 1:
    #     plt.legend()
    #     plt.show()
    #     pshape = predictions.shape
    #     tshape = targets.shape
    #     print("Loss: {:.6f}".format(self.criterion
    #                                 (torch.from_numpy(self.scaler.inverse_transform(predictions.reshape(-1, 1))
    #                                                   .reshape(pshape)),
    #                                  torch.from_numpy(self.scaler.inverse_transform(targets.reshape(-1, 1))
    #                                                   .reshape(tshape))).item()))
