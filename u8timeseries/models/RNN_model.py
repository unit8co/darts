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
import matplotlib.pyplot as plt
from sklearn.base import TransformerMixin
from sklearn.preprocessing import MinMaxScaler

from tqdm.notebook import tqdm  # add check for different tqdm

from typing import List


# TODO add batch norm
class RNN(nn.Module):
    def __init__(self, name, input_size, output_length, hidden_dim, n_layers, hidden_linear=[], dropout=0,
                 many=False):
        """
        PyTorch nn module implementing a simple RNN with the specified `name` layer.

        :param name: The name of the specific PyTorch RNN layer.
        :param input_size: The number of feature in th time series.
        :param output_length: The number of steps to predict in the future.
        :param hidden_dim: The dimension of the hidden layer.
        :param n_layers: The number of RNN layers.
        :param hidden_linear: A list containing the dimension of the hidden layers of the fully connected NN.
        :param dropout: The percentage of neurons that are dropped in the non-last RNN layers.
        :param many: If True, will compare the output of all time-steps instead of only the last one.
        """
        super(RNN, self).__init__()

        self.device = torch.device("cpu")
        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.out_len = output_length
        self.name = name
        self.many = many
        # self.teach_force = 0
        # if teacher_forcing:
        #     self.teach_force = self.out_len - 1

        # Defining the layers
        # RNN Layer
        # TODO: should we implement different hiddensize for RNN?
        self.rnn = getattr(nn, name)(input_size, hidden_dim, n_layers, batch_first=True, dropout=dropout)
        # Fully connected layer
        last = hidden_dim
        feats = []
        for feature in hidden_linear + [output_length]:
            feats.append(nn.Linear(last, feature))
            last = feature
        self.fc = nn.Sequential(*feats)  # nn.Linear(hidden_dim, output_length)

    def forward(self, x, y=None, epoch=0):
        # data is batch_size X input_length X input_size
        batch_size = x.size(0)

        # Initializing hidden state for first input using method defined below
        hidden = self.init_hidden(batch_size)

        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.rnn(x, hidden)

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

        # initial_pred = predictions

        # for i in range(self.teach_force):  # TODO: still buggy
        #     future = x.roll(-1, 1)
        #     if y is None:
        #         break
        #     if np.random.random() > epoch/600 * 0.5:
        #         future[:, -1, :] = y[:, i].unsqueeze(1)
        #     else:
        #         future[:, -1, :] = initial_pred[:, i].unsqueeze(1)
        #     out, hidden = self.rnn(future, hidden)
        #
        #     if self.many:
        #         predictions = self.fc(out.contiguous().view(-1, self.hidden_dim))
        #         predictions = predictions.view(batch_size, x.size(1), self.out_len)
        #         initial_pred[:, :, i + 1] = predictions[:, :, i]
        #     else:
        #         if self.name == "LSTM":
        #             hidden = hidden[0]
        #         predictions = hidden[-1, :, :]
        #         predictions = self.fc(predictions)
        #         initial_pred[:, i+1] = predictions[:, 0]

        # predictions is batch_size X output_length
        return predictions

    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        if self.name == 'LSTM':
            hidden = (torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(self.device),
                      torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(self.device))
        else:
            hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(self.device)
        # self.register_buffer("hidden_zero", hidden)
        return hidden

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.device = args[0]
        return self


class RNNModel(AutoRegressiveModel):
    def __init__(self, model: [str, nn.Module] = 'RNN', input_size: int = 1, output_length: int = 1,
                 input_length: int = 12, hidden_size: int = 25, n_rnn_layers: int = 1,
                 hidden_fc_size: list = [], dropout: float = 0., batch_size: int = None, n_epochs: int = 800,
                 scaler: TransformerMixin = MinMaxScaler(feature_range=(0, 1)), full: bool = False,
                 loss: nn.modules.loss._Loss = nn.MSELoss(), exp_name: str = "RNN_run", vis_tb: bool = False,
                 work_dir: str = './'):
        """
        Implementation of different RNN for forecasting.

        :param model: kind of RNN module, or custom pytorch module
        :param input_size: number of features/channels in input (Must be identical to module in size)
        :param output_length: number of steps to predict (Must be identical to module out size)
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
        :param work_dir: Path of the current working directory, where to save checkpoints and tensorboard summaries.
        """
        super().__init__()
        self._torch_device()
        self.scaler = scaler
        self.in_size = input_size
        self.out_size = output_length
        self.seq_len = input_length
        self.vis_tb = vis_tb
        self.full = full

        if model in ['RNN', 'LSTM', 'GRU']:
            self.model = RNN(name=model, input_size=input_size, output_length=output_length, hidden_dim=hidden_size,
                             n_layers=n_rnn_layers, hidden_linear=hidden_fc_size, dropout=dropout, many=full)
        elif model is 'Seq2Seq':
            raise NotImplementedError("seq2seq is not yet implemented")
        elif isinstance(model, nn.Module):
            self.model = model
        else:
            raise ValueError('{} is not a possible RNN model.\n Please choose' \
                             ' between "RNN", "LSTM", "Seq2Seq" or give your own nn.Module'.
                             format(model.__class__.__name__))
        self.model = self.model.to(self.device)
        self.exp_name = exp_name
        self.cwd = work_dir
        self.save_path = os.path.join(self.cwd, 'checkpoints', exp_name)
        self.start_epoch = 0
        self.n_epochs = n_epochs
        self.bsize = batch_size
        self.epoch = None
        self.dataset = None
        self.train_loader = None
        self.val_dataset = None
        self.val_loader = None
        self.from_scratch = True  # do we train the model from scratch

        # Define Loss, Optimizer
        self.criterion = loss
        self.optimizer = None
        self.scheduler = None
        self.writer = None
        # self.set_optimizer()
        # self.set_scheduler()

        # self.load_from_checkpoint()

    def fit(self, dataset: torch.utils.data.dataset):
        # TODO: cannot pass only one timeseries. be better to pass a dataset, and and can transform to dataloader
        # TODO: is it better to have a function to construct a dataset from timeseries? it is, in fact, the class
        # TODO: how to incorporate the scaler? add transform function inside dataset? may be a good idea
        if self.scheduler is None or self.optimizer is None:
            raise AssertionError("optimizer and scheduler must be set to launch the training")
        if type(dataset) is TimeSeries or type(dataset) is list:
            self.set_train_dataset(dataset)
            dataset = self.dataset
        super().fit(dataset.series[0])
        self.dataset = dataset
        self.scaler = self.dataset.fit_scaler(self.scaler)
        if self.from_scratch:
            shutil.rmtree(self.cwd+'checkpoints/{}/'.format(self.exp_name), ignore_errors=True)
        #     if not hasattr(self.scaler, "scale_"):
        #         self.scaler.fit(series.values().reshape(-1, 1))
        # self.set_train_dataset(series)
        if self.bsize is None:
            self.bsize = len(self.dataset)
        self.train_loader = DataLoader(self.dataset, batch_size=self.bsize, shuffle=True,
                                       num_workers=0, pin_memory=True, drop_last=True)

        # Tensorboard
        if self.vis_tb:
            if self.from_scratch:
                shutil.rmtree(self.cwd+'runs/{}/'.format(self.exp_name), ignore_errors=True)
                self.writer = SummaryWriter(self.cwd+'runs/{}'.format(self.exp_name))
                dummy_input = torch.empty(self.bsize, self.seq_len, self.in_size).to(self.device)
                self.writer.add_graph(self.model, dummy_input)
            else:
                self.writer = SummaryWriter(self.cwd+'runs/{}'.format(self.exp_name), purge_step=self.start_epoch)

        if self.val_dataset is not None:
            self.val_dataset.transform(self.scaler)
            self.val_loader = DataLoader(self.val_dataset, batch_size=self.bsize, shuffle=False,
                                         num_workers=0, pin_memory=True, drop_last=False)

        self._train()

        if self.vis_tb:
            self.writer.flush()
            self.writer.close()

    def predict(self, series: 'TimeSeries' = None, n: int = None, is_best: bool = False):
        # TODO: merge the different functions
        if n is None:
            return self.true_predict(series, is_best)
        self.load_from_checkpoint(is_best=is_best)
        if series is None:
            series = self.training_series
        else:
            self.training_series = series
        super().predict(n)

        scaled_series = self.scaler.transform(series.values()[-self.seq_len:].reshape(-1, 1))
        pred_in = torch.from_numpy(scaled_series).float().view(1, -1, 1).to(self.device)
        # try:
        #     pred_in = self.dataset[len(self.dataset)-1+self.out_size][0].unsqueeze(0).to(self.device)
        # except TypeError:
        #     raise AssertionError("Must call set_train_dataset before predict if the model is loaded from checkpoint")
        test_out = []
        self.model.eval()
        for i in range(n):
            out = self.model(pred_in)
            if self.full:
                out = out[:, -1, :]
            pred_in = pred_in.roll(-1, 1)
            pred_in[:, -1, :] = out[:, 0]
            test_out.append(out.cpu().detach().numpy()[0, 0])
        test_out = self.scaler.inverse_transform(np.stack(test_out).reshape(-1, 1))
        return self._build_forecast_series(test_out.squeeze())

    def set_val_series(self, val_series: List[TimeSeries]):
        if type(val_series) is not list:
            val_series = [val_series]
        self.val_dataset = TimeSeriesDataset1D(val_series, self.seq_len, self.out_size, full=self.full)

    def set_train_dataset(self, train_series: List[TimeSeries]):
        # todo: can pass a dataset object too
        if type(train_series) is not list:
            train_series = [train_series]
        self.training_series = train_series[0]
        self.dataset = TimeSeriesDataset1D(train_series, self.seq_len, self.out_size, full=self.full)
        self.scaler = self.dataset.fit_scaler(self.scaler)

    def set_val_dataset(self, dataset: torch.utils.data.dataset):
        self.val_dataset = dataset

    def set_optimizer(self, optimizer: torch.optim.Optimizer = torch.optim.Adam, learning_rate: float = 1e-2,
                      swa: bool = False, **kwargs):
        """
        Set the optimizer from pytorch.

        :param optimizer: optimizer from pytorch (default: torch.optim.Adam)
        :param learning_rate: base learning rate value (default: 1e-2)
        :param swa: If True, use Stochastic Weight Averaging
        :param kwargs: Other parameters to pass to optimizer
        """
        self.optimizer = optimizer(self.model.parameters(), lr=learning_rate, **kwargs)
        if swa:
            from torchcontrib.optim import SWA
            self.optimizer = SWA(self.optimizer, swa_start=10, swa_freq=5, swa_lr=learning_rate / 2)
            # TODO change these values? to test

    def set_scheduler(self, learning_rate: torch.optim.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR, **kwargs):
        """
        Set the learning rate scheduler from pytorch

        :param learning_rate: scheduler from pytorch (default: ExponentialLR with gamma = 1)
        :param kwargs: Other parameters to pass to scheduler
        """
        if self.optimizer is None:
            raise AssertionError("The optimizer must be set before the scheduler")
        # default scheduler is not decreasing
        if learning_rate == torch.optim.lr_scheduler.ExponentialLR and 'gamma' not in kwargs.keys():
            kwargs['gamma'] = 1.
        self.scheduler = learning_rate(self.optimizer, **kwargs)

    def load_from_checkpoint(self, checkpoint: str = None, file: str = None, is_best: bool = True):
        """
        Load the model from the given checkpoint.
        Warning: all hyper-parameters must be the same
        if file is not given, will try to restore the most recent checkpoint

        :param checkpoint: path where the checkpoints are stored.
        :param file: the name of the checkpoint file. If None, find the most recent one.
        :param is_best: if True, will retrieve the best model instead of the most recent one.
        """
        if self.scheduler is None or self.optimizer is None:
            raise AssertionError("optimizer and scheduler must be set to load the parameters")
        self.from_scratch = False
        if checkpoint is None:
            checkpoint = self.save_path
        # if filename is none, find most recent file in savepath that is a checkpoint
        if file is None:
            path = os.path.join(checkpoint, "model_best_*" if is_best else "checkpoint_*")
            checklist = glob(path)
            file = max(checklist, key=os.path.getctime)  # latest file
            file = os.path.basename(file)
        self._load_model(checkpoint, file)
        self._fit_called = True

    def _torch_device(self):
        is_cuda = torch.cuda.is_available()
        if is_cuda:
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

    def _get_learning_rate(self):
        for p in self.optimizer.param_groups:
            return p['lr']

    def _train(self):
        self.best_loss = np.inf
        for epoch in tqdm(range(self.start_epoch, self.n_epochs)):
            self.epoch = epoch
            tot_loss_mse = 0
            tot_loss_diff = 0

            for batch_idx, (data, target) in enumerate(self.train_loader):
                self.model.train()
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                # print(target.size(), output.size())
                loss_mse = self.criterion(output, target)
                if self.out_size == 1 and not self.full:
                    loss_diff = self.criterion(output[1:] - output[:-1], target[1:] - target[:-1])
                else:
                    loss_diff = self.criterion(output[:, 1:] - output[:, :-1], target[:, 1:] - target[:, :-1])
                loss = loss_mse + loss_diff
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                tot_loss_mse += loss_mse.item()
                tot_loss_diff += loss_diff.item()
            self.scheduler.step()
            if self.vis_tb:
                for name, param in self.model.named_parameters():
                    self.writer.add_histogram(name + '/gradients', param.grad.data.cpu().numpy(), epoch)
                self.writer.add_scalar("loss/training_loss", (tot_loss_mse + tot_loss_diff) / (batch_idx + 1), epoch)
                self.writer.add_scalar("training/training_loss_diff", tot_loss_diff / (batch_idx + 1), epoch)
                self.writer.add_scalar("training/training_loss", tot_loss_mse / (batch_idx + 1), epoch)
                self.writer.add_scalar("training/learning_rate", self._get_learning_rate(), epoch)
            # print("<Loss>: {:.4f}".format((tot_loss_mse + tot_loss_diff) / (batch_idx + 1)), end="\r")

            self._save_model(False, self.save_path)

            if epoch % 10 == 0:
                val_loss = self._val()
                print("Training loss: {:.4f}, validation loss: {:.4f}".
                      format((tot_loss_mse + tot_loss_diff) / (batch_idx + 1), val_loss), end="\r")

    def _val(self):
        if self.val_loader is None:
            return np.nan
        tot_loss_mse = 0
        tot_loss_diff = 0
        self.model.eval()
        for batch_idx, (data, target) in enumerate(self.val_loader):
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            loss_mse = self.criterion(output, target)
            if self.out_size == 1 and not self.full:
                loss_diff = self.criterion(output[1:] - output[:-1], target[1:] - target[:-1])
            else:
                loss_diff = self.criterion(output[:, 1:] - output[:, :-1], target[:, 1:] - target[:, :-1])
            tot_loss_mse += loss_mse.item()
            tot_loss_diff += loss_diff.item()
        if self.vis_tb:
            self.writer.add_scalar("loss/validation_loss", (tot_loss_mse + tot_loss_diff) / (batch_idx + 1), self.epoch)
            self.writer.add_scalar("validation/validation_loss_diff", tot_loss_diff / (batch_idx + 1), self.epoch)
            self.writer.add_scalar("validation/validation_loss", tot_loss_mse / (batch_idx + 1), self.epoch)
        # print("               ,Validation Loss: {:.4f}".format((tot_loss_mse + tot_loss_diff) / (batch_idx + 1)),
        #       end="\r")

        if (tot_loss_mse + tot_loss_diff) / (batch_idx + 1) < self.best_loss:
            self.best_loss = (tot_loss_mse + tot_loss_diff) / (batch_idx + 1)
            self._save_model(True, self.save_path)
        return (tot_loss_mse + tot_loss_diff) / (batch_idx + 1)

    def plot_result_train(self):
        self._plot_result(self.dataset)

    def test_series(self, tseries: TimeSeries):
        if issubclass(type(tseries), torch.utils.data.Dataset):
            self._plot_result(tseries)
            return
        if type(tseries) is not list:
            tseries = [tseries]
        test_dataset = TimeSeriesDataset1D(tseries, self.seq_len, self.out_size, full=self.full, scaler=self.scaler)
        test_dataset.transform()
        self._plot_result(test_dataset)

    def _plot_result(self, dataset):
        targets = []
        predictions = []
        data_loader = DataLoader(dataset, batch_size=self.bsize, shuffle=False,
                                 num_workers=0, pin_memory=True, drop_last=False)
        for batch_idx, (data, target) in enumerate(data_loader):
            self.model.eval()
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            targets.append(target.cpu().numpy())
            predictions.append(output.cpu().detach().numpy())
        targets = np.vstack(targets)
        predictions = np.vstack(predictions)

        if self.full:
            index_p = np.arange(dataset.tw)
            true_labels_p = targets[:1, :, 0].T
            predictions_p = predictions[:1, :, 0].T
        else:
            label_per_serie = dataset.len_series - dataset.tw - dataset.lw + 1
            index_p = np.arange(label_per_serie)
            # print(label_per_serie)
            true_labels_p = targets[:label_per_serie, 0:1, 0]
            predictions_p = predictions[:label_per_serie, 0:1, 0]
        # if self.full:
        #     targets = targets[:, -1, :]
        #     predictions = predictions[:, -1, :]
        # index_p = np.stack([np.arange(self.out_size) + i for i in range(dataset.__len__())])
        # index_p = index_p.reshape(-1, self.out_size)
        true_labels_p = self.scaler.inverse_transform(true_labels_p)
        predictions_p = self.scaler.inverse_transform(predictions_p)
        if self.out_size != 1 and not self.full:
        #     index_p = index_p[:, 0]  # .T
            true_labels_p = true_labels_p[:, 0]  # .T
            predictions_p = predictions_p[:, 0]  # .T

        plt.plot(index_p, true_labels_p, label="true labels")
        # if self.out_size != 1:
        #     plt.figure()
        plt.plot(index_p, predictions_p, label="predictions")
        # if self.out_size == 1:
        plt.legend()
        plt.show()
        print("Loss: {:.6f}".format(self.criterion
                                   (torch.from_numpy(self.scaler.inverse_transform(predictions.reshape(-1, 1))),
                                    torch.from_numpy(self.scaler.inverse_transform(targets.reshape(-1, 1)))).item()))

    def _save_model(self, is_best: bool, save_path: str):
        state = {
            'epoch': self.epoch + 1,  # state in the loop (why +1 ?)
            'state_dict': self.model.state_dict(),  # model params state
            'optimizer': self.optimizer.state_dict(),  # adam optimizer state
            'scheduler': self.scheduler.state_dict(),  # learning rate state  ## state_dict?
            #  maybe use multistepLR with milestones. Seems rather easy and what i search
            'scaler': self.scaler
        }
        checklist = glob(os.path.join(save_path, "checkpoint_*"))
        checklist = sorted(checklist, key=lambda x: float(re.findall('(\d+)', x)[-1]))
        filename = 'checkpoint_{0}.pth.tar'
        filename = filename.format(self.epoch)
        os.makedirs(save_path, exist_ok=True)
        filename = os.path.join(save_path, filename)
        torch.save(state, filename)
        if len(checklist) >= 5:
            # remove older files
            for chkpt in checklist[:-4]:
                os.remove(chkpt)
        if is_best:
            best_name = os.path.join(save_path, 'model_best_{0}.pth.tar'.format(self.epoch))
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
            self.scaler = checkpoint['scaler']
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(os.path.join(save_path, filename)))

    def true_predict(self, series: 'TimeSeries', is_best: bool = False):
        # TODO: future deprecation
        self.load_from_checkpoint(is_best=is_best)
        n = self.out_size
        super().predict(n)
        self.training_series = series

        scaled_serie = self.scaler.transform(series.values()[-self.seq_len:].reshape(-1, 1))
        pred_in = torch.from_numpy(scaled_serie).float().view(1, -1, 1).to(self.device)
        # try:
        #     pred_in = self.dataset[len(self.dataset)-self.out_size][0].unsqueeze(0).to(self.device)
        # except TypeError:
        #     raise AssertionError("Must call set_train_dataset before predict if the model is loaded from checkpoint")
        out = self.model(pred_in)
        out = out.cpu().detach().numpy()[0, :, :]
        test_out = self.scaler.inverse_transform(out)
        return self._build_forecast_series(test_out.squeeze())
