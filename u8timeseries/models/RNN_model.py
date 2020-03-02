from ..timeseries import TimeSeries
from ..utils import TimeSeriesDataset
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


# TODO add batch norm
class VanillaRNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers, hidden_linear=[], dropout=0):
        super(VanillaRNN, self).__init__()

        self.device = torch.device("cpu")
        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # Defining the layers
        # RNN Layer
        # TODO: should we implement different hiddensize for RNN?
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True, dropout=dropout)
        # Fully connected layer
        last = hidden_dim
        feats = []
        for feature in hidden_linear + [output_size]:
            feats.append(nn.Linear(last, feature))
            last = feature
        self.fc = nn.Sequential(*feats)  # nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        batch_size = x.size(0)

        # Initializing hidden state for first input using method defined below
        hidden = self.init_hidden(batch_size)

        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.rnn(x, hidden)

        # Reshaping the outputs such that it can be fit into the fully connected layer
        # out = out.contiguous().view(self.batch_size, -1, self.hidden_dim)
        hidden = hidden[-1, :, :]
        hidden = self.fc(hidden)

        return hidden

    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        # self.register_buffer("hidden_zero", hidden)
        hidden = hidden.to(self.device)
        return hidden

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.device = args[0]
        return self


class LSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers, hidden_linear=[], dropout=0):
        super(LSTM, self).__init__()

        self.device = torch.device("cpu")
        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # Defining the layers
        # LSTM Layer
        # TODO: should we implement different hiddensize for RNN?
        self.lstm = nn.LSTM(input_size, hidden_dim, n_layers, batch_first=True, dropout=dropout)
        # Fully connected layer
        last = hidden_dim
        feats = []
        for feature in hidden_linear + [output_size]:
            feats.append(nn.Linear(last, feature))
            last = feature
        self.fc = nn.Sequential(*feats)  # nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        batch_size = x.size(0)

        # Initializing hidden state for first input using method defined below
        hidden_cell = self.init_hidden(batch_size)

        # Passing in the input and hidden state into the model and obtaining outputs
        _, hidden_cell = self.lstm(x, hidden_cell)

        # Reshaping the outputs such that it can be fit into the fully connected layer
        # out = out.contiguous().view(self.batch_size, -1, self.hidden_dim)
        hidden = hidden_cell[0][-1, :, :]
        predictions = self.fc(hidden)

        return predictions

    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden = (torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(self.device),
                  torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(self.device))
        return hidden

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.device = args[0]
        return self


class RNNModel(AutoRegressiveModel):
    def __init__(self, model: [str, nn.Module] = 'RNN', input_size: int = 1, output_size: int = 1,
                 sequence_length: int = 12, hidden_size: int = 25, n_rnn_layers: int = 1,
                 hidden_fc_size: list = [], dropout: float = 0., batch_size: int = None, n_epochs: int = 800,
                 scaler: TransformerMixin = MinMaxScaler(feature_range=(0, 1)),
                 loss: nn.modules.loss._Loss = nn.MSELoss(), exp_name: str = "RNN_run", vis_tb: bool = False):
        """
        Implementation of different RNN for forecasting.

        model: kind of RNN module, or custom pytorch module
        input_size: number of features/channels in input (Must be identical to module in size)
        output_size: number of steps to predict (Must be identical to module out size)
        sequence_length: number of previous time stamps taken into account
        hidden_size: size for feature maps for each RNN layer (h_n) (unnecessary if module given)
        n_rnn_layers: number of rnn layers (unnecessary if module given)
        hidden_fc_size: size of hidden layers for the fully connected part (unnecessary if module given)
        dropout: percent of neuron dropped in RNN hidden layers (unnecessary if module given)
        batch_size: number of time series used in each training pass
        n_epochs: number of epochs to train the model
        loss: pytorch loss used during training (default: torch.nn.MSELoss())
        exp_name: name of the save and tensorboard directory
        vis_tb: if True, use tensorboard to log the different parameters
        """
        super().__init__()
        self._torch_device()
        self.scaler = scaler
        self.in_size = input_size
        self.out_size = output_size
        self.seq_len = sequence_length
        self.vis_tb = vis_tb

        if model is 'RNN':
            self.model = VanillaRNN(input_size=input_size, output_size=output_size, hidden_dim=hidden_size,
                                    n_layers=n_rnn_layers, hidden_linear=hidden_fc_size, dropout=dropout)
        elif model is 'LSTM':
            self.model = LSTM(input_size=input_size, output_size=output_size, hidden_dim=hidden_size,
                              n_layers=n_rnn_layers, hidden_linear=hidden_fc_size, dropout=dropout)
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
        self.save_path = os.path.join('./', 'checkpoints', exp_name)
        self.start_epoch = 0
        self.n_epochs = n_epochs
        self.bsize = batch_size
        self.epoch = None
        self.dataset = None
        self.train_loader = None
        self.val_series = None
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

    def fit(self, series: TimeSeries):
        if self.scheduler is None or self.optimizer is None:
            raise AssertionError("optimizer and scheduler must be set to launch the training")
        super().fit(series)
        if self.from_scratch:
            self.scaler.fit(series.values().reshape(-1, 1))
        self.dataset = TimeSeriesDataset(series, self.scaler, self.seq_len, self.out_size)
        if self.bsize is None:
            self.bsize = len(self.dataset)
        self.train_loader = DataLoader(self.dataset, batch_size=self.bsize, shuffle=True,
                                       num_workers=0, pin_memory=True, drop_last=True)

        # Tensorboard
        if self.vis_tb:
            if self.from_scratch:
                shutil.rmtree('runs/{}/'.format(self.exp_name), ignore_errors=True)
                self.writer = SummaryWriter('runs/{}'.format(self.exp_name))
                dummy_input = torch.empty(self.bsize, self.seq_len, self.in_size).to(self.device)
                self.writer.add_graph(self.model, dummy_input)
            else:
                self.writer = SummaryWriter('runs/{}'.format(self.exp_name), purge_step=self.start_epoch)

        if self.val_series is not None:
            val_dataset = TimeSeriesDataset(self.val_series, self.scaler, self.seq_len, self.out_size)
            self.val_loader = DataLoader(val_dataset, batch_size=self.bsize, shuffle=False,
                                         num_workers=0, pin_memory=True, drop_last=False)

        self._train()

        if self.vis_tb:
            self.writer.flush()
            self.writer.close()

    def predict(self, n: int = None):
        if n is None:
            n = self.out_size
        super().predict(n)

        pred_in = self.dataset[len(self.dataset)][0].unsqueeze(0).to(self.device)
        test_out = []
        for i in range(n):
            out = self.model(pred_in)
            pred_in = pred_in.roll(-1, 1)
            pred_in[:, -1, :] = out[:, 0]
            test_out.append(out.cpu().detach().numpy()[0])
        test_out = self.scaler.inverse_transform(np.stack(test_out))
        return self._build_forecast_series(test_out.squeeze())

    def set_val_series(self, val_series: TimeSeries):
        self.val_series = val_series

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
        best_loss = np.inf
        for epoch in tqdm(range(self.start_epoch, self.n_epochs)):
            self.epoch = epoch
            tot_loss_mse = 0
            tot_loss_diff = 0
            is_best = False

            for batch_idx, (data, target) in enumerate(self.train_loader):
                self.model.train()
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss_mse = self.criterion(output, target)
                loss_diff = self.criterion(output[1:] - output[:-1], target[1:] - target[:-1])
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
                self.writer.add_scalar("training_loss", (tot_loss_mse + tot_loss_diff) / (batch_idx + 1), epoch)
                self.writer.add_scalar("training_loss_diff", tot_loss_diff / (batch_idx + 1), epoch)
                self.writer.add_scalar("training_loss_mse", tot_loss_mse / (batch_idx + 1), epoch)
                self.writer.add_scalar("learning_rate", self._get_learning_rate(), epoch)
            print("<Loss>: {:.4f}".format((tot_loss_mse + tot_loss_diff) / (batch_idx + 1)), end="\r")

            if (tot_loss_mse + tot_loss_diff) / (batch_idx + 1) < best_loss:
                best_loss = (tot_loss_mse + tot_loss_diff) / (batch_idx + 1)
                is_best = True
            self._save_model(is_best, self.save_path)

            if epoch % 10 == 0:
                self._val()

    def _val(self):
        if self.val_loader is None:
            return
        tot_loss_mse = 0
        tot_loss_diff = 0
        self.model.eval()
        for batch_idx, (data, target) in enumerate(self.val_loader):
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            loss_mse = self.criterion(output, target)
            loss_diff = self.criterion(output[1:] - output[:-1], target[1:] - target[:-1])
            tot_loss_mse += loss_mse.item()
            tot_loss_diff += loss_diff.item()
        if self.vis_tb:
            self.writer.add_scalar("validation_loss", (tot_loss_mse + tot_loss_diff) / (batch_idx + 1), self.epoch)
            self.writer.add_scalar("validation_loss_diff", tot_loss_diff / (batch_idx + 1), self.epoch)
            self.writer.add_scalar("validation_loss_mse", tot_loss_mse / (batch_idx + 1), self.epoch)
        print("Validation Loss: {:.4f}".format((tot_loss_mse + tot_loss_diff) / (batch_idx + 1)), end="\r")

    def plot_result_train(self):
        self._plot_result(self.dataset)

    def test_series(self, tseries: TimeSeries):
        test_dataset = TimeSeriesDataset(tseries, self.scaler, self.seq_len, self.out_size)
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
        index_p = np.stack([np.arange(self.out_size) + i for i in range(dataset.__len__())])
        index_p = index_p.reshape(-1, self.out_size)
        true_labels_p = self.scaler.inverse_transform(targets)
        predictions_p = self.scaler.inverse_transform(predictions)
        if self.out_size != 1:
            index_p = index_p.T
            true_labels_p = true_labels_p.T
            predictions_p = predictions_p.T
        plt.plot(index_p, true_labels_p, label="true labels")
        if self.out_size != 1:
            plt.figure()
        plt.plot(index_p, predictions_p, label="predictions")
        if self.out_size == 1:
            plt.legend()
        plt.show()
        print("MSE: {:.6f}".format(torch.nn.MSELoss()(torch.from_numpy(predictions_p),
                                                      torch.from_numpy(true_labels_p)).item()))

    def _save_model(self, is_best: bool, save_path: str):
        # maybe move to cpu to save
        # TODO: should we save the dataloader state? (i think not)
        state = {
            'epoch': self.start_epoch + 1,  # state in the loop (why +1 ?)
            'state_dict': self.model.state_dict(),  # model params state
            'optimizer': self.optimizer.state_dict(),  # adam optimizer state
            'scheduler': self.scheduler.state_dict(),  # learning rate state  ## state_dict?
            #  maybe use multistepLR with milestones. Seems rather easy and what i search
            'scaler': self.scaler
        }
        checklist = glob(os.path.join(save_path, "checkpoint_*"))
        checklist = sorted(checklist, key=lambda x: float(re.findall('(\d+)', x)[0]))
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
            checklist = sorted(checklist, key=lambda x: float(re.findall('(\d+)', x)[0]))
            if len(checklist) >= 2:
                # remove older files
                for chkpt in checklist[:-1]:
                    os.remove(chkpt)

    def _load_model(self, save_path: str, filename: str):
        if os.path.isfile(save_path + filename):
            print("=> loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(save_path + filename, map_location=self.device)
            self.start_epoch = checkpoint['epoch']
            self.scaler = checkpoint['scaler']
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(filename, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(save_path + filename))
