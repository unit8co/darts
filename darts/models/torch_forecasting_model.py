"""
Torch Forecasting Model
-----------------------
"""

import numpy as np
import os
import re
import math
from glob import glob
import shutil
from typing import Optional, Dict, Tuple, Union
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from ..timeseries import TimeSeries
from ..utils import _build_tqdm_iterator
from ..utils.torch import random_method
from ..utils.data.timeseries_dataset import TimeSeriesDataset
from ..utils.data.sequential_dataset import SequentialDataset
from ..utils.data.simple_dataset import SimpleTimeSeriesDataset
from ..logging import raise_if_not, get_logger, raise_log, raise_if
from .forecasting_model import ForecastingModel

CHECKPOINTS_FOLDER = os.path.join('.darts', 'checkpoints')
RUNS_FOLDER = os.path.join('.darts', 'runs')
UNTRAINED_MODELS_FOLDER = os.path.join('.darts', 'untrained_models')

logger = get_logger(__name__)


def _get_checkpoint_folder(work_dir, model_name):
    return os.path.join(work_dir, CHECKPOINTS_FOLDER, model_name)


def _get_untrained_models_folder(work_dir, model_name):
    return os.path.join(work_dir, UNTRAINED_MODELS_FOLDER, model_name)


def _get_runs_folder(work_dir, model_name):
    return os.path.join(work_dir, RUNS_FOLDER, model_name)


class TimeSeriesTorchDataset(Dataset):
    def __init__(self,
                 ts_dataset: TimeSeriesDataset,
                 device,
                 always_produce_tuples: bool = False,
                 never_produce_tuples: bool = False):
        """
        Wraps around `TimeSeriesDataset`, in order to provide translation
        from `TimeSeries` to torch tensors. Inherits from torch `Dataset`.

        Parameters
        ----------
        ts_dataset
            the `TimeSeriesDataset` underlying this torch Dataset.
        always_produce_tuples
            Whether to always emmit tuples. If True and the underlying `ts_dataset` does not contain tuples,
            this will duplicate the emitted time series into a two-tuple.
            Cannot be set if `never_produce_tuples` is set.
        never_produce_tuples
            Whether to always emmit simple `TimeSeries`. If True and the underlying `ts_dataset` does contain tuples,
            this will only return the first element of the tuple.
            Cannot be set if `always_produce_tuples` is set.
        """
        raise_if(always_produce_tuples and never_produce_tuples,
                 'Only one of `always_produce_tuples` and `never_produce_tuples` can be set.')
        self.ts_dataset = ts_dataset
        self.device = device
        self.always_produce_tuples = always_produce_tuples
        self.never_produce_tuples = never_produce_tuples

    def _ts_to_tensor(self, ts: TimeSeries):
        return torch.from_numpy(ts.values(copy=False)).float().to(self.device)

    def __len__(self):
        return len(self.ts_dataset)

    def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Cast the content of the dataset to torch tensors
        """
        item = self.ts_dataset[idx]
        if isinstance(item, Tuple):
            if self.never_produce_tuples:
                return self._ts_to_tensor(item[0])
            else:
                return (self._ts_to_tensor(item[0]),
                        self._ts_to_tensor(item[1]))
        else:
            if self.always_produce_tuples:
                tsr = self._ts_to_tensor(item)
                return tsr, tsr
            else:
                return self._ts_to_tensor(item)


class TorchForecastingModel(ForecastingModel):
    # TODO: add is_stochastic & reset methods
    def __init__(self,
                 batch_size: int = 32,
                 target_length: int = 1,
                 input_length: int = 10,
                 input_size: int = 1,
                 output_size: int = 1,
                 n_epochs: int = 800,
                 optimizer_cls: torch.optim.Optimizer = torch.optim.Adam,
                 optimizer_kwargs: Dict = None,
                 lr_scheduler_cls: torch.optim.lr_scheduler._LRScheduler = None,
                 lr_scheduler_kwargs: Optional[Dict] = None,
                 loss_fn: nn.modules.loss._Loss = nn.MSELoss(),
                 model_name: str = "torch_model_run",  # TODO: uid
                 work_dir: str = os.getcwd(),
                 log_tensorboard: bool = False,
                 nr_epochs_val_period: int = 10,
                 torch_device_str: Optional[str] = None):

        """ Pytorch-based Forecasting Model.

        This class is meant to be inherited to create a new pytorch-based forecasting module.
        When subclassing this class, please make sure to set the self.model attribute
        in the __init__ function and then call super().__init__ while passing the kwargs.

        Parameters
        ----------
        target_length
            Number of time steps to be output by the forecasting module.
        input_length
            Number of past time steps that are fed to the forecasting module.
        input_size
            The dimensionality of the TimeSeries instances that will be fed to the fit function.
        output_size
            The dimensionality of the output time series.
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
            `[work_dir]/.darts/runs/`.
        nr_epochs_val_period
            Number of epochs to wait before evaluating the validation loss (if a validation
            `TimeSeries` is passed to the `fit()` method).
        torch_device_str
            Optionally, a string indicating the torch device to use. (default: "cuda:0" if a GPU
            is available, otherwise "cpu")
        """
        super().__init__()

        raise_if_not(isinstance(self.model, nn.Module), 'Please make sure that self.model is set to a valid '
                     'nn.Module subclass when subclassing TorchForecastingModel',
                     logger)

        if torch_device_str is None:
            self.device = self._get_best_torch_device()
        else:
            self.device = torch.device(torch_device_str)

        self.input_length = input_length
        self.input_size = input_size
        self.target_length = target_length
        self.output_size = output_size
        self.log_tensorboard = log_tensorboard
        self.nr_epochs_val_period = nr_epochs_val_period

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

        self._save_untrained_model(_get_untrained_models_folder(work_dir, model_name))

    def build_ts_dataset_from_single_series(self, series):
        """ Inherit this method in your model if there is a better default dataset
        """
        return SequentialDataset(series, input_length=self.input_length, target_length=self.target_length)

    @random_method
    def fit(self,
            series: TimeSeries,
            val_series: Optional[TimeSeries] = None,
            verbose: bool = False) -> None:
        """ Fit method for torch modules. This is the entry point to fit the model on one time series only,
            and it wraps around the `multi_fit()` method.
            If you need to fit over several time series, or to differentiate between the input
            and target dimensions of your series, consider building a `TimeSeriesDataset` and calling
            `multi_fit()` instead.

        Parameters
        ----------
        series
            A series to train the model on.
        val_series
            Optionally, a validation training time series, which will be used to compute the validation
            loss throughout training and keep track of the best performing models.
        verbose
            Optionally, whether to print progress.
        """
        super().fit(series)

        raise_if_not(self.training_series.width == self.input_size, "The number of components of the training series "
                     "must be equal to the `input_size` defined when instantiating the current model.", logger)
        raise_if_not(self.training_series.width == self.output_size, "The number of components in the training series "
                     "be equal to the `output_size` defined when instantiating the current model.", logger)

        train_dataset = self.build_ts_dataset_from_single_series(series)
        val_dataset = None if val_series is None else self.build_ts_dataset_from_single_series(val_series)
        self.multi_fit(train_dataset, val_dataset, verbose)

    @random_method
    def multi_fit(self,
                  train_dataset: TimeSeriesDataset,
                  val_dataset: Optional[TimeSeriesDataset] = None,
                  verbose: bool = False) -> None:

        raise_if(len(train_dataset) == 0,
                 'The provided training time series dataset is too short for obtaining even one training point.',
                 logger)
        raise_if(val_dataset is not None and len(val_dataset) == 0,
                 'The provided validation time series dataset is too short for obtaining even one training point.',
                 logger)

        if self.from_scratch:
            shutil.rmtree(_get_checkpoint_folder(self.work_dir, self.model_name), ignore_errors=True)

        torch_train_dataset = TimeSeriesTorchDataset(train_dataset, self.device, always_produce_tuples=True)
        torch_val_dataset = TimeSeriesTorchDataset(val_dataset, self.device, always_produce_tuples=True)

        train_loader = DataLoader(torch_train_dataset,
                                  batch_size=self.batch_size,
                                  shuffle=True,
                                  num_workers=0,
                                  pin_memory=True,
                                  drop_last=True)

        # Prepare validation data
        val_loader = None if val_dataset is None else DataLoader(torch_val_dataset,
                                                                 batch_size=self.batch_size,
                                                                 shuffle=False,
                                                                 num_workers=0,
                                                                 pin_memory=True,
                                                                 drop_last=False)

        # Prepare tensorboard writer
        tb_writer = self._prepare_tensorboard_writer()

        # Train model
        self._train(train_loader, val_loader, tb_writer, verbose)

        # Close tensorboard writer
        if tb_writer is not None:
            tb_writer.flush()
            tb_writer.close()

    def predict(self,
                n: int,
                use_full_target_length: bool = False,
                input_series: Optional[TimeSeries] = None) -> TimeSeries:
        """
        Predicts values for a certain number of time steps after the end of the training series,
        or after the end of a specified `input_series`.

        In the case of univariate training series, `n` can assume any integer value greater than 0.
        If `use_full_target_length` is set to `False`, the model will perform `n` predictions, where in each iteration
        the first predicted value is kept as output while at the same time being fed into the input for
        the next prediction (the first value of the previous input is discarded). This way, the input sequence
        'rolls over' by 1 step for every prediction in 'n'.
        If `use_full_target_length` is set to `True`, the model will predict not one, but `self.target_length` values
        in every iteration. This means that `ceil(n / self.target_length)` iterations will be required. After
        every iteration the input sequence 'rolls over' by `self.target_length` steps, meaning that the last
        `self.target_length` entries in the input sequence will correspond to the prediction of the previous
        iteration.

        In the case of multivariate training series, `n` cannot exceed `self.target_length` and `use_full_target_length`
        has to be set to `True`. In this case, only one forward pass of predictions will be performed.

        Parameters
        ----------
        n
            The number of time steps after the end of the training time series for which to produce predictions
        use_full_target_length
            Boolean value indicating whether or not the full output sequence of the model prediction should be
            used to produce the output of this function.
        input_series
            Optionally, the input TimeSeries instance fed to the trained TorchForecastingModel to produce the
            prediction. If it is not passed, the training TimeSeries instance will be used as input.

        Returns
        -------
        TimeSeries
            A time series containing the `n` next points, starting after the end of the training time series
        """
        super().predict(n)

        raise_if_not(input_series is None or input_series.width == self.training_series.width,
                     "'input_series' must have same width as series used to fit model.", logger)

        raise_if_not(use_full_target_length or self.training_series.width == 1,
                     "Please set 'use_full_target_length' to 'True' and 'n' smaller or equal to 'target_length'"
                     " when using a multivariate TimeSeries instance as input.", logger)

        if input_series is None:
            dataset = SimpleTimeSeriesDataset(self.training_series)
        else:
            dataset = SimpleTimeSeriesDataset(input_series)
        return self.multi_predict(n, dataset)[0]

    def multi_predict(self,
                      n: int,
                      input_series_dataset: TimeSeriesDataset,
                      use_full_target_length: bool = False) -> TimeSeriesDataset:

        self.model.eval()

        # TODO use a torch Dataset and DataLoader for parallel loading and batching
        # TODO also currently we assume all forecasts fit in memory...
        ts_forecasts = []
        for input_ts in input_series_dataset:
            if isinstance(input_ts, Tuple):
                input_ts = input_ts[0]

            raise_if_not(len(input_ts) >= self.input_length,
                         'All input series must have length >= `input_length` ({}).'.format(self.input_length))
            in_sequence = input_ts.values(copy=False)[-self.input_length:]
            in_sequence = torch.from_numpy(in_sequence).float().to(self.device).view(1, self.input_length, -1)

            out_sequence = self._produce_prediction(in_sequence, n, use_full_target_length)

            # translate to numpy
            out_sequence = out_sequence.cpu().detach().numpy()
            # test_out = np.stack(out_seq)
            ts_forecasts.append(self._build_forecast_series(out_sequence.reshape(n, -1),
                                                            input_series=input_ts))
        return SimpleTimeSeriesDataset(ts_forecasts)

    def _produce_prediction(self, in_sequence: torch.Tensor, n: int, use_full_out_length: bool) -> torch.Tensor:
        # produces output tensor for one time series
        # TODO: make it work for batches

        prediction = []  # (length x width)
        if not use_full_out_length:
            for i in range(n):
                out = self.model(in_sequence)  # (1, length, width)
                in_sequence = in_sequence.roll(-1, 1)
                in_sequence[:, -1, :] = out[:, self.first_prediction_index, :]
                prediction.append(out[0, self.first_prediction_index, :])  # (0-th batch element, one timestamp, all dims)
        else:
            num_iterations = int(math.ceil(n / self.target_length))
            for i in range(num_iterations):
                out = self.model(in_sequence)  # (1, length, width)
                in_sequence = in_sequence.roll(-self.target_length, 1)
                in_sequence[:, -self.target_length:, :] = out[:, -self.target_length:, :]
                prediction.append(out[0, -self.target_length:, :])  # TODO check
        prediction = torch.cat(prediction)
        prediction = prediction[:n]  # prediction[:n, :]
        return prediction

    def multi_backtest(self):
        # TODO
        raise NotImplementedError()

    def multi_gridsearch(self):
        # TODO
        raise NotImplementedError()

    def untrained_model(self):
        return self._load_untrained_model(_get_untrained_models_folder(self.work_dir, self.model_name))

    @property
    def first_prediction_index(self) -> int:
        """
        Returns the index of the first predicted within the output of self.model.
        """
        return 0

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
                if self.target_length == 1:
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
                if self.target_length == 1:
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
        Saves the whole torch model object to a file

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
            torch.save(self, f)

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

    def _save_untrained_model(self, folder):
        os.makedirs(folder, exist_ok=True)
        filename = os.path.join(folder, 'model.pth.tar')

        with open(filename, 'wb') as f:
            torch.save(self, f)

    def _load_untrained_model(self, folder):
        filename = os.path.join(folder, 'model.pth.tar')

        with open(filename, 'rb') as f:
            model = torch.load(f)
        return model

    def _prepare_validation_data(self, val_training_series, val_target_series):
        val_dataset = self._create_dataset(val_training_series, val_target_series)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False,
                                num_workers=0, pin_memory=True, drop_last=False)
        raise_if_not(len(val_dataset) > 0 and len(val_loader) > 0,
                     'The provided validation time series is too short for this model output length.',
                     logger)
        return val_loader

    def _prepare_tensorboard_writer(self):
        runs_folder = _get_runs_folder(self.work_dir, self.model_name)
        if self.log_tensorboard:
            if self.from_scratch:
                shutil.rmtree(runs_folder, ignore_errors=True)
                tb_writer = SummaryWriter(runs_folder)
                dummy_input = torch.empty(self.batch_size, self.input_length, self.input_size).to(self.device)
                tb_writer.add_graph(self.model, dummy_input)
            else:
                tb_writer = SummaryWriter(runs_folder, purge_step=self.start_epoch)
        else:
            tb_writer = None
        return tb_writer

    @staticmethod
    def load_from_checkpoint(model_name: str,
                             work_dir: str = os.getcwd(),
                             filename: str = None,
                             best: bool = True) -> 'TorchForecastingModel':
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
        TorchForecastingModel
            The corresponding trained `TorchForecastingModel`.
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
            model = torch.load(f)
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
