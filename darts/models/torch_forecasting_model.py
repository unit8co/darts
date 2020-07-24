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
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from typing import Optional, Dict

from ..timeseries import TimeSeries
from ..utils import _build_tqdm_iterator
from ..utils.torch import random_method
from ..logging import raise_if_not, get_logger, raise_log, raise_if
from .forecasting_model import MultivariateForecastingModel

CHECKPOINTS_FOLDER = os.path.join('.darts', 'checkpoints')
RUNS_FOLDER = os.path.join('.darts', 'runs')

logger = get_logger(__name__)


def _get_checkpoint_folder(work_dir, model_name):
    return os.path.join(work_dir, CHECKPOINTS_FOLDER, model_name)


def _get_runs_folder(work_dir, model_name):
    return os.path.join(work_dir, RUNS_FOLDER, model_name)


class _TimeSeriesSequentialDataset(Dataset):

    def __init__(self,
                 covariate_series: TimeSeries,
                 target_series: TimeSeries,
                 data_length: int = 1,
                 target_length: int = 1):
        """
        A PyTorch Dataset from a univariate TimeSeries.
        The Dataset iterates a moving window over the time series. The resulting slices contain `(data, target)`,
        where `data` is a 1-D sub-sequence of length `data_length` and target is the 1-D sub-sequence of length
        `target_length` following it in the time series.

        Parameters
        ----------
        covariate_series
            The time series to be included in the dataset.
        target_series
            The time series used has target.
        data_length
            The length of the training sub-sequences.
        target_length
            The length of the target sub-sequences, starting at the end of the training sub-sequence.
        """

        self.covariate_series_values = covariate_series.values()
        self.target_series_values = target_series.values()

        # self.series = torch.from_numpy(self.series).float()  # not possible to cast in advance
        self.len_series = len(covariate_series)
        self.data_length = len(covariate_series) - 1 if data_length is None else data_length
        self.target_length = target_length

        raise_if_not(self.data_length > 0,
                     "The input sequence length must be positive. It is {}".format(self.data_length),
                     logger)
        raise_if_not(self.target_length > 0,
                     "The output sequence length must be positive. It is {}".format(self.target_length),
                     logger)

    def __len__(self):
        return self.len_series - self.data_length - self.target_length + 1

    def __getitem__(self, index):
        # TODO: Cast to PyTorch tensors on the right device in advance
        idx = index % (self.len_series - self.data_length - self.target_length + 1)
        data = self.covariate_series_values[idx:idx + self.data_length]
        target = self.target_series_values[idx + self.data_length:idx + self.data_length + self.target_length]
        return torch.from_numpy(data).float(), torch.from_numpy(target).float()


class _TimeSeriesShiftedDataset(Dataset):

    def __init__(self,
                 covariate_series: TimeSeries,
                 target_series: TimeSeries,
                 length: int = 3,
                 shift: int = 1):
        """
        A PyTorch Dataset from a univariate TimeSeries.
        The Dataset iterates a moving window over the time series. The resulting slices contain `(data, target)`,
        where `data` and `target` are both 1-D sub-sequences of length `ength`. The sequence contained in
        target is shifted forward by `shift` positions, meaning that `target` contains the last
        `length` - `shift` entries of `data` and then the `shift` following ones.

        Parameters
        ----------
        covariate_series
            The time series to be included in the dataset.
        target_series
            The time series used has target.
        length
            The length of the training and target sub-sequences.
        shift
            The number of positions that the target sequence is shifted forward compared to the training sequence.
        """

        self.covariate_series_values = covariate_series.values()
        self.target_series_values = target_series.values()
        self.len_series = len(covariate_series)
        self.length = len(covariate_series) - 1 if length is None else length
        self.shift = shift

        raise_if_not(self.length > 0,
                     "The input sequence length must be positive. It is {}".format(self.length),
                     logger)

        raise_if_not(self.shift > 0,
                     "The shift value must be positive. It is {}".format(self.length),
                     logger)

    def __len__(self):
        return self.len_series - self.length - self.shift + 1

    def __getitem__(self, index):
        idx = index % self.__len__()
        data = self.covariate_series_values[idx:idx + self.length]
        target = self.target_series_values[idx + self.shift:idx + self.length + self.shift]
        return torch.from_numpy(data).float(), torch.from_numpy(target).float()


class TorchForecastingModel(MultivariateForecastingModel):
    # TODO: add is_stochastic & reset methods
    # TODO: transparent support for multivariate time series
    def __init__(self,
                 batch_size: int = 32,
                 output_length: int = 1,
                 input_length: int = 10,
                 input_size: int = 1,
                 output_size: int = 1,
                 n_epochs: int = 800,
                 optimizer_cls: torch.optim.Optimizer = torch.optim.Adam,
                 optimizer_kwargs: Dict = None,
                 lr_scheduler_cls: torch.optim.lr_scheduler._LRScheduler = None,
                 lr_scheduler_kwargs: Optional[Dict] = None,
                 loss_fn: nn.modules.loss._Loss = nn.MSELoss(),
                 model_name: str = "torch_model_run",  # TODO uid
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
        output_length
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
            `[work_dir]/.u8timeseries/runs/`.
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
        self.output_length = output_length
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

    @random_method
    def fit(self,
            covariate_series: TimeSeries,
            target_series: Optional[TimeSeries] = None,
            val_covariate_series: Optional[TimeSeries] = None,
            val_target_series: Optional[TimeSeries] = None,
            verbose: bool = False) -> None:
        """ Fit method for torch modules

        Parameters
        ----------
        covariate_series
            A series of covariate values used to predict the target series (can also include the target).
        target_series
            A series of target (dependent) value(s) predicted using the covariate_series if None specified, use the
            covariate series as target.
        val_covariate_series
            Optionally, a validation covariate time series, which will be used to compute the validation
            loss throughout training and keep track of the best performing models.
        val_target_series
            Optionally, a validation target time series, which will be used to compute the validation
            loss throughout training and keep track of the best performing models.
        verbose
            Optionally, whether to print progress.
        target_indices
            A list of integers indicating which component(s) of the time series should be used
            as targets for forecasting.
        """
        super().fit(covariate_series, target_series)

        raise_if_not(self.covariate_series.width == self.input_size, "The number of components of the covariate series "
                     "must be equal to the `input_size` defined when instantiating the current model.", logger)
        raise_if_not(self.target_series.width == self.output_size, "The number of components in the target series must "
                     "be equal to the `output_size` defined when instantiating the current model.", logger)

        # perform checks on the validation series
        raise_if(val_covariate_series is None and val_target_series is not None, "`val_target_series` can not be "
                 "specified without a `val_covariate_series`.")
        if val_covariate_series is not None:
            val_covariate_series, val_target_series = self._make_fitable_series(val_covariate_series, val_target_series)
            raise_if_not(self.covariate_series.width == val_covariate_series.width, "covariate_series must have the "
                         "same number of component(s) as val_covariate_series.", logger)
            raise_if_not(self.target_series.width == val_target_series.width, "target_series must have the same number "
                         "of component(s) as val_target_series.", logger)

        if self.from_scratch:
            shutil.rmtree(_get_checkpoint_folder(self.work_dir, self.model_name), ignore_errors=True)

        # Prepare training data
        dataset = self._create_dataset(self.covariate_series, self.target_series)
        train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=0, pin_memory=True,
                                  drop_last=True)
        raise_if_not(len(train_loader) > 0,
                     'The provided training time series is too short for obtaining even one training point.',
                     logger)

        # Prepare validation data
        val_loader = None if val_covariate_series is None else self._prepare_validation_data(val_covariate_series,
                                                                                             val_target_series)

        # Prepare tensorboard writer
        tb_writer = self._prepare_tensorboard_writer()

        # Train model
        self._train(train_loader, val_loader, tb_writer, verbose)

        # Close tensorboard writer
        if tb_writer is not None:
            tb_writer.flush()
            tb_writer.close()

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
        iteration.

        In the case of multivariate training series, `n` cannot exceed `self.output_length` and `use_full_output_length`
        has to be set to `True`. In this case, only one iteration of predictions will be performed.

        Parameters
        ----------
        n
            The number of time steps after the end of the training time series for which to produce predictions
        use_full_output_length
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

        raise_if_not(input_series is None or input_series.width == self.covariate_series.width,
                     "'input_series' must have same width as series used to fit model.", logger)

        raise_if_not(use_full_output_length or self.covariate_series.width == 1, "Please set 'use_full_output_length'"
                     " to 'True' and 'n' smaller or equal to 'output_length' when using a multivariate"
                     "TimeSeries instance as input.", logger)

        # create input sequence for prediction
        pred_in = self._create_predict_input(input_series)

        # produce output
        self.model.eval()
        if (use_full_output_length):
            test_out = self._produce_predict_output_with_full_output_length(pred_in, n)
        else:
            test_out = self._produce_predict_output_with_single_output_steps(pred_in, n)

        test_out = np.stack(test_out)
        return self._build_forecast_series(test_out.reshape(n, -1))

    @property
    def first_prediction_index(self) -> int:
        """
        Returns the index of the first predicted within the output of self.model.
        """
        return 0

    def _create_dataset(self, covariate_series, target_series):
        return _TimeSeriesSequentialDataset(covariate_series, target_series, self.input_length, self.output_length)

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

    def _prepare_validation_data(self, val_covariate_series, val_target_series):
        val_dataset = self._create_dataset(val_covariate_series, val_target_series)
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

    def _create_predict_input(self, input_series):
        if input_series is None:
            input_sequence = self.covariate_series.values()[-self.input_length:]
        else:
            raise_if_not(len(input_series) >= self.input_length,
                         "'input_series' must at least be as long as 'self.input_length'", logger)
            input_sequence = input_series.values()[-self.input_length:]
            super().fit(input_series)
        pred_in = torch.from_numpy(input_sequence).float().view(1, self.input_length, -1).to(self.device)

        return pred_in

    def _produce_predict_output_with_full_output_length(self, pred_in, n):
        test_out = []
        num_iterations = int(math.ceil(n / self.output_length))
        for i in range(num_iterations):
            raise_if_not(self.covariate_series.width == 1 or num_iterations == 1, "Only univariate time series"
                         " support predictions with n > output_length", logger)
            out = self.model(pred_in)
            if (num_iterations > 1):
                pred_in = pred_in.roll(-self.output_length, 1)
                pred_in[:, -self.output_length:, 0] = out[:, -self.output_length:].view(1, -1)
            test_out.append(out.cpu().detach().numpy()[:, -self.output_length:])
        test_out = np.concatenate(test_out, axis=1)
        test_out = test_out[:, :n, :]
        return test_out

    def _produce_predict_output_with_single_output_steps(self, pred_in, n):
        test_out = []
        for i in range(n):
            out = self.model(pred_in)
            pred_in = pred_in.roll(-1, 1)
            pred_in[:, -1, 0] = out[:, self.first_prediction_index]
            test_out.append(out.cpu().detach().numpy()[0, self.first_prediction_index])
        return test_out

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
