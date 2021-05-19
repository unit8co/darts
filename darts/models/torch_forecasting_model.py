"""
Torch Forecasting Model Base Class
----------------------------------
This is the super class for all PyTorch-based forecasting models.
"""

import numpy as np
import os
import re
from glob import glob
import shutil
from joblib import Parallel, delayed
from typing import Optional, Dict, Tuple, Union, Sequence
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from ..timeseries import TimeSeries
from ..utils import _build_tqdm_iterator
from ..utils.torch import random_method
from ..utils.data.timeseries_dataset import TimeSeriesInferenceDataset, TrainingDataset
from ..utils.data.sequential_dataset import SequentialDataset
from ..utils.data.simple_inference_dataset import SimpleInferenceDataset
from ..logging import raise_if_not, get_logger, raise_log, raise_if
from .forecasting_model import GlobalForecastingModel

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
    def __init__(self, ts_dataset: Union[TimeSeriesInferenceDataset, TrainingDataset], device):
        """
        Wraps around `TimeSeriesDataset`, in order to provide translation
        from `TimeSeries` to torch tensors and stack target series with covariates when needed.
        Inherits from torch `Dataset`.

        Parameters
        ----------
        ts_dataset
            the `TimeSeriesDataset` or `TrainingDataset` underlying this torch Dataset.
        """
        self.ts_dataset = ts_dataset
        self.device = device

    @staticmethod
    def _cat_with_optional(tsr1: torch.Tensor, tsr2: Optional[torch.Tensor]):
        if tsr2 is None:
            return tsr1
        else:
            return torch.cat([tsr1, tsr2], dim=1)

    def __len__(self):
        return len(self.ts_dataset)

    def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Cast the content of the dataset to torch tensors
        """
        item = self.ts_dataset[idx]

        if len(item) == 2:
            # the dataset contains (input_target, input_covariate) only
            input_tgt = torch.from_numpy(item[0]).float()
            input_cov = torch.from_numpy(item[1]).float() if item[1] is not None else None
            return self._cat_with_optional(input_tgt, input_cov)

        elif len(item) == 3:
            # the dataset contains (input_target, output_target, input_covariate)
            input_tgt, output_tgt = torch.from_numpy(item[0]).float(), torch.from_numpy(item[1]).float()
            input_cov = torch.from_numpy(item[2]).float() if item[2] is not None else None
            return self._cat_with_optional(input_tgt, input_cov), output_tgt

        else:
            raise ValueError('The dataset has to contain tuples of size 2 or 4')


class TorchForecastingModel(GlobalForecastingModel, ABC):
    # TODO: add is_stochastic & reset methods
    def __init__(self,
                 input_chunk_length: int,
                 output_chunk_length: int,
                 batch_size: int = 32,
                 n_epochs: int = 100,
                 optimizer_cls: torch.optim.Optimizer = torch.optim.Adam,
                 optimizer_kwargs: Optional[Dict] = None,
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
        input_chunk_length
            Number of past time steps that are fed to the internal forecasting module.
        output_chunk_length
            Number of time steps to be output by the internal forecasting module.
        batch_size
            Number of time series (input and output sequences) used in each training pass.
        n_epochs
            Number of epochs over which to train the model.
        optimizer_cls
            The PyTorch optimizer class to be used (default: `torch.optim.Adam`).
        optimizer_kwargs
            Optionally, some keyword arguments for the PyTorch optimizer (e.g., `{'lr': 1e-3}`
            for specifying a learning rate). Otherwise the default values of the selected `optimizer_cls`
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

        if torch_device_str is None:
            self.device = self._get_best_torch_device()
        else:
            self.device = torch.device(torch_device_str)

        # We will fill these dynamically, upon first call of fit_from_dataset():
        self.model = None
        self.input_dim = None
        self.output_dim = None

        self.input_chunk_length = input_chunk_length
        self.output_chunk_length = output_chunk_length
        self.log_tensorboard = log_tensorboard
        self.nr_epochs_val_period = nr_epochs_val_period

        self.model_name = model_name
        self.work_dir = work_dir

        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.from_scratch = True  # do we train the model from scratch  # TODO clean this

        # Define the loss function
        self.criterion = loss_fn

        # The tensorboard writer
        self.tb_writer = None

        # Persist optimiser and LR scheduler parameters
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = dict() if optimizer_kwargs is None else optimizer_kwargs
        self.lr_scheduler_cls = lr_scheduler_cls
        self.lr_scheduler_kwargs = dict() if lr_scheduler_kwargs is None else lr_scheduler_kwargs

    def _init_model(self) -> None:
        """
        Init self.model - the torch module of this class, based on examples of input/output tensors (to get the
        sizes right).
        """

        # the tensors have shape (chunk_length, nr_dimensions)
        model = self._create_model(self.input_dim, self.output_dim)
        self.model = model.to(self.device)

        # A utility function to create optimizer and lr scheduler from desired classes
        def _create_from_cls_and_kwargs(cls, kws):
            try:
                return cls(**kws)
            except (TypeError, ValueError) as e:
                raise_log(ValueError('Error when building the optimizer or learning rate scheduler;'
                                     'please check the provided class and arguments'
                                     '\nclass: {}'
                                     '\narguments (kwargs): {}'
                                     '\nerror:\n{}'.format(cls, kws, e)),
                          logger)

        # Create the optimizer and (optionally) the learning rate scheduler
        # we have to create copies because we cannot save model.parameters into object state (not serializable)
        optimizer_kws = {k: v for k, v in self.optimizer_kwargs.items()}
        optimizer_kws['params'] = self.model.parameters()
        self.optimizer = _create_from_cls_and_kwargs(self.optimizer_cls, optimizer_kws)

        if self.lr_scheduler_cls is not None:
            lr_sched_kws = {k: v for k, v in self.lr_scheduler_kwargs.items()}
            lr_sched_kws['optimizer'] = self.optimizer
            self.lr_scheduler = _create_from_cls_and_kwargs(self.lr_scheduler_cls, lr_sched_kws)
        else:
            self.lr_scheduler = None  # We won't use a LR scheduler

        self._save_untrained_model(_get_untrained_models_folder(self.work_dir, self.model_name))

    @abstractmethod
    def _create_model(self, input_dim: int, output_dim: int) -> torch.nn.Module:
        """
        This method has to be implemented by all children. It is in charge of instantiating the actual torch model,
        based on examples input/output tensors (i.e. implement a model with the right input/output sizes).
        """
        pass

    def _build_train_dataset(self,
                             target: Sequence[TimeSeries],
                             covariates: Optional[Sequence[TimeSeries]]) -> TrainingDataset:
        return SequentialDataset(target_series=target,
                                 covariates=covariates,
                                 input_chunk_length=self.input_chunk_length,
                                 output_chunk_length=self.output_chunk_length)

    @random_method
    def fit(self,
            series: Union[TimeSeries, Sequence[TimeSeries]],
            covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
            val_series: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
            val_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
            verbose: bool = False) -> None:
        """
        The fit method for torch models.
        It wraps around `fit_from_dataset()`.

        *** Currently future covariates are not yet supported ***

        Parameters
        ----------
        series
            A series or sequence of series serving as target (i.e. what the model will be trained to forecast)
        covariates
            Optionally, a series or sequence of series specifying covariates
        val_series
            Optionally, one or a sequence of validation target series, which will be used to compute the validation
            loss throughout training and keep track of the best performing models.
        val_covariates
            Optionally, the covariates corresponding to the validation series (must match `covariates`)
        verbose
            Optionally, whether to print progress.
        """
        super().fit(series, covariates)

        wrap_fn = lambda ts: [ts] if isinstance(ts, TimeSeries) else ts
        series = wrap_fn(series)
        covariates = wrap_fn(covariates)
        val_series = wrap_fn(val_series)
        val_covariates = wrap_fn(val_covariates)
        # TODO - if one covariate is provided, we could repeat it N times for each of the N target series

        # Check that dimensions of train and val set match; on first series only
        if val_series is not None:
            train_set_dim = (series[0].width + (0 if covariates is None else covariates[0].width))
            val_set_dim = (val_series[0].width + (0 if val_covariates is None else val_covariates[0].width))
            raise_if_not(train_set_dim == val_set_dim, 'The dimensions of the series in the training set '
                                                       'and the validation set do not match. {} != {}'.format(
                                                        train_set_dim, val_set_dim))

        train_dataset = self._build_train_dataset(series, covariates)
        val_dataset = self._build_train_dataset(val_series, val_covariates) if val_series is not None else None

        logger.info('Train dataset contains {} samples.'.format(len(train_dataset)))

        self.fit_from_dataset(train_dataset, val_dataset, verbose)

    @random_method
    def fit_from_dataset(self,
                         train_dataset: TrainingDataset,
                         val_dataset: Optional[TrainingDataset] = None,
                         verbose: bool = False) -> None:
        raise_if(len(train_dataset) == 0,
                 'The provided training time series dataset is too short for obtaining even one training point.',
                 logger)
        raise_if(val_dataset is not None and len(val_dataset) == 0,
                 'The provided validation time series dataset is too short for obtaining even one training point.',
                 logger)

        if self.from_scratch:
            shutil.rmtree(_get_checkpoint_folder(self.work_dir, self.model_name), ignore_errors=True)

        torch_train_dataset = TimeSeriesTorchDataset(train_dataset, self.device)
        torch_val_dataset = TimeSeriesTorchDataset(val_dataset, self.device)

        input_dim, output_dim = torch_train_dataset[0][0].shape[1], torch_train_dataset[0][1].shape[1]
        if self.model is None:
            # Build model, based on the dimensions of the first series in the train set.
            self.input_dim, self.output_dim = input_dim, output_dim
            self._init_model()
        else:
            # Check existing model has input/output dim matching what's provided in the training set.
            raise_if_not(input_dim == self.input_dim and output_dim == self.output_dim,
                         'The dimensionality of the series in the training set do not match the dimensionality'
                         'of the series the model has previously been trained on. '
                         'Model input/output dimensions = {}/{}, provided input/ouptput dimensions = {}/{}'.format(
                             self.input_dim, self.output_dim, input_dim, output_dim
                         ))

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
                series: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
                covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
                batch_size: Optional[int] = None,
                verbose: bool = False,
                n_jobs=1
                ) -> Union[TimeSeries, Sequence[TimeSeries]]:
        """
        Predicts values for a certain number of time steps after the end of the training series,
        or after the end of the specified `series`.

        If `n` is larger than the model `output_chunk_length`, the predictions will be computed in an
        auto-regressive way, by iteratively feeding the last `output_chunk_length` forecast points as
        inputs to the model until a forecast of length `n` is obtained. This is at the moment only
        supported when covariates are not used, as this functionality requires future covariates,
        which are not supported yet.

        If some time series in the ``series`` argument have more time steps than the model was trained with,
        only the last ``input_chunk_length`` time steps will be considered.

        Parameters
        ----------
        n
            The number of time steps after the end of the training time series for which to produce predictions
        series
            Optionally, one or several input `TimeSeries`, representing the history of the target series' whose
            future is to be predicted. If specified, the method returns the forecasts of these
            series. Otherwise, the method returns the forecast of the (single) training series.
        covariates
            Optionally, the covariates series needed as inputs for the model. They must match the covariates used
            for training.
        batch_size
            Size of batches during prediction. Defaults to the models `batch_size` value.
        verbose
            Optionally, whether to print progress.
        n_jobs
            The number of jobs to run in parallel. Defaults to `1`. `-1` means using all processors.

        Returns
        -------
        Union[TimeSeries, Sequence[TimeSeries]]
            One or several time series containing the forecasts of `series`, or the forecast of the training series
            if `series` is not specified and the model has been trained on a single series.
        """
        super().predict(n, series, covariates)

        raise_if(covariates is not None and n > self.output_chunk_length,
                 'The horizon `n` must be smaller or equal to the model output length when covariates are used. '
                 'n: {}, output_chunk_length: {}'.format(n, self.output_chunk_length))

        if series is None:
            series = self.training_series

        if covariates is None and self.covariate_series is not None:
            covariates = self.covariate_series

        called_with_single_series = False
        if isinstance(series, TimeSeries):
            called_with_single_series = True
            series = [series]

        covariates = [covariates] if isinstance(covariates, TimeSeries) else covariates

        dataset = SimpleInferenceDataset(series, covariates)
        predictions = self.predict_from_dataset(n, dataset, verbose=verbose, batch_size=batch_size, n_jobs=n_jobs)
        return predictions[0] if called_with_single_series else predictions

    def predict_from_dataset(self,
                             n: int,
                             input_series_dataset: TimeSeriesInferenceDataset,
                             batch_size: Optional[int] = None,
                             verbose: bool = False,
                             n_jobs=1,
                             ) -> Sequence[TimeSeries]:

        """
        Predicts values for a certain number of time steps after the end of the series appearing in the specified
        ``input_series_dataset``.

        If ``n`` is larger than the model ``output_chunk_length``, the predictions will be computed in an
        auto-regressive way, by iteratively feeding the last ``output_chunk_length`` forecast points as
        inputs to the model until a forecast of length ``n`` is obtained. This is at the moment only
        supported when covariates are not used, as this functionality requires future covariates,
        which are not supported yet.

        If some series in the ``input_series_dataset`` have more time steps than the model was trained with,
        only the last ``input_chunk_length`` time steps will be considered.

        Parameters
        ----------
        n
            The number of time steps after the end of the training time series for which to produce predictions
        input_series_dataset
            Optionally, one or several input `TimeSeries`, representing the history of the target series' whose
            future is to be predicted. If specified, the method returns the forecasts of these
            series. Otherwise, the method returns the forecast of the (single) training series.
        batch_size
            Size of batches during prediction. Defaults to the models `batch_size` value.
        verbose
            Shows the progress bar for batch predicition. Off by default.
        n_jobs
            The number of jobs to run in parallel. Defaults to `1`. `-1` means using all processors.

        Returns
        -------
        Sequence[TimeSeries]
            Returns one or more forecasts for time series.
        """
        self.model.eval()

        # preprocessing
        raise_if_not(isinstance(input_series_dataset, TimeSeriesInferenceDataset),
                     'Only TimeSeriesInferenceDataset is accepted as input type')

        # check that the input sizes match
        sample = input_series_dataset[0]

        in_dim = sum(map(lambda ts: (ts.width if ts is not None else 0), sample))
        raise_if_not(in_dim == self.input_dim,
                     'The dimensionality of the series provided for prediction does not match the dimensionality '
                     'of the series this model has been trained on. Provided input dim = {}, '
                     'model input dim = {}'.format(in_dim, self.input_dim))

        # TODO currently we assume all forecasts fit in memory
        in_tsr_arr = []
        for target_series, covariate_series in input_series_dataset:
            raise_if_not(len(target_series) >= self.input_chunk_length,
                         'All input series must have length >= `input_chunk_length` ({}).'.format(
                self.input_chunk_length))

            # TODO: here we could be smart and handle cases where target and covariates do not have same time axis.
            # TODO: e.g. by taking their latest common timestamp.

            in_tsr_sample = target_series.values(copy=False)[-self.input_chunk_length:]
            in_tsr_sample = torch.from_numpy(in_tsr_sample).float().to(self.device)
            if covariate_series is not None:
                in_cov_tsr = covariate_series.values(copy=False)[-self.input_chunk_length:]
                in_cov_tsr = torch.from_numpy(in_cov_tsr).float().to(self.device)
                in_tsr_sample = torch.cat([in_tsr_sample, in_cov_tsr], dim=1)
            in_tsr_sample = in_tsr_sample.view(1, self.input_chunk_length, -1)

            in_tsr_arr.append(in_tsr_sample)

        # concatenate to one tensor of size [len(input_series_dataset), input_chunk_length, 1 + # of covariates)]
        in_tsr = torch.cat(in_tsr_arr, dim=0)

        # prediction
        pred_loader = DataLoader(in_tsr,
                                 batch_size=batch_size or self.batch_size,
                                 shuffle=False,
                                 num_workers=0,
                                 pin_memory=False,
                                 drop_last=False)
        predictions = []

        iterator = _build_tqdm_iterator(pred_loader, verbose=verbose)

        with torch.no_grad():
            for batch in iterator:
                batch_prediction = []  # (num_batches, n % output_chunk_length)
                out = self.model(batch)[:, self.first_prediction_index:, :]  # (batch_size, output_chunk_length, width)
                batch_prediction.append(out)
                while sum(map(lambda t: t.shape[1], batch_prediction)) < n:
                    roll_size = min(self.output_chunk_length, self.input_chunk_length)
                    batch = torch.roll(batch, -roll_size, 1)
                    batch[:, -roll_size:, :] = out[:, :roll_size, :]
                    # take only last part of the output sequence where needed
                    out = self.model(batch)[:, self.first_prediction_index:, :]
                    batch_prediction.append(out)

                batch_prediction = torch.cat(batch_prediction, dim=1)
                batch_prediction = batch_prediction[:, :n, :]
                batch_prediction = batch_prediction.cpu().detach().numpy()
                
                ts_forecasts = Parallel(n_jobs=n_jobs)(delayed(self._build_forecast_series)(prediction, input_series[0])
                                                       for prediction, input_series in zip(batch_prediction,
                                                                                           input_series_dataset))
                
                predictions.extend(ts_forecasts)

        return predictions

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

            for batch_idx, (data, target) in enumerate(train_loader):
                self.model.train()
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            if tb_writer is not None:
                for name, param in self.model.named_parameters():
                    tb_writer.add_histogram(name + '/gradients', param.grad.data.cpu().numpy(), epoch)
                tb_writer.add_scalar("training/loss", total_loss / (batch_idx + 1), epoch)
                tb_writer.add_scalar("training/loss_total", total_loss / (batch_idx + 1), epoch)
                tb_writer.add_scalar("training/learning_rate", self._get_learning_rate(), epoch)

            self._save_model(False, _get_checkpoint_folder(self.work_dir, self.model_name), epoch)

            if epoch % self.nr_epochs_val_period == 0:
                training_loss = total_loss / len(train_loader)
                if val_loader is not None:
                    validation_loss = self._evaluate_validation_loss(val_loader)
                    if tb_writer is not None:
                        tb_writer.add_scalar("validation/loss_total", validation_loss, epoch)

                    if validation_loss < best_loss:
                        best_loss = validation_loss
                        self._save_model(True, _get_checkpoint_folder(self.work_dir, self.model_name), epoch)

                    if verbose:
                        print("Training loss: {:.4f}, validation loss: {:.4f}, best val loss: {:.4f}".
                              format(training_loss, validation_loss, best_loss), end="\r")
                elif verbose:
                    print("Training loss: {:.4f}".format(training_loss), end="\r")

    def _evaluate_validation_loss(self, val_loader: DataLoader):
        total_loss = 0
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                total_loss += loss.item()

        validation_loss = total_loss / (batch_idx + 1)
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

    def _prepare_tensorboard_writer(self):
        runs_folder = _get_runs_folder(self.work_dir, self.model_name)
        if self.log_tensorboard:
            if self.from_scratch:
                shutil.rmtree(runs_folder, ignore_errors=True)
                tb_writer = SummaryWriter(runs_folder)
                dummy_input = torch.empty(self.batch_size, self.input_chunk_length, self.input_dim).to(self.device)
                tb_writer.add_graph(self.model, dummy_input)
            else:
                tb_writer = SummaryWriter(runs_folder, purge_step=self.start_epoch)
        else:
            tb_writer = None
        return tb_writer

    @staticmethod
    def load_from_checkpoint(model_name: str,
                             work_dir: str = None,
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

        if work_dir is None:
            work_dir = os.getcwd()

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
