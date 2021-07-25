"""
Torch Forecasting Model Base Classes
------------------------------------
This file contains several abstract classes:

    * TorchForecastingModel is the super-class of all torch (deep learning) darts forecasting models.

    * PastCovariatesTorchModel(TorchForecastingModel) for torch models consuming only past-observed covariates.
    * FutureCovariatesTorchModel(TorchForecastingModel) for torch models consuming only future values of
      future covariates.
    * DualCovariatesTorchModel(TorchForecastingModel) for torch models consuming past and future values of some single
      future covariates.
    * MixedCovariatesTorchModel(TorchForecastingModel) for torch models consuming both past-observed
      as well as past and future values of some future covariates.
    * SplitCovariatesTorchModel(TorchForecastingModel) for torch models consuming past-observed as well as future
      values of some future covariates.

    * RecurrentModel(TorchForecastingModel) is the super-class of all recurrent models.

    * TorchParametricProbabilisticForecastingModel(TorchForecastingModel) is the super-class of all probabilistic torch
      forecasting models.
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
from torch import Tensor
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import datetime

from ..timeseries import TimeSeries
from ..utils import _build_tqdm_iterator
from ..utils.torch import random_method

from ..utils.data.training_dataset import (TrainingDataset,
                                           PastCovariatesTrainingDataset,
                                           FutureCovariatesTrainingDataset,
                                           DualCovariatesTrainingDataset,
                                           MixedCovariatesTrainingDataset,
                                           SplitCovariatesTrainingDataset)
from ..utils.data.inference_dataset import (InferenceDataset,
                                            PastCovariatesInferenceDataset,
                                            FutureCovariatesInferenceDataset,
                                            DualCovariatesInferenceDataset,
                                            MixedCovariatesInferenceDataset,
                                            SplitCovariatesInferenceDataset)
from ..utils.data.sequential_dataset import (PastCovariatesSequentialDataset,
                                             FutureCovariatesSequentialDataset,
                                             DualCovariatesSequentialDataset,
                                             MixedCovariatesSequentialDataset,
                                             SplitCovariatesSequentialDataset)

from ..utils.likelihood_models import LikelihoodModel
from ..logging import raise_if_not, get_logger, raise_log, raise_if
from .forecasting_model import GlobalForecastingModel

DEFAULT_DARTS_FOLDER = '.darts'
CHECKPOINTS_FOLDER = 'checkpoints'
RUNS_FOLDER = 'runs'
UNTRAINED_MODELS_FOLDER = 'untrained_models'

logger = get_logger(__name__)


def _get_checkpoint_folder(work_dir, model_name):
    return os.path.join(work_dir, CHECKPOINTS_FOLDER, model_name)


def _get_untrained_models_folder(work_dir, model_name):
    return os.path.join(work_dir, UNTRAINED_MODELS_FOLDER, model_name)


def _get_runs_folder(work_dir, model_name):
    return os.path.join(work_dir, RUNS_FOLDER, model_name)


class TimeSeriesTorchDataset(Dataset):
    def __init__(self, ts_dataset: Union[InferenceDataset, TrainingDataset]):
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

    @staticmethod
    def _cat_with_optional(tsr1: torch.Tensor, tsr2: Optional[torch.Tensor]):
        if tsr2 is None:
            return tsr1
        else:
            return torch.cat([tsr1, tsr2], dim=1)

    def __len__(self):
        return len(self.ts_dataset)

    def __getitem__(self, idx: int):
        """
        Cast the content of the dataset to torch tensors
        """
        item = self.ts_dataset[idx]

        if isinstance(self.ts_dataset, InferenceDataset):
            # the dataset contains (input_target, input_covariate) only
            past_tgt = torch.from_numpy(item[0].values(copy=False)).float()
            past_cov = torch.from_numpy(item[1].values(copy=False)).float() if item[1] is not None else None
            future_cov = torch.from_numpy(item[2].values(copy=False)).float() if item[2] is not None else None

            if future_cov is not None:
                return self._cat_with_optional(past_tgt, past_cov), future_cov, idx
            else:
                return self._cat_with_optional(past_tgt, past_cov), idx

        elif isinstance(self.ts_dataset, TrainingDataset):
            # the dataset contains (input_target, output_target, input_covariate)
            past_tgt, output_tgt = torch.from_numpy(item[0]).float(), torch.from_numpy(item[1]).float()
            past_cov = torch.from_numpy(item[2]).float() if item[2] is not None else None
            return self._cat_with_optional(past_tgt, past_cov), output_tgt

        else:
            raise ValueError('The dataset must be of type `TrainingDataset` or `InferenceDataset`')


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
                 model_name: str = None,
                 work_dir: str = os.path.join(os.getcwd(), DEFAULT_DARTS_FOLDER),
                 log_tensorboard: bool = False,
                 nr_epochs_val_period: int = 10,
                 torch_device_str: Optional[str] = None,
                 force_reset=False):

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
            PyTorch loss function used for training.
            This parameter will be ignored for probabilistic models if the `likelihood` parameter is specified.
            Default: `torch.nn.MSELoss()`.
        model_name
            Name of the model. Used for creating checkpoints and saving tensorboard data. If not specified,
            defaults to the following string "YYYY-mm-dd_HH:MM:SS_torch_model_run_PID", where the initial part of the
            name is formatted with the local date and time, while PID is the processed ID (preventing models spawned at
            the same time by different processes to share the same model_name). E.g.,
            2021-06-14_09:53:32_torch_model_run_44607.
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
        force_reset
            If set to `True`, any previously-existing model with the same name will be reset (all checkpoints will
            be discarded).
        """
        super().__init__()

        if torch_device_str is None:
            self.device = self._get_best_torch_device()
        else:
            self.device = torch.device(torch_device_str)

        # We will fill these dynamically, upon first call of fit_from_dataset():
        self.model = None
        self.train_sample = None

        self.input_chunk_length = input_chunk_length
        self.output_chunk_length = output_chunk_length
        self.log_tensorboard = log_tensorboard
        self.nr_epochs_val_period = nr_epochs_val_period

        if model_name is None:
            current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S.%f")
            model_name = current_time + "_torch_model_run_" + str(os.getpid())

        self.model_name = model_name
        self.work_dir = work_dir

        self.n_epochs = n_epochs
        self.total_epochs = 0  # 0 means it wasn't trained yet.
        self.batch_size = batch_size

        # Define the loss function
        self.criterion = loss_fn

        # The tensorboard writer
        self.tb_writer = None

        # Persist optimiser and LR scheduler parameters
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = dict() if optimizer_kwargs is None else optimizer_kwargs
        self.lr_scheduler_cls = lr_scheduler_cls
        self.lr_scheduler_kwargs = dict() if lr_scheduler_kwargs is None else lr_scheduler_kwargs

        # by default models are block models (i.e. not recurrent)
        self.is_recurrent = False

        # by default models are deterministic (i.e. not probabilistic)
        self.likelihood = None

        self.force_reset = force_reset
        checkpoints_folder = _get_checkpoint_folder(self.work_dir, self.model_name)
        self.checkpoint_exists = \
            os.path.exists(checkpoints_folder) and len(glob(os.path.join(checkpoints_folder, "checkpoint_*"))) > 0

        if self.checkpoint_exists:
            if self.force_reset:
                self.reset_model()
            else:
                raise AttributeError("You already have model data for the '{}' name. Either load model to continue"
                                     " training or use `force_reset=True` to initialize anyway to start"
                                     " training from scratch and remove all the model data".format(self.model_name)
                                     )

    def reset_model(self):
        """ Resets the model object and removes all the stored data - model, checkpoints and training history.
        """
        shutil.rmtree(_get_checkpoint_folder(self.work_dir, self.model_name), ignore_errors=True)
        shutil.rmtree(_get_runs_folder(self.work_dir, self.model_name), ignore_errors=True)
        shutil.rmtree(_get_untrained_models_folder(self.work_dir, self.model_name), ignore_errors=True)

        self.checkpoint_exists = False
        self.total_epochs = 0
        self.model = None
        self.train_sample = None

    def _init_model(self) -> None:
        """
        Init self.model - the torch module of this class, based on examples of input/output tensors (to get the
        sizes right).
        """

        # the tensors have shape (chunk_length, nr_dimensions)
        model = self._create_model(self.train_sample)
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
    def _create_model(self, train_sample: Tuple[Tensor]) -> torch.nn.Module:
        """
        This method has to be implemented by all children. It is in charge of instantiating the actual torch model,
        based on examples input/output tensors (i.e. implement a model with the right input/output sizes).
        """
        pass

    @abstractmethod
    def _build_train_dataset(self,
                             target: Sequence[TimeSeries],
                             past_covariates: Optional[Sequence[TimeSeries]],
                             future_covariates: Optional[Sequence[TimeSeries]]) -> TrainingDataset:
        """
        Each model must specify the default training dataset to use.
        """
        pass

    @abstractmethod
    def _build_inference_dataset(self,
                                 target: Sequence[TimeSeries],
                                 n: int,
                                 past_covariates: Optional[Sequence[TimeSeries]],
                                 future_covariates: Optional[Sequence[TimeSeries]]) -> InferenceDataset:
        """
        Each model must specify the default training dataset to use.
        """
        pass

    @abstractmethod
    def _verify_train_dataset_type(self, train_dataset: TrainingDataset):
        pass

    @abstractmethod
    def _verify_inference_dataset_type(self, inference_dataset: InferenceDataset):
        pass

    @random_method
    def fit(self,
            series: Union[TimeSeries, Sequence[TimeSeries]],
            past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
            future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
            val_series: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
            val_past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
            val_future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
            verbose: bool = False,
            epochs: int = 0) -> None:
        """
        The fit method for torch models.
        It wraps around `fit_from_dataset()`.

        **Important**: if `epochs=0` (default), running `fit()` or `fit_from_dataset()` removes previously trained model - all it's checkpoints
        and tensorboard data. If you want to train your model for more epochs, set the `epochs` parameter to value
        greater than 0.

        **Note**: If your model wasn't yet trained and you requested to train for more epochs with `epoch` parameter,
        it will be treated as trained for 0 epochs.

        *** Future covariates are not yet supported ***

        Parameters
        ----------
        series
            A series or sequence of series serving as target (i.e. what the model will be trained to forecast)
        past_covariates
            Optionally, a series or sequence of series specifying past-observed covariates
        future_covariates
            Optionally, a series or sequence of series specifying future-known covariates
        val_series
            Optionally, one or a sequence of validation target series, which will be used to compute the validation
            loss throughout training and keep track of the best performing models.
        val_past_covariates
            Optionally, the covariates corresponding to the validation series (must match `covariates`)
        verbose
            Optionally, whether to print progress.
        epochs
            If specified, will train the model for `epochs` (additional) epochs, irrespective of what `n_epochs`
            was provided to the model constructor.
        """
        super().fit(series, past_covariates)

        wrap_fn = lambda ts: [ts] if isinstance(ts, TimeSeries) else ts
        series = wrap_fn(series)
        past_covariates = wrap_fn(past_covariates)
        future_covariates = wrap_fn(future_covariates)
        val_series = wrap_fn(val_series)
        val_past_covariates = wrap_fn(val_past_covariates)
        val_future_covariates = wrap_fn(val_future_covariates)

        # Check that dimensions of train and val set match; on first series only
        if val_series is not None:
            match = (series[0].width == val_series[0].width and
                     (past_covariates[0].width if past_covariates is not None else None) ==
                     (val_past_covariates[0].width if val_past_covariates is not None else None) and
                     (future_covariates[0].width if future_covariates is not None else None) ==
                     (val_future_covariates[0].width if val_future_covariates is not None else None))
            raise_if_not(match, 'The dimensions of the series in the training set '
                                'and the validation set do not match.')

        train_dataset = self._build_train_dataset(series, past_covariates, future_covariates)
        val_dataset = self._build_train_dataset(val_series, val_past_covariates, future_covariates) if val_series is not None else None

        logger.info('Train dataset contains {} samples.'.format(len(train_dataset)))

        self.fit_from_dataset(train_dataset, val_dataset, verbose, epochs)

    @random_method
    def fit_from_dataset(self,
                         train_dataset: TrainingDataset,
                         val_dataset: Optional[TrainingDataset] = None,
                         verbose: bool = False,
                         epochs: int = 0) -> None:

        self._verify_train_dataset_type(train_dataset)
        raise_if(len(train_dataset) == 0,
                 'The provided training time series dataset is too short for obtaining even one training point.',
                 logger)
        raise_if(val_dataset is not None and len(val_dataset) == 0,
                 'The provided validation time series dataset is too short for obtaining even one training point.',
                 logger)

        train_sample = train_dataset[0]
        if self.model is None:
            # Build model, based on the dimensions of the first series in the train set.
            self.train_sample = train_sample
            self._init_model()
        else:
            # Check existing model has input/output dims matching what's provided in the training set.
            raise_if_not(len(train_sample) == len(self.train_sample),
                         'The size of the training set samples (tuples) does not match what the model has been '
                         'previously trained one. Trained on tuples of length {}, received tuples of length {}.'.format(
                             len(self.train_sample), len(train_sample)
                         ))
            raise_if_not((s.shape[1] for s in train_sample) == (s.shape[1] for s in self.train_sample),
                         'The dimensionality of the series in the training set do not match the dimensionality'
                         ' of the series the model has previously been trained on. '
                         'Model input/output dimensions = {}, provided input/ouptput dimensions = {}'.format(
                             (s.shape[1] for s in self.train_sample), (s.shape[1] for s in train_sample)
                         ))

        # Setting drop_last to False makes the model see each sample at least once, and guarantee the presence of at
        # least one batch no matter the chosen batch size
        train_loader = DataLoader(torch_train_dataset,
                                  batch_size=self.batch_size,
                                  shuffle=True,
                                  num_workers=0,
                                  pin_memory=True,
                                  drop_last=False)

        # Prepare validation data
        val_loader = None if val_dataset is None else DataLoader(torch_val_dataset,
                                                                 batch_size=self.batch_size,
                                                                 shuffle=False,
                                                                 num_workers=0,
                                                                 pin_memory=True,
                                                                 drop_last=False)

        # Prepare tensorboard writer
        tb_writer = self._prepare_tensorboard_writer()

        # if user wants to train the model for more epochs, ignore the n_epochs parameter
        train_num_epochs = epochs if epochs > 0 else self.n_epochs

        # Train model
        self._train(train_loader, val_loader, tb_writer, verbose, train_num_epochs)

        # Close tensorboard writer
        if tb_writer is not None:
            tb_writer.flush()
            tb_writer.close()

    @random_method
    def predict(self,
                n: int,
                series: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
                past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
                future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
                batch_size: Optional[int] = None,
                verbose: bool = False,
                n_jobs: int = 1,
                roll_size: Optional[int] = None,
                num_samples: int = 1,
                ) -> Union[TimeSeries, Sequence[TimeSeries]]:
        """
        Predicts values for a certain number of time steps after the end of the training series,
        or after the end of the specified `series`.

        TODO: update doc

        Models relying on past covariates:
        If `n` is larger than the model `output_chunk_length`, the predictions will be computed in an
        auto-regressive way, by iteratively feeding the last `roll_size` forecast points as
        inputs to the model until a forecast of length `n` is obtained. If the model was trained with
        covariates, all of the covariate time series need to have a time index that extends at least
        `n - output_chunk_length` into the future. In other words, if `n` is larger than `output_chunk_length`
        then covariates need to be available in the future.

        Recurrent models:
        All predictions are produced in a recurrent way by taking as input
        - the previous target value, which will be set to the last known target value for the first prediction,
          and for all other predictions it will be set to the previous prediction
        - the previous hidden state
        - the current covariates (if the model was trained with covariates)
        As a result, if covariates were used, `n` covariates have to be available into the future.

        If some time series in the `series` argument have more than `input_chunk_length` time steps,
        only the last `input_chunk_length` time steps will be considered.

        Parameters
        ----------
        n
            The number of time steps after the end of the training time series for which to produce predictions
        series
            Optionally, one or several input `TimeSeries`, representing the history of the target series whose
            future is to be predicted. If specified, the method returns the forecasts of these
            series. Otherwise, the method returns the forecast of the (single) training series.
        past_covariates
            Optionally, the past-observed covariates series needed as inputs for the model.
            They must match the covariates used for training in terms of dimension and type.
        future_covariates
            Optionally, the future-known covariates series needed as inputs for the model.
            They must match the covariates used for training in terms of dimension and type.
        batch_size
            Size of batches during prediction. Defaults to the models `batch_size` value.
        verbose
            Optionally, whether to print progress.
        n_jobs
            The number of jobs to run in parallel. Defaults to `1`. `-1` means using all processors.
        roll_size
            For self-consuming predictions, i.e. `n > self.output_chunk_length`, determines how many
            outputs of the model are fed back into it at every iteration of feeding the predicted target
            (and optionally future covariates) back into the model. If this parameter is not provided,
            it will be set `self.output_chunk_length` by default.
        num_samples
            Number of times a prediction is sampled from a probabilistic model. Should be left set to 1
            for deterministic models.

        Returns
        -------
        Union[TimeSeries, Sequence[TimeSeries]]
            One or several time series containing the forecasts of `series`, or the forecast of the training series
            if `series` is not specified and the model has been trained on a single series.
        """
        super().predict(n, series, past_covariates, future_covariates)

        if series is None:
            raise_if(self.training_series is None, "Input series has to be provided after fitting on multiple series.")
            series = self.training_series

        if past_covariates is None and self.past_covariate_series is not None:
            past_covariates = self.past_covariate_series
        if future_covariates is None and self.future_covariate_series is not None:
            future_covariates = self.future_covariate_series

        called_with_single_series = False
        if isinstance(series, TimeSeries):
            called_with_single_series = True
            series = [series]

        past_covariates = [past_covariates] if isinstance(past_covariates, TimeSeries) else past_covariates
        future_covariates = [future_covariates] if isinstance(future_covariates, TimeSeries) else future_covariates

        # check that the input sizes match
        # TODO: move this check in predict_from_dataset() by checking index 0 of torch dataset
        # in_dim = (0 if covariates is None else covariates[0].width) + series[0].width
        # raise_if_not(in_dim == self.input_dim,
        #              'The dimensionality of the series provided for prediction does not match the dimensionality '
        #              'of the series this model has been trained on. Provided input dim = {}, '
        #              'model input dim = {}'.format(in_dim, self.input_dim))

        dataset = self._build_inference_dataset(target=series,
                                                n=n,
                                                past_covariates=past_covariates,
                                                future_covariates=future_covariates)

        predictions = self.predict_from_dataset(n, dataset, verbose=verbose, batch_size=batch_size, n_jobs=n_jobs,
                                                roll_size=roll_size, num_samples=num_samples)
        return predictions[0] if called_with_single_series else predictions

    def predict_from_dataset(self,
                             n: int,
                             input_series_dataset: InferenceDataset,
                             batch_size: Optional[int] = None,
                             verbose: bool = False,
                             n_jobs: int = 1,
                             roll_size: Optional[int] = None,
                             num_samples: int = 1,
                             ) -> Sequence[TimeSeries]:

        """
        Predicts values for a certain number of time steps after the end of the series appearing in the specified
        `input_series_dataset`.

        TODO: update doc

        Block models:
        If `n` is larger than the model `output_chunk_length`, the predictions will be computed in a
        recurrent way, by iteratively feeding the last `roll_size` forecast points as
        inputs to the model until a forecast of length `n` is obtained. If the model was trained with
        covariates, all of the covariate time series need to have a time index that extends at least
        `n - output_chunk_length` into the future. In other words, if `n` is larger than `output_chunk_length`
        then covariates need to be available in the future.

        Recurrent models:
        All predictions are produced in a recurrent way by taking as input
        - the previous target value, which will be set to the last known target value for the first prediction,
          and for all other predictions it will be set to the previous prediction
        - the previous hidden state
        - the current covariates (if the model was trained with covariates)
        As a result, if covariates were used, `n` covariates have to be available into the future.

        If some series in the `input_series_dataset` have more time steps than `input_chunk_length`,
        only the last `input_chunk_length` time steps will be considered.

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
        roll_size
            For self-consuming predictions, i.e. `n > self.output_chunk_length`, determines how many
            outputs of the model are fed back into it at every iteration of feeding the predicted target
            (and optionally future covariates) back into the model. If this parameter is not provided,
            it will be set `self.output_chunk_length` by default.
        num_samples
            Number of times a prediction is sampled from a probabilistic model. Should be left set to 1
            for deterministic models.

        Returns
        -------
        Sequence[TimeSeries]
            Returns one or more forecasts for time series.
        """
        self._verify_inference_dataset_type(input_series_dataset)

        self.model.eval()

        if roll_size is None:
            roll_size = self.output_chunk_length
        else:
            raise_if_not(0 < roll_size <= self.output_chunk_length,
                         '`roll_size` must be an integer between 1 and `self.output_chunk_length`.')

        # check input data type
        raise_if_not(isinstance(input_series_dataset, InferenceDataset),
                     'Only InferenceDataset is accepted as input type.')

        # check that `num_samples` is a positive integer
        raise_if_not(num_samples > 0, '`num_samples` must be a positive integer.')

        # iterate through batches to produce predictions
        batch_size = batch_size or self.batch_size
        torch_dataset = input_series_dataset.to_torch_dataset()  # TODO: self.device?

        pred_loader = DataLoader(torch_dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=0,
                                 pin_memory=False,
                                 drop_last=False)
        predictions = []
        iterator = _build_tqdm_iterator(pred_loader, verbose=verbose)
        with torch.no_grad():
            for batch_tuple in iterator:

                # at this point `input_series` contains both the past target series and past covariates
                input_series = batch_tuple[0].to(self.device)
                cov_future = batch_tuple[1] if len(batch_tuple) == 3 else None

                # repeat prediction procedure for every needed sample
                batch_predictions = []
                for i in range(num_samples):
                    if self.is_recurrent:
                        batch_prediction = self._predict_batch_recurrent_model(n, input_series, cov_future)
                    else:
                        batch_prediction = self._predict_batch_block_model(n, input_series, cov_future, roll_size)

                    # bring predictions into desired format and drop unnecessary values
                    batch_prediction = torch.cat(batch_prediction, dim=1)
                    batch_prediction = batch_prediction[:, :n, :]
                    batch_prediction = batch_prediction.cpu().detach().numpy()

                    batch_predictions.append(batch_prediction)

                # TODO: this isn't very good as here we are calling again the input_series_dataset (which
                # TODO: causes some slicing again). Instead, we should return Tensors only in the __getitem__()
                # TODO: of the dataset, and offer a get_past_target(idx) method to return the past_target.
                batch_indices = batch_tuple[-1]
                ts_forecasts = Parallel(n_jobs=n_jobs)(
                    delayed(self._build_forecast_series)(
                        [batch_prediction[batch_idx] for batch_prediction in batch_predictions],
                        input_series_dataset[dataset_idx][0]
                    )
                    for batch_idx, dataset_idx in enumerate(batch_indices)
                )

                predictions.extend(ts_forecasts)

        return predictions

    def _predict_batch_block_model(self,
                                   n,
                                   input_series,
                                   cov_future,
                                   roll_size) -> Sequence[TimeSeries]:

        batch_prediction = []
        out = self._produce_predict_output(input_series)[:, self.first_prediction_index:, :]
        batch_prediction.append(out[:, :roll_size, :])
        prediction_length = roll_size

        while prediction_length < n:

            # roll over input series to contain latest target and covariate
            input_series = torch.roll(input_series, -roll_size, 1)

            # update target input to include next `roll_size` predictions
            if self.input_chunk_length >= roll_size:
                input_series[:, -roll_size:, :self.output_dim] = out[:, :roll_size, :]
            else:
                input_series[:, :, :self.output_dim] = out[:, -self.input_chunk_length:, :]

            # update covariates to include next `roll_size` predictions into the future
            if cov_future is not None and self.input_chunk_length >= roll_size:
                input_series[:, -roll_size:, self.output_dim:] = (
                    cov_future[:, prediction_length - roll_size:prediction_length, :]
                )
            elif cov_future is not None:
                input_series[:, :, self.output_dim:] = (
                    cov_future[:, prediction_length - self.input_chunk_length:prediction_length, :]
                )

            # take only last part of the output sequence where needed
            out = self._produce_predict_output(input_series)[:, self.first_prediction_index:, :]

            # update predictions depending on how many data points have been predicted
            if prediction_length <= n - self.output_chunk_length - roll_size:
                batch_prediction.append(out[:, :roll_size, :])
                prediction_length += roll_size
            elif prediction_length < n - self.output_chunk_length:
                # if we produce have `n - output_chunk_length < #predictions < n` we want to only use
                # the predictions and covariates necessary to exactly reach `n - output_chunk_length`,
                # so that the final forecast produces exactly the right number of predictions to reach `n`
                spillover_prediction_length = (prediction_length + roll_size) - (n - self.output_chunk_length)
                roll_size -= spillover_prediction_length
                batch_prediction.append(out[:, :roll_size, :])
                prediction_length += roll_size
            else:
                batch_prediction.append(out)
                prediction_length += self.output_chunk_length

        return batch_prediction

    def _predict_batch_recurrent_model(self, n, input_series, cov_future):
        batch_prediction = []
        out, last_hidden_state = self._produce_predict_output(input_series)
        batch_prediction.append(out[:, -1:, :])
        prediction_length = 1

        while prediction_length < n:

            # create new input to model from last prediction and current covariates, if available
            new_input = (
                torch.cat([out[:, -1:, :], cov_future[:, prediction_length - 1:prediction_length, :]], dim=2)
                if cov_future is not None else out[:, -1:, :]
            )

            # feed new input to model, including the last hidden state from the previous iteration
            out, last_hidden_state = self._produce_predict_output(new_input, last_hidden_state)

            # append prediction to batch prediction array, increase counter
            batch_prediction.append(out[:, -1:, :])
            prediction_length += 1

        return batch_prediction

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
               verbose: bool,
               epochs: int = 0
               ) -> None:
        """
        Performs the actual training
        :param train_loader: the training data loader feeding the training data and targets
        :param val_loader: optionally, a validation set loader
        :param tb_writer: optionally, a TensorBoard writer
        :param epochs: value >0 means we're retraining model
        """

        best_loss = np.inf

        iterator = _build_tqdm_iterator(
            range(self.total_epochs, self.total_epochs + epochs),
            verbose=verbose,
        )

        for epoch in iterator:
            total_loss = 0

            for batch_idx, (data, target) in enumerate(train_loader):
                self.model.train()
                data, target = data.to(self.device), target.to(self.device)
                output = self._produce_train_output(data)
                loss = self._compute_loss(output, target)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            if tb_writer is not None:
                for name, param in self.model.named_parameters():
                    # if the param doesn't require gradient, then param.grad = None and param.grad.data will crash
                    if param.requires_grad:
                        tb_writer.add_histogram(name + '/gradients', param.grad.data.cpu().numpy(), epoch)

                tb_writer.add_scalar("training/loss", total_loss / (batch_idx + 1), epoch)
                tb_writer.add_scalar("training/loss_total", total_loss / (batch_idx + 1), epoch)
                tb_writer.add_scalar("training/learning_rate", self._get_learning_rate(), epoch)

            self.total_epochs = epoch + 1
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

    def _produce_train_output(self, data):
        return self.model(data)

    def _compute_loss(self, output, target):
        return self.criterion(output, target)

    def _produce_predict_output(self, input):
        return self.model(input)

    def _evaluate_validation_loss(self, val_loader: DataLoader):
        total_loss = 0
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self._produce_train_output(data)
                loss = self._compute_loss(output, target)
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
            if self.total_epochs > 0:
                tb_writer = SummaryWriter(runs_folder, purge_step=self.total_epochs)
            else:
                tb_writer = SummaryWriter(runs_folder)
                dummy_input = torch.empty(self.batch_size, self.input_chunk_length, self.input_dim).to(self.device)
                tb_writer.add_graph(self.model, dummy_input)
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
            work_dir = os.path.join(os.getcwd(), DEFAULT_DARTS_FOLDER)

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


class RecurrentModel(TorchForecastingModel, ABC):
    # TODO: extract recurrent specific logic here (override produce_block_forecast() etc).
    pass


class TorchParametricProbabilisticForecastingModel(TorchForecastingModel, ABC):
    def __init__(self, likelihood: Optional[LikelihoodModel] = None, **kwargs):
        """ Pytorch Parametric Probabilistic Forecasting Model.

        This is a base class for pytroch parametric probabilistic models. "Parametric"
        means that these models are based on some predefined parametric distribution, say Gaussian.
        Make sure that subclasses contain the *likelihood* parameter in __init__ method
        and it is passed to the superclass via calling super().__init__. If the likelihood is not
        provided, the model is considered as deterministic.

        Parameters
        ----------
        likelihood
            The likelihood model to be used for probabilistic forecasts.
        """
        super().__init__(**kwargs)
        self.likelihood = likelihood

    def _is_probabilistic(self):
        return self.likelihood is not None

    def _compute_loss(self, output, target):
        if self.likelihood:
            return self.likelihood._compute_loss(output, target)
        else:
            return super()._compute_loss(output, target)

    @abstractmethod
    def _produce_predict_output(self, input):
        """
        This method has to be implemented by all children.

        TODO: rename parameter as it shadows input name
        """
        pass


def _raise_if_wrong_type(obj, exp_type, msg='expected type {}, got: {}'):
    raise_if_not(isinstance(obj, exp_type), msg.format(exp_type, type(obj)))


"""
Below we define the 5 torch model types:
    * PastCovariatesTorchModel
    * FutureCovariatesTorchModel
    * DualCovariatesTorchModel
    * MixedCovariatesTorchModel
    * SplitCovariatesTorchModel
"""
# TODO: there's a lot of repetition below... is there a cleaner way to do this in Python- Using eg generics or something


class PastCovariatesTorchModel(TorchForecastingModel, ABC):
    def _build_train_dataset(self,
                             target: Sequence[TimeSeries],
                             past_covariates: Optional[Sequence[TimeSeries]],
                             future_covariates: Optional[Sequence[TimeSeries]]) -> PastCovariatesTrainingDataset:

        raise_if_not(future_covariates is None,
                     'Specified future_covariates for a PastCovariatesModel (only past_covariates are expected).')

        return PastCovariatesSequentialDataset(target_series=target,
                                               covariates=past_covariates,
                                               input_chunk_length=self.input_chunk_length,
                                               output_chunk_length=self.output_chunk_length)

    def _build_inference_dataset(self,
                                 target: Sequence[TimeSeries],
                                 n: int,
                                 past_covariates: Optional[Sequence[TimeSeries]],
                                 future_covariates: Optional[Sequence[TimeSeries]]) -> PastCovariatesInferenceDataset:

        raise_if_not(future_covariates is None,
                     'Specified future_covariates for a PastCovariatesModel (only past_covariates are expected).')

        return PastCovariatesInferenceDataset(target_series=target,
                                              covariates=past_covariates,
                                              n=n,
                                              input_chunk_length=self.input_chunk_length,
                                              output_chunk_length=self.output_chunk_length)

    def _verify_train_dataset_type(self, train_dataset: TrainingDataset):
        _raise_if_wrong_type(train_dataset, PastCovariatesTrainingDataset)

    def _verify_inference_dataset_type(self, inference_dataset: InferenceDataset):
        _raise_if_wrong_type(inference_dataset, PastCovariatesInferenceDataset)


class FutureCovariatesTorchModel(TorchForecastingModel, ABC):
    def _build_train_dataset(self,
                             target: Sequence[TimeSeries],
                             past_covariates: Optional[Sequence[TimeSeries]],
                             future_covariates: Optional[Sequence[TimeSeries]]) -> FutureCovariatesTrainingDataset:
        raise_if_not(past_covariates is None,
                     'Specified past_covariates for a FutureCovariatesModel (only future_covariates are expected).')

        return FutureCovariatesSequentialDataset(target_series=target,
                                                 covariates=future_covariates,
                                                 input_chunk_length=self.input_chunk_length,
                                                 output_chunk_length=self.output_chunk_length)

    def _build_inference_dataset(self,
                                 target: Sequence[TimeSeries],
                                 n: int,
                                 past_covariates: Optional[Sequence[TimeSeries]],
                                 future_covariates: Optional[Sequence[TimeSeries]]) -> FutureCovariatesInferenceDataset:
        raise_if_not(past_covariates is None,
                     'Specified past_covariates for a FutureCovariatesModel (only future_covariates are expected).')

        return FutureCovariatesInferenceDataset(target_series=target,
                                                covariates=future_covariates,
                                                n=n,
                                                input_chunk_length=self.input_chunk_length)

    def _verify_train_dataset_type(self, train_dataset: TrainingDataset):
        _raise_if_wrong_type(train_dataset, FutureCovariatesTrainingDataset)

    def _verify_inference_dataset_type(self, inference_dataset: InferenceDataset):
        _raise_if_wrong_type(inference_dataset, FutureCovariatesInferenceDataset)


class DualCovariatesTorchModel(TorchForecastingModel, ABC):
    def _build_train_dataset(self,
                             target: Sequence[TimeSeries],
                             past_covariates: Optional[Sequence[TimeSeries]],
                             future_covariates: Optional[Sequence[TimeSeries]]) -> DualCovariatesTrainingDataset:

        return DualCovariatesSequentialDataset(target_series=target,
                                               covariates=future_covariates,
                                               input_chunk_length=self.input_chunk_length,
                                               output_chunk_length=self.output_chunk_length)

    def _build_inference_dataset(self,
                                 target: Sequence[TimeSeries],
                                 n: int,
                                 past_covariates: Optional[Sequence[TimeSeries]],
                                 future_covariates: Optional[Sequence[TimeSeries]]) -> DualCovariatesInferenceDataset:

        return DualCovariatesInferenceDataset(target_series=target,
                                              covariates=future_covariates,
                                              n=n,
                                              input_chunk_length=self.input_chunk_length,
                                              output_chunk_length=self.output_chunk_length)

    def _verify_train_dataset_type(self, train_dataset: TrainingDataset):
        _raise_if_wrong_type(train_dataset, DualCovariatesTrainingDataset)

    def _verify_inference_dataset_type(self, inference_dataset: InferenceDataset):
        _raise_if_wrong_type(inference_dataset, DualCovariatesInferenceDataset)


class MixedCovariatesTorchModel(TorchForecastingModel, ABC):
    def _build_train_dataset(self,
                             target: Sequence[TimeSeries],
                             past_covariates: Optional[Sequence[TimeSeries]],
                             future_covariates: Optional[Sequence[TimeSeries]]) -> MixedCovariatesTrainingDataset:

        return MixedCovariatesSequentialDataset(target_series=target,
                                                past_covariates=past_covariates,
                                                future_covariates=future_covariates,
                                                input_chunk_length=self.input_chunk_length,
                                                output_chunk_length=self.output_chunk_length)

    def _build_inference_dataset(self,
                                 target: Sequence[TimeSeries],
                                 n: int,
                                 past_covariates: Optional[Sequence[TimeSeries]],
                                 future_covariates: Optional[Sequence[TimeSeries]]) -> MixedCovariatesInferenceDataset:

        return MixedCovariatesInferenceDataset(target_series=target,
                                               past_covariates=past_covariates,
                                               future_covariates=future_covariates,
                                               n=n,
                                               input_chunk_length=self.input_chunk_length,
                                               output_chunk_length=self.output_chunk_length)

    def _verify_train_dataset_type(self, train_dataset: TrainingDataset):
        _raise_if_wrong_type(train_dataset, MixedCovariatesTrainingDataset)

    def _verify_inference_dataset_type(self, inference_dataset: InferenceDataset):
        _raise_if_wrong_type(inference_dataset, MixedCovariatesInferenceDataset)


class SplitCovariatesTorchModel(TorchForecastingModel, ABC):
    def _build_train_dataset(self,
                             target: Sequence[TimeSeries],
                             past_covariates: Optional[Sequence[TimeSeries]],
                             future_covariates: Optional[Sequence[TimeSeries]]) -> SplitCovariatesTrainingDataset:

        return SplitCovariatesSequentialDataset(target_series=target,
                                                past_covariates=past_covariates,
                                                future_covariates=future_covariates,
                                                input_chunk_length=self.input_chunk_length,
                                                output_chunk_length=self.output_chunk_length)

    def _build_inference_dataset(self,
                                 target: Sequence[TimeSeries],
                                 n: int,
                                 past_covariates: Optional[Sequence[TimeSeries]],
                                 future_covariates: Optional[Sequence[TimeSeries]]) -> SplitCovariatesInferenceDataset:

        return SplitCovariatesInferenceDataset(target_series=target,
                                               past_covariates=past_covariates,
                                               future_covariates=future_covariates,
                                               n=n,
                                               input_chunk_length=self.input_chunk_length,
                                               output_chunk_length=self.output_chunk_length)

    def _verify_train_dataset_type(self, train_dataset: TrainingDataset):
        _raise_if_wrong_type(train_dataset, SplitCovariatesTrainingDataset)

    def _verify_inference_dataset_type(self, inference_dataset: InferenceDataset):
        _raise_if_wrong_type(inference_dataset, SplitCovariatesInferenceDataset)
