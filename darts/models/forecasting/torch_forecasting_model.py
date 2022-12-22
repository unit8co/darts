"""
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

    * TorchParametricProbabilisticForecastingModel(TorchForecastingModel) is the super-class of all probabilistic torch
      forecasting models.
"""

import datetime
import inspect
import os
import shutil
import sys
from abc import ABC, abstractmethod
from glob import glob
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning import loggers as pl_loggers
from torch import Tensor
from torch.utils.data import DataLoader

from darts.logging import (
    get_logger,
    raise_if,
    raise_if_not,
    raise_log,
    suppress_lightning_warnings,
)
from darts.models.forecasting.forecasting_model import (
    ForecastingModel,
    GlobalForecastingModel,
)
from darts.models.forecasting.pl_forecasting_module import PLForecastingModule
from darts.timeseries import TimeSeries
from darts.utils.data.inference_dataset import (
    DualCovariatesInferenceDataset,
    FutureCovariatesInferenceDataset,
    InferenceDataset,
    MixedCovariatesInferenceDataset,
    PastCovariatesInferenceDataset,
    SplitCovariatesInferenceDataset,
)
from darts.utils.data.sequential_dataset import (
    DualCovariatesSequentialDataset,
    FutureCovariatesSequentialDataset,
    MixedCovariatesSequentialDataset,
    PastCovariatesSequentialDataset,
    SplitCovariatesSequentialDataset,
)
from darts.utils.data.training_dataset import (
    DualCovariatesTrainingDataset,
    FutureCovariatesTrainingDataset,
    MixedCovariatesTrainingDataset,
    PastCovariatesTrainingDataset,
    SplitCovariatesTrainingDataset,
    TrainingDataset,
)
from darts.utils.likelihood_models import Likelihood
from darts.utils.torch import random_method
from darts.utils.utils import seq2series, series2seq

DEFAULT_DARTS_FOLDER = "darts_logs"
CHECKPOINTS_FOLDER = "checkpoints"
RUNS_FOLDER = "runs"
INIT_MODEL_NAME = "_model.pth.tar"

# pickling a TorchForecastingModel will not save below attributes: the keys specify the
# attributes to be ignored, and the values are the default values getting assigned upon loading
TFM_ATTRS_NO_PICKLE = {"model": None, "trainer": None}

logger = get_logger(__name__)


def _get_checkpoint_folder(work_dir, model_name):
    return os.path.join(work_dir, model_name, CHECKPOINTS_FOLDER)


def _get_logs_folder(work_dir, model_name):
    return os.path.join(work_dir, model_name)


def _get_runs_folder(work_dir, model_name):
    return os.path.join(work_dir, model_name)


def _get_checkpoint_fname(work_dir, model_name, best=False):
    checkpoint_dir = _get_checkpoint_folder(work_dir, model_name)
    path = os.path.join(checkpoint_dir, "best-*" if best else "last-*")

    checklist = glob(path)
    if len(checklist) == 0:
        raise_log(
            FileNotFoundError(
                "There is no file matching prefix {} in {}".format(
                    "best-*" if best else "last-*", checkpoint_dir
                )
            ),
            logger,
        )

    file_name = max(checklist, key=os.path.getctime)
    return os.path.basename(file_name)


class TorchForecastingModel(GlobalForecastingModel, ABC):
    @random_method
    def __init__(
        self,
        batch_size: int = 32,
        n_epochs: int = 100,
        model_name: str = None,
        work_dir: str = os.path.join(os.getcwd(), DEFAULT_DARTS_FOLDER),
        log_tensorboard: bool = False,
        nr_epochs_val_period: int = 1,
        force_reset: bool = False,
        save_checkpoints: bool = False,
        add_encoders: Optional[dict] = None,
        random_state: Optional[int] = None,
        pl_trainer_kwargs: Optional[dict] = None,
        show_warnings: bool = False,
    ):

        """Pytorch Lightning (PL)-based Forecasting Model.

        This class is meant to be inherited to create a new PL-based forecasting model.
        It governs the interactions between:
            - Darts forecasting models (module) :class:`PLTorchForecastingModel`
            - Darts integrated PL Lightning Trainer :class:`pytorch_lightning.Trainer` or custom PL Trainers
            - Dataset loaders :class:`TrainingDataset` and :class:`InferenceDataset` or custom Dataset Loaders.

        When subclassing this class, please make sure to set the self.model attribute
        in the __init__ function and then call super().__init__ while passing the kwargs.

        Parameters
        ----------
        batch_size
            Number of time series (input and output sequences) used in each training pass. Default: ``32``.
        n_epochs
            Number of epochs over which to train the model. Default: ``100``.
        model_name
            Name of the model. Used for creating checkpoints and saving tensorboard data. If not specified,
            defaults to the following string ``"YYYY-mm-dd_HH:MM:SS_torch_model_run_PID"``, where the initial part
            of the name is formatted with the local date and time, while PID is the processed ID (preventing models
            spawned at the same time by different processes to share the same model_name). E.g.,
            ``"2021-06-14_09:53:32_torch_model_run_44607"``.
        work_dir
            Path of the working directory, where to save checkpoints and Tensorboard summaries.
            Default: current working directory.
        log_tensorboard
            If set, use Tensorboard to log the different parameters. The logs will be located in:
            ``"{work_dir}/darts_logs/{model_name}/logs/"``. Default: ``False``.
        nr_epochs_val_period
            Number of epochs to wait before evaluating the validation loss (if a validation
            ``TimeSeries`` is passed to the :func:`fit()` method). Default: ``1``.
        force_reset
            If set to ``True``, any previously-existing model with the same name will be reset (all checkpoints will
            be discarded). Default: ``False``.
        save_checkpoints
            Whether or not to automatically save the untrained model and checkpoints from training.
            To load the model from checkpoint, call :func:`MyModelClass.load_from_checkpoint()`, where
            :class:`MyModelClass` is the :class:`TorchForecastingModel` class that was used (such as :class:`TFTModel`,
            :class:`NBEATSModel`, etc.). If set to ``False``, the model can still be manually saved using
            :func:`save()` and loaded using :func:`load()`. Default: ``False``.
        add_encoders
            A large number of past and future covariates can be automatically generated with `add_encoders`.
            This can be done by adding multiple pre-defined index encoders and/or custom user-made functions that
            will be used as index encoders. Additionally, a transformer such as Darts' :class:`Scaler` can be added to
            transform the generated covariates. This happens all under one hood and only needs to be specified at
            model creation.
            Read :meth:`SequentialEncoder <darts.dataprocessing.encoders.SequentialEncoder>` to find out more about
            ``add_encoders``. Default: ``None``. An example showing some of ``add_encoders`` features:

            .. highlight:: python
            .. code-block:: python

                add_encoders={
                    'cyclic': {'future': ['month']},
                    'datetime_attribute': {'future': ['hour', 'dayofweek']},
                    'position': {'past': ['relative'], 'future': ['relative']},
                    'custom': {'past': [lambda idx: (idx.year - 1950) / 50]},
                    'transformer': Scaler()
                }
            ..
        random_state
            Control the randomness of the weights initialization. Check this
            `link <https://scikit-learn.org/stable/glossary.html#term-random_state>`_ for more details.
            Default: ``None``.
        pl_trainer_kwargs
            By default :class:`TorchForecastingModel` creates a PyTorch Lightning Trainer with several useful presets
            that performs the training, validation and prediction processes. These presets include automatic
            checkpointing, tensorboard logging, setting the torch device and more.
            With ``pl_trainer_kwargs`` you can add additional kwargs to instantiate the PyTorch Lightning trainer
            object. Check the `PL Trainer documentation
            <https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html>`_ for more information about the
            supported kwargs. Default: ``None``.
            Running on GPU(s) is also possible using ``pl_trainer_kwargs`` by specifying keys ``"accelerator",
            "devices", and "auto_select_gpus"``. Some examples for setting the devices inside the ``pl_trainer_kwargs``
            dict:


            - ``{"accelerator": "cpu"}`` for CPU,
            - ``{"accelerator": "gpu", "devices": [i]}`` to use only GPU ``i`` (``i`` must be an integer),
            - ``{"accelerator": "gpu", "devices": -1, "auto_select_gpus": True}`` to use all available GPUS.

            For more info, see here:
            `trainer flags
            <https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#trainer-flags>`_,
            and `training on multiple gpus
            <https://pytorch-lightning.readthedocs.io/en/stable/accelerators/gpu_basic.html#train-on-multiple-gpus>`_.

            With parameter ``"callbacks"`` you can add custom or PyTorch-Lightning built-in callbacks to Darts'
            :class:`TorchForecastingModel`. Below is an example for adding EarlyStopping to the training process.
            The model will stop training early if the validation loss `val_loss` does not improve beyond
            specifications. For more information on callbacks, visit:
            `PyTorch Lightning Callbacks
            <https://pytorch-lightning.readthedocs.io/en/stable/extensions/callbacks.html>`_

            .. highlight:: python
            .. code-block:: python

                from pytorch_lightning.callbacks.early_stopping import EarlyStopping

                # stop training when validation loss does not decrease more than 0.05 (`min_delta`) over
                # a period of 5 epochs (`patience`)
                my_stopper = EarlyStopping(
                    monitor="val_loss",
                    patience=5,
                    min_delta=0.05,
                    mode='min',
                )

                pl_trainer_kwargs={"callbacks": [my_stopper]}
            ..

            Note that you can also use a custom PyTorch Lightning Trainer for training and prediction with optional
            parameter ``trainer`` in :func:`fit()` and :func:`predict()`.
        show_warnings
            whether to show warnings raised from PyTorch Lightning. Useful to detect potential issues of
            your forecasting use case. Default: ``False``.
        """
        super().__init__(add_encoders=add_encoders)
        suppress_lightning_warnings(suppress_all=not show_warnings)

        # model will get created in first call of fit_from_dataset()
        self.model: Optional[PLForecastingModule] = None
        # to retrieve the PLForecastingModule upon loading, we store the module path, and class name
        self._module_path = self.__module__
        # class name will be set in fit_from_dataset()
        self._module_name: Optional[str] = ""

        self.train_sample: Optional[Tuple] = None
        self.output_dim: Optional[int] = None

        self.n_epochs = n_epochs
        self.batch_size = batch_size

        # get model name and work dir
        if model_name is None:
            current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S.%f")
            model_name = current_time + "_torch_model_run_" + str(os.getpid())

        self.model_name = model_name
        self.work_dir = work_dir

        # setup model save dirs
        self.save_checkpoints = save_checkpoints
        checkpoints_folder = _get_checkpoint_folder(self.work_dir, self.model_name)
        log_folder = _get_logs_folder(self.work_dir, self.model_name)
        checkpoint_exists = (
            os.path.exists(checkpoints_folder)
            and len(glob(os.path.join(checkpoints_folder, "*"))) > 0
        )

        # setup model save dirs
        if checkpoint_exists and save_checkpoints:
            raise_if_not(
                force_reset,
                f"Some model data already exists for `model_name` '{self.model_name}'. Either load model to continue "
                f"training or use `force_reset=True` to initialize anyway to start training from scratch and remove "
                f"all the model data",
                logger,
            )
            self.reset_model()
        elif save_checkpoints:
            self._create_save_dirs()
        else:
            pass

        # save best epoch on val_loss and last epoch under 'darts_logs/model_name/checkpoints/'
        if save_checkpoints:
            checkpoint_callback = pl.callbacks.ModelCheckpoint(
                dirpath=checkpoints_folder,
                save_last=True,
                monitor="val_loss",
                filename="best-{epoch}-{val_loss:.2f}",
            )
            checkpoint_callback.CHECKPOINT_NAME_LAST = "last-{epoch}"
        else:
            checkpoint_callback = None

        # save tensorboard under 'darts_logs/model_name/logs/'
        model_logger = (
            pl_loggers.TensorBoardLogger(save_dir=log_folder, name="", version="logs")
            if log_tensorboard
            else False
        )

        # setup trainer parameters from model creation parameters
        self.trainer_params = {
            "logger": model_logger,
            "max_epochs": n_epochs,
            "check_val_every_n_epoch": nr_epochs_val_period,
            "enable_checkpointing": save_checkpoints,
            "callbacks": [cb for cb in [checkpoint_callback] if cb is not None],
        }

        # update trainer parameters with user defined `pl_trainer_kwargs`
        if pl_trainer_kwargs is not None:
            pl_trainer_kwargs_copy = {
                key: val for key, val in pl_trainer_kwargs.items()
            }
            self.n_epochs = pl_trainer_kwargs_copy.get("max_epochs", self.n_epochs)
            self.trainer_params["callbacks"] += pl_trainer_kwargs_copy.pop(
                "callbacks", []
            )
            self.trainer_params = dict(self.trainer_params, **pl_trainer_kwargs_copy)

        # pytorch lightning trainer will be created at training time
        # keep a reference of the trainer, to avoid weak reference errors
        self.trainer: Optional[pl.Trainer] = None
        self.load_ckpt_path: Optional[str] = None

        # pl_module_params must be set in __init__ method of TorchForecastingModel subclass
        self.pl_module_params: Optional[dict] = None

    @classmethod
    def _validate_model_params(cls, **kwargs):
        """validate that parameters used at model creation are part of :class:`TorchForecastingModel`,
        :class:`PLForecastingModule` or cls __init__ methods.
        """
        valid_kwargs = (
            set(inspect.signature(TorchForecastingModel.__init__).parameters.keys())
            | set(inspect.signature(PLForecastingModule.__init__).parameters.keys())
            | set(inspect.signature(cls.__init__).parameters.keys())
        )

        invalid_kwargs = [kwarg for kwarg in kwargs if kwarg not in valid_kwargs]

        raise_if(
            len(invalid_kwargs) > 0,
            f"Invalid model creation parameters. Model `{cls.__name__}` has no args/kwargs `{invalid_kwargs}`",
            logger=logger,
        )

    @classmethod
    def _extract_torch_model_params(cls, **kwargs):
        """extract params from model creation to set up TorchForecastingModels"""
        cls._validate_model_params(**kwargs)
        get_params = list(
            inspect.signature(TorchForecastingModel.__init__).parameters.keys()
        )
        get_params.remove("self")
        return {kwarg: kwargs.get(kwarg) for kwarg in get_params if kwarg in kwargs}

    @staticmethod
    def _extract_pl_module_params(**kwargs):
        """Extract params from model creation to set up PLForecastingModule (the actual torch.nn.Module)"""
        get_params = list(
            inspect.signature(PLForecastingModule.__init__).parameters.keys()
        )
        get_params.remove("self")
        return {kwarg: kwargs.get(kwarg) for kwarg in get_params if kwarg in kwargs}

    def _create_save_dirs(self):
        """Create work dir and model dir"""
        if not os.path.exists(self.work_dir):
            os.mkdir(self.work_dir)
        if not os.path.exists(_get_runs_folder(self.work_dir, self.model_name)):
            os.mkdir(_get_runs_folder(self.work_dir, self.model_name))

    def _remove_save_dirs(self):
        shutil.rmtree(
            _get_runs_folder(self.work_dir, self.model_name), ignore_errors=True
        )

    def reset_model(self):
        """Resets the model object and removes all stored data - model, checkpoints, loggers and training history."""
        self._remove_save_dirs()
        self._create_save_dirs()

        self.model = None
        self.train_sample = None

    def _init_model(self, trainer: Optional[pl.Trainer] = None) -> None:
        """Initializes model and trainer based on examples of input/output tensors (to get the sizes right):"""

        raise_if(
            self.pl_module_params is None,
            "`pl_module_params` must be extracted in __init__ method of `TorchForecastingModel` subclass after "
            "calling `super.__init__(...)`. Do this with `self._extract_pl_module_params(**self.model_params).`",
        )

        # the tensors have shape (chunk_length, nr_dimensions)
        self.model = self._create_model(self.train_sample)
        self._module_name = self.model.__class__.__name__

        precision = None
        dtype = self.train_sample[0].dtype
        if np.issubdtype(dtype, np.float32):
            logger.info("Time series values are 32-bits; casting model to float32.")
            precision = 32
        elif np.issubdtype(dtype, np.float64):
            logger.info("Time series values are 64-bits; casting model to float64.")
            precision = 64

        precision_user = (
            self.trainer_params.get("precision", None)
            if trainer is None
            else trainer.precision
        )
        raise_if(
            precision_user is not None and precision_user != precision,
            f"User-defined trainer_kwarg `precision={precision_user}`-bit does not match dtype: `{dtype}` of the "
            f"underlying TimeSeries. Set `precision` to `{precision}` or cast your data to `{precision_user}-"
            f"bit` with `TimeSeries.astype(np.float{precision_user})`.",
            logger,
        )

        self.trainer_params["precision"] = precision

        # we need to save the initialized TorchForecastingModel as PyTorch-Lightning only saves module checkpoints
        if self.save_checkpoints:
            self.save(
                os.path.join(
                    _get_runs_folder(self.work_dir, self.model_name), INIT_MODEL_NAME
                )
            )

    def _setup_trainer(
        self,
        trainer: Optional[pl.Trainer],
        verbose: Optional[bool] = None,
        epochs: int = 0,
    ) -> pl.Trainer:
        """Sets up a PyTorch-Lightning trainer (if not already provided) for training or prediction."""
        if trainer is not None:
            return trainer

        trainer_params = {key: val for key, val in self.trainer_params.items()}
        if verbose is not None:
            trainer_params["enable_model_summary"] = (
                verbose if self.model.epochs_trained == 0 else False
            )
            trainer_params["enable_progress_bar"] = verbose

        return self._init_trainer(trainer_params=trainer_params, max_epochs=epochs)

    @staticmethod
    def _init_trainer(
        trainer_params: dict, max_epochs: Optional[int] = None
    ) -> pl.Trainer:
        """Initializes a PyTorch-Lightning trainer for training or prediction from `trainer_params`."""
        trainer_params_copy = {param: val for param, val in trainer_params.items()}
        if max_epochs is not None:
            trainer_params_copy["max_epochs"] = max_epochs

        return pl.Trainer(**trainer_params_copy)

    @abstractmethod
    def _create_model(self, train_sample: Tuple[Tensor]) -> torch.nn.Module:
        """
        This method has to be implemented by all children. It is in charge of instantiating the actual torch model,
        based on examples input/output tensors (i.e. implement a model with the right input/output sizes).
        """
        pass

    @abstractmethod
    def _build_train_dataset(
        self,
        target: Sequence[TimeSeries],
        past_covariates: Optional[Sequence[TimeSeries]],
        future_covariates: Optional[Sequence[TimeSeries]],
        max_samples_per_ts: Optional[int],
    ) -> TrainingDataset:
        """
        Each model must specify the default training dataset to use.
        """
        pass

    @abstractmethod
    def _build_inference_dataset(
        self,
        target: Sequence[TimeSeries],
        n: int,
        past_covariates: Optional[Sequence[TimeSeries]],
        future_covariates: Optional[Sequence[TimeSeries]],
    ) -> InferenceDataset:
        """
        Each model must specify the default training dataset to use.
        """
        pass

    @abstractmethod
    def _verify_train_dataset_type(self, train_dataset: TrainingDataset):
        """
        Verify that the provided train dataset is of the correct type
        """
        pass

    @abstractmethod
    def _verify_inference_dataset_type(self, inference_dataset: InferenceDataset):
        """
        Verify that the provided inference dataset is of the correct type
        """
        pass

    @abstractmethod
    def _verify_predict_sample(self, predict_sample: Tuple):
        """
        verify that the (first) sample contained in the inference dataset matches the model type and the
        data the model has been trained on.
        """
        pass

    @abstractmethod
    def _verify_past_future_covariates(self, past_covariates, future_covariates):
        """
        Verify that any non-None covariates comply with the model type.
        """
        pass

    @staticmethod
    @abstractmethod
    def _supports_static_covariates() -> bool:
        """
        Whether model supports static covariates
        """
        pass

    def _verify_static_covariates(self, static_covariates: Optional[pd.DataFrame]):
        """
        Verify that all static covariates are numeric.
        """
        if static_covariates is not None and self._supports_static_covariates():
            numeric_mask = static_covariates.columns.isin(
                static_covariates.select_dtypes(include=np.number)
            )
            raise_if(
                sum(~numeric_mask),
                "TorchForecastingModels can only interpret numeric static covariate data. Consider "
                "encoding/transforming categorical static covariates with "
                "`darts.dataprocessing.transformers.static_covariates_transformer.StaticCovariatesTransformer`.",
                logger,
            )

    @random_method
    def fit(
        self,
        series: Union[TimeSeries, Sequence[TimeSeries]],
        past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        val_series: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        val_past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        val_future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        trainer: Optional[pl.Trainer] = None,
        verbose: Optional[bool] = None,
        epochs: int = 0,
        max_samples_per_ts: Optional[int] = None,
        num_loader_workers: int = 0,
    ):
        """Fit/train the model on one or multiple series.

        This method wraps around :func:`fit_from_dataset()`, constructing a default training
        dataset for this model. If you need more control on how the series are sliced for training, consider
        calling :func:`fit_from_dataset()` with a custom :class:`darts.utils.data.TrainingDataset`.

        Training is performed with a PyTorch Lightning Trainer. It uses a default Trainer object from presets and
        ``pl_trainer_kwargs`` used at model creation. You can also use a custom Trainer with optional parameter
        ``trainer``. For more information on PyTorch Lightning Trainers check out `this link
        <https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html>`_ .

        This function can be called several times to do some extra training. If ``epochs`` is specified, the model
        will be trained for some (extra) ``epochs`` epochs.

        Below, all possible parameters are documented, but not all models support all parameters. For instance,
        all the :class:`PastCovariatesTorchModel` support only ``past_covariates`` and not ``future_covariates``.
        Darts will complain if you try fitting a model with the wrong covariates argument.

        When handling covariates, Darts will try to use the time axes of the target and the covariates
        to come up with the right time slices. So the covariates can be longer than needed; as long as the time axes
        are correct Darts will handle them correctly. It will also complain if their time span is not sufficient.

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
            Optionally, the past covariates corresponding to the validation series (must match ``covariates``)
        val_future_covariates
            Optionally, the future covariates corresponding to the validation series (must match ``covariates``)
        trainer
            Optionally, a custom PyTorch-Lightning Trainer object to perform training. Using a custom ``trainer`` will
            override Darts' default trainer.
        verbose
            Optionally, whether to print progress.
        epochs
            If specified, will train the model for ``epochs`` (additional) epochs, irrespective of what ``n_epochs``
            was provided to the model constructor.
        max_samples_per_ts
            Optionally, a maximum number of samples to use per time series. Models are trained in a supervised fashion
            by constructing slices of (input, output) examples. On long time series, this can result in unnecessarily
            large number of training samples. This parameter upper-bounds the number of training samples per time
            series (taking only the most recent samples in each series). Leaving to None does not apply any
            upper bound.
        num_loader_workers
            Optionally, an integer specifying the ``num_workers`` to use in PyTorch ``DataLoader`` instances,
            both for the training and validation loaders (if any).
            A larger number of workers can sometimes increase performance, but can also incur extra overheads
            and increase memory usage, as more batches are loaded in parallel.

        Returns
        -------
        self
            Fitted model.
        """
        # guarantee that all inputs are either list of `TimeSeries` or `None`
        series = series2seq(series)
        past_covariates = series2seq(past_covariates)
        future_covariates = series2seq(future_covariates)
        val_series = series2seq(val_series)
        val_past_covariates = series2seq(val_past_covariates)
        val_future_covariates = series2seq(val_future_covariates)

        self.encoders = self.initialize_encoders()
        if self.encoders.encoding_available:
            past_covariates, future_covariates = self.generate_fit_encodings(
                series=series,
                past_covariates=past_covariates,
                future_covariates=future_covariates,
            )

        if past_covariates is not None:
            self._uses_past_covariates = True
        if future_covariates is not None:
            self._uses_future_covariates = True

        self._verify_past_future_covariates(
            past_covariates=past_covariates, future_covariates=future_covariates
        )
        self._verify_static_covariates(series[0].static_covariates)

        # Check that dimensions of train and val set match; on first series only
        if val_series is not None:
            if self.encoders.encoding_available:
                (
                    val_past_covariates,
                    val_future_covariates,
                ) = self.generate_fit_encodings(
                    series=val_series,
                    past_covariates=val_past_covariates,
                    future_covariates=val_future_covariates,
                )
            self._verify_past_future_covariates(
                past_covariates=val_past_covariates,
                future_covariates=val_future_covariates,
            )
            self._verify_static_covariates(val_series[0].static_covariates)

            match = (
                series[0].width == val_series[0].width
                and (past_covariates[0].width if past_covariates is not None else None)
                == (
                    val_past_covariates[0].width
                    if val_past_covariates is not None
                    else None
                )
                and (
                    future_covariates[0].width
                    if future_covariates is not None
                    else None
                )
                == (
                    val_future_covariates[0].width
                    if val_future_covariates is not None
                    else None
                )
            )
            raise_if_not(
                match,
                "The dimensions of the series in the training set "
                "and the validation set do not match.",
            )

        train_dataset = self._build_train_dataset(
            target=series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            max_samples_per_ts=max_samples_per_ts,
        )

        if val_series is not None:
            val_dataset = self._build_train_dataset(
                target=val_series,
                past_covariates=val_past_covariates,
                future_covariates=val_future_covariates,
                max_samples_per_ts=max_samples_per_ts,
            )
        else:
            val_dataset = None

        # Pro-actively catch length exceptions to display nicer messages
        length_ok = True
        try:
            len(train_dataset)
        except ValueError:
            length_ok = False
        raise_if(
            not length_ok or len(train_dataset) == 0,  # mind the order
            "The train dataset does not contain even one training sample. "
            + "This is likely due to the provided training series being too short. "
            + "This model expect series of length at least {}.".format(
                self.min_train_series_length
            ),
        )
        logger.info(f"Train dataset contains {len(train_dataset)} samples.")

        super().fit(
            series=seq2series(series),
            past_covariates=seq2series(past_covariates),
            future_covariates=seq2series(future_covariates),
        )

        return self.fit_from_dataset(
            train_dataset, val_dataset, trainer, verbose, epochs, num_loader_workers
        )

    @random_method
    def fit_from_dataset(
        self,
        train_dataset: TrainingDataset,
        val_dataset: Optional[TrainingDataset] = None,
        trainer: Optional[pl.Trainer] = None,
        verbose: Optional[bool] = None,
        epochs: int = 0,
        num_loader_workers: int = 0,
    ):
        """
        Train the model with a specific :class:`darts.utils.data.TrainingDataset` instance.
        These datasets implement a PyTorch ``Dataset``, and specify how the target and covariates are sliced
        for training. If you are not sure which training dataset to use, consider calling :func:`fit()` instead,
        which will create a default training dataset appropriate for this model.

        Training is performed with a PyTorch Lightning Trainer. It uses a default Trainer object from presets and
        ``pl_trainer_kwargs`` used at model creation. You can also use a custom Trainer with optional parameter
        ``trainer``. For more information on PyTorch Lightning Trainers check out `this link
        <https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html>`_ .

        This function can be called several times to do some extra training. If ``epochs`` is specified, the model
        will be trained for some (extra) ``epochs`` epochs.

        Parameters
        ----------
        train_dataset
            A training dataset with a type matching this model (e.g. :class:`PastCovariatesTrainingDataset` for
            :class:`PastCovariatesTorchModel`).
        val_dataset
            A training dataset with a type matching this model (e.g. :class:`PastCovariatesTrainingDataset` for
            :class:`PastCovariatesTorchModel`s), representing the validation set (to track the validation loss).
        trainer
            Optionally, a custom PyTorch-Lightning Trainer object to perform prediction. Using a custom `trainer` will
            override Darts' default trainer.
        verbose
            Optionally, whether to print progress.
        epochs
            If specified, will train the model for ``epochs`` (additional) epochs, irrespective of what ``n_epochs``
            was provided to the model constructor.
        num_loader_workers
            Optionally, an integer specifying the ``num_workers`` to use in PyTorch ``DataLoader`` instances,
            both for the training and validation loaders (if any).
            A larger number of workers can sometimes increase performance, but can also incur extra overheads
            and increase memory usage, as more batches are loaded in parallel.

        Returns
        -------
        self
            Fitted model.
        """
        self._fit_called = True
        self._verify_train_dataset_type(train_dataset)

        # Pro-actively catch length exceptions to display nicer messages
        train_length_ok, val_length_ok = True, True
        try:
            len(train_dataset)
        except ValueError:
            train_length_ok = False
        if val_dataset is not None:
            try:
                len(val_dataset)
            except ValueError:
                val_length_ok = False

        raise_if(
            not train_length_ok or len(train_dataset) == 0,  # mind the order
            "The provided training time series dataset is too short for obtaining even one training point.",
            logger,
        )
        raise_if(
            val_dataset is not None and (not val_length_ok or len(val_dataset) == 0),
            "The provided validation time series dataset is too short for obtaining even one training point.",
            logger,
        )

        train_sample = train_dataset[0]
        if self.model is None:
            # Build model, based on the dimensions of the first series in the train set.
            self.train_sample, self.output_dim = train_sample, train_sample[-1].shape[1]
            self._init_model(trainer)
        else:
            # Check existing model has input/output dims matching what's provided in the training set.
            raise_if_not(
                len(train_sample) == len(self.train_sample),
                "The size of the training set samples (tuples) does not match what the model has been "
                "previously trained on. Trained on tuples of length {}, received tuples of length {}.".format(
                    len(self.train_sample), len(train_sample)
                ),
            )
            same_dims = tuple(
                s.shape[1] if s is not None else None for s in train_sample
            ) == tuple(s.shape[1] if s is not None else None for s in self.train_sample)
            raise_if_not(
                same_dims,
                "The dimensionality of the series in the training set do not match the dimensionality"
                " of the series the model has previously been trained on. "
                "Model input/output dimensions = {}, provided input/ouptput dimensions = {}".format(
                    tuple(
                        s.shape[1] if s is not None else None for s in self.train_sample
                    ),
                    tuple(s.shape[1] if s is not None else None for s in train_sample),
                ),
            )

        # Setting drop_last to False makes the model see each sample at least once, and guarantee the presence of at
        # least one batch no matter the chosen batch size
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_loader_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=self._batch_collate_fn,
        )

        # Prepare validation data
        val_loader = (
            None
            if val_dataset is None
            else DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=num_loader_workers,
                pin_memory=True,
                drop_last=False,
                collate_fn=self._batch_collate_fn,
            )
        )

        # if user wants to train the model for more epochs, ignore the n_epochs parameter
        train_num_epochs = epochs if epochs > 0 else self.n_epochs

        # setup trainer
        self.trainer = self._setup_trainer(trainer, verbose, train_num_epochs)

        # TODO: multiple training without loading from checkpoint is not trivial (I believe PyTorch-Lightning is still
        #  working on that, see https://github.com/PyTorchLightning/pytorch-lightning/issues/9636)
        if self.epochs_trained > 0 and not self.load_ckpt_path:
            logger.warning(
                "Attempting to retrain the model without resuming from a checkpoint. This is currently "
                "discouraged. Consider setting `save_checkpoints` to `True` and specifying `model_name` at model "
                f"creation. Then call `model = {self.__class__.__name__}.load_from_checkpoint(model_name, "
                "best=False)`. Finally, train the model with `model.fit(..., epochs=new_epochs)` where "
                "`new_epochs` is the sum of (epochs already trained + some additional epochs)."
            )

        # Train model
        self._train(train_loader, val_loader)
        return self

    def _train(
        self, train_loader: DataLoader, val_loader: Optional[DataLoader]
    ) -> None:
        """
        Performs the actual training

        Parameters
        ----------
        train_loader
            the training data loader feeding the training data and targets
        val_loader
            optionally, a validation set loader
        """

        # if model was loaded from checkpoint (when `load_ckpt_path is not None`) and model.fit() is called,
        # we resume training
        ckpt_path = self.load_ckpt_path
        self.load_ckpt_path = None

        self.trainer.fit(
            self.model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
            ckpt_path=ckpt_path,
        )

    @random_method
    def predict(
        self,
        n: int,
        series: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        trainer: Optional[pl.Trainer] = None,
        batch_size: Optional[int] = None,
        verbose: Optional[bool] = None,
        n_jobs: int = 1,
        roll_size: Optional[int] = None,
        num_samples: int = 1,
        num_loader_workers: int = 0,
        mc_dropout: bool = False,
    ) -> Union[TimeSeries, Sequence[TimeSeries]]:
        """Predict the ``n`` time step following the end of the training series, or of the specified ``series``.

        Prediction is performed with a PyTorch Lightning Trainer. It uses a default Trainer object from presets and
        ``pl_trainer_kwargs`` used at model creation. You can also use a custom Trainer with optional parameter
        ``trainer``. For more information on PyTorch Lightning Trainers check out `this link
        <https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html>`_ .

        Below, all possible parameters are documented, but not all models support all parameters. For instance,
        all the :class:`PastCovariatesTorchModel` support only ``past_covariates`` and not ``future_covariates``.
        Darts will complain if you try calling :func:`predict()` on a model with the wrong covariates argument.

        Darts will also complain if the provided covariates do not have a sufficient time span.
        In general, not all models require the same covariates' time spans:

        * | Models relying on past covariates require the last ``input_chunk_length`` of the ``past_covariates``
          | points to be known at prediction time. For horizon values ``n > output_chunk_length``, these models
          | require at least the next ``n - output_chunk_length`` future values to be known as well.
        * | Models relying on future covariates require the next ``n`` values to be known.
          | In addition (for :class:`DualCovariatesTorchModel` and :class:`MixedCovariatesTorchModel`), they also
          | require the "historic" values of these future covariates (over the past ``input_chunk_length``).

        When handling covariates, Darts will try to use the time axes of the target and the covariates
        to come up with the right time slices. So the covariates can be longer than needed; as long as the time axes
        are correct Darts will handle them correctly. It will also complain if their time span is not sufficient.

        Parameters
        ----------
        n
            The number of time steps after the end of the training time series for which to produce predictions
        series
            Optionally, a series or sequence of series, representing the history of the target series whose
            future is to be predicted. If specified, the method returns the forecasts of these
            series. Otherwise, the method returns the forecast of the (single) training series.
        past_covariates
            Optionally, the past-observed covariates series needed as inputs for the model.
            They must match the covariates used for training in terms of dimension.
        future_covariates
            Optionally, the future-known covariates series needed as inputs for the model.
            They must match the covariates used for training in terms of dimension.
        trainer
            Optionally, a custom PyTorch-Lightning Trainer object to perform prediction. Using a custom ``trainer``
            will override Darts' default trainer.
        batch_size
            Size of batches during prediction. Defaults to the models' training ``batch_size`` value.
        verbose
            Optionally, whether to print progress.
        n_jobs
            The number of jobs to run in parallel. ``-1`` means using all processors. Defaults to ``1``.
        roll_size
            For self-consuming predictions, i.e. ``n > output_chunk_length``, determines how many
            outputs of the model are fed back into it at every iteration of feeding the predicted target
            (and optionally future covariates) back into the model. If this parameter is not provided,
            it will be set ``output_chunk_length`` by default.
        num_samples
            Number of times a prediction is sampled from a probabilistic model. Should be left set to 1
            for deterministic models.
        num_loader_workers
            Optionally, an integer specifying the ``num_workers`` to use in PyTorch ``DataLoader`` instances,
            for the inference/prediction dataset loaders (if any).
            A larger number of workers can sometimes increase performance, but can also incur extra overheads
            and increase memory usage, as more batches are loaded in parallel.
        mc_dropout
            Optionally, enable monte carlo dropout for predictions using neural network based models.
            This allows bayesian approximation by specifying an implicit prior over learned models.

        Returns
        -------
        Union[TimeSeries, Sequence[TimeSeries]]
            One or several time series containing the forecasts of ``series``, or the forecast of the training series
            if ``series`` is not specified and the model has been trained on a single series.
        """
        if series is None:
            raise_if(
                self.training_series is None,
                "Input series has to be provided after fitting on multiple series.",
            )
            series = self.training_series

        called_with_single_series = True if isinstance(series, TimeSeries) else False

        # guarantee that all inputs are either list of TimeSeries or None
        series = series2seq(series)
        past_covariates = series2seq(past_covariates)
        future_covariates = series2seq(future_covariates)

        self._verify_static_covariates(series[0].static_covariates)

        # encoders are set when calling fit(), but not when calling fit_from_dataset()
        # additionally, do not generate encodings when covariates were loaded as they already
        # contain the encodings
        if self.encoders is not None and self.encoders.encoding_available:
            past_covariates, future_covariates = self.generate_predict_encodings(
                n=n,
                series=series,
                past_covariates=past_covariates,
                future_covariates=future_covariates,
            )

        if past_covariates is None and self.past_covariate_series is not None:
            past_covariates = series2seq(self.past_covariate_series)
        if future_covariates is None and self.future_covariate_series is not None:
            future_covariates = series2seq(self.future_covariate_series)

        super().predict(n, series, past_covariates, future_covariates)

        dataset = self._build_inference_dataset(
            target=series,
            n=n,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
        )

        predictions = self.predict_from_dataset(
            n,
            dataset,
            trainer=trainer,
            verbose=verbose,
            batch_size=batch_size,
            n_jobs=n_jobs,
            roll_size=roll_size,
            num_samples=num_samples,
            num_loader_workers=num_loader_workers,
            mc_dropout=mc_dropout,
        )

        return predictions[0] if called_with_single_series else predictions

    @random_method
    def predict_from_dataset(
        self,
        n: int,
        input_series_dataset: InferenceDataset,
        trainer: Optional[pl.Trainer] = None,
        batch_size: Optional[int] = None,
        verbose: Optional[bool] = None,
        n_jobs: int = 1,
        roll_size: Optional[int] = None,
        num_samples: int = 1,
        num_loader_workers: int = 0,
        mc_dropout: bool = False,
    ) -> Sequence[TimeSeries]:

        """
        This method allows for predicting with a specific :class:`darts.utils.data.InferenceDataset` instance.
        These datasets implement a PyTorch ``Dataset``, and specify how the target and covariates are sliced
        for inference. In most cases, you'll rather want to call :func:`predict()` instead, which will create an
        appropriate :class:`InferenceDataset` for you.

        Prediction is performed with a PyTorch Lightning Trainer. It uses a default Trainer object from presets and
        ``pl_trainer_kwargs`` used at model creation. You can also use a custom Trainer with optional parameter
        ``trainer``. For more information on PyTorch Lightning Trainers check out `this link
        <https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html>`_ .

        Parameters
        ----------
        n
            The number of time steps after the end of the training time series for which to produce predictions
        input_series_dataset
            Optionally, a series or sequence of series, representing the history of the target series' whose
            future is to be predicted. If specified, the method returns the forecasts of these
            series. Otherwise, the method returns the forecast of the (single) training series.
        trainer
            Optionally, a custom PyTorch-Lightning Trainer object to perform prediction.  Using a custom ``trainer``
            will override Darts' default trainer.
        batch_size
            Size of batches during prediction. Defaults to the models ``batch_size`` value.
        verbose
            Optionally, whether to print progress.
        n_jobs
            The number of jobs to run in parallel. ``-1`` means using all processors. Defaults to ``1``.
        roll_size
            For self-consuming predictions, i.e. ``n > output_chunk_length``, determines how many
            outputs of the model are fed back into it at every iteration of feeding the predicted target
            (and optionally future covariates) back into the model. If this parameter is not provided,
            it will be set ``output_chunk_length`` by default.
        num_samples
            Number of times a prediction is sampled from a probabilistic model. Should be left set to 1
            for deterministic models.
        num_loader_workers
            Optionally, an integer specifying the ``num_workers`` to use in PyTorch ``DataLoader`` instances,
            for the inference/prediction dataset loaders (if any).
            A larger number of workers can sometimes increase performance, but can also incur extra overheads
            and increase memory usage, as more batches are loaded in parallel.
        mc_dropout
            Optionally, enable monte carlo dropout for predictions using neural network based models.
            This allows bayesian approximation by specifying an implicit prior over learned models.

        Returns
        -------
        Sequence[TimeSeries]
            Returns one or more forecasts for time series.
        """

        # We need to call super's super's method directly, because GlobalForecastingModel expects series:
        ForecastingModel.predict(self, n, num_samples)

        self._verify_inference_dataset_type(input_series_dataset)

        # check that covariates and dimensions are matching what we had during training
        self._verify_predict_sample(input_series_dataset[0])

        if roll_size is None:
            roll_size = self.output_chunk_length
        else:
            raise_if_not(
                0 < roll_size <= self.output_chunk_length,
                "`roll_size` must be an integer between 1 and `self.output_chunk_length`.",
            )

        # check that `num_samples` is a positive integer
        raise_if_not(num_samples > 0, "`num_samples` must be a positive integer.")

        # iterate through batches to produce predictions
        batch_size = batch_size or self.batch_size

        # set prediction parameters
        self.model.set_predict_parameters(
            n=n,
            num_samples=num_samples,
            roll_size=roll_size,
            batch_size=batch_size,
            n_jobs=n_jobs,
        )

        pred_loader = DataLoader(
            input_series_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_loader_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=self._batch_collate_fn,
        )

        # set mc_dropout rate
        self.model.set_mc_dropout(mc_dropout)

        # set up trainer. use user supplied trainer or create a new trainer from scratch
        self.trainer = self._setup_trainer(
            trainer=trainer, verbose=verbose, epochs=self.n_epochs
        )

        # prediction output comes as nested list: list of predicted `TimeSeries` for each batch.
        predictions = self.trainer.predict(self.model, pred_loader)
        # flatten and return
        return [ts for batch in predictions for ts in batch]

    @property
    def first_prediction_index(self) -> int:
        """
        Returns the index of the first predicted within the output of self.model.
        """
        return 0

    @property
    def min_train_series_length(self) -> int:
        """
        Class property defining the minimum required length for the training series;
        overriding the default value of 3 of ForecastingModel
        """
        return self.input_chunk_length + self.output_chunk_length

    @staticmethod
    def _batch_collate_fn(batch: List[Tuple]) -> Tuple:
        """
        Returns a batch Tuple from a list of samples
        """
        aggregated = []
        first_sample = batch[0]
        for i in range(len(first_sample)):
            elem = first_sample[i]
            if isinstance(elem, np.ndarray):
                aggregated.append(
                    torch.from_numpy(np.stack([sample[i] for sample in batch], axis=0))
                )
            elif elem is None:
                aggregated.append(None)
            elif isinstance(elem, TimeSeries):
                aggregated.append([sample[i] for sample in batch])
        return tuple(aggregated)

    def save(self, path: Optional[str] = None) -> None:
        """
        Saves the model under a given path.

        Creates two files under ``path`` (model object) and ``path``.ckpt (checkpoint).

        Example for saving and loading a :class:`RNNModel`:

            .. highlight:: python
            .. code-block:: python

                from darts.models import RNNModel

                model = RNNModel(input_chunk_length=4)

                model.save("my_model.pt")
                model_loaded = RNNModel.load("my_model.pt")
            ..

        Parameters
        ----------
        path
            Path under which to save the model at its current state. If no path is specified, the model is automatically
            saved under ``"{ModelClass}_{YYYY-mm-dd_HH:MM:SS}.pt"``. E.g., ``"RNNModel_2020-01-01_12:00:00.pt"``.
        """
        if path is None:
            # default path
            path = self._default_save_path() + ".pt"

        # save the TorchForecastingModel (does not save the PyTorch LightningModule, and Trainer)
        with open(path, "wb") as f_out:
            torch.save(self, f_out)

        # save the LightningModule checkpoint
        if self.trainer is not None:
            path_ptl_ckpt = path + ".ckpt"
            self.trainer.save_checkpoint(path_ptl_ckpt)

    @staticmethod
    def load(path: str, **kwargs) -> "TorchForecastingModel":
        """
        Loads a model from a given file path.

        Example for loading a general save from :class:`RNNModel`:

            .. highlight:: python
            .. code-block:: python

                from darts.models import RNNModel

                model_loaded = RNNModel.load(path)
            ..

        Example for loading an :class:`RNNModel` to CPU that was saved on GPU:

            .. highlight:: python
            .. code-block:: python

                from darts.models import RNNModel

                model_loaded = RNNModel.load(path, map_location="cpu")
                model_loaded.to_cpu()
            ..

        Parameters
        ----------
        path
            Path from which to load the model. If no path was specified when saving the model, the automatically
            generated path ending with ".pt" has to be provided.
        **kwargs
            Additional kwargs for PyTorch Lightning's :func:`LightningModule.load_from_checkpoint()` method,
            such as ``map_location`` to load the model onto a different device than the one from which it was saved.
            For more information, read the `official documentation <https://pytorch-lightning.readthedocs.io/en/stable/
            common/lightning_module.html#load-from-checkpoint>`_.
        """

        # load the base TorchForecastingModel (does not contain the actual PyTorch LightningModule)
        with open(path, "rb") as fin:
            model: TorchForecastingModel = torch.load(
                fin, map_location=kwargs.get("map_location", None)
            )

        # if a checkpoint was saved, we also load the PyTorch LightningModule from checkpoint
        path_ptl_ckpt = path + ".ckpt"
        if os.path.exists(path_ptl_ckpt):
            model.model = model._load_from_checkpoint(path_ptl_ckpt, **kwargs)
        return model

    @staticmethod
    def load_from_checkpoint(
        model_name: str,
        work_dir: str = None,
        file_name: str = None,
        best: bool = True,
        **kwargs,
    ) -> "TorchForecastingModel":
        """
        Load the model from automatically saved checkpoints under '{work_dir}/darts_logs/{model_name}/checkpoints/'.
        This method is used for models that were created with ``save_checkpoints=True``.

        If you manually saved your model, consider using :meth:`load() <TorchForecastingModel.load()>`.

        Example for loading a :class:`RNNModel` from checkpoint (``model_name`` is the ``model_name`` used at model
        creation):

            .. highlight:: python
            .. code-block:: python

                from darts.models import RNNModel

                model_loaded = RNNModel.load_from_checkpoint(model_name, best=True)
            ..

        If ``file_name`` is given, returns the model saved under
        '{work_dir}/darts_logs/{model_name}/checkpoints/{file_name}'.

        If ``file_name`` is not given, will try to restore the best checkpoint (if ``best`` is ``True``) or the most
        recent checkpoint (if ``best`` is ``False`` from '{work_dir}/darts_logs/{model_name}/checkpoints/'.

        Example for loading an :class:`RNNModel` checkpoint to CPU that was saved on GPU:

            .. highlight:: python
            .. code-block:: python

                from darts.models import RNNModel

                model_loaded = RNNModel.load_from_checkpoint(model_name, best=True, map_location="cpu")
                model_loaded.to_cpu()
            ..

        Parameters
        ----------
        model_name
            The name of the model (used to retrieve the checkpoints folder's name).
        work_dir
            Working directory (containing the checkpoints folder). Defaults to current working directory.
        file_name
            The name of the checkpoint file. If not specified, use the most recent one.
        best
            If set, will retrieve the best model (according to validation loss) instead of the most recent one. Only
            is ignored when ``file_name`` is given.
        **kwargs
            Additional kwargs for PyTorch Lightning's :func:`LightningModule.load_from_checkpoint()` method,
            such as ``map_location`` to load the model onto a different device than the one from which it was saved.
            For more information, read the `official documentation <https://pytorch-lightning.readthedocs.io/en/stable/
            common/lightning_module.html#load-from-checkpoint>`_.


        Returns
        -------
        TorchForecastingModel
            The corresponding trained :class:`TorchForecastingModel`.
        """

        if work_dir is None:
            work_dir = os.path.join(os.getcwd(), DEFAULT_DARTS_FOLDER)

        checkpoint_dir = _get_checkpoint_folder(work_dir, model_name)
        model_dir = _get_runs_folder(work_dir, model_name)

        # load the base TorchForecastingModel (does not contain the actual PyTorch LightningModule)
        base_model_path = os.path.join(model_dir, INIT_MODEL_NAME)
        raise_if_not(
            os.path.exists(base_model_path),
            f"Could not find base model save file `{INIT_MODEL_NAME}` in {model_dir}.",
            logger,
        )
        model = TorchForecastingModel.load(base_model_path, **kwargs)

        # load PyTorch LightningModule from checkpoint
        # if file_name is None, find most recent file in savepath that is a checkpoint
        if file_name is None:
            file_name = _get_checkpoint_fname(work_dir, model_name, best=best)

        file_path = os.path.join(checkpoint_dir, file_name)
        logger.info(f"loading {file_name}")

        model.model = model._load_from_checkpoint(file_path, **kwargs)
        model.load_ckpt_path = file_path
        return model

    def _load_from_checkpoint(self, file_path, **kwargs):
        """Loads a checkpoint for the underlying :class:`PLForecastingModule` (PLM) model.
        The PLM object is not stored when saving a :class:`TorchForecastingModel` (TFM) to avoid saving
        the model twice. Instead, we recover the module class with the module path and class name stored
        in the TFM object. With the recovered module class, we can load the checkpoint.
        """
        pl_module_cls = getattr(sys.modules[self._module_path], self._module_name)
        return pl_module_cls.load_from_checkpoint(file_path, **kwargs)

    def to_cpu(self):
        """Updates the PyTorch Lightning Trainer parameters to move the model to CPU the next time :fun:`fit()` or
        :func:`predict()` is called.
        """
        self.trainer_params["accelerator"] = "cpu"
        self.trainer_params = {
            k: v
            for k, v in self.trainer_params.items()
            if k not in ["devices", "auto_select_gpus"]
        }

    @property
    def model_created(self) -> bool:
        return self.model is not None

    @property
    def epochs_trained(self) -> int:
        return self.model.epochs_trained if self.model_created else 0

    @property
    def likelihood(self) -> Likelihood:
        return (
            self.model.likelihood
            if self.model_created
            else self.pl_module_params.get("likelihood", None)
        )

    @property
    def input_chunk_length(self) -> int:
        return (
            self.model.input_chunk_length
            if self.model_created
            else self.pl_module_params["input_chunk_length"]
        )

    @property
    def output_chunk_length(self) -> int:
        return (
            self.model.output_chunk_length
            if self.model_created
            else self.pl_module_params["output_chunk_length"]
        )

    def _is_probabilistic(self) -> bool:
        return (
            self.model._is_probabilistic()
            if self.model_created
            else True  # all torch models can be probabilistic (via Dropout)
        )

    def __getstate__(self):
        # do not pickle the PyTorch LightningModule, and Trainer
        return {k: v for k, v in self.__dict__.items() if k not in TFM_ATTRS_NO_PICKLE}

    def __setstate__(self, d):
        self.__dict__ = d
        # upon loading the pickled object, add back the PyTorch LightningModule, and Trainer attribute with
        # default values
        for attr, default_val in TFM_ATTRS_NO_PICKLE.items():
            setattr(self, attr, default_val)


def _raise_if_wrong_type(obj, exp_type, msg="expected type {}, got: {}"):
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


def _basic_compare_sample(train_sample: Tuple, predict_sample: Tuple):
    """
    For all models relying on one type of covariates only (Past, Future, Dual), we can rely on the fact
    that training/inference datasets have target and covariates in first and second position to do the checks.
    """
    tgt_train, cov_train, static_train = train_sample[:2] + (train_sample[-2],)
    tgt_pred, cov_pred, static_pred = predict_sample[:2] + (predict_sample[-2],)
    raise_if_not(
        tgt_train.shape[-1] == tgt_pred.shape[-1],
        "The provided target has a dimension (width) that does not match the dimension "
        "of the target this model has been trained on.",
    )
    for (c_train, c_pred), c_descr in zip(
        [(cov_train, cov_pred), (static_train, static_pred)],
        ["past or future covariates", "static covariates"],
    ):
        raise_if(
            c_train is not None and c_pred is None,
            f"This model has been trained with {c_descr}; covariates of matching dimensionality are required "
            f"for prediction.",
        )
        raise_if(
            c_train is None and c_pred is not None,
            f"This model has been trained without {c_descr}. No {c_descr} should be provided for prediction.",
        )
        raise_if(
            c_train is not None
            and c_pred is not None
            and (
                c_train.shape[-1] != c_pred.shape[-1]
                if c_descr != "static covariates"
                else c_train.shape != c_pred.shape
            ),
            f"The provided {c_descr} must have dimensionality matching that of the covariates used for training "
            "the model.",
        )


def _mixed_compare_sample(train_sample: Tuple, predict_sample: Tuple):
    """
    For models relying on MixedCovariates.

    Parameters
    ----------
    train_sample
        (past_target, past_covariates, historic_future_covariates, future_covariates, future_target)
    predict_sample
        (past_target, past_covariates, historic_future_covariates, future_covariates, future_past_covariates, ts_target)
    """
    # datasets; we skip future_target for train and predict, and skip future_past_covariates for predict datasets
    ds_names = [
        "past_target",
        "past_covariates",
        "historic_future_covariates",
        "future_covariates",
        "static_covariates",
    ]

    train_has_ds = [ds is not None for ds in train_sample[:-1]]
    predict_has_ds = [
        ds is not None for ds in predict_sample[:4] + (predict_sample[5],)
    ]

    train_datasets = train_sample[:-1]
    predict_datasets = predict_sample[:4] + (predict_sample[5],)

    tgt_train, tgt_pred = train_datasets[0], predict_datasets[0]
    raise_if_not(
        tgt_train.shape[-1] == tgt_pred.shape[-1],
        "The provided target has a dimension (width) that does not match the dimension "
        "of the target this model has been trained on.",
    )

    for idx, (ds_in_train, ds_in_predict, ds_name) in enumerate(
        zip(train_has_ds, predict_has_ds, ds_names)
    ):
        raise_if(
            ds_in_train and not ds_in_predict,
            f"This model has been trained with `{ds_name}`; some `{ds_name}` of matching dimensionality are needed "
            f"for prediction.",
        )
        raise_if(
            not ds_in_train and ds_in_predict,
            f"This model has been trained without `{ds_name}`; No `{ds_name}` should be provided for prediction.",
        )
        raise_if(
            ds_in_train
            and ds_in_predict
            and (
                train_datasets[idx].shape[-1] != predict_datasets[idx].shape[-1]
                if ds_name != "static_covariates"
                else train_datasets[idx].shape != predict_datasets[idx].shape
            ),
            f"The provided `{ds_name}` must have equal dimensionality as the `{ds_name}` used for training the model.",
        )


class PastCovariatesTorchModel(TorchForecastingModel, ABC):

    supports_future_covariates = False

    def _build_train_dataset(
        self,
        target: Sequence[TimeSeries],
        past_covariates: Optional[Sequence[TimeSeries]],
        future_covariates: Optional[Sequence[TimeSeries]],
        max_samples_per_ts: Optional[int],
    ) -> PastCovariatesTrainingDataset:

        raise_if_not(
            future_covariates is None,
            "Specified future_covariates for a PastCovariatesModel (only past_covariates are expected).",
        )

        return PastCovariatesSequentialDataset(
            target_series=target,
            covariates=past_covariates,
            input_chunk_length=self.input_chunk_length,
            output_chunk_length=self.output_chunk_length,
            max_samples_per_ts=max_samples_per_ts,
            use_static_covariates=self._supports_static_covariates(),
        )

    def _build_inference_dataset(
        self,
        target: Sequence[TimeSeries],
        n: int,
        past_covariates: Optional[Sequence[TimeSeries]],
        future_covariates: Optional[Sequence[TimeSeries]],
    ) -> PastCovariatesInferenceDataset:

        raise_if_not(
            future_covariates is None,
            "Specified future_covariates for a PastCovariatesModel (only past_covariates are expected).",
        )

        return PastCovariatesInferenceDataset(
            target_series=target,
            covariates=past_covariates,
            n=n,
            input_chunk_length=self.input_chunk_length,
            output_chunk_length=self.output_chunk_length,
            use_static_covariates=self._supports_static_covariates(),
        )

    def _verify_train_dataset_type(self, train_dataset: TrainingDataset):
        _raise_if_wrong_type(train_dataset, PastCovariatesTrainingDataset)

    def _verify_inference_dataset_type(self, inference_dataset: InferenceDataset):
        _raise_if_wrong_type(inference_dataset, PastCovariatesInferenceDataset)

    def _verify_predict_sample(self, predict_sample: Tuple):
        _basic_compare_sample(self.train_sample, predict_sample)

    def _verify_past_future_covariates(self, past_covariates, future_covariates):
        raise_if_not(
            future_covariates is None,
            "Some future_covariates have been provided to a PastCovariates model. These models "
            "support only past_covariates.",
        )

    @property
    def _model_encoder_settings(
        self,
    ) -> Tuple[int, int, bool, bool, Optional[List[int]], Optional[List[int]]]:
        input_chunk_length = self.input_chunk_length
        output_chunk_length = self.output_chunk_length
        takes_past_covariates = True
        takes_future_covariates = False
        return (
            input_chunk_length,
            output_chunk_length,
            takes_past_covariates,
            takes_future_covariates,
            None,
            None,
        )

    @property
    def extreme_lags(self):
        return (
            -self.input_chunk_length,
            self.output_chunk_length,
            -self.input_chunk_length if self.uses_past_covariates else None,
            None,
            None,
        )


class FutureCovariatesTorchModel(TorchForecastingModel, ABC):

    supports_past_covariates = False

    def _build_train_dataset(
        self,
        target: Sequence[TimeSeries],
        past_covariates: Optional[Sequence[TimeSeries]],
        future_covariates: Optional[Sequence[TimeSeries]],
        max_samples_per_ts: Optional[int],
    ) -> FutureCovariatesTrainingDataset:
        raise_if_not(
            past_covariates is None,
            "Specified past_covariates for a FutureCovariatesModel (only future_covariates are expected).",
        )

        return FutureCovariatesSequentialDataset(
            target_series=target,
            covariates=future_covariates,
            input_chunk_length=self.input_chunk_length,
            output_chunk_length=self.output_chunk_length,
            max_samples_per_ts=max_samples_per_ts,
            use_static_covariates=self._supports_static_covariates(),
        )

    def _build_inference_dataset(
        self,
        target: Sequence[TimeSeries],
        n: int,
        past_covariates: Optional[Sequence[TimeSeries]],
        future_covariates: Optional[Sequence[TimeSeries]],
    ) -> FutureCovariatesInferenceDataset:
        raise_if_not(
            past_covariates is None,
            "Specified past_covariates for a FutureCovariatesModel (only future_covariates are expected).",
        )

        return FutureCovariatesInferenceDataset(
            target_series=target,
            covariates=future_covariates,
            n=n,
            input_chunk_length=self.input_chunk_length,
            use_static_covariates=self._supports_static_covariates(),
        )

    def _verify_train_dataset_type(self, train_dataset: TrainingDataset):
        _raise_if_wrong_type(train_dataset, FutureCovariatesTrainingDataset)

    def _verify_inference_dataset_type(self, inference_dataset: InferenceDataset):
        _raise_if_wrong_type(inference_dataset, FutureCovariatesInferenceDataset)

    def _verify_predict_sample(self, predict_sample: Tuple):
        _basic_compare_sample(self.train_sample, predict_sample)

    def _verify_past_future_covariates(self, past_covariates, future_covariates):
        raise_if_not(
            past_covariates is None,
            "Some past_covariates have been provided to a PastCovariates model. These models "
            "support only future_covariates.",
        )

    @property
    def _model_encoder_settings(
        self,
    ) -> Tuple[int, int, bool, bool, Optional[List[int]], Optional[List[int]]]:
        input_chunk_length = self.input_chunk_length
        output_chunk_length = self.output_chunk_length
        takes_past_covariates = False
        takes_future_covariates = True
        return (
            input_chunk_length,
            output_chunk_length,
            takes_past_covariates,
            takes_future_covariates,
            None,
            None,
        )

    @property
    def extreme_lags(self):
        return (
            -self.input_chunk_length,
            self.output_chunk_length,
            None,
            0 if self.uses_future_covariates else None,
            self.output_chunk_length if self.uses_future_covariates else None,
        )


class DualCovariatesTorchModel(TorchForecastingModel, ABC):
    supports_past_covariates = False

    def _build_train_dataset(
        self,
        target: Sequence[TimeSeries],
        past_covariates: Optional[Sequence[TimeSeries]],
        future_covariates: Optional[Sequence[TimeSeries]],
        max_samples_per_ts: Optional[int],
    ) -> DualCovariatesTrainingDataset:

        return DualCovariatesSequentialDataset(
            target_series=target,
            covariates=future_covariates,
            input_chunk_length=self.input_chunk_length,
            output_chunk_length=self.output_chunk_length,
            max_samples_per_ts=max_samples_per_ts,
            use_static_covariates=self._supports_static_covariates(),
        )

    def _build_inference_dataset(
        self,
        target: Sequence[TimeSeries],
        n: int,
        past_covariates: Optional[Sequence[TimeSeries]],
        future_covariates: Optional[Sequence[TimeSeries]],
    ) -> DualCovariatesInferenceDataset:

        return DualCovariatesInferenceDataset(
            target_series=target,
            covariates=future_covariates,
            n=n,
            input_chunk_length=self.input_chunk_length,
            output_chunk_length=self.output_chunk_length,
            use_static_covariates=self._supports_static_covariates(),
        )

    def _verify_train_dataset_type(self, train_dataset: TrainingDataset):
        _raise_if_wrong_type(train_dataset, DualCovariatesTrainingDataset)

    def _verify_inference_dataset_type(self, inference_dataset: InferenceDataset):
        _raise_if_wrong_type(inference_dataset, DualCovariatesInferenceDataset)

    def _verify_predict_sample(self, predict_sample: Tuple):
        _basic_compare_sample(self.train_sample, predict_sample)

    def _verify_past_future_covariates(self, past_covariates, future_covariates):
        raise_if_not(
            past_covariates is None,
            "Some past_covariates have been provided to a DualCovariates Torch model. These models "
            "support only future_covariates.",
        )

    @property
    def _model_encoder_settings(
        self,
    ) -> Tuple[int, int, bool, bool, Optional[List[int]], Optional[List[int]]]:
        input_chunk_length = self.input_chunk_length
        output_chunk_length = self.output_chunk_length
        takes_past_covariates = False
        takes_future_covariates = True
        return (
            input_chunk_length,
            output_chunk_length,
            takes_past_covariates,
            takes_future_covariates,
            None,
            None,
        )

    @property
    def extreme_lags(self):
        return (
            -self.input_chunk_length,
            self.output_chunk_length,
            None,
            -self.input_chunk_length if self.uses_future_covariates else None,
            self.output_chunk_length if self.uses_future_covariates else None,
        )


class MixedCovariatesTorchModel(TorchForecastingModel, ABC):
    def _build_train_dataset(
        self,
        target: Sequence[TimeSeries],
        past_covariates: Optional[Sequence[TimeSeries]],
        future_covariates: Optional[Sequence[TimeSeries]],
        max_samples_per_ts: Optional[int],
    ) -> MixedCovariatesTrainingDataset:

        return MixedCovariatesSequentialDataset(
            target_series=target,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            input_chunk_length=self.input_chunk_length,
            output_chunk_length=self.output_chunk_length,
            max_samples_per_ts=max_samples_per_ts,
            use_static_covariates=self._supports_static_covariates(),
        )

    def _build_inference_dataset(
        self,
        target: Sequence[TimeSeries],
        n: int,
        past_covariates: Optional[Sequence[TimeSeries]],
        future_covariates: Optional[Sequence[TimeSeries]],
    ) -> MixedCovariatesInferenceDataset:

        return MixedCovariatesInferenceDataset(
            target_series=target,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            n=n,
            input_chunk_length=self.input_chunk_length,
            output_chunk_length=self.output_chunk_length,
            use_static_covariates=self._supports_static_covariates(),
        )

    def _verify_train_dataset_type(self, train_dataset: TrainingDataset):
        _raise_if_wrong_type(train_dataset, MixedCovariatesTrainingDataset)

    def _verify_inference_dataset_type(self, inference_dataset: InferenceDataset):
        _raise_if_wrong_type(inference_dataset, MixedCovariatesInferenceDataset)

    def _verify_predict_sample(self, predict_sample: Tuple):
        _mixed_compare_sample(self.train_sample, predict_sample)

    def _verify_past_future_covariates(self, past_covariates, future_covariates):
        # both covariates are supported; do nothing
        pass

    @property
    def _model_encoder_settings(
        self,
    ) -> Tuple[int, int, bool, bool, Optional[List[int]], Optional[List[int]]]:
        input_chunk_length = self.input_chunk_length
        output_chunk_length = self.output_chunk_length
        takes_past_covariates = True
        takes_future_covariates = True
        return (
            input_chunk_length,
            output_chunk_length,
            takes_past_covariates,
            takes_future_covariates,
            None,
            None,
        )

    @property
    def extreme_lags(self):
        return (
            -self.input_chunk_length,
            self.output_chunk_length,
            -self.input_chunk_length if self.uses_past_covariates else None,
            -self.input_chunk_length if self.uses_future_covariates else None,
            self.output_chunk_length if self.uses_future_covariates else None,
        )


class SplitCovariatesTorchModel(TorchForecastingModel, ABC):
    def _build_train_dataset(
        self,
        target: Sequence[TimeSeries],
        past_covariates: Optional[Sequence[TimeSeries]],
        future_covariates: Optional[Sequence[TimeSeries]],
        max_samples_per_ts: Optional[int],
    ) -> SplitCovariatesTrainingDataset:

        return SplitCovariatesSequentialDataset(
            target_series=target,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            input_chunk_length=self.input_chunk_length,
            output_chunk_length=self.output_chunk_length,
            max_samples_per_ts=max_samples_per_ts,
            use_static_covariates=self._supports_static_covariates(),
        )

    def _build_inference_dataset(
        self,
        target: Sequence[TimeSeries],
        n: int,
        past_covariates: Optional[Sequence[TimeSeries]],
        future_covariates: Optional[Sequence[TimeSeries]],
    ) -> SplitCovariatesInferenceDataset:

        return SplitCovariatesInferenceDataset(
            target_series=target,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            n=n,
            input_chunk_length=self.input_chunk_length,
            output_chunk_length=self.output_chunk_length,
            use_static_covariates=self._supports_static_covariates(),
        )

    def _verify_train_dataset_type(self, train_dataset: TrainingDataset):
        _raise_if_wrong_type(train_dataset, SplitCovariatesTrainingDataset)

    def _verify_inference_dataset_type(self, inference_dataset: InferenceDataset):
        _raise_if_wrong_type(inference_dataset, SplitCovariatesInferenceDataset)

    def _verify_past_future_covariates(self, past_covariates, future_covariates):
        # both covariates are supported; do nothing
        pass

    def _verify_predict_sample(self, predict_sample: Tuple):
        # TODO: we have to check both past and future covariates
        raise NotImplementedError()

    @property
    def _model_encoder_settings(
        self,
    ) -> Tuple[int, int, bool, bool, Optional[List[int]], Optional[List[int]]]:
        input_chunk_length = self.input_chunk_length
        output_chunk_length = self.output_chunk_length
        takes_past_covariates = True
        takes_future_covariates = True
        return (
            input_chunk_length,
            output_chunk_length,
            takes_past_covariates,
            takes_future_covariates,
            None,
            None,
        )

    @property
    def extreme_lags(self):
        return (
            -self.input_chunk_length,
            self.output_chunk_length,
            -self.input_chunk_length if self.uses_past_covariates else None,
            0 if self.uses_future_covariates else None,
            self.output_chunk_length if self.uses_future_covariates else None,
        )
