"""
    This file contains the basic structures to define, run and backup benchmarking experiments
"""

import datetime
import inspect
import os
import shutil
from abc import ABC, abstractmethod
from glob import glob
from typing import List, Optional, Sequence, Tuple, Union
from tqdm import tqdm
from darts.utils.utils import series2seq
from darts.dataprocessing.pipeline import Pipeline
from darts.metrics import mse, mae, smape, rmse, mape
import torch
from pytorch_lightning.callbacks import Callback, EarlyStopping
from typing import Callable
from darts import TimeSeries

import optuna
from optuna.integration import PyTorchLightningPruningCallback
from optuna.visualization import (
    plot_optimization_history,
    plot_contour,
    plot_param_importances,
)

from darts.logging import (
    get_logger,
    raise_if,
    raise_if_not,
    raise_log,
)

from darts.models import (
    StatsForecastAutoARIMA, StatsForecastETS, LinearRegressionModel, FourTheta, Theta, DLinearModel, NBEATSModel,
    NHiTSModel, NLinearModel, TFTModel, TransformerModel, LightGBMModel, CatBoostModel, XGBModel, RegressionEnsembleModel,
    TCNModel
)
from darts.datasets import (
    ETTh1Dataset, ETTh2Dataset, ETTm1Dataset, ETTm2Dataset, ElectricityDataset, TrafficDataset, WeatherDataset,
    ILINetDataset, ExchangeRateDataset
)

logger = get_logger(__name__)

DEFAULT_RADNOM_SEED = 42
DEFAULT_EXP_ROOT = "./experiment_runs/"

DATASETS = [ETTh1Dataset, ETTh2Dataset, ETTm1Dataset, ETTm2Dataset, ElectricityDataset, TrafficDataset, WeatherDataset,
    ILINetDataset, ExchangeRateDataset]

METRICS = [mse, mae, smape, rmse, mape]

class BaseExperiment():
    """
    Abstract class to define the generic structure of an experiment.

    Parameters
    ----------
    experiment_root
        Root directory path where all experiments information are going to be stored.
    dataset
        The dataset to train on.
    """
    # list here all possible attributes of an experiment (of any type)
    experiment_root = None
    experiment_dir = None
    random_state = None
    start_exp_time = None
    experiment_name = None
    val_len = None
    split = None
    orig_train = None # dataset
    orig_val = None # dataset
    train = None # dataset
    val = None # dataset
    test = None # dataset
    models_cls = None
    dataset = None
    transformers_pipeline = None
    transformed_val = None #bool
    test_predictions = {} #dict with keys as model names and values as timeseries predictions
    test_stats = {}
    verbose = None

    def __init__(self, experiment_root:Optional[str] = None, random_state: Optional[int] = None,
                 experiment_name: Optional[str] = None, dataset: Union[TimeSeries, Sequence[TimeSeries]] = None):
        super().__init__()

        self.experiment_root = experiment_root if experiment_root is not None else DEFAULT_EXP_ROOT
        self.random_state = random_state if random_state is not None else DEFAULT_RADNOM_SEED
        self.dataset = dataset

        #check if experiment_dir exists and create it if not
        if not os.path.exists(self.experiment_root):
            os.makedirs(self.experiment_root)

        #store experiment start time
        self.start_exp_time = datetime.datetime.now()

        self.experiment_name = experiment_name

        #make experiment_root and specific experiment directory
        try:
            os.mkdir(self.experiment_root)
            print(f"Root directory {self.experiment_root} created")
        except:
            print(f"Root directory {self.experiment_root} already exists")

        try:
            self.experiment_root = os.path.join(self.experiment_root, dataset.__name__)
            os.mkdir(self.experiment_root)
            print(f"Root directory {self.experiment_root} created")
        except:
            print(f"Root directory {self.experiment_root} already exists")

        try:
            self.experiment_dir = os.path.join(self.experiment_root, experiment_name)
            os.mkdir(self.experiment_dir)
            print(f"Experiment directory {self.experiment_dir} created")
        except:
            print(f"Experiment directory {self.experiment_dir} already exists")

    def run(self):
        """
        Run the experiment.
        """
        pass


    def backup(self):
        """
        Backup the experiment.
        """
        pass

    def _get_data(self, load_multivariate = False, split = 0.8, val_len = None):
        """
        Get the data.
        """
        # read data
        if "multivariate" in self.dataset.__init__.__code__.co_varnames:
            dataset = self.dataset(multivariate=load_multivariate).load()
        else:
            dataset = self.dataset().load()

        dataset = series2seq(dataset)

        # split data
        if val_len is not None:
            train = [s[:-(2 * val_len)] for s in dataset]
            val = [s[-(2 * val_len):-val_len] for s in dataset]
            test = [s[-val_len:] for s in dataset]
            self.val_len = val_len
        else:
            all_splits = [list(s.split_after(split)) for s in dataset]
            train = [split[0] for split in all_splits]
            vals = [split[1] for split in all_splits]
            vals = [list(s.split_after(0.5)) for s in vals]
            test = [s[1] for s in vals]
            val = [s[0] for s in vals]
            self.split = split

        self.orig_train = train
        self.orig_val = val
        self.test = test

        self.train = train
        self.val = val

        return self

    def _preprocess_data(self, transformers: List[dict], transform_val: bool = False):
        """
        Preprocess the data with transformers.

        Parameters
        ----------
            transforms: List[dict] a list of dictionaries containing the transformers and their corresponding parameters.
            It is expected to be of the form [{"transformer": transformer_cls, "params": params_kwargs}, ...]
            e.g., [{"transformer": Scaler, "params": {"scaler": MaxAbsScaler()}},
                  {"transformer": MissingValuesFiller, "params": {"fill": 0}}]
            The transformers in the list will be chained together is a pipeline.
        """
        transformers_instances = [transformer["transformer"](**transformer["params"]) for transformer in transformers]
        transformers_pipeline = Pipeline(transformers_instances)

        # fit transforms on train and transform train
        self.train = transformers_pipeline.fit_transform(self.orig_train)

        # transform val
        if transform_val:
            self.transformed_val = True
            self.val = transformers_pipeline.transform(self.orig_val)
        else:
            self.transformed_val = False

        # save the transformers
        self.transformers_pipeline = transformers_pipeline
        return self

    def _postprocess_predictions(self, model):
        """
        Post-process the predictions.
        """

        if self.transformers_pipeline.invertible():

            self.val_predictions[model] = self.transformers_pipeline.inverse_transform(self.val_predictions[model])

        return self

    def _explore_data(self):
        """
        Explore the data.
        """
        pass

    # load model
    def _load_model(self):
        """
        Load the model.
        """
        pass

    # optimize hyper-parameters
    def _optimize_hyperparameters(self):
        """
        Optimize hyper-parameters.
        """
        pass

    # train model
    def _train_model(self):
        """
        Train the model.
        """
        self.models_trained = True
        pass

    def _predict_model(self):
        """
        Predict the model.
        """
        pass

    # validate model
    def _compute_validation_stats(self, predictions, ground_truth, metric):
        """
        Validate the model.
        """
        metric_eval = metric(ground_truth, predictions, n_jobs = -1, verbose = self.verbose)

        return np.mean(metric_eval)

    # test model
    def _test_model(self):
        """
        Test the model.
        """
        pass

    def _backup_exp(self):
        """
        Backup the experiment.
        """
        pass

class StatsExperiment(BaseExperiment):
    """
    Experiment class to run the baseline statistical models.
    """

    MODELS = [StatsForecastAutoARIMA, StatsForecastETS, FourTheta, Theta] # TODO: addoptimized theta, complex ES
    NAME = "Stats_Experiment"

    def __init__(self, experiment_root:Optional[str] = None, random_state: Optional[int] = None,
                 experiment_name: Optional[str] = None, datasets: Optional[str] = None,
                 models: Optional[List[str]] = None, verbose: Optional[bool] = False):

        experiment_name = experiment_name if experiment_name is not None else self.NAME

        super().__init__(experiment_root, random_state, experiment_name)

        self.datasets_cls = datasets if datasets is not None else DATASETS
        self.models_cls = models if models is not None else self.MODELS
        self.verbose = verbose

    def run(self):
        super().run()

        # load the data
        self._get_data()
        # train model
        self._train_model(self.train)
        # predict model
        self._predict_model(self.val)
        # validate model
        self._compute_validation_stats()
        # backup experiment
        self._backup_exp()


    def _backup_exp(self):
        """
        Backup the experiment.
        """
        pass

class MLExperiment(BaseExperiment):
    """
        Experiment class to run machine learning models.
    """

    MODELS = [LinearRegressionModel, LightGBMModel, CatBoostModel, XGBModel]
    NAME = "ML_Experiments"

    def __init__(self, experiment_root:Optional[str] = None, random_state: Optional[int] = None,
                 experiment_name: Optional[str] = None, datasets: Optional[str] = None,
                 models: Optional[List[str]] = None, verbose: Optional[bool] = False):

        experiment_name = experiment_name if experiment_name is not None else self.NAME

        super().__init__(experiment_root, random_state, experiment_name)

        self.datasets_cls = datasets if datasets is not None else DATASETS
        self.models_cls = models if models is not None else self.MODELS
        self.verbose = verbose

def DLinearModelBuilder(self):
    pass

def NLinearModelBuilder(self):
    pass

def NBEATSModelBuilder(self):
    pass

def TFTModelBuilder(self):
    pass

def TransformerModelBuilder(self):
    pass

def NHiTSModelBuilder(self):
    pass

def TCNModelBuilder(experiment, in_len, out_len, kernel_size, num_filters, weight_norm, dilation_base,
dropout, lr, encoders = None, likelihood=None):
    torch.manual_seed(experiment.random_state)

    # detect if a GPU is available
    if torch.cuda.is_available():
        experiment.pl_trainer_kwargs = {
            "accelerator": "gpu",
            "gpus": -1,
            "auto_select_gpus": True,
            "callbacks": experiment.callbacks,
        }
        experiment.num_workers = 4
    else:
        experiment.pl_trainer_kwargs = {"callbacks": experiment.callbacks}
        experiment.num_workers = 0

    # build the model
    model = TCNModel(
        input_chunk_length=in_len,
        output_chunk_length=out_len,
        batch_size=experiment.BATCH_SIZE,
        n_epochs=experiment.MAX_N_EPOCHS,
        nr_epochs_val_period=experiment.NR_EPOCHS_VAL_PERIOD,
        kernel_size=kernel_size,
        num_filters=num_filters,
        weight_norm=weight_norm,
        dilation_base=dilation_base,
        dropout=dropout,
        optimizer_kwargs={"lr": lr},
        add_encoders=encoders,
        likelihood=likelihood,
        pl_trainer_kwargs=experiment.pl_trainer_kwargs,
        model_name="tcn_model",
        force_reset=True,
        save_checkpoints=True,
    )
    train = experiment.train
    val = experiment.val
    # train the model
    model.fit(
        series=train,
        val_series=val,
        max_samples_per_ts=experiment.MAX_SAMPLES_PER_TS,
        num_loader_workers=experiment.num_workers,
    )

    # reload best model over course of training
    model = TCNModel.load_from_checkpoint("tcn_model")

    return model

class DLExperiment(BaseExperiment):
    """
            Experiment class to run deep learning models.
    """

    MODELS = [DLinearModel, NLinearModel, NBEATSModel, TFTModel, TransformerModel, NHiTSModel, TCNModel]
    NAME = "DL_Experiments"

    # parameters shared by all models
    BATCH_SIZE = 1024
    MAX_N_EPOCHS = 30
    NR_EPOCHS_VAL_PERIOD = 1
    MAX_SAMPLES_PER_TS = 1000

    def __init__(self, experiment_root: Optional[str] = None, random_state: Optional[int] = None,
                 experiment_name: Optional[str] = None, dataset: Optional[str] = None,
                 models: Optional[List[str]] = None, verbose: Optional[bool] = False,
                 metrics: Optional[List[Callable]] = None,):
        experiment_name = experiment_name if experiment_name is not None else self.NAME

        super().__init__(experiment_root, random_state, experiment_name, dataset = dataset)

        self.models_cls = models if models is not None else self.MODELS
        self.verbose = verbose
        self.metrics = metrics if metrics is not None else METRICS

        # early stopping by monitoring validation loss
        self.early_stopper = EarlyStopping("val_loss", min_delta=0.001, patience=3, verbose=True)

        self.callbacks = [self.early_stopper]

    MODEL_BUILDERS = {DLinearModel: DLinearModelBuilder,
                      NLinearModel: NLinearModelBuilder,
                      NBEATSModel: NBEATSModelBuilder,
                      TFTModel: TFTModelBuilder,
                      TransformerModel: TransformerModelBuilder,
                      NHiTSModel: NHiTSModelBuilder,
                      TCNModel: TCNModelBuilder}

    def _preprocess_data(self, transformers: List[dict], transform_val: bool = True):
        super()._preprocess_data(transformers, transform_val)
        return self

    def _define_optuna_objective(self, params_dict, model_builder, metric):
        """
        Define the objective function for optuna.

        Parameters
        ----------
            params_dict: dict, a dictionary containing the parameters to be optimized.
            It is expected to be of the form {"param_name": (string, list), ...}
            e.g., {"lr": ("float", [0.01, 0.05]), "optimizer": ("categorical", ["Adam", "RMSprop", "SGD"]),
            "batch_size": ("int", [32])}
        """
        def objective(trial):
            # sample the parameters
            params = {}
            for param_name, (param_type, param_range) in params_dict.items():
                if param_type == "float":
                    params[param_name] = trial.suggest_float(param_name, param_range[0], param_range[1])
                elif param_type == "int":
                    params[param_name] = trial.suggest_int(param_name, param_range[0], param_range[1])
                elif param_type == "categorical":
                    params[param_name] = trial.suggest_categorical(param_name, param_range)
                else:
                    raise ValueError(f"Unknown parameter type {param_type}")

            self.pruning_callback = optuna.integration.PyTorchLightningPruningCallback(trial, "val_loss")
            self.callbacks.append(self.pruning_callback)

            params.update({"experiment":self})

            # build the model
            model = model_builder(**params)

            # train the model
            model.fit(self.train)

            # predict the model
            val_predictions = model.predict(n = self.val_len)

            val_predictions = self._postprocess_predictions(val_predictions)

            # compute the validation stats
            val_stats = self._compute_validation_stats(self.val, val_predictions, metric)

            return val_stats

        return objective

    def _optimize_hyperparameters(self, objective_fn, n_trials = None, timeout = 7200, direction = "minimize",
                                  load_if_exists = True):
        """
        Optimize hyper-parameters with optuna

        Parameters
        ----------
            objective_fn: function, the objective function to be optimized
            n_trials: int, the number of trials to be run
            timeout: int, the maximum number of seconds to run the optimization
            direction: str, the direction of the optimization, either "minimize" or "maximize"
            load_if_exists: bool, whether to load the study if it already exists and continue the optimization
        """
        study_name = f"{self.experiment_name}_optuna_study"
        study = optuna.create_study(study_name = study_name, direction=direction, load_if_exists=load_if_exists)
        study.optimize(objective_fn, n_trials=n_trials, timeout=timeout)

        best_prams = study.best_trial.params

        return best_prams

    def _postprocess_predictions(self):
        super()._postprocess_predictions()
        return self

    def run(self, dataset, hyperparams, transformers: List[dict], transform_val: bool = True):
        super().run()

        self._get_data()
        self._preprocess_data(transformers, transform_val)
        for model in self.models_cls:
            for metric in self.metrics:
                optuna_objective = self._define_optuna_objective(hyperparams, self.MODEL_BUILDERS[model], metric)
                best_params = self._optimize_hyperparameters(optuna_objective)
                best_model = self.MODEL_BUILDERS[model](**best_params) # returns a fitted model
                test_predictions = best_model.predict(series = self.val, n = self.val_len)
                test_predictions = self._postprocess_predictions(test_predictions)
                self.test_predictions[model] = test_predictions
                self.test_stats[f"{model.__class__.__name__}_{metric}"] = self._compute_validation_stats(self.test, test_predictions, metric)
                self._backup_exp(dataset, model, metric, best_params)

    def _backup_exp(self):
        pass
