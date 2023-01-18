"""
    This file contains the basic structures to define, run and backup benchmark experiments
"""

from datetime import datetime
import inspect
import random
import os
import shutil
from abc import ABC, abstractmethod
from glob import glob
from typing import List, Optional, Sequence, Tuple, Union
from tqdm import tqdm
from darts.utils.utils import series2seq
from darts.dataprocessing.pipeline import Pipeline
from darts.metrics import mse, mae, smape, rmse, mape, mase
import torch
from pytorch_lightning.callbacks import Callback, EarlyStopping
from typing import Callable
from darts import TimeSeries
import numpy as np
import pickle
from ray import tune, air
from ray.tune.search.optuna import OptunaSearch
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.tune.schedulers import ASHAScheduler
from ray.air import session
import matplotlib.pyplot as plt
import optuna

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
DEFAULT_EXP_ROOT = "experiment_runs"

DATASETS = [ETTh1Dataset, ETTh2Dataset, ETTm1Dataset, ETTm2Dataset, ElectricityDataset, TrafficDataset, WeatherDataset,
    ILINetDataset, ExchangeRateDataset]

METRICS = [mse, mae, smape, rmse, mape]

class BenchmarckStudy():
    """
    Per dataset study, where we read all models (statistics models, ML and DL models) pre-run experiments.
    """
    stats_experiment_results = None
    ml_experiment_results = None
    dl_experiment_resutls = None

    def __init__(self, dataset, experiments_location):
        # get location of all experiments, open folders and read results into objects
        pass
    def plot_metric_vs_models(self):
        pass

    def plot_training_time_vs_models(self):
        pass

    def plot_inference_time_vs_models(self):
        pass


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
    val_predictions = {} #dict with keys as model names and values as timeseries predictions
    test_predictions = {} #dict with keys as model names and values as timeseries predictions
    test_stats = {} #dict with keys as model names and values as evaluation metric values
    models_train_time = {} #dict with keys as model names and values as training time values
    models_inference_time = {} #dict with keys as model names and values as inference time values
    models_best_params = {}
    verbose = None

    def __init__(self, experiment_root:Optional[str] = None, random_state: Optional[int] = None,
                 experiment_name: Optional[str] = None, dataset: Union[TimeSeries, Sequence[TimeSeries]] = None):
        super().__init__()

        self.experiment_root = experiment_root if experiment_root is not None else os.path.join(os.getcwd(), DEFAULT_EXP_ROOT)
        self.random_state = random_state if random_state is not None else DEFAULT_RADNOM_SEED
        self.dataset = dataset

        #Fix random states
        #https://pytorch.org/docs/stable/notes/randomness.html
        random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        torch.use_deterministic_algorithms(True)

        #check if experiment_dir exists and create it if not
        if not os.path.exists(self.experiment_root):
            os.makedirs(self.experiment_root)


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
            self.experiment_dir = os.path.join(self.experiment_root, f"{self.experiment_name}_seed_{self.random_state}")
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
            self.test_len = val_len
        else:
            all_splits = [list(s.split_after(split)) for s in dataset]
            train = [split[0] for split in all_splits]
            vals = [split[1] for split in all_splits]
            vals = [list(s.split_after(0.5)) for s in vals]
            test = [s[1] for s in vals]
            val = [s[0] for s in vals]
            self.split = split
            self.val_len = len(val[0])
            self.test_len = self.val_len

        self.orig_train = train
        self.orig_val = val
        self.test = test

        self.train = train # used when data is transformed
        self.val = val # used when data is transformed

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


    def _postprocess_predictions(self, key, predictions, test = False):
        """
        Post-process the predictions.
        """

        if self.transformers_pipeline is not None and self.transformers_pipeline.invertible():
            transformed = self.transformers_pipeline.inverse_transform(predictions)
            if not test:
                self.val_predictions[key] = transformed
            else:
                self.test_predictions[key] = transformed

        return transformed

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
    def _compute_metric(self, ground_truth, predictions, metric=smape, reduction = np.mean):
        """
        Validate the model.
        """
        if metric.__name__ == "mase":
            metric_evals = metric(ground_truth, predictions, self.train, n_jobs=-1, verbose=True)
        else:
            metric_evals = metric(ground_truth, predictions, n_jobs=-1, verbose=True)

        metric_evals_reduced = reduction(metric_evals) if metric_evals != np.nan else float("inf")
        metric_evals_reduced = metric_evals_reduced if metric_evals_reduced != np.nan else float("inf")

        _std = np.std(metric_evals) if reduction == np.mean else None # over multiple series

        return metric_evals_reduced, _std

    # test model
    def _test_model(self):
        """
        Test the model.
        """
        pass

    def _backup_exp(self):
        with open(f"{self.experiment_dir}/full_{self.experiment_name}_bkp.pkl", 'wb') as f:
            pickle.dump(self, f)

    def check_exp_outputs(self, models, metrics, max_series_to_plot = 5, plot_series = False, comp = 0):
        if metrics is None:
            metrics = self.metrics

        if models is None:
            models = self.models_cls

        if isinstance(comp, int) and comp < len(self.train[0].columns):
            comp = self.orig_train[0].columns[comp]
        elif isinstance(comp, str):
            comp = comp
        else:
            comp = self.orig_train[0].columns[0]

        for model_cl in models:
            model_name = model_cl.__name__
            for metric in metrics:
                metric_name = metric.__name__

                print(f"{model_name} {metric_name}: {self.test_stats[f'{model_name}_{metric_name}']}")

                if plot_series:
                    if len(self.test) < max_series_to_plot:
                        max_series_to_plot = len(self.test)

                    idx_vec = np.random.randint(0, len(self.test), max_series_to_plot)

                    for idx in idx_vec:
                        plt.figure(figsize=(15, 5))
                        self.orig_val[idx][comp][-self.val_len:].plot()
                        self.test[idx][comp].plot(label="actual")
                        selector = f"{model_name}_{metric_name}"
                        self.test_predictions[selector][idx][comp].plot(label="forecast")
                        plt.title(f"{self.dataset.__name__}_{model_name}_{metric_name}_{comp}")
                        plt.show()
                        plt.close()

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

def TCNModelBuilder(in_len, out_len, kernel_size, num_filters, weight_norm, dilation_base,
dropout, lr, experiment, encoders = None, likelihood=None, callbacks = None, work_dir = None):

    early_stopper = EarlyStopping("val_loss", min_delta=0.001, patience=3, verbose=True)
    if callbacks is None:
        callbacks = [early_stopper]
    else:
        callbacks.append(early_stopper)

    # detect if a GPU is available
    if torch.cuda.is_available():
        pl_trainer_kwargs = {
            "accelerator": "gpu",
            "gpus": -1,
            "auto_select_gpus": True,
            "callbacks": callbacks,
        }
        #num_workers = 4
    else:
        pl_trainer_kwargs = {"callbacks": callbacks}
        #num_workers = 0

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
        pl_trainer_kwargs=pl_trainer_kwargs,
        model_name=TCNModel.__name__,
        force_reset=True,
        save_checkpoints=True,
        work_dir = os.path.join(os.getcwd()) if work_dir is None else work_dir
    )

    return model

MODEL_BUILDERS = {DLinearModel.__name__: DLinearModelBuilder,
                      NLinearModel.__name__: NLinearModelBuilder,
                      NBEATSModel.__name__: NBEATSModelBuilder,
                      TFTModel.__name__: TFTModelBuilder,
                      TransformerModel.__name__: TransformerModelBuilder,
                      NHiTSModel.__name__: NHiTSModelBuilder,
                      TCNModel.__name__: TCNModelBuilder}

class DLExperiment(BaseExperiment):
    """
            Experiment class to run deep learning models.
    """

    MODELS = [DLinearModel, NLinearModel, NBEATSModel, TFTModel, TransformerModel, NHiTSModel, TCNModel]

    # parameters shared by all models
    BATCH_SIZE = 1024
    MAX_N_EPOCHS = 30
    NR_EPOCHS_VAL_PERIOD = 1
    MAX_SAMPLES_PER_TS = 1000

    def __init__(self, experiment_root: Optional[str] = None, random_state: Optional[int] = None,
                 experiment_name: Optional[str] = None, dataset: Optional[str] = None,
                 models: Optional[List[str]] = None, verbose: Optional[bool] = False,
                 metrics: Optional[List[Callable]] = None,):

        # store experiment start time
        self.start_exp_time = datetime.now()

        if experiment_name is None:
            experiment_name = f"DL_Experiments_{self.start_exp_time.strftime('%Y-%m-%d')}_pid{os.getpid()}"

        experiment_name = experiment_name

        super().__init__(experiment_root, random_state, experiment_name, dataset = dataset)

        self.models_cls = models if models is not None else self.MODELS
        self.verbose = verbose
        self.metrics = metrics if metrics is not None else METRICS


    @staticmethod
    def _val_loss_objective(config, model_cl, experiment):

        metrics = {"val_metric":"val_loss"}

        callbacks = [TuneReportCallback(metrics, on="validation_end")]

        model = MODEL_BUILDERS[model_cl](**config, callbacks=callbacks, experiment = experiment)

        # train the model
        model.fit(
            series=experiment.train,
            val_series=experiment.val,
            max_samples_per_ts=experiment.MAX_SAMPLES_PER_TS
        )

    @staticmethod
    def _val_metric_objective(config, model_cl, experiment, metric, reduction = np.mean):

        model = MODEL_BUILDERS[model_cl.__name__](**config, experiment=experiment)

        # train the model
        model.fit(
            series=experiment.train,
            val_series=experiment.val,
            max_samples_per_ts=experiment.MAX_SAMPLES_PER_TS
        )

        # use best model for subsequent evaluation
        model = model_cl.load_from_checkpoint(model_cl.__name__, work_dir = os.getcwd(), best = True)

        preds = model.predict(series=experiment.train, n=experiment.val_len)

        metric_evals,_ = experiment._compute_metric(experiment.val, preds, metric, reduction=reduction)

        session.report({"val_metric":metric_evals})

    @staticmethod
    def tune_hyperparameters(experiment, params_space, model_cl, eval_mode = "val_metric_eval", metric = None,
                             reduction = None, sampler = None, num_samples=-1,
                            max_concurrent_trials = 1, time_budget_s = 60, scheduler = None,
                            scheduler_kwargs = None, min_max_mode = "min"):

        metric = metric if metric is not None else smape
        if eval_mode == "val_metric_eval":
            objective = DLExperiment._val_metric_objective
            objective_with_params = tune.with_parameters(objective, model_cl=model_cl, experiment=experiment,
                                                         metric = metric,
                                                         reduction = reduction if reduction is not None else np.mean)
        elif eval_mode == "val_loss":
            objective = DLExperiment._val_loss_objective
            objective_with_params = tune.with_parameters(objective, model_cl=model_cl, experiment=experiment)

        """
        if scheduler is None:
            scheduler_kwargs = {}
            scheduler_kwargs["max_t"] = experiment.MAX_N_EPOCHS
            scheduler_kwargs["grace_period"] = 1
            scheduler_kwargs["reduction_factor"] = 2

            scheduler = ASHAScheduler(**scheduler_kwargs)
        else:
            scheduler = scheduler(**scheduler_kwargs)
        """

        tuner = tune.Tuner(
            objective_with_params,
            tune_config = tune.TuneConfig(
                metric = "val_metric" if sampler is None else None,
                mode = min_max_mode if sampler is None else None,
                search_alg = sampler,
                scheduler = scheduler,
                num_samples = num_samples,
                max_concurrent_trials = max_concurrent_trials,
                time_budget_s = time_budget_s,
                reuse_actors = True,
            ),
            run_config = air.RunConfig(local_dir = experiment.experiment_dir, name = f"{model_cl.__name__}_tuner_{metric.__name__}"),
            param_space = params_space,
        )
        results = tuner.fit()

        return results

    def OLD_define_objective(self, params_dict, model_cl, metric):
        """
        Define the Ray tune objective function.

        Parameters
        ----------
            params_dict: dict, a dictionary containing the parameters to be optimized.
            It is expected to be of the form {"param_name": (string, list), ...}
            e.g., {"lr": ("float", [0.01, 0.05]), "optimizer": ("categorical", ["Adam", "RMSprop", "SGD"]),
            "batch_size": ("int", [32])}
        """
        model_builder = self.MODEL_BUILDERS[model_cl]

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

            # early stopping by monitoring validation loss

            pruning_callback = optuna.integration.PyTorchLightningPruningCallback(trial, "val_loss")
            callbacks = [pruning_callback]

            params.update({"experiment":self})
            params.update({"callbacks": callbacks})

            # build the model
            model = model_builder(**params)

            # train the model
            model.fit(self.train)

            # predict the model
            val_predictions = model.predict(n = self.val_len)

            val_predictions = self._postprocess_predictions(model_cl.__name__, val_predictions, test = False)

            val = self.val
            # compute the validation stats
            if metric == mase:
                val_stats = mase(val, val_predictions, insample=self.train, n_jobs = -1, verbose = self.verbose)
            else :
                val_stats = metric(val, val_predictions, n_jobs = -1, verbose = self.verbose)

            val_stats = np.mean(val_stats)

            return val_stats if val_stats != np.nan else float("inf")

        return objective

    def OLD_optimize_hyperparameters(self, objective_fn, n_trials = None, timeout = 7200, direction = "minimize",
                                  load_if_exists = True):
        """
        Optimize hyper-parameters with Ray tune.

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


    def run(self, params_space, transformers: List[dict] = None,
            transform_val: bool = True, **tuner_kwargs):
        super().run()

        self._get_data()

        if transformers is not None:
            self._preprocess_data(transformers, transform_val)

        scheduler = tuner_kwargs.get("scheduler")
        sampler = tuner_kwargs.get("sampler")

        for model_cl in self.models_cls:
            model_name = model_cl.__name__
            for metric in self.metrics:
                metric_name = metric.__name__

                print(f"Running {model_name} with metric {metric_name}")
                work_dir = os.path.join(os.getcwd(), f"{self.experiment_dir}/{metric_name}/{model_name}")
                os.makedirs(work_dir, exist_ok=True)
                print(f"Final model checkpoints directory: {work_dir}")


                tuner_results = self.tune_hyperparameters(experiment= self, params_space=params_space[model_name],
                                                        model_cl=model_cl, scheduler=scheduler, sampler=sampler,
                                                        metric=metric)
                best_params = tuner_results.get_best_result(metric = "val_metric", mode = "min").config

                self.models_best_params[f"{model_name}_{metric_name}"] = best_params

                best_model = MODEL_BUILDERS[model_name](**best_params, work_dir=work_dir, experiment=self)

                time_start = datetime.now()
                # train the model
                best_model.fit(
                    series=self.train,
                    val_series=self.val,
                    max_samples_per_ts=self.MAX_SAMPLES_PER_TS
                )
                time_end = datetime.now()
                self.models_train_time[f"{model_name}_{metric_name}"] = (time_end - time_start).total_seconds()

                # use best model for subsequent evaluation
                best_model = model_cl.load_from_checkpoint(model_name, work_dir=work_dir, best=True)

                time_start = datetime.now()
                test_predictions = best_model.predict(series = self.val, n = self.test_len)
                time_end = datetime.now()
                self.models_inference_time[f"{model_name}_{metric_name}"] = (time_end - time_start).total_seconds()

                self.test_predictions[f"{model_name}_{metric_name}"] = test_predictions
                test_predictions = self._postprocess_predictions(f"{model_name}_{metric_name}",
                                                                 test_predictions, test = True)
                self.test_stats[f"{model_name}_{metric_name}"] = self._compute_metric(self.test, test_predictions, metric)

        self.end_exp_time = datetime.now()
        self._backup_exp()













