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
    NHiTSModel, NLinearModel, TFTModel, TransformerModel, LightGBMModel, CatBoostModel, XGBModel, RegressionEnsembleModel
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

class BaseExperiment(ABC):
    """
    Abstract class to define the generic structure of an experiment.

    Parameters
    ----------
    experiment_root
        Root directory path where all experiments information are going to be stored.
    dataset
        The dataset to train on.
    """

    def __init__(self, experiment_root:Optional[str] = None, random_state: Optional[int] = None,
                 experiment_name: Optional[str] = None):
        super().__init__()

        self.experiment_root = experiment_root if experiment_root is not None else DEFAULT_EXP_ROOT
        self.experiment_dir = os.path.join(self.experiment_root, experiment_name)
        self.random_state = random_state if random_state is not None else DEFAULT_RADNOM_SEED

        #check if experiment_dir exists and create it if not
        if not os.path.exists(self.experiment_root):
            os.makedirs(self.experiment_root)

        #store experiment start time
        self.start_exp_time = datetime.datetime.now()

        self.experiment_name = experiment_name

        #make experiment_root and specific experiment directory directory
        try:
            os.mkdir(self.experiment_root)
            print(f"Root directory {self.experiment_root} created")
        except:
            print(f"Root directory {self.experiment_root} already exists")

        try:
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

    def _get_data(self, dataset_cls, load_multivariate = False, split = 0.8, horizon = None):
        """
        Get the data.
        """
        # read data
        if "multivariate" in dataset_cls.__init__.__code__.co_varnames:
            dataset = dataset_cls(multivariate=load_multivariate).load()
        else:
            dataset = dataset_cls().load()

        dataset = series2seq(dataset)

        # split data
        if horizon is not None:
            train = [s[:-horizon] for s in dataset]
            val = [s[-horizon:] for s in dataset]
            self.horizon = horizon
        else:
            all_splits = [list(s.split_after(split)) for s in dataset]
            train = [split[0] for split in all_splits]
            val = [split[1] for split in all_splits]

            self.split = split

        return train, val

    def _preprocess_data(self):
        """
        Transform the data.
        """
        pass

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
        pass

    # validate model
    def _validate_model(self):
        """
        Validate the model.
        """
        pass

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

        self.datasets = datasets if datasets is not None else DATASETS
        self.models = models if models is not None else self.MODELS
        self.verbose = verbose

    def run(self):
        """
        Run the experiment.
        """
        # load the data
        self._get_data()
        # train model
        self._train_model()
        # validate model
        self._validate_model()
        # test model
        self._test_model()
        pass

    def backup(self):
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

        self.datasets = datasets if datasets is not None else DATASETS
        self.models = models if models is not None else self.MODELS
        self.verbose = verbose


class DLExperiment(BaseExperiment):
    """
            Experiment class to run deep learning models.
    """

    MODELS = [DLinearModel, NLinearModel, NBEATSModel, TFTModel, TransformerModel, NHiTSModel]
    NAME = "DL_Experiments"

    def __init__(self, experiment_root: Optional[str] = None, random_state: Optional[int] = None,
                 experiment_name: Optional[str] = None, datasets: Optional[str] = None,
                 models: Optional[List[str]] = None, verbose: Optional[bool] = False):
        experiment_name = experiment_name if experiment_name is not None else self.NAME

        super().__init__(experiment_root, random_state, experiment_name)

        self.datasets = datasets if datasets is not None else DATASETS
        self.models = models if models is not None else self.MODELS
        self.verbose = verbose