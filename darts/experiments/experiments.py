"""
    This file contains the basic structure to define, run and backup and experiment
"""

import datetime
import inspect
import os
import shutil
from abc import ABC, abstractmethod
from glob import glob
from typing import List, Optional, Sequence, Tuple, Union
from tqdm import tqdm

from darts.logging import (
    get_logger,
    raise_if,
    raise_if_not,
    raise_log,
)

logger = get_logger(__name__)

DEFAULT_RADNOM_SEED = 42
DEFAULT_EXP_ROOT = "./experiment_runs/"

class BaseExperiment(ABC):
    """
    Abstract class to define the generic structure of an experiment.

    Parameters
    ----------
    experiment_root
        Root directory path where all experiments information are going to be stored.
    dataset
        The dataset to train on.
    optimizer
        The optimizer to use.
    epochs
        The number of epochs to train for.
    split
        The split of the dataset to use for training, validation and testing.
    verbose
        Whether to print progress during training.
    """

    def __init__(self, experiment_root:Optional[str] = None, random_state: Optional[int] = None):
        super.__init__()

        self.experiment_root = experiment_root if experiment_root is not None else DEFAULT_EXP_ROOT
        self.random_state = random_state if random_state is not None else DEFAULT_RADNOM_SEED

        #check if experiment_dir exists and create it if not
        if not os.path.exists(self.experiment_root):
            os.makedirs(self.experiment_root)

        #store start time
        self.start_time = datetime.datetime.now()

    @abstractmethod
    def run(self):
        """
        Run the experiment.
        """
        pass

    @abstractmethod
    def backup(self):
        """
        Backup the experiment.
        """
        pass

    @abstractmethod
    def _get_data(self):
        """
        Get the data.
        """
        pass

    # load model
    @abstractmethod
    def _load_model(self):
        """
        Load the model.
        """
        pass

    # optimize hyper-parameters
    @abstractmethod
    def _optimize_hyperparameters(self):
        """
        Optimize hyper-parameters.
        """
        pass

    # train model
    @abstractmethod
    def _train_model(self):
        """
        Train the model.
        """
        pass

    # validate model
    @abstractmethod
    def _validate_model(self):
        """
        Validate the model.
        """
        pass

    # test model
    @abstractmethod
    def _test_model(self):
        """
        Test the model.
        """
        pass

class BaselinesExperiment(BaseExperiment):
    """
    Experiment class to run the baseline models.
    """

    models = []

    def __init__(self, experiment_root:Optional[str] = None, random_state: Optional[int] = None,
                 experiment_name: Optional[str] = None, dataset: Optional[str] = None,
                 models: Optional[List[str]] = None, epochs: Optional[int] = None,
                split: Optional[Tuple[float, float, float]] = None, verbose: Optional[bool] = None):
        super.__init__(experiment_root, random_state)

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