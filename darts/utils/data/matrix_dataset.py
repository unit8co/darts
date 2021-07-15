"""
Matrix Dataset Base Classes
---------------------------
"""
from abc import ABC, abstractmethod
import numpy as np

from typing import Tuple, Sequence


class MatrixTrainingDataset(ABC, Sequence):
    def __init__(self):
        """
        Abstract class for a matrix training dataset.

        This dataset is meant to be used for training (or validation) models requiring matrix data as input, in a
        sklearn-like style.
        """
        pass

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def get_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        This method returns:
            - np.ndarray of shape (n_samples, n_features), a matrix containing the
            training data
            - np.array of shape (n_samples,), containing training labels

        Where n_samples is the sum of subsamples extracted by each TimeSeries and
        n_features will be # lags +  # lags_covariates.
        """
        pass
