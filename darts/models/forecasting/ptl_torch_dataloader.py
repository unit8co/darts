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

import numpy as np
import os
import re
from glob import glob
import shutil
from joblib import Parallel, delayed
from typing import Any, Optional, Dict, Tuple, Union, Sequence, List
from abc import ABC, abstractmethod
import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import datetime

from darts.timeseries import TimeSeries
from darts.utils import _build_tqdm_iterator
from darts.utils.torch import random_method

from darts.utils.data.training_dataset import (TrainingDataset,
                                               PastCovariatesTrainingDataset,
                                               FutureCovariatesTrainingDataset,
                                               DualCovariatesTrainingDataset,
                                               MixedCovariatesTrainingDataset,
                                               SplitCovariatesTrainingDataset)
from darts.utils.data.inference_dataset import (InferenceDataset,
                                                PastCovariatesInferenceDataset,
                                                FutureCovariatesInferenceDataset,
                                                DualCovariatesInferenceDataset,
                                                MixedCovariatesInferenceDataset,
                                                SplitCovariatesInferenceDataset)
from darts.utils.data.sequential_dataset import (PastCovariatesSequentialDataset,
                                                 FutureCovariatesSequentialDataset,
                                                 DualCovariatesSequentialDataset,
                                                 MixedCovariatesSequentialDataset,
                                                 SplitCovariatesSequentialDataset)
from darts.utils.data.encoders import SequentialEncoder

from darts.utils.likelihood_models import Likelihood
from darts.logging import raise_if_not, get_logger, raise_log, raise_if
from darts.models.forecasting.forecasting_model import GlobalForecastingModel

import pytorch_lightning as pl

from darts.models.forecasting.helper_functions import (_get_checkpoint_folder,
                                                       _get_runs_folder,
                                                       _raise_if_wrong_type,
                                                       _cat_with_optional,
                                                       _basic_compare_sample,
                                                       _mixed_compare_sample,
                                                       )


class PLMixedCovariatesTorchModel(pl.LightningDataModule, ABC):
    def prepare_data(self) -> None:
        self.train_dataset = self._build_train_dataset()

    def _build_train_dataset(self,
                             target: Sequence[TimeSeries],
                             past_covariates: Optional[Sequence[TimeSeries]],
                             future_covariates: Optional[Sequence[TimeSeries]],
                             max_samples_per_ts: Optional[int]) -> MixedCovariatesTrainingDataset:
        return MixedCovariatesSequentialDataset(target_series=target,
                                                past_covariates=past_covariates,
                                                future_covariates=future_covariates,
                                                input_chunk_length=self.input_chunk_length,
                                                output_chunk_length=self.output_chunk_length,
                                                max_samples_per_ts=max_samples_per_ts)

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

    def _verify_predict_sample(self, predict_sample: Tuple):
        _mixed_compare_sample(self.train_sample, predict_sample)

    def _verify_past_future_covariates(self, past_covariates, future_covariates):
        # both covariates are supported; do nothing
        pass

    def _get_batch_prediction(self, n: int, input_batch: Tuple, roll_size: int) -> Tensor:
        raise NotImplementedError("TBD: Darts doesn't contain such a model yet.")

    @property
    def _model_encoder_settings(self) -> Tuple[int, int, bool, bool]:
        input_chunk_length = self.input_chunk_length
        output_chunk_length = self.output_chunk_length
        takes_past_covariates = True
        takes_future_covariates = True
        return input_chunk_length, output_chunk_length, takes_past_covariates, takes_future_covariates