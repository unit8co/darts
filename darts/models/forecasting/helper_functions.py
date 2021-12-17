import os
from typing import Any, Optional, Dict, Tuple, Union, Sequence, List
import torch
from darts.logging import raise_if_not, get_logger, raise_log, raise_if


logger = get_logger(__name__)

DEFAULT_DARTS_FOLDER = '.darts'
CHECKPOINTS_FOLDER = 'checkpoints'
RUNS_FOLDER = 'runs'


def _get_checkpoint_folder(work_dir, model_name):
    return os.path.join(work_dir, CHECKPOINTS_FOLDER, model_name)


def _get_runs_folder(work_dir, model_name):
    return os.path.join(work_dir, RUNS_FOLDER, model_name)


def _raise_if_wrong_type(obj, exp_type, msg='expected type {}, got: {}'):
    raise_if_not(isinstance(obj, exp_type), msg.format(exp_type, type(obj)))


def _cat_with_optional(tsr1: torch.Tensor, tsr2: Optional[torch.Tensor]):
    if tsr2 is None:
        return tsr1
    else:
        # dimensions are (batch, length, width), we concatenate along the widths.
        return torch.cat([tsr1, tsr2], dim=2)


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
    that training/inference datasets have target and a covariate in first and second position to do the checks.
    """
    tgt_train, cov_train = train_sample[:2]
    tgt_pred, cov_pred = predict_sample[:2]
    raise_if_not(tgt_train.shape[-1] == tgt_pred.shape[-1],
                 'The provided target has a dimension (width) that does not match the dimension '
                 'of the target this model has been trained on.')
    raise_if(cov_train is not None and cov_pred is None,
             'This model has been trained with covariates; some covariates of matching dimensionality are needed '
             'for prediction.')
    raise_if(cov_train is None and cov_pred is not None,
             'This model has been trained without covariates. No covariates should be provided for prediction.')
    raise_if(cov_train is not None and cov_pred is not None and
             cov_train.shape[-1] != cov_pred.shape[-1],
             'The provided covariates must have dimensionality matching that of the covariates used for training '
             'the model.')


def _mixed_compare_sample(train_sample: Tuple, predict_sample: Tuple):
    """
    For models relying on MixedCovariates.

    Parameters:
    ----------
    train_sample
        (past_target, past_covariates, historic_future_covariates, future_covariates, future_target)
    predict_sample
        (past_target, past_covariates, historic_future_covariates, future_covariates, future_past_covariates, ts_target)
    """
    # datasets; we skip future_target for train and predict, and skip future_past_covariates for predict datasets
    ds_names = ['past_target', 'past_covariates', 'historic_future_covariates', 'future_covariates']

    train_has_ds = [ds is not None for ds in train_sample[:-1]]
    predict_has_ds = [ds is not None for ds in predict_sample[:4]]

    train_datasets = train_sample[:-1]
    predict_datasets = predict_sample[:4]

    tgt_train, tgt_pred = train_datasets[0], predict_datasets[0]
    raise_if_not(tgt_train.shape[-1] == tgt_pred.shape[-1],
                 'The provided target has a dimension (width) that does not match the dimension '
                 'of the target this model has been trained on.')

    for idx, (ds_in_train, ds_in_predict, ds_name) in enumerate(zip(train_has_ds, predict_has_ds, ds_names)):
        raise_if(ds_in_train and not ds_in_predict and ds_in_train,
                 f'This model has been trained with {ds_name}; some {ds_name} of matching dimensionality are needed '
                 f'for prediction.')
        raise_if(ds_in_train and not ds_in_predict and ds_in_predict,
                 f'This model has been trained without {ds_name}; No {ds_name} should be provided for prediction.')
        raise_if(ds_in_train and ds_in_predict and train_datasets[idx].shape[-1] != predict_datasets[idx].shape[-1],
                 f'The provided {ds_name} must have dimensionality that of the {ds_name} used for training the model.')