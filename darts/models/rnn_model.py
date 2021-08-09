"""
Recurrent Neural Networks
-------------------------
"""

import torch.nn as nn
import torch
from numpy.random import RandomState
from typing import Sequence, Optional, Union, Tuple
from ..timeseries import TimeSeries

from ..logging import raise_if_not, get_logger
from .torch_forecasting_model import TorchParametricProbabilisticForecastingModel, DualCovariatesTorchModel
from ..utils.torch import random_method
from ..utils.data import DualCovariatesShiftedDataset, TrainingDataset
from ..utils.likelihood_models import LikelihoodModel

logger = get_logger(__name__)


# TODO add batch norm
class _RNNModule(nn.Module):
    def __init__(self,
                 name: str,
                 input_size: int,
                 hidden_dim: int,
                 num_layers: int,
                 target_size: int = 1,
                 dropout: float = 0.):

        """ PyTorch module implementing an RNN to be used in `RNNModel`.

        PyTorch module implementing a simple RNN with the specified `name` type.
        This module combines a PyTorch RNN module, together with one fully connected layer which
        maps the hidden state of the RNN at each step to the output value of the model at that
        time step.

        Parameters
        ----------
        name
            The name of the specific PyTorch RNN module ("RNN", "GRU" or "LSTM").
        input_size
            The dimensionality of the input time series.
        hidden_dim
            The number of features in the hidden state `h` of the RNN module.
        num_layers
            The number of recurrent layers.
        target_size
            The dimensionality of the output time series.
        dropout
            The fraction of neurons that are dropped in all-but-last RNN layers.
        likelihood
            Optionally, the likelihood model to be used for probabilistic forecasts.
            Expects an instance of 'darts.utils.likelihood_model.LikelihoodModel'.

        Inputs
        ------
        x of shape `(batch_size, input_length, input_size)`
            Tensor containing the features of the input sequence. The `input_length` is not fixed.

        Outputs
        -------
        y of shape `(batch_size, input_length, output_size)`
            Tensor containing the outputs of the RNN at every time step of the input sequence.
            During training the whole tensor is used as output, whereas during prediction we only use y[:, -1, :].
            However, this module always returns the whole Tensor.
        """

        super(_RNNModule, self).__init__()

        # Defining parameters
        self.target_size = target_size
        self.name = name

        # Defining the RNN module
        self.rnn = getattr(nn, name)(input_size, hidden_dim, num_layers, batch_first=True, dropout=dropout)

        # The RNN module needs a linear layer V that transforms hidden states into outputs, individually
        self.V = nn.Linear(hidden_dim, target_size)

    def forward(self, x, h=None):
        # data is of size (batch_size, input_length, input_size)
        batch_size = x.shape[0]

        # out is of size (batch_size, input_length, hidden_dim)
        out, last_hidden_state = self.rnn(x) if h is None else self.rnn(x, h)

        # Here, we apply the V matrix to every hidden state to produce the outputs
        predictions = self.V(out)

        # predictions is of size (batch_size, input_length, target_size)
        predictions = predictions.view(batch_size, -1, self.target_size)

        # returns outputs for all inputs, only the last one is needed for prediction time
        return predictions, last_hidden_state


class RNNModel(TorchParametricProbabilisticForecastingModel, DualCovariatesTorchModel):
    @random_method
    def __init__(self,
                 model: Union[str, nn.Module] = 'RNN',
                 input_chunk_length: int = 12,
                 hidden_dim: int = 25,
                 n_rnn_layers: int = 1,
                 dropout: float = 0.,
                 training_length: int = 24,
                 likelihood: Optional[LikelihoodModel] = None,
                 random_state: Optional[Union[int, RandomState]] = None,
                 **kwargs):

        """ Recurrent Neural Network Model (RNNs).

        This class provides three variants of RNNs:

        * Vanilla RNN

        * LSTM

        * GRU

        RNNModel is fully recurrent in the sense that, at prediction time, an output is computed using these inputs:

        - previous target value, which will be set to the last known target value for the first prediction,
          and for all other predictions it will be set to the previous prediction (in an auto-regressive fashion),
        - the previous hidden state,
        - the covariates at time `t` for forecasting the target at time `t` (if the model was trained with covariates),

        This model supports future covariates; and it requires these covariates to extend far enough in the past
        and the future (it's a so-called "dual covariates" model as the future covariates have to be provided both
        in the past and the future). The model will complain if the provided `future_covariates` series doesn't have
        an appropriate time span.

        For a block version using an RNN model as an encoder only and supporting past
        covariates, checkout `BlockRNNModel`.

        Parameters
        ----------
        model
            Either a string specifying the RNN module type ("RNN", "LSTM" or "GRU"),
            or a PyTorch module with the same specifications as
            `darts.models.rnn_model.RNNModule`.
        input_chunk_length
            Number of past time steps that are fed to the forecasting module at prediction time.
        hidden_dim
            Size for feature maps for each hidden RNN layer (:math:`h_n`).
        n_rnn_layers
            The number of recurrent layers.
        dropout
            Fraction of neurons afected by Dropout.
        training_length
            The length of both input (target and covariates) and output (target) time series used during
            training. Generally speaking, `training_length` should have a higher value than `input_chunk_length`
            because otherwise during training the RNN is never run for as many iterations as it will during
            training. For more information on this parameter, please see `darts.utils.data.ShiftedDataset`
        likelihood
            Optionally, the likelihood model to be used for probabilistic forecasts.
            If no likelihood model is provided, forecasts will be deterministic.
        random_state
            Control the randomness of the weights initialization. Check this
            `link <https://scikit-learn.org/stable/glossary.html#term-random-state>`_ for more details.

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

        kwargs['input_chunk_length'] = input_chunk_length
        kwargs['output_chunk_length'] = 1
        super().__init__(likelihood=likelihood, **kwargs)

        # check we got right model type specified:
        if model not in ['RNN', 'LSTM', 'GRU']:
            raise_if_not(isinstance(model, nn.Module), '{} is not a valid RNN model.\n Please specify "RNN", "LSTM", '
                                                       '"GRU", or give your own PyTorch nn.Module'.format(
                                                        model.__class__.__name__), logger)

        self.rnn_type_or_module = model
        self.dropout = dropout
        self.hidden_dim = hidden_dim
        self.n_rnn_layers = n_rnn_layers
        self.training_length = training_length

    def _create_model(self, train_sample: Tuple[torch.Tensor]) -> torch.nn.Module:
        # samples are made of (past_target, historic_future_covariates, future_covariates, future_target)
        # historic_future_covariates and future_covariates have the same width
        input_dim = train_sample[0].shape[1] + (train_sample[1].shape[1] if train_sample[1] is not None else 0)
        output_dim = train_sample[-1].shape[1]

        target_size = (
            self.likelihood._num_parameters * output_dim if self.likelihood is not None else output_dim
        )
        if self.rnn_type_or_module in ['RNN', 'LSTM', 'GRU']:
            model = _RNNModule(name=self.rnn_type_or_module,
                               input_size=input_dim,
                               target_size=target_size,
                               hidden_dim=self.hidden_dim,
                               dropout=self.dropout,
                               num_layers=self.n_rnn_layers)
        else:
            model = self.rnn_type_or_module
        return model

    def _build_train_dataset(self,
                             target: Sequence[TimeSeries],
                             past_covariates: Optional[Sequence[TimeSeries]],
                             future_covariates: Optional[Sequence[TimeSeries]]) -> DualCovariatesShiftedDataset:

        return DualCovariatesShiftedDataset(target_series=target,
                                            covariates=future_covariates,
                                            length=self.training_length,
                                            shift=1)

    def _verify_train_dataset_type(self, train_dataset: TrainingDataset):
        raise_if_not(isinstance(train_dataset, DualCovariatesShiftedDataset),
                     'RNNModel requires a training dataset of type DualCovariatesShiftedDataset.')
        raise_if_not(train_dataset.ds_past.shift == 1, 'RNNModel requires a shifted training dataset with shift=1.')

    def _produce_train_output(self, input_batch: Tuple):
        past_target, historic_future_covariates, future_covariates = input_batch
        # For the RNN we concatenate the past_target with the future_covariates
        # (they have the same length because we enforce a Shift dataset for RNNs)
        model_input = torch.cat([past_target, future_covariates],
                                dim=2) if future_covariates is not None else past_target
        return self.model(model_input)[0]

    @random_method
    def _produce_predict_output(self, input, last_hidden_state=None):
        if self.likelihood:
            output, hidden = self.model(input, last_hidden_state)
            return self.likelihood._sample(output), hidden
        else:
            return self.model(input, last_hidden_state)

    def _get_batch_prediction(self, n: int, input_batch: Tuple, roll_size: int) -> torch.Tensor:
        """
        This model is recurrent, so we have to write a specific way to obtain the time series forecasts of length n.
        """
        past_target, historic_future_covariates, future_covariates = input_batch

        if historic_future_covariates is not None:
            # RNNs need as inputs (target[t] and covariates[t+1]) so here we shift the covariates
            all_covariates = torch.cat([historic_future_covariates[:, 1:, :], future_covariates], dim=1)
            cov_past, cov_future = all_covariates[:, :past_target.shape[1], :], all_covariates[:, past_target.shape[1]:, :]
            input_series = torch.cat([past_target, cov_past], dim=2)
        else:
            input_series = past_target
            cov_future = None

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

        # bring predictions into desired format and drop unnecessary values
        batch_prediction = torch.cat(batch_prediction, dim=1)
        batch_prediction = batch_prediction[:, :n, :]

        return batch_prediction
