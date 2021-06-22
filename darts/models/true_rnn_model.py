"""
Recurrent Neural Networks
-------------------------
"""

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from numpy.random import RandomState
from joblib import Parallel, delayed
from typing import Sequence, Optional, Union
from ..timeseries import TimeSeries

from ..logging import raise_if_not, get_logger
from .torch_forecasting_model import TorchForecastingModel, TimeSeriesTorchDataset
from ..utils.torch import random_method
from ..utils.data.timeseries_dataset import TimeSeriesInferenceDataset
from ..utils.data import ShiftedDataset
from ..utils import _build_tqdm_iterator

logger = get_logger(__name__)


# TODO add batch norm
class _TrueRNNModule(nn.Module):
    def __init__(self,
                 name: str,
                 input_size: int,
                 hidden_dim: int,
                 target_size: int = 1,
                 dropout: float = 0.):

        """ PyTorch module implementing a RNN to be used in `RNNModel`.

        PyTorch module implementing a simple RNN with the specified `name` layer.
        This module combines a PyTorch RNN module, together with a fully connected network, which maps the
        last hidden layers to output of the desired size `output_chunk_length` and makes it compatible with
        `RNNModel`s.

        Parameters
        ----------
        name
            The name of the specific PyTorch RNN module ("RNN", "GRU" or "LSTM").
        input_size
            The dimensionality of the input time series.
        hidden_dim
            The number of features in the hidden state `h` of the RNN module.
        target_size
            The dimensionality of the output time series.
        num_layers_out_fc
            A list containing the dimensions of the hidden layers of the fully connected NN.
            This network connects the last hidden layer of the PyTorch RNN module to the output.
        dropout
            The fraction of neurons that are dropped in all-but-last RNN layers.

        Inputs
        ------
        x of shape `(batch_size, input_length, input_size)`
            Tensor containing the features of the input sequence. The `input_length` is not fixed.

        Outputs
        -------
        y of shape `(batch_size, output_length, output_size)`
            The `output_length` is equal to 1 at prediction time, but equal to the `input_length` during training.
        """

        super(_TrueRNNModule, self).__init__()

        # Defining parameters
        self.target_size = target_size
        self.name = name

        # Defining the RNN module
        self.rnn = getattr(nn, name)(input_size, hidden_dim, 1, batch_first=True, dropout=dropout)

        # The RNN module needs a linear layer V that transforms hidden states into outputs, individually
        self.V = nn.Linear(hidden_dim, target_size)

    def forward(self, x, h=None):
        # data is of size (batch_size, input_length, input_size)
        batch_size = x.size(0)

        # out is of size (batch_size, input_length, hidden_dim)
        out, last_hidden_state = self.rnn(x) if h is None else self.rnn(x, h)
        # TODO: confirm shape of out

        """ Here, we apply the V matrix to every hidden state to produce the outputs
        """
        # TODO: for one layer out should be equal to hidden, make sure that's true
        predictions = self.V(out)  # TODO: make sure this matrix multiplication is correct

        # predictions is of size (batch_size, input_length, target_size)
        predictions = predictions.view(batch_size, -1, self.target_size)

        # returns outputs for all inputs, only the last one is needed for prediction time
        return predictions, last_hidden_state


class TrueRNNModel(TorchForecastingModel):
    @random_method
    def __init__(self,
                 model: Union[str, nn.Module] = 'RNN',
                 input_chunk_length: int = 12,
                 hidden_dim: int = 25,
                 dropout: float = 0.,
                 training_length: int = 24,
                 random_state: Optional[Union[int, RandomState]] = None,
                 **kwargs):

        """ Recurrent Neural Network Model (RNNs).

        This class provides three variants of RNNs:

        * Vanilla RNN

        * LSTM

        * GRU

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
        dropout
            Fraction of neurons afected by Dropout.
        training_length
            The length of 1 training sample time series.
        random_state
            Control the randomness of the weights initialization. Check this
            `link <https://scikit-learn.org/stable/glossary.html#term-random-state>`_ for more details.
        """

        kwargs['input_chunk_length'] = input_chunk_length
        kwargs['output_chunk_length'] = 1
        super().__init__(**kwargs)

        # check we got right model type specified:
        if model not in ['RNN', 'LSTM', 'GRU']:
            raise_if_not(isinstance(model, nn.Module), '{} is not a valid RNN model.\n Please specify "RNN", "LSTM", '
                                                       '"GRU", or give your own PyTorch nn.Module'.format(
                                                        model.__class__.__name__), logger)

        self.rnn_type_or_module = model
        self.dropout = dropout
        self.hidden_dim = hidden_dim
        self.training_length = training_length
        self.is_recurrent = True

    def _create_model(self, input_dim: int, output_dim: int) -> torch.nn.Module:
        if self.rnn_type_or_module in ['RNN', 'LSTM', 'GRU']:
            model = _TrueRNNModule(name=self.rnn_type_or_module,
                                   input_size=input_dim,
                                   target_size=output_dim,
                                   hidden_dim=self.hidden_dim,
                                   dropout=self.dropout)
        else:
            model = self.rnn_type_or_module
        return model

    def _build_train_dataset(self,
                             target: Sequence[TimeSeries],
                             covariates: Optional[Sequence[TimeSeries]]) -> ShiftedDataset:
        return ShiftedDataset(target_series=target,
                              covariates=covariates,
                              length=self.training_length,
                              shift_covariates=True,
                              shift=1)

    @property
    def first_prediction_index(self) -> int:
        return -1

    def _produce_train_output(self, data):
        return self.model(data)[0]

    def predict_from_dataset(self,
                             n: int,
                             input_series_dataset: TimeSeriesInferenceDataset,
                             batch_size: Optional[int] = None,
                             verbose: bool = False,
                             n_jobs: int = 1,
                             roll_size: Optional[int] = None
                             ) -> Sequence[TimeSeries]:

        """
        Predicts values for a certain number of time steps after the end of the series appearing in the specified
        ``input_series_dataset``.

        If ``n`` is larger than the model ``output_chunk_length``, the predictions will be computed in an
        auto-regressive way, by iteratively feeding the last ``roll_size`` forecast points as
        inputs to the model until a forecast of length ``n`` is obtained. If the model was trained with
        covariates, all of the covariate time series need to have a time index that extends at least
        `n - output_chunk_length` into the future. In other words, if `n` is larger than `output_chunk_length`
        then covariates need to be available in the future.

        If some series in the ``input_series_dataset`` have more time steps than the model was trained with,
        only the last ``input_chunk_length`` time steps will be considered.

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

        Returns
        -------
        Sequence[TimeSeries]
            Returns one or more forecasts for time series.
        """
        self.model.eval()

        if roll_size is None:
            roll_size = self.output_chunk_length
        else:
            raise_if_not(0 < roll_size <= self.output_chunk_length,
                         '`roll_size` must be an integer between 1 and `self.output_chunk_length`')

        # check input data type
        raise_if_not(isinstance(input_series_dataset, TimeSeriesInferenceDataset),
                     'Only TimeSeriesInferenceDataset is accepted as input type')

        # TODO currently we assume all forecasts fit in memory

        # iterate through batches to produce predictions
        batch_size = batch_size or self.batch_size
        pred_loader = DataLoader(TimeSeriesTorchDataset(input_series_dataset, self.device),
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=0,
                                 pin_memory=False,
                                 drop_last=False)
        predictions = []
        iterator = _build_tqdm_iterator(pred_loader, verbose=verbose)
        with torch.no_grad():
            for batch_tuple in iterator:

                input_series, indices = batch_tuple[0], batch_tuple[-1]
                cov_future = batch_tuple[1] if len(batch_tuple) == 3 else None

                batch_prediction = []
                out, last_hidden_state = self.model(input_series)
                batch_prediction.append(out[:, -1:, :])
                prediction_length = 1

                while prediction_length < n:

                    # create new input to model from last prediction and current covariates, if available
                    new_input = (
                        torch.cat([out[:, -1:, :], cov_future[:, prediction_length - 1:prediction_length, :]], dim=2)
                        if cov_future is not None else out[:, -1:, :]
                    )

                    # feed new input to model, including the last hidden state from the previous iteration
                    out, last_hidden_state = self.model(new_input, last_hidden_state)

                    # append prediction to batch prediction array, increase counter
                    batch_prediction.append(out[:, -1:, :])
                    prediction_length += 1

                # bring predictions into desired format and drop unnecessary values
                batch_prediction = torch.cat(batch_prediction, dim=1)
                batch_prediction = batch_prediction[:, :n, :]
                batch_prediction = batch_prediction.cpu().detach().numpy()

                ts_forecasts = Parallel(n_jobs=n_jobs)(
                    delayed(self._build_forecast_series)(
                        batch_prediction[batch_idx],
                        input_series_dataset[dataset_idx][0]
                    )
                    for batch_idx, dataset_idx in enumerate(indices)
                )

                predictions.extend(ts_forecasts)

        return predictions
