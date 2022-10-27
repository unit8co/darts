"""
Block Recurrent Neural Networks
-------------------------------
"""

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from darts.logging import get_logger, raise_if_not
from darts.models.forecasting.pl_forecasting_module import PLPastCovariatesModule
from darts.models.forecasting.torch_forecasting_model import PastCovariatesTorchModel

logger = get_logger(__name__)


# TODO add batch norm
class _BlockRNNModule(PLPastCovariatesModule):
    def __init__(
        self,
        name: str,
        input_size: int,
        hidden_dim: int,
        num_layers: int,
        target_size: int,
        nr_params: int,
        num_layers_out_fc: Optional[List] = None,
        dropout: float = 0.0,
        **kwargs,
    ):

        """PyTorch module implementing a block RNN to be used in `BlockRNNModel`.

        PyTorch module implementing a simple block RNN with the specified `name` layer.
        This module combines a PyTorch RNN module, together with a fully connected network, which maps the
        last hidden layers to output of the desired size `output_chunk_length` and makes it compatible with
        `BlockRNNModel`s.

        This module uses an RNN to encode the input sequence, and subsequently uses a fully connected
        network as the decoder which takes as input the last hidden state of the encoder RNN.
        The final output of the decoder is a sequence of length `output_chunk_length`. In this sense,
        the `_BlockRNNModule` produces 'blocks' of forecasts at a time (which is different
        from `_RNNModule` used by the `RNNModel`).

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
        nr_params
            The number of parameters of the likelihood (or 1 if no likelihood is used).
        num_layers_out_fc
            A list containing the dimensions of the hidden layers of the fully connected NN.
            This network connects the last hidden layer of the PyTorch RNN module to the output.
        dropout
            The fraction of neurons that are dropped in all-but-last RNN layers.
        **kwargs
            all parameters required for :class:`darts.model.forecasting_models.PLForecastingModule` base class.

        Inputs
        ------
        x of shape `(batch_size, input_chunk_length, input_size, nr_params)`
            Tensor containing the features of the input sequence.

        Outputs
        -------
        y of shape `(batch_size, output_chunk_length, target_size, nr_params)`
            Tensor containing the prediction at the last time step of the sequence.
        """

        super().__init__(**kwargs)

        # Defining parameters
        self.hidden_dim = hidden_dim
        self.n_layers = num_layers
        self.target_size = target_size
        self.nr_params = nr_params
        num_layers_out_fc = [] if num_layers_out_fc is None else num_layers_out_fc
        self.out_len = self.output_chunk_length
        self.name = name

        # Defining the RNN module
        self.rnn = getattr(nn, name)(
            input_size, hidden_dim, num_layers, batch_first=True, dropout=dropout
        )

        # The RNN module is followed by a fully connected layer, which maps the last hidden layer
        # to the output of desired length
        last = hidden_dim
        feats = []
        for feature in num_layers_out_fc + [
            self.output_chunk_length * target_size * nr_params
        ]:
            feats.append(nn.Linear(last, feature))
            last = feature
        self.fc = nn.Sequential(*feats)

    def forward(self, x_in: Tuple):
        x, _ = x_in
        # data is of size (batch_size, input_chunk_length, input_size)
        batch_size = x.size(0)

        out, hidden = self.rnn(x)

        """ Here, we apply the FC network only on the last output point (at the last time step)
        """
        if self.name == "LSTM":
            hidden = hidden[0]
        predictions = hidden[-1, :, :]
        predictions = self.fc(predictions)
        predictions = predictions.view(
            batch_size, self.out_len, self.target_size, self.nr_params
        )

        # predictions is of size (batch_size, output_chunk_length, 1)
        return predictions


class BlockRNNModel(PastCovariatesTorchModel):
    def __init__(
        self,
        input_chunk_length: int,
        output_chunk_length: int,
        model: Union[str, nn.Module] = "RNN",
        hidden_dim: int = 25,
        n_rnn_layers: int = 1,
        hidden_fc_sizes: Optional[List] = None,
        dropout: float = 0.0,
        **kwargs,
    ):

        """Block Recurrent Neural Network Model (RNNs).

        This is a neural network model that uses an RNN encoder to encode fixed-length input chunks, and
        a fully connected network to produce fixed-length outputs.

        This model supports past covariates (known for `input_chunk_length` points before prediction time).

        This class provides three variants of RNNs:

        * Vanilla RNN

        * LSTM

        * GRU

        Parameters
        ----------
        input_chunk_length
            The number of time steps that will be fed to the internal forecasting module
        output_chunk_length
            Number of time steps to be output by the internal forecasting module.
        model
            Either a string specifying the RNN module type ("RNN", "LSTM" or "GRU"),
            or a PyTorch module with the same specifications as
            :class:`darts.models.block_rnn_model._BlockRNNModule`.
        hidden_dim
            Size for feature maps for each hidden RNN layer (:math:`h_n`).
            In Darts version <= 0.21, hidden_dim was referred as hidden_size.
        n_rnn_layers
            Number of layers in the RNN module.
        hidden_fc_sizes
            Sizes of hidden layers connecting the last hidden layer of the RNN module to the output, if any.
        dropout
            Fraction of neurons afected by Dropout.
        **kwargs
            Optional arguments to initialize the pytorch_lightning.Module, pytorch_lightning.Trainer, and
            Darts' :class:`TorchForecastingModel`.

        loss_fn
            PyTorch loss function used for training.
            This parameter will be ignored for probabilistic models if the ``likelihood`` parameter is specified.
            Default: ``torch.nn.MSELoss()``.
        likelihood
            One of Darts' :meth:`Likelihood <darts.utils.likelihood_models.Likelihood>` models to be used for
            probabilistic forecasts. Default: ``None``.
        torch_metrics
            A torch metric or a ``MetricCollection`` used for evaluation. A full list of available metrics can be found
            at https://torchmetrics.readthedocs.io/en/latest/. Default: ``None``.
        optimizer_cls
            The PyTorch optimizer class to be used. Default: ``torch.optim.Adam``.
        optimizer_kwargs
            Optionally, some keyword arguments for the PyTorch optimizer (e.g., ``{'lr': 1e-3}``
            for specifying a learning rate). Otherwise the default values of the selected ``optimizer_cls``
            will be used. Default: ``None``.
        lr_scheduler_cls
            Optionally, the PyTorch learning rate scheduler class to be used. Specifying ``None`` corresponds
            to using a constant learning rate. Default: ``None``.
        lr_scheduler_kwargs
            Optionally, some keyword arguments for the PyTorch learning rate scheduler. Default: ``None``.
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
            :func:`save_model()` and loaded using :func:`load_model()`. Default: ``False``.
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
            https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#trainer-flags , and
            https://pytorch-lightning.readthedocs.io/en/stable/accelerators/gpu_basic.html#train-on-multiple-gpus

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
        super().__init__(**self._extract_torch_model_params(**self.model_params))

        # extract pytorch lightning module kwargs
        self.pl_module_params = self._extract_pl_module_params(**self.model_params)

        # check we got right model type specified:
        if model not in ["RNN", "LSTM", "GRU"]:
            raise_if_not(
                isinstance(model, nn.Module),
                '{} is not a valid RNN model.\n Please specify "RNN", "LSTM", '
                '"GRU", or give your own PyTorch nn.Module'.format(
                    model.__class__.__name__
                ),
                logger,
            )

        self.rnn_type_or_module = model
        self.hidden_fc_sizes = hidden_fc_sizes
        self.hidden_dim = hidden_dim
        self.n_rnn_layers = n_rnn_layers
        self.dropout = dropout

    @staticmethod
    def _supports_static_covariates() -> bool:
        return False

    def _create_model(self, train_sample: Tuple[torch.Tensor]) -> torch.nn.Module:
        # samples are made of (past_target, past_covariates, future_target)
        input_dim = train_sample[0].shape[1] + (
            train_sample[1].shape[1] if train_sample[1] is not None else 0
        )
        output_dim = train_sample[-1].shape[1]
        nr_params = 1 if self.likelihood is None else self.likelihood.num_parameters

        if self.rnn_type_or_module in ["RNN", "LSTM", "GRU"]:
            hidden_fc_sizes = (
                [] if self.hidden_fc_sizes is None else self.hidden_fc_sizes
            )
            model = _BlockRNNModule(
                name=self.rnn_type_or_module,
                input_size=input_dim,
                target_size=output_dim,
                nr_params=nr_params,
                hidden_dim=self.hidden_dim,
                num_layers=self.n_rnn_layers,
                num_layers_out_fc=hidden_fc_sizes,
                dropout=self.dropout,
                **self.pl_module_params,
            )
        else:
            model = self.rnn_type_or_module
        return model
