"""
Recurrent Neural Networks
-------------------------
"""

import inspect
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Optional, Union

import torch
import torch.nn as nn

from darts.logging import get_logger, raise_if_not, raise_log
from darts.models.forecasting.pl_forecasting_module import (
    PLDualCovariatesModule,
    io_processor,
)
from darts.models.forecasting.torch_forecasting_model import DualCovariatesTorchModel
from darts.timeseries import TimeSeries
from darts.utils.data import DualCovariatesShiftedDataset, TrainingDataset

logger = get_logger(__name__)


class CustomRNNModule(PLDualCovariatesModule, ABC):
    def __init__(
        self,
        input_size: int,
        hidden_dim: int,
        num_layers: int,
        target_size: int,
        nr_params: int,
        dropout: float = 0.0,
        **kwargs,
    ):
        """This class allows to create custom RNN modules that can later be used with Darts' :class:`RNNModel`.
        It adds the backbone that is required to be used with Darts' :class:`TorchForecastingModel` and
        :class:`RNNModel`.

        To create a new module, subclass from :class:`CustomRNNModule` and:

        * Define the architecture in the module constructor (`__init__()`)

        * Add the `forward()` method and define the logic of your module's forward pass

        * Use the custom module class when creating a new :class:`RNNModel` with parameter `model`.

        You can use `darts.models.forecasting.rnn_model._RNNModule` as an example.

        Parameters
        ----------
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
        dropout
            The fraction of neurons that are dropped in all-but-last RNN layers.
        **kwargs
            all parameters required for :class:`darts.models.forecasting.pl_forecasting_module.PLForecastingModule`
            base class.
        """
        # RNNModule doesn't really need input and output_chunk_length for PLModule
        super().__init__(**kwargs)

        # Defining parameters
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.target_size = target_size
        self.nr_params = nr_params
        self.dropout = dropout

    @io_processor
    @abstractmethod
    def forward(
        self, x_in: tuple, h: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """RNN Module forward.

        Parameters
        ----------
        x_in
            Tuple of Tensors containing the features of the input sequence. The tuple has elements (past target,
            historic future covariates, future covariates, static covariates). The shape of the past target is
            `(batch_size, input_length, input_size)`.
        h
            Optionally, the hidden state.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Tuple of Tensors with elements (RNN output, hidden state). The RNN output Tensor has shape
            `(batch_size, output_chunk_length, target_size, nr_params)`. It contains the outputs at every
            time step of the input sequence. During training the whole tensor is used as output, whereas during
            prediction we only use y[:, -1, :]. However, this module always returns the whole Tensor.
        """
        pass

    def _produce_train_output(self, input_batch: tuple) -> torch.Tensor:
        # only return the forecast, not the hidden state
        return self(self._process_input_batch(input_batch))[0]

    def _process_input_batch(
        self, input_batch: tuple
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        (
            past_target,
            historic_future_covariates,
            future_covariates,
            static_covariates,
        ) = input_batch
        # For the RNN we concatenate the past_target with the future_covariates
        # (they have the same length because we enforce a Shift dataset for RNNs)
        return (
            (
                torch.cat([past_target, future_covariates], dim=2)
                if future_covariates is not None
                else past_target
            ),
            static_covariates,
        )

    def _produce_predict_output(
        self, x: tuple, last_hidden_state: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """overwrite parent classes `_produce_predict_output` method"""
        output, hidden = self(x, last_hidden_state)
        if self.likelihood:
            if self.predict_likelihood_parameters:
                return self.likelihood.predict_likelihood_parameters(output), hidden
            else:
                return self.likelihood.sample(output), hidden
        else:
            return output.squeeze(dim=-1), hidden

    def _get_batch_prediction(
        self, n: int, input_batch: tuple, roll_size: int
    ) -> torch.Tensor:
        """
        This model is recurrent, so we have to write a specific way to
        obtain the time series forecasts of length n.
        """
        (
            past_target,
            historic_future_covariates,
            future_covariates,
            static_covariates,
        ) = input_batch

        if historic_future_covariates is not None:
            # RNNs need as inputs (target[t] and covariates[t+1]) so here we shift the covariates
            all_covariates = torch.cat(
                [historic_future_covariates[:, 1:, :], future_covariates], dim=1
            )
            cov_past, cov_future = (
                all_covariates[:, : past_target.shape[1], :],
                all_covariates[:, past_target.shape[1] :, :],
            )
            input_series = torch.cat([past_target, cov_past], dim=2)
        else:
            input_series = past_target
            cov_future = None

        batch_prediction = []
        out, last_hidden_state = self._produce_predict_output((
            input_series,
            static_covariates,
        ))
        batch_prediction.append(out[:, -1:, :])
        prediction_length = 1

        while prediction_length < n:
            # create new input to model from last prediction and current covariates, if available
            new_input = (
                torch.cat(
                    [
                        out[:, -1:, :],
                        cov_future[:, prediction_length - 1 : prediction_length, :],
                    ],
                    dim=2,
                )
                if cov_future is not None
                else out[:, -1:, :]
            )

            # feed new input to model, including the last hidden state from the previous iteration
            out, last_hidden_state = self._produce_predict_output(
                (new_input, static_covariates), last_hidden_state
            )

            # append prediction to batch prediction array, increase counter
            batch_prediction.append(out[:, -1:, :])
            prediction_length += 1

        # bring predictions into desired format and drop unnecessary values
        batch_prediction = torch.cat(batch_prediction, dim=1)
        batch_prediction = batch_prediction[:, :n, :]
        return batch_prediction


# TODO add batch norm
class _RNNModule(CustomRNNModule):
    def __init__(
        self,
        name: str,
        **kwargs,
    ):
        """PyTorch module implementing an RNN to be used in `RNNModel`.

        PyTorch module implementing a simple RNN with the specified `name` type.
        This module combines a PyTorch RNN module, together with one fully connected layer which
        maps the hidden state of the RNN at each step to the output value of the model at that
        time step.

        Parameters
        ----------
        name
            The name of the specific PyTorch RNN module ("RNN", "GRU" or "LSTM").
        **kwargs
            all parameters required for the :class:`darts.models.forecasting.CustomRNNModule` base class.

        Inputs
        ------
        x of shape `(batch_size, input_length, input_size)`
            Tensor containing the features of the input sequence. The `input_length` is not fixed.

        Outputs
        -------
        y of shape `(batch_size, output_chunk_length, target_size, nr_params)`
            Tensor containing the outputs of the RNN at every time step of the input sequence.
            During training the whole tensor is used as output, whereas during prediction we only use y[:, -1, :].
            However, this module always returns the whole Tensor.
        """

        # RNNModule doesn't really need input and output_chunk_length for PLModule
        super().__init__(**kwargs)
        self.name = name

        # Defining the RNN module
        self.rnn = getattr(nn, name)(
            self.input_size,
            self.hidden_dim,
            self.num_layers,
            batch_first=True,
            dropout=self.dropout,
        )

        # The RNN module needs a linear layer V that transforms hidden states into outputs, individually
        self.V = nn.Linear(self.hidden_dim, self.target_size * self.nr_params)

    @io_processor
    def forward(
        self, x_in: tuple, h: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x, _ = x_in
        # data is of size (batch_size, input_length, input_size)
        batch_size = x.shape[0]

        # out is of size (batch_size, input_length, hidden_dim)
        out, last_hidden_state = self.rnn(x) if h is None else self.rnn(x, h)

        # Here, we apply the V matrix to every hidden state to produce the outputs
        predictions = self.V(out)

        # predictions is of size (batch_size, input_length, target_size)
        predictions = predictions.view(batch_size, -1, self.target_size, self.nr_params)

        # returns outputs for all inputs, only the last one is needed for prediction time
        return predictions, last_hidden_state


class RNNModel(DualCovariatesTorchModel):
    def __init__(
        self,
        input_chunk_length: int,
        model: Union[str, type[CustomRNNModule]] = "RNN",
        hidden_dim: int = 25,
        n_rnn_layers: int = 1,
        dropout: float = 0.0,
        training_length: int = 24,
        **kwargs,
    ):
        """Recurrent Neural Network Model (RNNs).

        This class provides three variants of RNNs:

        * Vanilla RNN

        * LSTM

        * GRU

        RNNModel is fully recurrent in the sense that, at prediction time, an output is computed using these inputs:

        - previous target value, which will be set to the last known target value for the first prediction,
          and for all other predictions it will be set to the previous prediction (in an autoregressive fashion),
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
        input_chunk_length
            Number of past time steps that are fed to the forecasting module at prediction time.
        model
            Either a string specifying the RNN module type ("RNN", "LSTM" or "GRU"), or a subclass of
            :class:`CustomRNNModule` (the class itself, not an object of the class) with a custom logic.
        hidden_dim
            Size for feature maps for each hidden RNN layer (:math:`h_n`).
        n_rnn_layers
            The number of recurrent layers.
        dropout
            Fraction of neurons affected by Dropout.
        training_length
            The length of both input (target and covariates) and output (target) time series used during
            training. Must have a larger value than `input_chunk_length`, because otherwise during training
            the RNN is never run for as many iterations as it will during inference. For more information on
            this parameter, please see `darts.utils.data.ShiftedDataset`.
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
            defaults to the following string ``"YYYY-mm-dd_HH_MM_SS_torch_model_run_PID"``, where the initial part
            of the name is formatted with the local date and time, while PID is the processed ID (preventing models
            spawned at the same time by different processes to share the same model_name). E.g.,
            ``"2021-06-14_09_53_32_torch_model_run_44607"``.
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
            Whether to automatically save the untrained model and checkpoints from training.
            To load the model from checkpoint, call :func:`MyModelClass.load_from_checkpoint()`, where
            :class:`MyModelClass` is the :class:`TorchForecastingModel` class that was used (such as :class:`TFTModel`,
            :class:`NBEATSModel`, etc.). If set to ``False``, the model can still be manually saved using
            :func:`save()` and loaded using :func:`load()`. Default: ``False``.
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

                def encode_year(idx):
                    return (idx.year - 1950) / 50

                add_encoders={
                    'cyclic': {'future': ['month']},
                    'datetime_attribute': {'future': ['hour', 'dayofweek']},
                    'position': {'past': ['relative'], 'future': ['relative']},
                    'custom': {'past': [encode_year]},
                    'transformer': Scaler(),
                    'tz': 'CET'
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

        Examples
        --------
        >>> from darts.datasets import WeatherDataset
        >>> from darts.models import RNNModel
        >>> series = WeatherDataset().load()
        >>> # predicting atmospheric pressure
        >>> target = series['p (mbar)'][:100]
        >>> # optionally, use future temperatures (pretending this component is a forecast)
        >>> future_cov = series['T (degC)'][:106]
        >>> # `training_length` > `input_chunk_length` to mimic inference constraints
        >>> model = RNNModel(
        >>>     model="RNN",
        >>>     input_chunk_length=6,
        >>>     training_length=18,
        >>>     n_epochs=20,
        >>> )
        >>> model.fit(target, future_covariates=future_cov)
        >>> pred = model.predict(6)
        >>> pred.values()
        array([[ 3.18922903],
               [ 1.17791019],
               [ 0.39992814],
               [ 0.13277921],
               [ 0.02523252],
               [-0.01829086]])

        .. note::
            `RNN example notebook <https://unit8co.github.io/darts/examples/04-RNN-examples.html>`_ presents techniques
            that can be used to improve the forecasts quality compared to this simple usage example.
        """
        if training_length < input_chunk_length:
            raise_log(
                ValueError(
                    f"`training_length` ({training_length}) must be `>=input_chunk_length` ({input_chunk_length})."
                ),
                logger=logger,
            )
        # create copy of model parameters
        model_kwargs = {key: val for key, val in self.model_params.items()}

        for kwarg, default_value in zip(
            [
                "output_chunk_length",
                "use_reversible_instance_norm",
                "output_chunk_shift",
            ],
            [1, False, 0],
        ):
            if model_kwargs.get(kwarg) is not None:
                logger.warning(
                    f"ignoring user defined `{kwarg}`. RNNModel uses a fixed "
                    f"`{kwarg}={default_value}`."
                )
            model_kwargs[kwarg] = default_value

        super().__init__(**self._extract_torch_model_params(**model_kwargs))

        # extract pytorch lightning module kwargs
        self.pl_module_params = self._extract_pl_module_params(**model_kwargs)

        # check we got right model type specified:
        if model not in ["RNN", "LSTM", "GRU"]:
            if not inspect.isclass(model) or not issubclass(model, CustomRNNModule):
                raise_log(
                    ValueError(
                        "`model` is not a valid RNN model. Please specify 'RNN', 'LSTM', 'GRU', or give a subclass "
                        "(not an instance) of darts.models.forecasting.rnn_model.CustomRNNModule."
                    ),
                    logger=logger,
                )

        self.rnn_type_or_module = model
        self.dropout = dropout
        self.hidden_dim = hidden_dim
        self.n_rnn_layers = n_rnn_layers
        self.training_length = training_length

    def _create_model(self, train_sample: tuple[torch.Tensor]) -> torch.nn.Module:
        # samples are made of (past_target, historic_future_covariates, future_covariates, future_target)
        # historic_future_covariates and future_covariates have the same width
        input_dim = train_sample[0].shape[1] + (
            train_sample[1].shape[1] if train_sample[1] is not None else 0
        )
        output_dim = train_sample[-1].shape[1]
        nr_params = 1 if self.likelihood is None else self.likelihood.num_parameters

        kwargs = {}
        if isinstance(self.rnn_type_or_module, str):
            model_cls = _RNNModule
            kwargs["name"] = self.rnn_type_or_module
        else:
            model_cls = self.rnn_type_or_module
        return model_cls(
            input_size=input_dim,
            target_size=output_dim,
            nr_params=nr_params,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
            num_layers=self.n_rnn_layers,
            **self.pl_module_params,
            **kwargs,
        )

    def _build_train_dataset(
        self,
        target: Sequence[TimeSeries],
        past_covariates: Optional[Sequence[TimeSeries]],
        future_covariates: Optional[Sequence[TimeSeries]],
        sample_weight: Optional[Sequence[TimeSeries]],
        max_samples_per_ts: Optional[int],
    ) -> DualCovariatesShiftedDataset:
        return DualCovariatesShiftedDataset(
            target_series=target,
            covariates=future_covariates,
            length=self.training_length,
            shift=1,
            max_samples_per_ts=max_samples_per_ts,
            use_static_covariates=self.uses_static_covariates,
            sample_weight=sample_weight,
        )

    def _verify_train_dataset_type(self, train_dataset: TrainingDataset):
        raise_if_not(
            isinstance(train_dataset, DualCovariatesShiftedDataset),
            "RNNModel requires a training dataset of type DualCovariatesShiftedDataset.",
        )
        raise_if_not(
            train_dataset.ds_past.shift == 1,
            "RNNModel requires a shifted training dataset with shift=1.",
        )

    @property
    def supports_multivariate(self) -> bool:
        return True

    @property
    def min_train_series_length(self) -> int:
        return self.training_length + 1

    @property
    def extreme_lags(
        self,
    ) -> tuple[
        Optional[int],
        Optional[int],
        Optional[int],
        Optional[int],
        Optional[int],
        Optional[int],
        int,
        Optional[int],
    ]:
        return (
            -self.input_chunk_length,
            self.output_chunk_length - 1,
            None,
            None,
            -self.input_chunk_length,
            self.output_chunk_length - 1,
            self.output_chunk_shift,
            self.training_length - self.input_chunk_length,
        )
