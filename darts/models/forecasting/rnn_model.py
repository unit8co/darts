"""
Recurrent Neural Networks
-------------------------
.. autoclass:: CustomRNNModule
   :members: forward
   :no-inherited-members:
   :no-undoc-members:
   :no-special-members:
"""

import inspect
from abc import ABC, abstractmethod
from collections.abc import Sequence

import torch
import torch.nn as nn

from darts import TimeSeries
from darts.logging import get_logger, raise_log
from darts.models.forecasting.pl_forecasting_module import (
    PLForecastingModule,
    io_processor,
)
from darts.models.forecasting.torch_forecasting_model import DualCovariatesTorchModel
from darts.utils.data import ShiftedTorchTrainingDataset
from darts.utils.data.torch_datasets.utils import (
    ModuleStage,
    PLModuleInput,
    PLModuleOutput,
    TorchTrainingSample,
)

logger = get_logger(__name__)


class CustomRNNModule(PLForecastingModule, ABC):
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

        * Define the architecture in the module constructor (``__init__()``)

        * Add the ``forward()`` method and define the logic of your module's forward pass

        * Use the custom module class when creating a new :class:`RNNModel` with parameter ``model``.

        You can use ``darts.models.forecasting.rnn_model._RNNModule`` as an example.

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
    def forward(self, x_in: PLModuleInput) -> PLModuleOutput:
        """RNN Module forward.

        Parameters
        ----------
        x_in
            Named module input. For RNN models, future covariates are remapped into the past cov slot.
            The previous hidden state is in ``x_in.state``. Train / val
            (``x_in.stage is not ModuleStage.PREDICT``) uses the shifted-dataset
            layout; predict applies the inference covariate shift.

        Returns
        -------
        PLModuleOutput
            ``prediction`` has shape `(batch_size, output_chunk_length, target_size, nr_params)`
            and contains the outputs at every time step of the input sequence. During training the
            whole tensor is used as output, whereas during prediction we only use y[:, -1, :].
            ``state`` is the last hidden state, passed to the next ``forward``.
        """
        pass

    def _onnx_wrapper(self, input_sample: PLModuleInput):
        """Export a 1-step cell; inference warms up over the input window."""
        from darts.utils.onnx.export import _prepare_onnx_export

        def _last_step(tensor):
            return tensor[:, -1:] if tensor is not None else None

        # state is always a tensor in the graph, so only the last target is consumed
        input_sample = input_sample.replace(
            past_target=input_sample.past_target[:, -1:],
            past_covariates=_last_step(input_sample.past_covariates),
            historic_future_covariates=_last_step(
                input_sample.historic_future_covariates
            ),
        )
        bundle = _prepare_onnx_export(self, input_sample)
        bundle.spec.stepwise_state = True
        return bundle


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
    def forward(self, x_in: PLModuleInput) -> PLModuleOutput:
        past_target = x_in.past_target
        historic_future_covariates = x_in.historic_future_covariates
        future_covariates = x_in.future_covariates

        if x_in.stage is not ModuleStage.PREDICT:
            # during training, sanity checking and evaluation, RNN receives a ShiftedDataset sample;
            # concatenate `past_target` with `future_covariates` (they have the same length due to
            # shifted dataset)
            historic_future_covariates = future_covariates
        else:
            if x_in.state is not None:
                # a previous predict step already encoded the history into `state`
                past_target = past_target[:, -1:]
            if historic_future_covariates is not None and future_covariates is not None:
                # RNNs need as inputs (target[t] and covariates[t+1]) so here we shift the covariates
                # extract the relevant times depending on the length of `past_target`
                t_offset = (
                    historic_future_covariates.shape[1] - past_target.shape[1] + 1
                )
                historic_future_covariates = torch.cat(
                    [
                        historic_future_covariates[:, t_offset:],
                        future_covariates[:, :1],
                    ],
                    dim=1,
                )

        x_in = x_in.replace(
            past_target=past_target,
            historic_future_covariates=historic_future_covariates,
        )
        x = x_in.concatenate_past_features()
        # data shape (batch_size, input_length, input_size)
        batch_size = x.shape[0]

        # out shape (batch_size, input_length, hidden_dim)
        out, last_hidden_state = (
            self.rnn(x) if x_in.state is None else self.rnn(x, x_in.state)
        )

        # Here, we apply the V matrix to every hidden state to produce the outputs
        predictions = self.V(out)

        # predictions shape (batch_size, input_length, target_size)
        predictions = predictions.view(batch_size, -1, self.target_size, self.nr_params)

        # during prediction mode, only the last prediction is required;
        # otherwise, return outputs for all inputs
        if x_in.stage is ModuleStage.PREDICT:
            predictions = predictions[:, -1:, :]
        return PLModuleOutput(prediction=predictions, state=last_hidden_state)


class RNNModel(DualCovariatesTorchModel):
    def __init__(
        self,
        input_chunk_length: int,
        model: str | type[CustomRNNModule] = "RNN",
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
            training. Must be `>input_chunk_length`, because otherwise during training the RNN is never run for as
            many iterations as it will during inference. For training, a
            :class:`~darts.utils.data.torch_datasets.training_dataset.ShiftedTorchTrainingDataset` is used with
            parameters `input_chunk_length=output_chunk_length=training_length` and `shift=1`.
        **kwargs
            Optional arguments to initialize the pytorch_lightning.Module, pytorch_lightning.Trainer, and
            Darts' :class:`TorchForecastingModel`.

        loss_fn
            PyTorch loss function used for training.
            This parameter will be ignored for probabilistic models if the ``likelihood`` parameter is specified.
            Default: ``torch.nn.MSELoss()``.
        likelihood
            One of Darts' :meth:`Likelihood <darts.utils.likelihood_models.torch.TorchLikelihood>` models to be used for
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
            of the name is formatted with the local date and time, while PID is the process ID (preventing models
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
            Controls the randomness of the weights initialization and reproducible forecasting.
        pl_trainer_kwargs
            By default :class:`TorchForecastingModel` creates a PyTorch Lightning Trainer with several useful presets
            that performs the training, validation and prediction processes. These presets include automatic
            checkpointing, tensorboard logging, setting the torch device and more.
            With ``pl_trainer_kwargs`` you can add additional kwargs to instantiate the PyTorch Lightning trainer
            object. Check the `PL Trainer documentation
            <https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html>`__ for more information about the
            supported kwargs. Default: ``None``.
            Running on GPU(s) is also possible using ``pl_trainer_kwargs`` by specifying keys ``"accelerator",
            "devices", and "auto_select_gpus"``. Some examples for setting the devices inside the ``pl_trainer_kwargs``
            dict:

            - ``{"accelerator": "cpu"}`` for CPU,
            - ``{"accelerator": "gpu", "devices": [i]}`` to use only GPU ``i`` (``i`` must be an integer),
            - ``{"accelerator": "gpu", "devices": -1, "auto_select_gpus": True}`` to use all available GPUs.

            For more info, see here:
            https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#trainer-flags , and
            https://pytorch-lightning.readthedocs.io/en/stable/accelerators/gpu_basic.html#train-on-multiple-gpus

            With parameter ``"callbacks"`` you can add custom or PyTorch-Lightning built-in callbacks to Darts'
            :class:`TorchForecastingModel`. Below is an example for adding EarlyStopping to the training process.
            The model will stop training early if the validation loss `val_loss` does not improve beyond
            specifications. For more information on callbacks, visit:
            `PyTorch Lightning Callbacks
            <https://pytorch-lightning.readthedocs.io/en/stable/extensions/callbacks.html>`__

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
        enable_finetuning
            Enables model fine-tuning. Only effective if not ``None``.
            If a bool, specifies whether to perform full fine-tuning / training (all parameters are updated) or keep
            all parameters frozen. If a dict, specifies which parameters to fine-tune. Must only contain one key-value
            record. Can be used to:

            - Unfreeze specific parameters, while keeping everything else frozen:
              ``{"unfreeze": ["param.name.patterns.*"]}``
            - Freeze specific parameters, while keeping everything else unfrozen:
              ``{"freeze": ["param.name.patterns.*"]}``

            Default: ``None``.

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
        >>> print(pred.values())
        [[ 3.18922903]
         [ 1.17791019]
         [ 0.39992814]
         [ 0.13277921]
         [ 0.02523252]
         [-0.01829086]]

        .. note::
            `RNN example notebook <https://unit8co.github.io/darts/examples/04-RNN-examples.html>`__ presents techniques
            that can be used to improve the forecasts quality compared to this simple usage example.
        """
        if training_length < input_chunk_length:
            raise_log(
                ValueError(
                    f"`training_length` ({training_length}) must be `>=input_chunk_length` ({input_chunk_length})."
                ),
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
                )

        self.rnn_type_or_module = model
        self.dropout = dropout
        self.hidden_dim = hidden_dim
        self.n_rnn_layers = n_rnn_layers
        self.training_length = training_length

    def _create_model(self, train_sample: TorchTrainingSample) -> PLForecastingModule:
        past_target = train_sample.past_target
        future_covariates = train_sample.future_covariates
        input_dim = past_target.shape[1] + (
            future_covariates.shape[1] if future_covariates is not None else 0
        )
        output_dim = past_target.shape[1]
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
        series: Sequence[TimeSeries],
        past_covariates: Sequence[TimeSeries] | None,
        future_covariates: Sequence[TimeSeries] | None,
        sample_weight: Sequence[TimeSeries] | str | None,
        max_samples_per_ts: int | None,
        stride: int = 1,
    ) -> ShiftedTorchTrainingDataset:
        return ShiftedTorchTrainingDataset(
            series=series,
            future_covariates=future_covariates,
            input_chunk_length=self.training_length,
            output_chunk_length=self.training_length,
            shift=1,
            stride=stride,
            max_samples_per_ts=max_samples_per_ts,
            use_static_covariates=self.uses_static_covariates,
            sample_weight=sample_weight,
        )

    @staticmethod
    def _verify_train_dataset_type(train_dataset: ShiftedTorchTrainingDataset):
        if not isinstance(train_dataset, ShiftedTorchTrainingDataset):
            raise_log(
                ValueError(
                    "RNNModel requires a training dataset of type `GenericShiftDataset`. "
                    f"Got {type(train_dataset)} instead."
                ),
            )
        if train_dataset.shift != 1:
            raise_log(
                ValueError(
                    f"RNNModel requires a shifted training dataset with shift=1. Got shift={train_dataset.shift}."
                ),
            )

    @property
    def min_train_samples(self) -> int:
        return (
            super().min_train_samples + self.training_length - self.input_chunk_length
        )
