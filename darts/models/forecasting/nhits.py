"""
N-HiTS
------
"""

from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from darts.logging import get_logger, raise_if_not
from darts.models.forecasting.pl_forecasting_module import PLPastCovariatesModule
from darts.models.forecasting.torch_forecasting_model import PastCovariatesTorchModel

logger = get_logger(__name__)


class _Block(nn.Module):
    def __init__(
        self,
        input_chunk_length: int,
        output_chunk_length: int,
        num_layers: int,
        layer_width: int,
        nr_params: int,
        pooling_kernel_size: int,
        n_freq_downsample: int,
        batch_norm: bool,
        dropout: float,
    ):
        """PyTorch module implementing the basic building block of the N-BEATS architecture.

        The blocks produce outputs of size (target_length, nr_params); i.e.
        "one vector per parameter". The parameters are predicted only for forecast outputs.
        Backcast outputs are in the original "domain".

        Parameters
        ----------
        num_layers
            The number of fully connected layers preceding the final forking layers.
        layer_width
            The number of neurons that make up each fully connected layer.
        nr_params
            The number of parameters of the likelihood (or 1 if no likelihood is used)
        expansion_coefficient_dim
            The dimensionality of the waveform generator parameters, also known as expansion coefficients.
            Used in the generic architecture and the trend module of the interpretable architecture, where it determines
            the degree of the polynomial basis.
        input_chunk_length
            The length of the input sequence fed to the model.
        target_length
            The length of the forecast of the model.
        g_type
            The type of function that is implemented by the waveform generator.

        Inputs
        ------
        x of shape `(batch_size, input_chunk_length)`
            Tensor containing the input sequence.

        Outputs
        -------
        x_hat of shape `(batch_size, input_chunk_length)`
            Tensor containing the 'backcast' of the block, which represents an approximation of `x`
            given the constraints of the functional space determined by `g`.
        y_hat of shape `(batch_size, output_chunk_length)`
            Tensor containing the forward forecast of the block.

        """
        super().__init__()

        self.num_layers = num_layers
        self.layer_width = layer_width
        self.input_chunk_length = input_chunk_length
        self.output_chunk_length = output_chunk_length
        self.nr_params = nr_params
        self.pooling_kernel_size = pooling_kernel_size
        self.n_freq_downsample = n_freq_downsample
        self.batch_norm = batch_norm
        self.dropout = dropout

        self.activation = nn.ReLU()  # TODO: make configurable?

        # layer widths
        in_len = int(np.ceil(input_chunk_length / pooling_kernel_size))
        self.layer_widths = [in_len] + [self.layer_width] * self.num_layers

        # number of parameters theta for backcast and forecast
        """
        Note:
        -----
        We use two "last" layers, one for the backcast yielding n_theta_backcast outputs,
        and one for the forecast yielding n_theta_forecast outputs.

        In the original code, only one last layer yielding "input_chunk_length + n_theta_forecast" [1]
        outputs is used. So they don't use interpolation for the backcast [2], contrary to what is
        explained in the paper. Here we use what is explained in the paper.

        [1] https://github.com/cchallu/n-hits/blob/4e929ed31e1d3ff5169b4aa0d3762a0040abb8db/
        src/models/nhits/nhits.py#L263
        [2] https://github.com/cchallu/n-hits/blob/4e929ed31e1d3ff5169b4aa0d3762a0040abb8db/
        src/models/nhits/nhits.py#L66
        """
        n_theta_backcast = max(input_chunk_length // n_freq_downsample, 1)
        n_theta_forecast = max(output_chunk_length // n_freq_downsample, 1)

        # entry pooling layer
        # TODO: we could leave an option to do e.g. avg pooling
        self.pooling_layer = nn.MaxPool1d(
            kernel_size=self.pooling_kernel_size,
            stride=self.pooling_kernel_size,
            ceil_mode=True,
        )

        # FC layers
        layers = []
        for i in range(self.num_layers):
            layers.append(
                nn.Linear(
                    in_features=self.layer_widths[i],
                    out_features=self.layer_widths[i + 1],
                )
            )
            layers.append(self.activation)

            # TODO: also add these two for NBEATS
            if self.batch_norm:
                layers.append(nn.BatchNorm1d(num_features=self.layer_widths[i + 1]))

            if self.dropout > 0:
                layers.append(nn.Dropout(p=self.dropout))

        self.layers = nn.Sequential(*layers)

        # Fully connected layer producing forecast/backcast expansion coeffcients (waveform generator parameters).
        # The coefficients are emitted for each parameter of the likelihood for the forecast.
        self.backcast_linear_layer = nn.Linear(
            in_features=layer_width, out_features=n_theta_backcast
        )
        self.forecast_linear_layer = nn.Linear(
            in_features=layer_width, out_features=nr_params * n_theta_forecast
        )

    def forward(self, x):
        batch_size = x.shape[0]

        # pooling
        x = x.unsqueeze(1)
        x = self.pooling_layer(x)
        x = x.squeeze(1)

        # fully connected layer stack
        x = self.layers(x)

        # forked linear layers producing waveform generator parameters
        theta_backcast = self.backcast_linear_layer(x)
        theta_forecast = self.forecast_linear_layer(x)

        # set the expansion coefs in last dimension for the forecasts
        theta_forecast = theta_forecast.view(batch_size, self.nr_params, -1)

        # interpolate function expects (batch, "channels", time)
        theta_backcast = theta_backcast.unsqueeze(1)

        # interpolate both backcast and forecast from the thetas
        backcast = F.interpolate(
            theta_backcast, size=self.input_chunk_length, mode="linear"
        )
        forecast = F.interpolate(
            theta_forecast, size=self.output_chunk_length, mode="linear"
        )

        backcast = backcast.squeeze(1)  # TODO: Needed?

        # Set the distribution parameters as the last dimension
        forecast = forecast.reshape(
            x.shape[0], self.output_chunk_length, self.nr_params
        )

        return backcast, forecast


class _Stack(nn.Module):
    def __init__(
        self,
        input_chunk_length: int,
        output_chunk_length: int,
        num_blocks: int,
        num_layers: int,
        layer_width: int,
        nr_params: int,
        pooling_kernel_sizes: Tuple[int],
        n_freq_downsample: Tuple[int],
        batch_norm: bool,
        dropout: float,
    ):
        """PyTorch module implementing one stack of the N-BEATS architecture that comprises multiple basic blocks.

        Parameters
        ----------
        num_blocks
            The number of blocks making up this stack.
        num_layers
            The number of fully connected layers preceding the final forking layers in each block.
        layer_width
            The number of neurons that make up each fully connected layer in each block.
        nr_params
            The number of parameters of the likelihood (or 1 if no likelihood is used)
        expansion_coefficient_dim
            The dimensionality of the waveform generator parameters, also known as expansion coefficients.
        input_chunk_length
            The length of the input sequence fed to the model.
        target_length
            The length of the forecast of the model.
        g_type
            The function that is implemented by the waveform generators in each block.

        Inputs
        ------
        stack_input of shape `(batch_size, input_chunk_length)`
            Tensor containing the input sequence.

        Outputs
        -------
        stack_residual of shape `(batch_size, input_chunk_length)`
            Tensor containing the 'backcast' of the block, which represents an approximation of `x`
            given the constraints of the functional space determined by `g`.
        stack_forecast of shape `(batch_size, output_chunk_length)`
            Tensor containing the forward forecast of the stack.

        """
        super().__init__()

        self.input_chunk_length = input_chunk_length
        self.output_chunk_length = output_chunk_length
        self.nr_params = nr_params

        # TODO: leave option to share weights across blocks?
        self.blocks_list = [
            _Block(
                input_chunk_length,
                output_chunk_length,
                num_layers,
                layer_width,
                nr_params,
                pooling_kernel_sizes[i],
                n_freq_downsample[i],
                batch_norm=(
                    batch_norm and i == 0
                ),  # batch norm only on first block of first stack
                dropout=dropout,
            )
            for i in range(num_blocks)
        ]
        self.blocks = nn.ModuleList(self.blocks_list)

    def forward(self, x):
        # One forecast vector per parameter in the distribution
        stack_forecast = torch.zeros(
            x.shape[0],
            self.output_chunk_length,
            self.nr_params,
            device=x.device,
            dtype=x.dtype,
        )

        for block in self.blocks_list:
            # pass input through block
            x_hat, y_hat = block(x)

            # add block forecast to stack forecast
            stack_forecast = stack_forecast + y_hat

            # subtract backcast from input to produce residual
            x = x - x_hat

        stack_residual = x

        return stack_residual, stack_forecast


class _NHiTSModule(PLPastCovariatesModule):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        nr_params: int,
        num_stacks: int,
        num_blocks: int,
        num_layers: int,
        layer_widths: List[int],
        pooling_kernel_sizes: Tuple[Tuple[int]],
        n_freq_downsample: Tuple[Tuple[int]],
        batch_norm: bool,
        dropout: float,
        **kwargs,
    ):
        """PyTorch module implementing the N-BEATS architecture.

        Parameters
        ----------
        output_dim
            Number of output components in the target
        nr_params
            The number of parameters of the likelihood (or 1 if no likelihood is used).
        generic_architecture
            Boolean value indicating whether the generic architecture of N-BEATS is used.
            If not, the interpretable architecture outlined in the paper (consisting of one trend
            and one seasonality stack with appropriate waveform generator functions).
        num_stacks
            The number of stacks that make up the whole model. Only used if `generic_architecture` is set to `True`.
        num_blocks
            The number of blocks making up every stack.
        num_layers
            The number of fully connected layers preceding the final forking layers in each block of every stack.
            Only used if `generic_architecture` is set to `True`.
        layer_widths
            Determines the number of neurons that make up each fully connected layer in each block of every stack.
            If a list is passed, it must have a length equal to `num_stacks` and every entry in that list corresponds
            to the layer width of the corresponding stack. If an integer is passed, every stack will have blocks
            with FC layers of the same width.
        **kwargs
            all parameters required for :class:`darts.model.forecasting_models.PLForecastingModule` base class.

        Inputs
        ------
        x of shape `(batch_size, input_chunk_length)`
            Tensor containing the input sequence.

        Outputs
        -------
        y of shape `(batch_size, output_chunk_length, target_size/output_dim, nr_params)`
            Tensor containing the output of the NBEATS module.

        """
        super().__init__(**kwargs)

        # required for all modules -> saves hparams for checkpoints
        self.save_hyperparameters()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.nr_params = nr_params
        self.input_chunk_length_multi = self.input_chunk_length * input_dim

        # TODO: shouldn't this be output_dim?
        self.output_chunk_length_multi = self.output_chunk_length * input_dim

        self.stacks_list = [
            _Stack(
                self.input_chunk_length_multi,
                self.output_chunk_length_multi,
                num_blocks,
                num_layers,
                layer_widths[i],
                nr_params,
                pooling_kernel_sizes[i],
                n_freq_downsample[i],
                batch_norm=(
                    batch_norm and i == 0
                ),  # batch norm only on first block of first stack
                dropout=dropout,
            )
            for i in range(num_stacks)
        ]

        self.stacks = nn.ModuleList(self.stacks_list)

        # setting the last backcast "branch" to be not trainable (without next block/stack, it doesn't need to be
        # backpropagated). Removing this lines would cause logtensorboard to crash, since no gradient is stored
        # on this params (the last block backcast is not part of the final output of the net).
        self.stacks_list[-1].blocks[-1].backcast_linear_layer.requires_grad_(False)

    def forward(self, x):

        # if x1, x2,... y1, y2... is one multivariate ts containing x and y, and a1, a2... one covariate ts
        # we reshape into x1, y1, a1, x2, y2, a2... etc
        x = torch.reshape(x, (x.shape[0], self.input_chunk_length_multi, 1))
        # squeeze last dimension (because model is univariate)
        x = x.squeeze(dim=2)

        # One vector of length target_length per parameter in the distribution
        y = torch.zeros(
            x.shape[0],
            self.output_chunk_length_multi,
            self.nr_params,
            device=x.device,
            dtype=x.dtype,
        )

        for stack in self.stacks_list:
            # compute stack output
            stack_residual, stack_forecast = stack(x)

            # add stack forecast to final output
            y = y + stack_forecast

            # set current stack residual as input for next stack
            x = stack_residual

        # In multivariate case, we get a result [x1_param1, x1_param2], [y1_param1, y1_param2], [x2..], [y2..], ...
        # We want to reshape to original format. We also get rid of the covariates and keep only the target dimensions.
        # The covariates are by construction added as extra time series on the right side. So we need to get rid of this
        # right output (keeping only :self.output_dim).
        y = y.view(
            y.shape[0], self.output_chunk_length, self.input_dim, self.nr_params
        )[:, :, : self.output_dim, :]

        return y


class NHiTS(PastCovariatesTorchModel):
    def __init__(
        self,
        input_chunk_length: int,
        output_chunk_length: int,
        num_stacks: int = 3,
        num_blocks: int = 1,
        num_layers: int = 2,
        layer_widths: Union[int, List[int]] = 512,
        pooling_kernel_sizes: Optional[Tuple[Tuple[int]]] = None,
        n_freq_downsample: Optional[Tuple[Tuple[int]]] = None,
        batch_norm: bool = False,
        dropout: float = 0.0,
        **kwargs,
    ):
        """N-HiTS, as outlined in [X]_.

        In addition to the univariate version presented in the paper, our implementation also
        supports multivariate series (and covariates) by flattening the model inputs to a 1-D series
        and reshaping the outputs to a tensor of appropriate dimensions. Furthermore, it also
        supports producing probabilistic forecasts (by specifying a `likelihood` parameter).

        This model supports past covariates (known for `input_chunk_length` points before prediction time).

        Parameters
        ----------
        input_chunk_length
            The length of the input sequence fed to the model.
        output_chunk_length
            The length of the forecast of the model.
        num_stacks
            The number of stacks that make up the whole model. Only used if `generic_architecture` is set to `True`.
            The interpretable architecture always uses two stacks - one for trend and one for seasonality.
        num_blocks
            The number of blocks making up every stack.
        num_layers
            The number of fully connected layers preceding the final forking layers in each block of every stack.
            Only used if `generic_architecture` is set to `True`.
        layer_widths
            Determines the number of neurons that make up each fully connected layer in each block of every stack.
            If a list is passed, it must have a length equal to `num_stacks` and every entry in that list corresponds
            to the layer width of the corresponding stack. If an integer is passed, every stack will have blocks
            with FC layers of the same width.
        expansion_coefficient_dim
            The dimensionality of the waveform generator parameters, also known as expansion coefficients.
            Only used if `generic_architecture` is set to `True`.
        trend_polynomial_degree
            The degree of the polynomial used as waveform generator in trend stacks. Only used if
            `generic_architecture` is set to `False`.
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
        torch_device_str
            Optionally, a string indicating the torch device to use. By default, ``torch_device_str`` is ``None``
            which will run on CPU. Set it to ``"cuda"`` to use all available GPUs or ``"cuda:i"`` to only use
            GPU ``i`` (``i`` must be an integer). For example "cuda:0" will use the first GPU only.

            .. deprecated:: v0.17.0
                ``torch_device_str`` has been deprecated in v0.17.0 and will be removed in a future version.
                Instead, specify this with keys ``"accelerator", "gpus", "auto_select_gpus"`` in your
                ``pl_trainer_kwargs`` dict. Some examples for setting the devices inside the ``pl_trainer_kwargs``
                dict:

                - ``{"accelerator": "cpu"}`` for CPU,
                - ``{"accelerator": "gpu", "gpus": [i]}`` to use only GPU ``i`` (``i`` must be an integer),
                - ``{"accelerator": "gpu", "gpus": -1, "auto_select_gpus": True}`` to use all available GPUS.

                For more info, see here:
                https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#trainer-flags , and
                https://pytorch-lightning.readthedocs.io/en/stable/advanced/multi_gpu.html#select-gpu-devices
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
            Read :meth:`SequentialEncoder <darts.utils.data.encoders.SequentialEncoder>` to find out more about
            ``add_encoders``. Default: ``None``. An example showing some of ``add_encoders`` features:

            .. highlight:: python
            .. code-block:: python

                add_encoders={
                    'cyclic': {'future': ['month']},
                    'datetime_attribute': {'future': ['hour', 'dayofweek']},
                    'position': {'past': ['absolute'], 'future': ['relative']},
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

        References
        ----------
        .. [1] https://openreview.net/forum?id=r1ecqn4YwB
        """
        super().__init__(**self._extract_torch_model_params(**self.model_params))

        # extract pytorch lightning module kwargs
        self.pl_module_params = self._extract_pl_module_params(**self.model_params)

        raise_if_not(
            isinstance(layer_widths, int) or len(layer_widths) == num_stacks,
            "Please pass an integer or a list of integers with length `num_stacks`"
            "as value for the `layer_widths` argument.",
            logger,
        )

        self.num_stacks = num_stacks
        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.layer_widths = layer_widths
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.pooling_kernel_sizes = pooling_kernel_sizes
        self.n_freq_downsample = n_freq_downsample

        if self.pooling_kernel_sizes is None:
            # make stacks handle different frequencies
            # go from in_len to 1 in num_stacks steps:
            max_v = self.input_chunk_length // 2
            self.pooling_kernel_sizes = tuple(
                (int(v),) * num_blocks
                for v in max_v // np.geomspace(1, max_v, num_stacks)
            )
            logger.info(
                "(N-HiTS): Using automatic kernel pooling size: {}.".format(
                    self.pooling_kernel_sizes
                )
            )

        if self.n_freq_downsample is None:
            # go from out_len to 1 in num_stacks steps:
            max_v = self.output_chunk_length // 2
            self.n_freq_downsample = tuple(
                (int(v),) * num_blocks
                for v in max_v // np.geomspace(1, max_v, num_stacks)
            )
            logger.info(
                "(N-HiTS):  Using automatic downsampling coefficients: {}.".format(
                    self.n_freq_downsample
                )
            )

        if isinstance(layer_widths, int):
            self.layer_widths = [layer_widths] * num_stacks

    def _create_model(self, train_sample: Tuple[torch.Tensor]) -> torch.nn.Module:
        # samples are made of (past_target, past_covariates, future_target)
        input_dim = train_sample[0].shape[1] + (
            train_sample[1].shape[1] if train_sample[1] is not None else 0
        )
        output_dim = train_sample[-1].shape[1]
        nr_params = 1 if self.likelihood is None else self.likelihood.num_parameters

        return _NHiTSModule(
            input_dim=input_dim,
            output_dim=output_dim,
            nr_params=nr_params,
            num_stacks=self.num_stacks,
            num_blocks=self.num_blocks,
            num_layers=self.num_layers,
            layer_widths=self.layer_widths,
            pooling_kernel_sizes=self.pooling_kernel_sizes,
            n_freq_downsample=self.n_freq_downsample,
            batch_norm=self.batch_norm,
            dropout=self.dropout,
            **self.pl_module_params,
        )
