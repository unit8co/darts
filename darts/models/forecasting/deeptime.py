"""
DeepTime
-------
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from darts.logging import get_logger, raise_if_not
from darts.models.forecasting.pl_forecasting_module import PLPastCovariatesModule
from darts.models.forecasting.torch_forecasting_model import PastCovariatesTorchModel
from darts.utils.torch import MonteCarloDropout

logger = get_logger(__name__)


ACTIVATIONS = [
    "ReLU",
    "RReLU",
    "PReLU",
    "ELU",
    "Softplus",
    "Tanh",
    "SELU",
    "LeakyReLU",
    "Sigmoid",
    "GELU",
]


class GaussianFourierFeatureTransform(nn.Module):
    def __init__(self, input_dim: int, n_fourier_feats: int, scales: List[float]):
        """
        Implementation of the Gaussian Fourier features mapping.
        https://arxiv.org/abs/2006.10739
        https://github.com/ndahlquist/pytorch-fourier-feature-networks

        Parameters
        ----------
        input_dim
            The dimensionality of the input time series.
        n_fourier_feats
            Number of Fourier components to sample to represent to time-serie in the frequency domain
        scales
            Scaling factors applied to the normal distribution sampled for Fourier components' magnitude

        Inputs
        ------
        x of shape `(1, input_chunk_length+output_chunk_length, 1)`
            Tensor containing the [0,1] normalised time representation.

        Outputs
        -------
        y of shape `(1, input_chunk_length+output_chunk_length, n_fourier_feats)`
            Tensor containing the Gaussian Fourier features for the processed period.
        """
        super().__init__()
        self.input_dim = input_dim
        self.n_fourier_feats = n_fourier_feats
        self.scales = scales

        n_scale_feats = n_fourier_feats // (2 * len(scales))
        B_size = (input_dim, n_scale_feats)
        # Sample Fourier components
        B = torch.cat([torch.randn(B_size) * scale for scale in scales], dim=1)
        self.register_buffer("B", B)

    def forward(self, x: Tensor) -> Tensor:
        raise_if_not(
            x.dim() >= 2,
            f"Expected 2 or more dimensional input (got {x.dim()}D input)",
            logger,
        )
        x = torch.einsum("... t n, n d -> ... t d", [x, self.B])
        x = 2 * np.pi * x
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)


class INR(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_layers: int,
        hidden_layers_width: int,
        n_fourier_feats: int,
        scales: List[float],
        dropout: float,
        activation: str,
        nr_params: int,
    ):
        """Implicit Neural Representation, mapping values to their coordinates using a Multi-Layer Perceptron.

        Features can be encoded using either a Linear layer or a Gaussian Fourier Transform

        Parameters
        ----------
        input_dim
            The dimensionality of the input time series.
        num_layers
            The number of fully connected layers.
        hidden_layers_width
            Determines the number of neurons that make up each hidden fully connected layer.
            If a list is passed, it must have a length equal to `num_layers`. If an integer is passed,
            every layers will have the same width.
        n_fourier_feats
            Number of Fourier components to sample to represent to time-serie in the frequency domain
        scales
            Scaling factors applied to the normal distribution sampled for Fourier components' magnitude
        dropout
            The fraction of neurons that are dropped at each layer.
        activation
            The activation function of fully connected network intermediate layers.

        Inputs
        ------
        x of shape `(1, input_chunk_length+output_chunk_length, 1)`
            Tensor containing the [0,1] normalised time representation.

        Outputs
        -------
        y of shape `(1, input_chunk_length+output_chunk_length, hidden_layers_width)`
            Tensor containing the implicit neural representation of the time.
        """
        super().__init__()

        self.input_dim = input_dim
        self.num_layers = num_layers
        self.n_fourier_feats = n_fourier_feats
        self.scales = scales
        self.dropout = dropout
        self.nr_params = nr_params

        raise_if_not(
            activation in ACTIVATIONS, f"'{activation}' is not in {ACTIVATIONS}"
        )
        self.activation = getattr(nn, activation)()

        if isinstance(hidden_layers_width, int):
            self.hidden_layers_width = [hidden_layers_width] * self.num_layers
        else:
            self.hidden_layers_width = hidden_layers_width

        if n_fourier_feats == 0:
            feats_size = self.hidden_layers_width[0]
            self.features = nn.Linear(self.input_dim, feats_size)
        else:
            feats_size = self.n_fourier_feats
            self.features = GaussianFourierFeatureTransform(
                self.input_dim, feats_size, self.scales
            )

        # Fully Connected Network
        # TODO : solve ambiguity between the num of layers and the number of hidden layers
        last_width = feats_size
        linear_layer_stack_list = []
        for layer_width in self.hidden_layers_width[:-1]:
            linear_layer_stack_list.append(nn.Linear(last_width, layer_width))
            linear_layer_stack_list.append(self.activation)

            if self.dropout > 0:
                linear_layer_stack_list.append(MonteCarloDropout(p=self.dropout))

            linear_layer_stack_list.append(nn.LayerNorm(layer_width))

            last_width = layer_width

        # output width multiplied self.nr_params to have one time encoding per param
        linear_layer_stack_list.append(
            nn.Linear(last_width, self.hidden_layers_width[-1] * self.nr_params)
        )
        linear_layer_stack_list.append(self.activation)
        if self.dropout > 0:
            linear_layer_stack_list.append(MonteCarloDropout(p=self.dropout))
        linear_layer_stack_list.append(
            nn.LayerNorm(self.hidden_layers_width[-1] * self.nr_params)
        )

        self.layers = nn.Sequential(*linear_layer_stack_list)

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        return self.layers(x)


class RidgeRegressor(nn.Module):
    def __init__(self, lambda_init: float = 0.0):
        """Implementation of the closed form Ridge Regression with a regularization coefficient."""
        super().__init__()
        self._lambda = nn.Parameter(torch.as_tensor(lambda_init))

    def forward(self, reprs: Tensor, x: Tensor, reg_coeff: float = None) -> Tensor:
        if reg_coeff is None:
            reg_coeff = self.reg_coeff()
        w, b = self.get_weights(reprs, x, reg_coeff)
        return w, b

    def get_weights(self, X: Tensor, Y: Tensor, reg_coeff: float) -> Tensor:
        batch_size, n_samples, n_dim = X.shape
        ones = torch.ones(batch_size, n_samples, 1, device=X.device)
        X = torch.concat([X, ones], dim=-1)

        if n_samples >= n_dim:
            # standard
            A = torch.bmm(X.mT, X)
            A.diagonal(dim1=-2, dim2=-1).add_(reg_coeff)
            B = torch.bmm(X.mT, Y)
            weights = torch.linalg.solve(A, B)
        else:
            # Woodbury
            A = torch.bmm(X, X.mT)
            A.diagonal(dim1=-2, dim2=-1).add_(reg_coeff)
            weights = torch.bmm(X.mT, torch.linalg.solve(A, Y))

        return weights[:, :-1], weights[:, -1:]

    def reg_coeff(self) -> Tensor:
        return F.softplus(self._lambda)


class _DeepTimeModule(PLPastCovariatesModule):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        nr_params: int,
        inr_num_layers: int,
        inr_layers_width: Union[int, List[int]],
        n_fourier_feats: int,
        scales: List[float],
        dropout: float,
        activation: str,
        **kwargs,
    ):
        """PyTorch module implementing the DeepTIMe architecture.

        Parameters
        ----------
        input_chunk_length
            The length of the input sequence (lookback) fed to the model.
        output_chunk_length
            The length of the forecast (horizon) of the model.
        inr_num_layers
            The number of fully connected layers in the INR module.
        inr_layers_width
            Determines the number of neurons that make up each hidden fully connected layer of the INR module.
            If a list is passed, it must have a length equal to `num_layers`. If an integer is passed,
            every layers will have the same width.
        n_fourier_feats
            Number of Fourier components to sample to represent to time-serie in the frequency domain
        scales
            Scaling factors applied to the normal distribution sampled for Fourier components' magnitude
        dropout
            The dropout probability to be used in fully connected layers (default=0). This is compatible with
            Monte Carlo dropout at inference time for model uncertainty estimation (enabled with
            ``mc_dropout=True`` at prediction time).
        activation
            The activation function of encoder/decoder intermediate layer (default='ReLU').
            Supported activations: ['ReLU','RReLU', 'PReLU', 'Softplus', 'Tanh', 'SELU', 'LeakyReLU',  'Sigmoid']
        **kwargs
            Optional arguments to initialize the pytorch_lightning.Module, pytorch_lightning.Trainer, and
            Darts' :class:`TorchForecastingModel`.
        """
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.nr_params = nr_params
        self.inr_num_layers = inr_num_layers
        self.inr_layers_width = inr_layers_width
        self.n_fourier_feats = n_fourier_feats
        self.scales = scales

        self.dropout = dropout
        self.activation = activation

        self.inr = INR(
            input_dim=self.input_dim + 1,
            num_layers=self.inr_num_layers,
            hidden_layers_width=self.inr_layers_width,
            n_fourier_feats=self.n_fourier_feats,
            scales=self.scales,
            dropout=self.dropout,
            activation=self.activation,
            nr_params=self.nr_params,
        )

        self.adaptive_weights = RidgeRegressor()

    def forward(self, x_in: Tensor) -> Tensor:
        x, _ = x_in  # x_in: (past_target|past_covariate, static_covariates)
        batch_size, _, _ = x.shape  # x: (batch_size, in_len, in_dim)

        coords = self.get_coords(self.input_chunk_length, self.output_chunk_length)
        time_reprs = self.inr(coords)
        # time_reprs.shape = [batch_size, input_chunk_len+output_chunk_len, inr_layers_width[-1]*nr_params]
        time_reprs = time_reprs.repeat(batch_size, 1, 1)
        time_reprs = time_reprs.reshape(
            batch_size,
            self.input_chunk_length + self.output_chunk_length,
            -1,
            self.nr_params,
        )

        # must use a different time_reprs (A) for each nr_param so that the linear equation changes:
        # AX = B where A is the diag of lookback_reprs.T*lookback_reprs and B is lookback_reprs.T*x
        # the parameter lambda of the RidgeRegressor is shared across the nr_params
        forecasts = []
        for i in range(self.nr_params):
            lookback_reprs = time_reprs[:, : -self.output_chunk_length, :, i]
            horizon_reprs = time_reprs[:, -self.output_chunk_length :, :, i]

            # learn weights from the lookback
            w, b = self.adaptive_weights(lookback_reprs, x)
            # apply weights to the horizon
            forecast = torch.einsum("b d o, b t d -> b t o", [w, horizon_reprs]) + b
            # forecast.shape = [batch, output_chunk_size, input_dim]
            forecasts.append(forecast)

        # y.shape = [batch, output_chunk_size, input_dim, nr_params]
        y = torch.stack(forecasts, dim=-1)
        # retain forecast of target (exclude past/static covariates)
        y = y[:, :, : self.output_dim * self.nr_params, :]
        # TODO: check that target predictions are the first self.output_dim*self.nr_params values, change slicing?
        # TODO: run experiments to check if the model benefits from covariates (potentially in the INR?!)
        return y

    def get_coords(self, lookback_len: int, horizon_len: int) -> Tensor:
        """Return time axis encoded as float values between 0 and 1"""
        coords = torch.linspace(0, 1, lookback_len + horizon_len)
        # coords.shape = [1, lookback_len + horizon_len, 1]
        return coords.unsqueeze(dim=0).unsqueeze(dim=-1)

    def configure_optimizers(self):
        """Override the configure_optimizers to define three groups of parameters, one for the
        Ridge Regression weights, one for the biais and norm of the FCN and another of the other
        weights of the FCN.
        """
        # we have to create copies because we cannot save model.parameters into object state (not serializable)
        optimizer_kws = {k: v for k, v in self.optimizer_kwargs.items()}

        # define three parameters groups
        group1 = []  # lambda (RidgeRegressor)
        group2 = []  # no decay (bias and norm)
        group3 = []  # decay
        no_decay_list = (
            "bias",
            "norm",
        )
        for param_name, param in self.named_parameters():
            if "_lambda" in param_name:
                group1.append(param)
            elif any([mod in param_name for mod in no_decay_list]):
                group2.append(param)
            else:
                group3.append(param)
        optimizer = torch.optim.Adam(
            [
                {"params": group1, "weight_decay": 0, "lr": optimizer_kws["lambda_lr"]},
                {"params": group2, "weight_decay": 0},
                {"params": group3},
            ],
            lr=optimizer_kws["lr"],
            weight_decay=optimizer_kws["weight_decay"],
        )

        # define a scheduler for each optimizer
        lr_sched_kws = {k: v for k, v in self.lr_scheduler_kwargs.items()}

        total_epochs = lr_sched_kws["total_epochs"]
        warmup_epochs = lr_sched_kws["warmup_epochs"]
        eta_min = lr_sched_kws["eta_min"]
        scheduler_fns = []

        def no_scheduler(current_epoch):
            return 1

        def cosine_annealing(current_epoch):
            return (
                eta_min
                + 0.5
                * (eta_max - eta_min)
                * (
                    1.0
                    + np.cos(
                        (current_epoch - warmup_epochs)
                        / (total_epochs - warmup_epochs)
                        * np.pi
                    )
                )
            ) / lr

        def cosine_annealing_with_linear_warmup(current_epoch):
            if current_epoch < warmup_epochs:
                return current_epoch / warmup_epochs
            else:
                return (
                    eta_min
                    + 0.5
                    * (eta_max - eta_min)
                    * (
                        1.0
                        + np.cos(
                            (current_epoch - warmup_epochs)
                            / (total_epochs - warmup_epochs)
                            * np.pi
                        )
                    )
                ) / lr

        for param_group, scheduler in zip(
            optimizer.param_groups, lr_sched_kws["scheduler_names"]
        ):
            if scheduler == "none":
                fn = no_scheduler
            elif scheduler == "cosine_annealing":
                lr = eta_max = param_group["lr"]
                fn = cosine_annealing
            elif scheduler == "cosine_annealing_with_linear_warmup":
                lr = eta_max = param_group["lr"]
                fn = cosine_annealing_with_linear_warmup
            else:
                raise ValueError(f"No such scheduler, {scheduler}")
            scheduler_fns.append(fn)

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=scheduler_fns
        )

        return [optimizer], {
            "scheduler": lr_scheduler,
            "monitor": "val_loss",
        }


class DeepTimeModel(PastCovariatesTorchModel):
    def __init__(
        self,
        input_chunk_length: int,
        output_chunk_length: int,
        inr_num_layers: int = 5,
        inr_layers_width: Union[int, List[int]] = 256,
        n_fourier_feats: int = 4096,
        scales: List[float] = None,
        dropout: float = 0.1,
        activation: str = "ReLU",
        optimizer_kwargs: Optional[Dict] = None,
        lr_scheduler_kwargs: Optional[Dict] = None,
        **kwargs,
    ):
        """Deep time-index model with meta-learning (DeepTIMe).

        This is an implementation of the DeepTime architecture, as outlined in [1]_. The default arguments
        correspond to the hyper-parameters described in the article.

        This model supports past covariates (known for `input_chunk_length` points before prediction time).

        Parameters
        ----------
        input_chunk_length
            The length of the input sequence (lookback) fed to the model.
        output_chunk_length
            The length of the forecast (horizon) of the model.
        inr_num_layers
            The number of fully connected layers in the INR module.
        inr_layers_width
            Determines the number of neurons that make up each hidden fully connected layer of the INR module.
            If a list is passed, it must have a length equal to `num_layers`. If an integer is passed,
            every layers will have the same width.
        n_fourier_feats
            Number of Fourier components to sample to represent to time-serie in the frequency domain
        scales
            Scaling factors applied to the normal distribution sampled for Fourier components' magnitude
        legacy_optimiser
            Determine if the optimiser described in the original article should be used. Overwrites the
            parameters provided in optimizers_cls, optimizers_kwargs, scheduler_cls and scheduler_kwargs.
            Defaults to True.
        dropout
            The dropout probability to be used in fully connected layers (default=0). This is compatible with
            Monte Carlo dropout at inference time for model uncertainty estimation (enabled with
            ``mc_dropout=True`` at prediction time).
        activation
            The activation function of encoder/decoder intermediate layer (default='ReLU').
            Supported activations: ['ReLU','RReLU', 'PReLU', 'Softplus', 'Tanh', 'SELU', 'LeakyReLU',  'Sigmoid']
        **kwargs
            Optional arguments to initialize the pytorch_lightning.Module, pytorch_lightning.Trainer, and
            Darts' :class:`TorchForecastingModel`.

        loss_fn
            PyTorch loss function used for training.
            This parameter will be ignored for probabilistic models if the ``likelihood`` parameter is specified.
            Default: ``torch.nn.MSELoss()``.
        torch_metrics
            A torch metric or a ``MetricCollection`` used for evaluation. A full list of available metrics can be found
            at https://torchmetrics.readthedocs.io/en/latest/. Default: ``None``.
        optimizer_cls
            The PyTorch optimizer class to be used. Default: ``torch.optim.Adam``.
        optimizer_kwargs
            Optional keyword arguments for the PyTorch optimizer: ``'lr'`` for the INR networks weights and
            ``'lambda_lr'`` for the Ridge Regression regularisation term. Otherwise the values from the
            original publication will be used. Default: ``{"lr": 1e-3, "lambda_lr": 1.0, "weight_decay": 0.0}``.
        lr_scheduler_cls
            Due to the model architecture, distincts learning rate schedulers can be used for the three groups of
            parameters: Ridge Regression regularisation term, (biais and norm) and weights of the INR network.
            They must be provided using the lr_scheduler_kwargs argument.
        lr_scheduler_kwargs
            Optionally, names and keyword arguments for the three learning rate scheduler (respectively Ridge Regression
            regularisation term, INR biais and norm, and INR weights. Supported scheduler: "none",  "cosine_annealing"
            and "cosine_annealing_with_linear_warmup".  Default: {"warmup_epochs": 5, "total_epochs": self.n_epochs,
            "eta_min": 0.0, "scheduler_names": ["cosine_annealing", "cosine_annealing_with_linear_warmup",
            "cosine_annealing_with_linear_warmup"]}.
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

        References
        ----------
        .. [1] https://arxiv.org/abs/2207.06046
        """
        super().__init__(**self._extract_torch_model_params(**self.model_params))

        # extract pytorch lightning module kwargs
        self.pl_module_params = self._extract_pl_module_params(**self.model_params)

        if scales is None:
            scales = [0.01, 0.1, 1, 5, 10, 20, 50, 100]

        if self.pl_module_params["optimizer_kwargs"] is None:
            self.pl_module_params["optimizer_kwargs"] = {
                "lr": 1e-3,
                "lambda_lr": 1.0,
                "weight_decay": 0.0,
            }
        if self.pl_module_params["lr_scheduler_kwargs"] is None:
            self.pl_module_params["lr_scheduler_kwargs"] = {
                "warmup_epochs": 5,
                "eta_min": 0.0,
                "scheduler_names": [
                    "cosine_annealing",
                    "cosine_annealing_with_linear_warmup",
                    "cosine_annealing_with_linear_warmup",
                ],
            }

        raise_if_not(
            isinstance(inr_layers_width, int)
            or len(inr_layers_width) == inr_num_layers,
            "Please pass an integer or a list of integers with length `inr_num_layers`"
            "as value for the `inr_layers_width` argument.",
            logger,
        )

        raise_if_not(
            n_fourier_feats % (2 * len(scales)) == 0,
            f"n_fourier_feats: {n_fourier_feats} must be divisible by 2 * len(scales) = {2 * len(scales)}",
            logger,
        )

        # user can either use default arguments or must redefine all of them
        expected_params = {
            "weight_decay",
            "lambda_lr",
            "lr",
            "warmup_epochs",
            "eta_min",
            "scheduler_names",
        }
        optimizer_params = self.pl_module_params["optimizer_kwargs"].keys()
        scheduler_params = self.pl_module_params["lr_scheduler_kwargs"].keys()
        provided_params = set(optimizer_params).union(set(scheduler_params))
        missing_params = expected_params - provided_params
        raise_if_not(
            len(missing_params) == 0,
            f"Missing argument(s) for the optimiser: {missing_params}. `weight_decay`, "
            "`lambda_lr` and `lr` must be defined in `optimizer_kwargs` whereas `eta_min`, "
            "`scheduler_names` and `warmup_epochs` must be defined in `lr_scheduler_kwargs`.",
            logger,
        )

        self.pl_module_params["lr_scheduler_kwargs"]["total_epochs"] = self.n_epochs

        raise_if_not(
            self.n_epochs
            > self.pl_module_params["lr_scheduler_kwargs"]["warmup_epochs"],
            f"n_epochs ({self.n_epochs}) must be greater than the number of warmup epochs for the "
            f"learning rate scheduler ({self.pl_module_params['lr_scheduler_kwargs']['warmup_epochs']}). "
            f" This value is controlled by the `lr_scheduler_kwargs['warmup_epochs']` argument.",
            logger,
        )

        self.inr_num_layers = inr_num_layers
        self.inr_layers_width = inr_layers_width
        self.n_fourier_feats = n_fourier_feats
        self.scales = scales

        self.dropout = dropout
        self.activation = activation

    # TODO: might actually be True?
    @staticmethod
    def _supports_static_covariates() -> bool:
        return False

    def _create_model(
        self,
        train_sample: Tuple[torch.Tensor],
    ) -> torch.nn.Module:
        input_dim = train_sample[0].shape[1] + (
            train_sample[1].shape[1] if train_sample[1] is not None else 0
        )
        output_dim = train_sample[-1].shape[1]
        nr_params = 1 if self.likelihood is None else self.likelihood.num_parameters

        model = _DeepTimeModule(
            input_dim=input_dim,
            output_dim=output_dim,
            nr_params=nr_params,
            inr_num_layers=self.inr_num_layers,
            inr_layers_width=self.inr_layers_width,
            n_fourier_feats=self.n_fourier_feats,
            scales=self.scales,
            dropout=self.dropout,
            activation=self.activation,
            **self.pl_module_params,
        )
        return model
