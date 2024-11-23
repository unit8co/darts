"""
TimesNet Model
-------
The implementation is built upon the Time Series Library's TimesNet model
<https://github.com/thuml/Time-Series-Library/blob/main/models/TimesNet.py>

-------
MIT License

Copyright (c) 2021 THUML @ Tsinghua University

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version

from darts.models.components.embed import DataEmbedding
from darts.models.forecasting.pl_forecasting_module import (
    PLPastCovariatesModule,
    io_processor,
)
from darts.models.forecasting.torch_forecasting_model import PastCovariatesTorchModel


class Inception_Block_V1(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels

        self.kernels = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i)
            for i in range(self.num_kernels)
        ])

        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = torch.stack([kernel(x) for kernel in self.kernels], dim=-1)
        return torch.mean(res, dim=-1)


def FFT_for_Period(x, k: int = 2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)

    T = torch.tensor(x.shape[1], dtype=torch.int64)
    period = (T / top_list).to(torch.int64)
    return period, abs(xf).mean(-1)[:, top_list]


if version.parse(torch.__version__) >= version.parse("2.0.0"):
    import torch.jit

    FFT_for_Period = torch.jit.script(FFT_for_Period)


class TimesBlock(nn.Module):
    def __init__(self, seq_len, pred_len, top_k, d_model, d_ff, num_kernels):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.k = top_k
        self.conv = nn.Sequential(
            Inception_Block_V1(d_model, d_ff, num_kernels=num_kernels),
            nn.GELU(),
            Inception_Block_V1(d_ff, d_model, num_kernels=num_kernels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for period in period_list:
            # padding
            if (self.seq_len + self.pred_len) % period != 0:
                length = (((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([
                    x.shape[0],
                    (length - (self.seq_len + self.pred_len)),
                    x.shape[2],
                ]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = self.seq_len + self.pred_len
                out = x
            # reshape
            out = (
                out.reshape(B, length // period, period, N)
                .permute(0, 3, 1, 2)
                .contiguous()
            )
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, : (self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        # residual connection
        res = res + x
        return res


class _TimesNetModule(PLPastCovariatesModule):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        nr_params: int,
        hidden_size: int,
        num_layers: int,
        num_kernels: int,
        top_k: int,
        embed_type: str = "fixed",
        freq: str = "h",
        **kwargs,
    ):
        """
        input_size
            The dimensionality of the TimeSeries instances that will be fed to the the fit and predict functions.
        output_size
            The dimensionality of the output time series.
        nr_params
            The number of parameters of the likelihood (or 1 if no likelihood is used).
        hidden_size : int
            The size of the hidden layers in the model.
        num_layers : int
            The number of TimesBlock layers in the model.
        num_kernels : int
            The number of kernels in each Inception block within the TimesBlock.
        top_k : int
            The number of top frequencies to consider in the FFT analysis.
        embed_type : str, optional
            The type of embedding to use. Default is "fixed".
        freq : str, optional
            The frequency of the time series. Default is "h" (hourly).
        **kwargs
            Additional keyword arguments passed to the parent PLPastCovariatesModule.

        Notes
        -----
        - The `embed_type` and `freq` parameters are currently placeholders and are not fully utilized
          in the current implementation.
        """
        super().__init__(**kwargs)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.nr_params = nr_params

        # embed_type and freq are placeholders and are not used until the futures
        # covariate in the forward method are figured out
        self.embedding = DataEmbedding(
            input_dim, hidden_size, embed_type=embed_type, freq=freq, dropout=0.1
        )

        self.model = nn.ModuleList([
            TimesBlock(
                seq_len=self.input_chunk_length,
                pred_len=self.output_chunk_length,
                top_k=top_k,
                d_model=hidden_size,
                d_ff=hidden_size * 4,
                num_kernels=num_kernels,
            )
            for _ in range(num_layers)
        ])
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.predict_linear = nn.Linear(
            self.input_chunk_length, self.output_chunk_length + self.input_chunk_length
        )
        self.projection = nn.Linear(hidden_size, output_dim * nr_params)

    @io_processor
    def forward(self, x_in: Tuple) -> torch.Tensor:
        x, _ = x_in

        # Embedding
        x = self.embedding(x, None)  # TODO: future covariate would go here
        x = self.predict_linear(x.transpose(1, 2)).transpose(1, 2)

        # TimesNet
        for layer in self.model:
            x = self.layer_norm(layer(x))

        y = self.projection(x)

        y = y[:, -self.output_chunk_length :, :]
        y = y.view(
            y.shape[0], self.output_chunk_length, self.output_dim, self.nr_params
        )

        return y


class TimesNetModel(PastCovariatesTorchModel):
    def __init__(
        self,
        input_chunk_length: int,
        output_chunk_length: int,
        output_chunk_shift: int = 0,
        hidden_size: int = 32,
        num_layers: int = 2,
        num_kernels: int = 6,
        top_k: int = 5,
        **kwargs,
    ):
        """
        TimesNet model for time series forecasting.

        This model is based on the paper "TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis"
        by Haixu Wu et al. (2023). TimesNet uses a combination of 2D convolutions and frequency domain analysis
        to capture both temporal patterns and periodic variations in time series data.
        https://arxiv.org/abs/2210.02186

        Parameters
        ----------
        input_chunk_length
            Number of time steps in the past to take as a model input (per chunk). Applies to the target
            series, and past and/or future covariates (if the model supports it).
        output_chunk_length
            Number of time steps predicted at once (per chunk) by the internal model. Also, the number of future values
            from future covariates to use as a model input (if the model supports future covariates). It is not the same
            as forecast horizon `n` used in `predict()`, which is the desired number of prediction points generated
            using either a one-shot- or autoregressive forecast. Setting `n <= output_chunk_length` prevents
            auto-regression. This is useful when the covariates don't extend far enough into the future, or to prohibit
            the model from using future values of past and / or future covariates for prediction (depending on the
            model's covariate support).
        output_chunk_shift
            Optionally, the number of steps to shift the start of the output chunk into the future (relative to the
            input chunk end). This will create a gap between the input and output. If the model supports
            `future_covariates`, the future values are extracted from the shifted output chunk. Predictions will start
            `output_chunk_shift` steps after the end of the target `series`. If `output_chunk_shift` is set, the model
            cannot generate autoregressive predictions (`n > output_chunk_length`).
        hidden_size : int, optional
            The hidden size of the model, controlling the dimensionality of the internal representations.
            Default: 32.
        num_layers : int, optional
            The number of TimesBlock layers in the model. Each layer processes the input sequence
            using 2D convolutions and frequency domain analysis. Default: 2.
        num_kernels : int, optional
            The number of kernels in each Inception block within the TimesBlock. This controls
            the variety of convolution operations applied to the input. Default: 6.
        top_k : int, optional
            The number of top frequencies to consider in the FFT analysis. This parameter influences
            how many periodic components are extracted from the input sequence. Default: 5.
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
            for specifying a learning rate). Otherwise, the default values of the selected ``optimizer_cls``
            will be used. Default: ``None``.
        lr_scheduler_cls
            Optionally, the PyTorch learning rate scheduler class to be used. Specifying ``None`` corresponds
            to using a constant learning rate. Default: ``None``.
        lr_scheduler_kwargs
            Optionally, some keyword arguments for the PyTorch learning rate scheduler. Default: ``None``.
        use_reversible_instance_norm
            Whether to use reversible instance normalization `RINorm` against distribution shift as shown in [3]_.
            It is only applied to the features of the target series and not the covariates.
        batch_size
            Number of time series (input and output sequences) used in each training pass. Default: ``32``.
        n_epochs
            Number of epochs over which to train the model. Default: ``100``.
        model_name
            Name of the model. Used for creating checkpoints and saving torch.Tensorboard data. If not specified,
            defaults to the following string ``"YYYY-mm-dd_HH_MM_SS_torch_model_run_PID"``, where the initial part
            of the name is formatted with the local date and time, while PID is the processed ID (preventing models
            spawned at the same time by different processes to share the same model_name). E.g.,
            ``"2021-06-14_09_53_32_torch_model_run_44607"``.
        work_dir
            Path of the working directory, where to save checkpoints and torch.Tensorboard summaries.
            Default: current working directory.
        log_torch.Tensorboard
            If set, use torch.Tensorboard to log the different parameters. The logs will be located in:
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
            Control the randomness of the weight's initialization. Check this
            `link <https://scikit-learn.org/stable/glossary.html#term-random_state>`_ for more details.
            Default: ``None``.
        pl_trainer_kwargs
            By default :class:`TorchForecastingModel` creates a PyTorch Lightning Trainer with several useful presets
            that performs the training, validation and prediction processes. These presets include automatic
            checkpointing, torch.Tensorboard logging, setting the torch device and more.
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
        >>> from darts.models import TimesNetModel
        >>> series = WeatherDataset().load()
        >>> # predicting atmospheric pressure
        >>> target = series['p (mbar)'][:100]
        >>> # optionally, use past observed rainfall (pretending to be unknown beyond index 100)
        >>> past_cov = series['rain (mm)'][:100]
        >>> model = TimesNetModel(
        >>>     input_chunk_length=6,
        >>>     output_chunk_length=6,
        >>>     n_epochs=20
        >>> )
        >>> model.fit(target, past_covariates=past_cov)
        >>> pred = model.predict(6)
        >>> pred.values()
        array([[5.40498034],
               [5.36561899],
               [5.80616883],
               [6.48695488],
               [7.63158655],
               [5.65417736]])
        """
        super().__init__(**self._extract_torch_model_params(**self.model_params))

        # extract pytorch lightning module kwargs
        self.pl_module_params = self._extract_pl_module_params(**self.model_params)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_kernels = num_kernels
        self.top_k = top_k

    def _create_model(self, train_sample: Tuple[torch.Tensor]) -> torch.nn.Module:
        input_dim = train_sample[0].shape[1] + (
            train_sample[1].shape[1] if train_sample[1] is not None else 0
        )
        output_dim = train_sample[-1].shape[1]
        nr_params = 1 if self.likelihood is None else self.likelihood.num_parameters

        return _TimesNetModule(
            input_dim=input_dim,
            output_dim=output_dim,
            nr_params=nr_params,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            num_kernels=self.num_kernels,
            top_k=self.top_k,
            **self.pl_module_params,
        )

    @property
    def supports_multivariate(self) -> bool:
        return True
