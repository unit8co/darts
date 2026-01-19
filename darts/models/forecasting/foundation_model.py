"""
Time Series Foundation Model (TSFM)
-----------------------------------

This file contains several abstract classes:

    * FoundationModel: base class for foundation forecasting models with PyTorch Lightning backend,
        inheriting from :class:`MixedCovariatesTorchModel` and :class:`TorchForecastingModel`.
    * HuggingFaceModelMixin: mixin class for loading model configuration and weights from HuggingFace Hub.
"""

from abc import ABC
from copy import deepcopy
from functools import partial
from typing import Any, Callable

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from torch import nn

from darts.logging import get_logger, raise_log
from darts.models.forecasting.pl_forecasting_module import PLForecastingModule
from darts.models.forecasting.torch_forecasting_model import MixedCovariatesTorchModel

logger = get_logger(__name__)


class FoundationModel(MixedCovariatesTorchModel, ABC):
    _allows_finetuning: bool = False

    def __init__(
        self,
        enable_finetuning: bool = False,
        **kwargs,
    ):
        """Foundation Forecasting Model with PyTorch Lightning backend.

        This class is meant to be inherited to create a new foundation forecasting model.
        It governs the interactions between:
        - Darts forecasting models (module) :class:`PLTorchForecastingModel`
        - Darts integrated PL Lightning Trainer :class:`pytorch_lightning.Trainer` or custom PL Trainers
        - Dataset loaders :class:`TorchTrainingDataset` and :class:`TorchInferenceDataset` or custom Dataset
          Loaders.

        This class itself inherits from :class:`MixedCovariatesTorchModel`, which in turn inherits from
        :class:`TorchForecastingModel`. That allows :class:`FoundationModel` to use functionalities from both,
        such as optimized historical forecasting, model training (fine-tuning), checkpointing, and more.

        When subclassing this class, please make sure to perform necessary parameter validation and then call
        super().__init__(**kwargs). Also, please implement the abstract method :func:`_create_model()`.

        If the model requires downloading configuration files and model weights from HuggingFace, please
        also inherit from :class:`HuggingFaceModelMixin` and use its methods to load the model configuration
        inside :func:`__init__()` and to load the model weights inside :func:`_create_model()`.

        Parameters
        ----------
        enable_finetuning
            Whether to enable fine-tuning of the foundation model. If set to ``True``, calling :func:`fit()` will
            update the model weights. Default: ``False``.
        batch_size
            Number of time series (input and output sequences) used in each fine-tuning pass. Default: ``32``.
        n_epochs
            Number of epochs over which to fine-tune the model. Default: ``100``.
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
            - ``{"accelerator": "gpu", "devices": -1, "auto_select_gpus": True}`` to use all available GPUS.

            For more info, see here:
            `trainer flags
            <https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#trainer-flags>`__,
            and `training on multiple gpus
            <https://pytorch-lightning.readthedocs.io/en/stable/accelerators/gpu_basic.html#train-on-multiple-gpus>`__.

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
        """
        # initialize `TorchForecastingModel` base class
        super().__init__(**self._extract_torch_model_params(**self.model_params))

        # extract pytorch lightning module kwargs
        self.pl_module_params = self._extract_pl_module_params(**self.model_params)

        # validate and set fine-tuning flag
        if enable_finetuning and not self._allows_finetuning:
            raise_log(
                ValueError(
                    f"Fine-tuning is not supported for {self.__class__.__name__}."
                    " Please set `enable_finetuning=False`."
                ),
                logger,
            )

        self._enable_finetuning = enable_finetuning

    @property
    def _requires_training(self) -> bool:
        return self._enable_finetuning

    @property
    def internal_model(self) -> Any:
        """
        Returns the underlying PyTorch model (nn.Module).
        This gives access to the actual internal mechanics of the model, which can be useful
        for advanced usage like accessing PEFT adapters, inspecting weights or custom saving/loading.

        If the model has not been initialized yet, returns None.
        """
        if hasattr(self, "model") and hasattr(self.model, "model"):
            return self.model.model
        return None

    @internal_model.setter
    def internal_model(self, model: nn.Module):
        """
        Sets the underlying PyTorch model (nn.Module).
        This allows replacing the internal model, which can be useful for advanced usage like loading PEFT adapters.

        Parameters
        ----------
        model
            The new PyTorch nn.Module to set as the internal model.
        """
        if hasattr(self, "model"):
            self.model.model = model
        else:
            raise_log(
                AttributeError(
                    "The internal model cannot be set because the outer model is not initialized yet."
                ),
                logger,
            )


class FoundationPLModule(PLForecastingModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model: nn.Module


class ModelTransformCallback(Callback):
    def __init__(
        self,
        transform_fn: Callable[[nn.Module], nn.Module],
        model_attribute: str = "model",
        verbose: bool = False,
    ):
        """
        A PyTorch Lightning callback that applies a transformation function to an internal model
        within a LightningModule.

        This is useful for modifying model architectures (e.g., applying PEFT or freezing layers)
        just before the training starts, while ensuring the transformation is correctly handled
        during checkpoint saving and loading.

        Parameters
        ----------
        transform_fn
            A function that takes an ``nn.Module`` and returns a transformed ``nn.Module``.
        model_attribute
            The attribute name of the model within the LightningModule. Default: ``"model"``.
        verbose
            Whether to log information about the model transformation, such as the number of
            trainable parameters. Default: ``False``.
        """
        super().__init__()
        self.transform_fn = transform_fn
        self.model_attribute = model_attribute
        self.verbose = verbose
        self._transformed = False

    def _get_inner_model(self, pl_module: pl.LightningModule) -> nn.Module:
        """Get the inner model from the Lightning module."""
        return getattr(pl_module, self.model_attribute)

    def _set_inner_model(self, pl_module: pl.LightningModule, model: nn.Module):
        """Set the inner model on the Lightning module."""
        setattr(pl_module, self.model_attribute, model)

    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str):
        """Apply transformation before training begins (before optimizer setup)."""
        if not self._transformed:
            inner_model = self._get_inner_model(pl_module)
            transformed_model = self.transform_fn(inner_model)
            self._set_inner_model(pl_module, transformed_model)
            self._transformed = True
            if self.verbose:
                # Log trainable parameters
                trainable = sum(
                    p.numel() for p in pl_module.parameters() if p.requires_grad
                )
                total = sum(p.numel() for p in pl_module.parameters())
                logger.info(
                    f"Model transformed. Trainable: {trainable:,}/{total:,} ({100 * trainable / total:.2f}%)"
                )

    def on_save_checkpoint(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        checkpoint: dict[str, Any],
    ):
        """
        Handle checkpoint saving for transformed models.

        For PEFT models, we could optionally save just the adapter weights
        or mark the checkpoint as requiring transformation on load.
        """
        # Mark that this checkpoint was saved with a transformed model
        checkpoint["model_transform_applied"] = True

    def on_load_checkpoint(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        checkpoint: dict[str, Any],
    ):
        """
        Apply transformation before loading checkpoint weights.

        This ensures the model structure matches the saved weights.
        """
        if checkpoint.get("model_transform_applied", False) and not self._transformed:
            inner_model = self._get_inner_model(pl_module)
            transformed_model = self.transform_fn(inner_model)
            self._set_inner_model(pl_module, transformed_model)
            self._transformed = True


class LayerFreezeCallback(ModelTransformCallback):
    @classmethod
    def _freeze_layers(
        cls, model: nn.Module, freeze_patterns: list[str], unfreeze_patterns: list[str]
    ) -> nn.Module:
        for name, param in model.named_parameters():
            if any(name.startswith(layer) for layer in freeze_patterns):
                param.requires_grad = False
            if any(name.startswith(layer) for layer in unfreeze_patterns):
                param.requires_grad = True
        return model

    def __init__(
        self,
        freeze_patterns: list[str],
        unfreeze_patterns: list[str] = None,
        model_attribute: str = "model",
        verbose: bool = False,
    ):
        """
        A callback to freeze or unfreeze specific layers of a model based on name patterns.

        Parameters
        ----------
        freeze_patterns
            A list of strings. Parameters whose names start with any of these patterns will be frozen
            (``requires_grad=False``).
        unfreeze_patterns
            A list of strings. Parameters whose names start with any of these patterns will be unfrozen
            (``requires_grad=True``). This is applied after ``freeze_patterns``. Default: ``None``.
        model_attribute
            The attribute name of the model within the LightningModule. Default: ``"model"``.
        verbose
            Whether to log the trainable parameter count after freezing. Default: ``False``.
        """
        unfreeze_patterns = unfreeze_patterns or []

        super().__init__(
            transform_fn=partial(
                self._freeze_layers,
                freeze_patterns=freeze_patterns,
                unfreeze_patterns=unfreeze_patterns,
            ),
            model_attribute=model_attribute,
            verbose=verbose,
        )


class PeftCallback(ModelTransformCallback):
    @classmethod
    def _apply_peft(cls, model: nn.Module, peft_config) -> nn.Module:
        try:
            from peft import get_peft_model
        except ImportError:
            raise ImportError(
                "Please install the `peft` package to use PeftCallback: `pip install peft`."
            )
        peft_model = get_peft_model(model, peft_config)
        return peft_model

    def __init__(
        self,
        peft_config=None,
        model_attribute: str = "model",
        verbose: bool = False,
    ):
        """
        A callback to apply Parameter-Efficient Fine-Tuning (PEFT) to a model using the ``peft`` library.

        It wraps the internal model with a PEFT adapter (e.g., LoRA) and manages the merging of
        weights during checkpointing so that the saved state can be loaded as a standard model.

        Parameters
        ----------
        peft_config
            A PEFT configuration object (e.g., ``LoraConfig``) from the ``peft`` library.
        model_attribute
            The attribute name of the model within the LightningModule. Default: ``"model"``.
        verbose
            Whether to log the trainable parameter count after applying PEFT. Default: ``False``.
        """
        super().__init__(
            transform_fn=partial(self._apply_peft, peft_config=peft_config),
            model_attribute=model_attribute,
            verbose=verbose,
        )
        self.peft_config = peft_config

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        # We replace the state_dict in the checkpoint with the one from the base model
        # (with adapters merged), so that the model can be loaded as a regular model.
        peft_model = getattr(pl_module, self.model_attribute, None)
        try:
            from peft import PeftModel
        except ImportError:
            return

        if isinstance(peft_model, PeftModel):
            # Merge adapters into the base model weights
            # TODO: This might be inefficient for large models, think about a better way
            model_copy = deepcopy(peft_model)
            setattr(pl_module, self.model_attribute, peft_model.merge_and_unload())
            try:
                # Get the state dict of the base model
                # This returns the weights including the merged adapters
                # base_state_dict = peft_model.get_base_model().state_dict()

                # We need to prepend the model attribute name to the keys
                # because the PL module expects keys to start with `model.` (or `model_attribute.`)
                prefix = self.model_attribute + "."
                new_state_dict = {
                    prefix + k: v
                    for k, v in getattr(pl_module, self.model_attribute)
                    .state_dict()
                    .items()
                }

                # Update the checkpoint
                checkpoint["state_dict"] = new_state_dict

            finally:
                # Unmerge adapters to keep the current model in PEFT mode
                setattr(pl_module, self.model_attribute, model_copy)
