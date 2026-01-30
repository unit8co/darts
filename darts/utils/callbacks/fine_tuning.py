from copy import deepcopy
from functools import partial
from typing import Any, Callable, Optional

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from torch import nn

from darts.logging import get_logger

logger = get_logger(__name__)


class ModelTransformCallback(Callback):
    def __init__(
        self,
        transform_fn: Callable[[nn.Module], nn.Module],
        model_attribute: str = "model",
        verbose: Optional[bool] = None,
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
            trainable parameters. If ``None``, it will be set to ``True`` if the trainer has a
            progress bar callback enabled (e.g. when ``model.fit(..., verbose=True)``).
            Default: ``None``.
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

            verbose = self.verbose
            if verbose is None:
                verbose = trainer.progress_bar_callback is not None

            if verbose:
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
        verbose: Optional[bool] = None,
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
            Whether to log the trainable parameter count after freezing. If ``None``, it will be
            set to ``True`` if the trainer has a progress bar callback enabled
            (e.g. when ``model.fit(..., verbose=True)``). Default: ``None``.
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
        verbose: Optional[bool] = None,
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
            Whether to log the trainable parameter count after applying PEFT. If ``None``, it will be
            set to ``True`` if the trainer has a progress bar callback enabled
            (e.g. when ``model.fit(..., verbose=True)``). Default: ``None``.
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
        super().on_save_checkpoint(trainer, pl_module, checkpoint)
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
