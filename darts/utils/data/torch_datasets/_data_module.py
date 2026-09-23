"""
Torch Data Module
-----------------
"""

from collections.abc import Callable
from typing import Any

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from darts.utils.data import TorchInferenceDataset, TorchTrainingDataset
from darts.utils.data.torch_datasets.dataset import TorchDataset


class TorchDataModule(LightningDataModule):
    def __init__(
        self,
        train_dataset: TorchTrainingDataset | None = None,
        val_dataset: TorchTrainingDataset | None = None,
        predict_dataset: TorchInferenceDataset | None = None,
        batch_size: int = 32,
        collate_fn: Callable | None = None,
        dataloader_kwargs: dict[str, Any] | None = None,
        shuffle_seed: int | None = None,
    ):
        """LightningDataModule to handle train, val and predict dataloaders for ``TorchForecastingModel``.

        Parameters
        ----------
        train_dataset
            Dataset for training.
        val_dataset
            Dataset for validation.
        predict_dataset
            Dataset for prediction/inference.
        batch_size
            Number of time series (input and output sequences) used in each training/prediction pass.
        collate_fn
            Function to collate samples into a batch.
        dataloader_kwargs
            Additional keyword arguments for DataLoader.
        shuffle_seed
            Optional seed used to initialize the training DataLoader ``generator`` when ``shuffle=True``.
            Combined with checkpointing of the generator state, this makes shuffled training resumable.
        """
        super().__init__()

        dataloader_kwargs = dict(dataloader_kwargs or {})

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.predict_dataset = predict_dataset
        self.batch_size = dataloader_kwargs.pop("batch_size", batch_size)
        self.shuffle = dataloader_kwargs.pop("shuffle", True)

        self.generator: torch.Generator = dataloader_kwargs.pop(
            "generator", torch.Generator()
        )
        if self.shuffle and shuffle_seed is not None:
            self.generator.manual_seed(shuffle_seed)

        # self.generator: torch.Generator | None = dataloader_kwargs.pop("generator")
        # if (
        #     self.train_dataset is not None
        #     and self.shuffle
        #     and shuffle_seed is not None
        #     and self.generator is None
        # ):
        #     train_generator = torch.Generator()
        #     train_generator.manual_seed(shuffle_seed)
        #     self.generator = train_generator

        # setting drop_last to False makes the model see each sample at least once, and guarantee the presence of at
        # least one batch no matter the chosen batch size
        self.dataloader_kwargs: dict[str, Any] = dict(
            {
                "pin_memory": True,
                "drop_last": False,
                "generator": self.generator,
                **({"collate_fn": collate_fn} if collate_fn is not None else {}),
            },
            **dataloader_kwargs,
        )

    def state_dict(self) -> dict[str, Any]:
        """Return datamodule state for PyTorch Lightning checkpointing."""
        return {"generator_state": self.generator.get_state()}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore datamodule state from a PyTorch Lightning checkpoint."""
        state = state_dict.get("generator_state")
        if self.generator is not None and state is not None:
            self.generator.set_state(state)

    def train_dataloader(self) -> list | DataLoader:
        """Train dataloader."""
        return self._create_dataloader(self.train_dataset, shuffle=self.shuffle)

    def val_dataloader(self) -> list | DataLoader:
        """Validation dataloader."""
        return self._create_dataloader(self.val_dataset, shuffle=False)

    def predict_dataloader(self) -> list | DataLoader:
        """Predict/inference dataloader."""
        return self._create_dataloader(self.predict_dataset, shuffle=False)

    def _create_dataloader(
        self, dataset: TorchDataset | None, **kwargs
    ) -> list | DataLoader:
        """Create a dataloader."""
        if dataset is None:
            return []
        return DataLoader(
            dataset, batch_size=self.batch_size, **self.dataloader_kwargs, **kwargs
        )
