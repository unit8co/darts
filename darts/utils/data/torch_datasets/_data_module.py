"""
Torch Data Module
-----------------
"""

from collections.abc import Callable
from typing import Any

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
        """
        super().__init__()

        dataloader_kwargs = dataloader_kwargs or {}

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.predict_dataset = predict_dataset
        self.batch_size = dataloader_kwargs.pop("batch_size", batch_size)
        self.shuffle = dataloader_kwargs.pop("shuffle", True)

        # setting drop_last to False makes the model see each sample at least once, and guarantee the presence of at
        # least one batch no matter the chosen batch size
        self.dataloader_kwargs: dict[str, Any] = dict(
            {
                "pin_memory": True,
                "drop_last": False,
                **({"collate_fn": collate_fn} if collate_fn is not None else {}),
            },
            **dataloader_kwargs,
        )

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
