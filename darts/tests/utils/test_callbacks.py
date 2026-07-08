# Adapted from optuna-integration (MIT License, Copyright (c) 2018 Preferred Networks, Inc.)
# https://github.com/optuna/optuna-integration/blob/main/tests/pytorch_lightning/test_pytorch_lightning.py

from collections.abc import Sequence

import pytest

from darts.tests.conftest import OPTUNA_AVAILABLE, TORCH_AVAILABLE

if not OPTUNA_AVAILABLE:
    pytest.skip(
        f"Optuna not available. {__name__} tests will be skipped.",
        allow_module_level=True,
    )

if not TORCH_AVAILABLE:
    pytest.skip(
        f"Torch not available. {__name__} tests will be skipped.",
        allow_module_level=True,
    )

import optuna
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from optuna.testing.pruners import DeterministicPruner
from pytorch_lightning import LightningModule
from torch import nn

from darts.utils.callbacks import PyTorchLightningPruningCallback


class Model(LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self._model = nn.Sequential(nn.Linear(4, 8))
        self.validation_step_outputs: list[torch.Tensor] = []

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return self._model(data)

    def training_step(
        self, batch: Sequence[torch.Tensor], batch_nb: int
    ) -> dict[str, torch.Tensor]:
        data, target = batch
        output = self.forward(data)
        loss = F.nll_loss(output, target)
        return {"loss": loss}

    def validation_step(
        self, batch: Sequence[torch.Tensor], batch_nb: int
    ) -> torch.Tensor:
        data, target = batch
        output = self.forward(data)
        pred = output.argmax(dim=1, keepdim=True)
        accuracy = pred.eq(target.view_as(pred)).double().mean()
        self.validation_step_outputs.append(accuracy)
        return accuracy

    def on_validation_epoch_end(self) -> None:
        if not len(self.validation_step_outputs):
            return

        accuracy = sum(self.validation_step_outputs) / len(self.validation_step_outputs)
        self.log("accuracy", accuracy)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(self._model.parameters(), lr=1e-2)

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return self._generate_dummy_dataset()

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return self._generate_dummy_dataset()

    def _generate_dummy_dataset(self) -> torch.utils.data.DataLoader:
        data = torch.zeros(3, 4, dtype=torch.float32)
        target = torch.zeros(3, dtype=torch.int64)
        dataset = torch.utils.data.TensorDataset(data, target)
        return torch.utils.data.DataLoader(dataset, batch_size=1)


def test_pytorch_lightning_pruning_callback(tmpdir_fn) -> None:
    def objective(trial: optuna.trial.Trial) -> float:
        callback = PyTorchLightningPruningCallback(trial, monitor="accuracy")
        trainer = pl.Trainer(
            max_epochs=2,
            accelerator="cpu",
            enable_checkpointing=False,
            callbacks=[callback],
        )

        model = Model()
        trainer.fit(model)

        return 1.0

    study = optuna.create_study(pruner=DeterministicPruner(True))
    study.optimize(objective, n_trials=1)
    assert study.trials[0].state == optuna.trial.TrialState.PRUNED

    study = optuna.create_study(pruner=DeterministicPruner(False))
    study.optimize(objective, n_trials=1)
    assert study.trials[0].state == optuna.trial.TrialState.COMPLETE
    assert study.trials[0].value == 1.0


def test_pytorch_lightning_pruning_callback_monitor_is_invalid() -> None:
    study = optuna.create_study(pruner=DeterministicPruner(True))
    trial = study.ask()
    callback = PyTorchLightningPruningCallback(trial, "InvalidMonitor")

    trainer = pl.Trainer(
        max_epochs=1,
        enable_checkpointing=False,
        callbacks=[callback],
    )
    model = Model()

    with pytest.warns(UserWarning):
        callback.on_validation_end(trainer, model)
