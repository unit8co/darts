"""
Callbacks for TorchForecastingModel
-----------------------------------
"""

from __future__ import annotations

import sys
import warnings
from typing import TYPE_CHECKING

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback, TQDMProgressBar
from pytorch_lightning.callbacks.progress.tqdm_progress import Tqdm

if TYPE_CHECKING:
    import optuna


# Trial-system-attribute keys mirroring those used by
# ``optuna_integration.pytorch_lightning.PyTorchLightningPruningCallback`` so
# that DDP state remains portable between the two implementations.
_EPOCH_KEY = "ddp_pl:epoch"
_INTERMEDIATE_VALUE = "ddp_pl:intermediate_value"
_PRUNED_KEY = "ddp_pl:pruned"


class PyTorchLightningPruningCallback(Callback):
    """PyTorch Lightning callback to prune unpromising Optuna trials.

    Reports the latest value of ``monitor`` to the trial at the end of every
    validation epoch and raises :class:`optuna.TrialPruned` when the trial's
    pruner decides the trial is unpromising. See the Optuna `PyTorch
    Lightning example
    <https://github.com/optuna/optuna-examples/blob/main/pytorch/pytorch_lightning_simple.py>`__
    for end-to-end usage.

    This is a self-contained reimplementation that lets Darts depend only on
    ``optuna`` (without the ``optuna-integration[pytorch-lightning]`` extra)
    while continuing to support the same DDP behaviour.

    Parameters
    ----------
    trial
        An :class:`optuna.trial.Trial` corresponding to the current evaluation
        of the objective function.
    monitor
        Name of the metric to monitor for pruning, e.g. ``"val_loss"``. The
        metric must be logged by the ``LightningModule`` so that it appears
        in ``trainer.callback_metrics``.

    Notes
    -----
    For distributed-data-parallel (DDP) training, the
    :class:`~optuna.study.Study` must be instantiated with an
    :class:`~optuna.storages.RDBStorage` backend. After
    :meth:`pytorch_lightning.Trainer.fit` returns, callers must invoke
    :meth:`check_pruned` so that :class:`~optuna.exceptions.TrialPruned`
    is properly handled.

    Examples
    --------
    >>> import optuna
    >>> from darts.utils.callbacks import PyTorchLightningPruningCallback
    >>> def objective(trial):
    ...     pruner = PyTorchLightningPruningCallback(trial, monitor="val_loss")
    ...     # ... build and fit your TorchForecastingModel with `pruner` in
    ...     # `pl_trainer_kwargs={"callbacks": [pruner]}`
    """

    def __init__(self, trial: optuna.trial.Trial, monitor: str) -> None:
        # pragma: no cover - exercised only when optuna is missing
        try:
            import optuna  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "Optuna is required to use `PyTorchLightningPruningCallback`. "
                "Install it with: pip install optuna"
            ) from exc

        super().__init__()
        self._trial = trial
        self.monitor = monitor
        self.is_ddp_backend = False

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        import optuna
        from optuna.storages._cached_storage import _CachedStorage
        from optuna.storages._rdb.storage import RDBStorage

        self.is_ddp_backend = trainer._accelerator_connector.is_distributed
        if not self.is_ddp_backend:
            return

        # If it were not for this guard, fitting would be started even when an
        # unsupported storage is used. The ValueError would otherwise be
        # transformed into ``ProcessRaisedException`` inside torch.
        if not (
            isinstance(self._trial.study._storage, _CachedStorage)
            and isinstance(self._trial.study._storage._backend, RDBStorage)
        ):
            raise ValueError(
                "darts.utils.callbacks.PyTorchLightningPruningCallback supports"
                " only optuna.storages.RDBStorage in DDP."
            )
        # Intermediate values must be written directly to the backend storage
        # because they are not properly propagated to the main process through
        # the cached storage.
        if trainer.is_global_zero:
            self._trial.storage.set_trial_system_attr(
                self._trial._trial_id,
                _INTERMEDIATE_VALUE,
                dict(),
            )
        _ = optuna  # silence unused-import warnings if optuna is only re-imported lazily

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        import optuna

        # ``Trainer`` calls ``on_validation_end`` for the sanity check too.
        # Avoid calling ``trial.report`` multiple times at epoch 0; see
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/1391.
        if trainer.sanity_checking:
            return

        current_score = trainer.callback_metrics.get(self.monitor)
        if current_score is None:
            warnings.warn(
                f"The metric '{self.monitor}' is not in the evaluation logs for"
                " pruning. Please make sure you set the correct metric name."
            )
            return

        epoch = pl_module.current_epoch

        # Single-process: report and prune directly.
        if not self.is_ddp_backend:
            self._trial.report(current_score.item(), step=epoch)
            if not self._trial.should_prune():
                return
            raise optuna.TrialPruned(f"Trial was pruned at epoch {epoch}.")

        # DDP: only the global-zero process talks to the storage, then the
        # decision is broadcast to every other process.
        should_stop = False
        if trainer.is_global_zero:
            self._trial.report(current_score.item(), step=epoch)
            should_stop = self._trial.should_prune()

            _trial_id = self._trial._trial_id
            _study = self._trial.study
            _trial_system_attrs = _study._storage.get_trial_system_attrs(_trial_id)
            intermediate_values = _trial_system_attrs.get(_INTERMEDIATE_VALUE)
            intermediate_values[epoch] = current_score.item()  # type: ignore[index]
            self._trial.storage.set_trial_system_attr(
                self._trial._trial_id, _INTERMEDIATE_VALUE, intermediate_values
            )

        should_stop = trainer.strategy.broadcast(should_stop)
        trainer.should_stop = trainer.should_stop or should_stop
        if not should_stop:
            return

        if trainer.is_global_zero:
            self._trial.storage.set_trial_system_attr(
                self._trial._trial_id, _PRUNED_KEY, True
            )
            self._trial.storage.set_trial_system_attr(
                self._trial._trial_id, _EPOCH_KEY, epoch
            )

    def check_pruned(self) -> None:
        """Raise :class:`optuna.TrialPruned` manually if the trial was pruned.

        Currently, ``intermediate_values`` are not properly propagated between
        processes due to storage caching. Necessary information is kept in
        ``trial.system_attrs`` when the trial runs in a distributed
        situation. Call this method right after :meth:`Trainer.fit` returns.
        Outside of a DDP / RDB-storage context this method is a no-op.
        """
        import optuna
        from optuna.storages._cached_storage import _CachedStorage

        _trial_id = self._trial._trial_id
        _study = self._trial.study
        # If the storage is not the cached storage, we are not in a DDP run.
        if not isinstance(_study._storage, _CachedStorage):
            return

        _trial_system_attrs = _study._storage._backend.get_trial_system_attrs(_trial_id)
        is_pruned = _trial_system_attrs.get(_PRUNED_KEY)
        intermediate_values = _trial_system_attrs.get(_INTERMEDIATE_VALUE)

        # ``intermediate_values is None`` means we are not in a DDP run.
        if intermediate_values is None:
            return
        for epoch, score in intermediate_values.items():
            self._trial.report(score, step=int(epoch))
        if is_pruned:
            epoch = _trial_system_attrs.get(_EPOCH_KEY)
            raise optuna.TrialPruned(f"Trial was pruned at epoch {epoch}.")


class TFMProgressBar(TQDMProgressBar):
    def __init__(
        self,
        enable_sanity_check_bar: bool = True,
        enable_train_bar: bool = True,
        enable_validation_bar: bool = True,
        enable_prediction_bar: bool = True,
        enable_train_bar_only: bool = False,
        **kwargs,
    ):
        """Darts' Progress Bar for `TorchForecastingModels`.

        Allows to customize for which model stages (sanity checks, training, validation, prediction) to display a
        progress bar.

        This class is a PyTorch Lightning `Callback` and can be passed to the `TorchForecastingModel` constructor
        through the `pl_trainer_kwargs` parameter.

        Examples
        --------
        >>> from darts.models import NBEATSModel
        >>> from darts.utils.callbacks import TFMProgressBar
        >>> # only display the training bar and not the validation, prediction, and sanity check bars
        >>> prog_bar = TFMProgressBar(enable_train_bar_only=True)
        >>> model = NBEATSModel(1, 1, pl_trainer_kwargs={"callbacks": [prog_bar]})

        Parameters
        ----------
        enable_sanity_check_bar
            Whether to enable to progress bar for sanity checks.
        enable_train_bar
            Whether to enable to progress bar for training.
        enable_validation_bar
            Whether to enable to progress bar for validation.
        enable_prediction_bar
            Whether to enable to progress bar for prediction.
        enable_train_bar_only
            Whether to disable all progress bars except the bar for training.
        **kwargs
            Arguments passed to the PyTorch Lightning's `TQDMProgressBar
            <https://scikit-learn.org/stable/glossary.html#term-random_state>`__.
        """
        super().__init__(**kwargs)
        self.enable_sanity_check_bar = enable_sanity_check_bar
        self.enable_train_bar = enable_train_bar
        self.enable_validation_bar = enable_validation_bar
        self.enable_prediction_bar = enable_prediction_bar
        self.enable_train_bar_only = enable_train_bar_only

    def init_sanity_tqdm(self) -> Tqdm:
        """Override this to customize the tqdm bar for the validation sanity run."""
        return Tqdm(
            desc=self.sanity_check_description,
            position=(2 * self.process_position),
            disable=not self.enable_sanity_check_bar or self.enable_train_bar_only,
            leave=False,
            dynamic_ncols=True,
            file=sys.stdout,
        )

    def init_predict_tqdm(self) -> Tqdm:
        """Override this to customize the tqdm bar for predicting."""
        return Tqdm(
            desc=self.predict_description,
            position=(2 * self.process_position),
            disable=not self.enable_prediction_bar or self.enable_train_bar_only,
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
            smoothing=0,
        )

    def init_train_tqdm(self) -> Tqdm:
        """Override this to customize the tqdm bar for training."""
        return Tqdm(
            desc=self.train_description,
            position=(2 * self.process_position),
            disable=not self.enable_train_bar,
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
            smoothing=0,
        )

    def init_validation_tqdm(self) -> Tqdm:
        """Override this to customize the tqdm bar for validation."""
        # The train progress bar doesn't exist in `trainer.validate()`
        has_main_bar = self.trainer.state.fn != "validate"
        return Tqdm(
            desc=self.validation_description,
            position=(2 * self.process_position + has_main_bar),
            disable=not self.enable_validation_bar or self.enable_train_bar_only,
            leave=not has_main_bar,
            dynamic_ncols=True,
            file=sys.stdout,
        )
