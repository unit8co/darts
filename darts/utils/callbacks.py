"""
Callbacks for TorchForecastingModel
-----------------------------------
"""

import sys
import warnings

from pytorch_lightning.callbacks import Callback, TQDMProgressBar
from pytorch_lightning.callbacks.progress.tqdm_progress import Tqdm

from darts.logging import raise_log

# system attr keys used to coordinate DDP pruning across processes
_OPTUNA_EPOCH_KEY = "ddp_pl:epoch"
_OPTUNA_INTERMEDIATE_VALUE = "ddp_pl:intermediate_value"
_OPTUNA_PRUNED_KEY = "ddp_pl:pruned"


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


# Adapted from optuna-integration (MIT License, Copyright (c) 2018 Preferred Networks, Inc.)
# https://github.com/optuna/optuna-integration/blob/main/optuna_integration/pytorch_lightning/pytorch_lightning.py
class PyTorchLightningPruningCallback(Callback):
    """PyTorch Lightning callback to prune unpromising Optuna trials.

    Reports the monitored metric to the Optuna trial after each validation epoch
    and raises :class:`optuna.TrialPruned` when ``trial.should_prune()`` returns ``True``.

    For distributed (DDP) training, :class:`~optuna.study.Study` must use RDB storage, and
    :meth:`check_pruned` must be called manually after ``Trainer.fit()`` completes.

    Parameters
    ----------
    trial
        A :class:`~optuna.trial.Trial` corresponding to the current evaluation of the
        objective function.
    monitor
        An evaluation metric for pruning, e.g., ``val_loss`` or
        ``val_acc``. The metrics are obtained from the returned dictionaries from e.g.
        ``lightning.pytorch.LightningModule.training_step`` or
        ``lightning.pytorch.LightningModule.validation_epoch_end`` and the names thus depend on
        how this dictionary is formatted.

    Examples
    --------
    >>> import optuna
    >>> from darts.utils.callbacks import PyTorchLightningPruningCallback
    >>> def objective(trial):
    ...     pruner = PyTorchLightningPruningCallback(trial, monitor="val_loss")
    ...     model = TCNModel(..., pl_trainer_kwargs={"callbacks": [pruner]})
    ...     model.fit(...)
    """

    def __init__(self, trial, monitor: str) -> None:
        super().__init__()
        self._trial = trial
        self.monitor = monitor
        self.is_ddp_backend = False

    def on_fit_start(self, trainer, pl_module) -> None:
        self.is_ddp_backend = trainer._accelerator_connector.is_distributed
        if self.is_ddp_backend:
            from optuna.storages._cached_storage import _CachedStorage
            from optuna.storages._rdb.storage import RDBStorage

            # If it were not for this block, fitting is started even if unsupported storage
            # is used. Note that the ValueError is transformed into ProcessRaisedException inside
            # torch.
            if not (
                isinstance(self._trial.study._storage, _CachedStorage)
                and isinstance(self._trial.study._storage._backend, RDBStorage)
            ):
                raise_log(
                    ValueError(
                        "PyTorchLightningPruningCallback supports only "
                        "optuna.storages.RDBStorage in DDP."
                    ),
                )
            # It is necessary to store intermediate values directly in the backend storage because
            # they are not properly propagated to main process due to cached storage.
            if trainer.is_global_zero:
                self._trial.storage.set_trial_system_attr(
                    self._trial._trial_id,
                    _OPTUNA_INTERMEDIATE_VALUE,
                    dict(),
                )

    def on_validation_end(self, trainer, pl_module) -> None:
        import optuna

        # Trainer calls on_validation_end for sanity check — skip to avoid double-reporting
        # at epoch 0.
        if trainer.sanity_checking:
            return

        current_score = trainer.callback_metrics.get(self.monitor)
        if current_score is None:
            warnings.warn(
                f"The metric '{self.monitor}' is not in the evaluation logs for pruning. "
                "Please make sure you set the correct metric name."
            )
            return

        epoch = pl_module.current_epoch
        should_stop = False

        # Determine if the trial should be terminated in a single process.
        if not self.is_ddp_backend:
            self._trial.report(current_score.item(), step=epoch)
            if not self._trial.should_prune():
                return
            raise_log(optuna.TrialPruned(f"Trial was pruned at epoch {epoch}."))

        # Determine if the trial should be terminated in a DDP.
        if trainer.is_global_zero:
            self._trial.report(current_score.item(), step=epoch)
            should_stop = self._trial.should_prune()

            # Update intermediate value in the storage.
            _trial_id = self._trial._trial_id
            _study = self._trial.study
            _trial_system_attrs = _study._storage.get_trial_system_attrs(_trial_id)
            intermediate_values = _trial_system_attrs.get(_OPTUNA_INTERMEDIATE_VALUE)
            if intermediate_values is None:
                return

            intermediate_values[epoch] = current_score.item()
            self._trial.storage.set_trial_system_attr(
                self._trial._trial_id, _OPTUNA_INTERMEDIATE_VALUE, intermediate_values
            )

        # Terminate every process if any world process decides to stop.
        should_stop = trainer.strategy.broadcast(should_stop)
        trainer.should_stop = trainer.should_stop or should_stop
        if not should_stop:
            return

        if trainer.is_global_zero:
            # Update system_attr from global zero process.
            self._trial.storage.set_trial_system_attr(
                self._trial._trial_id, _OPTUNA_PRUNED_KEY, True
            )
            self._trial.storage.set_trial_system_attr(
                self._trial._trial_id, _OPTUNA_EPOCH_KEY, epoch
            )

    def check_pruned(self) -> None:
        """Raise :class:`optuna.TrialPruned` manually if pruned.

        Currently, ``intermediate_values`` are not properly propagated between processes due to
        storage cache. Therefore, necessary information is kept in ``trial.system_attrs`` when the
        trial runs in a distributed situation. Please call this method right after calling
        ``lightning.pytorch.Trainer.fit()``.
        If a callback doesn't have any backend storage for DDP, this method does nothing.
        """
        import optuna
        from optuna.storages._cached_storage import _CachedStorage

        _trial_id = self._trial._trial_id
        _study = self._trial.study
        # Confirm if storage is not InMemory in case this method is called in a non-distributed
        # situation by mistake.
        if not isinstance(_study._storage, _CachedStorage):
            return

        _trial_system_attrs = _study._storage._backend.get_trial_system_attrs(_trial_id)
        is_pruned = _trial_system_attrs.get(_OPTUNA_PRUNED_KEY)
        intermediate_values = _trial_system_attrs.get(_OPTUNA_INTERMEDIATE_VALUE)

        # Confirm if DDP backend is used in case this method is called from a non-DDP situation by
        # mistake.
        if intermediate_values is None:
            return
        for epoch, score in intermediate_values.items():
            self._trial.report(score, step=int(epoch))
        if is_pruned:
            epoch = _trial_system_attrs.get(_OPTUNA_EPOCH_KEY)
            raise_log(optuna.TrialPruned(f"Trial was pruned at epoch {epoch}."))
