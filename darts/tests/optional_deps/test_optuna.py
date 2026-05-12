import os
from itertools import product

import numpy as np
import pytest
from sklearn.preprocessing import MaxAbsScaler

from darts.dataprocessing.transformers import Scaler
from darts.datasets import AirPassengersDataset
from darts.metrics import smape
from darts.models import LinearRegressionModel
from darts.tests.conftest import OPTUNA_AVAILABLE, TORCH_AVAILABLE, tfm_kwargs

if not OPTUNA_AVAILABLE:
    pytest.skip(
        f"Optuna not available. {__name__} tests will be skipped.",
        allow_module_level=True,
    )

import optuna

if TORCH_AVAILABLE:
    import torch
    from pytorch_lightning.callbacks import EarlyStopping

    from darts.models import TCNModel
    from darts.utils.callbacks import PyTorchLightningPruningCallback
    from darts.utils.likelihood_models.torch import GaussianLikelihood


class TestOptuna:
    series = AirPassengersDataset().load().astype(np.float32)

    val_length = 36
    train, val = series.split_after(val_length)

    # scale
    scaler = Scaler(MaxAbsScaler())
    train = scaler.fit_transform(train)
    val = scaler.transform(val)

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="requires torch")
    def test_optuna_torch_model(self, tmpdir_fn):
        """Check that optuna works as expected with a torch-based model"""

        # define objective function
        def objective(trial):
            # select input and output chunk lengths
            in_len = trial.suggest_int("in_len", 4, 8)
            out_len = trial.suggest_int("out_len", 1, 3)

            # Other hyperparameters
            kernel_size = trial.suggest_int("kernel_size", 2, 3)
            num_filters = trial.suggest_int("num_filters", 1, 2)
            lr = trial.suggest_float("lr", 5e-5, 1e-3, log=True)
            include_year = trial.suggest_categorical("year", [False, True])

            # throughout training we'll monitor the validation loss for both pruning and early stopping
            pruner = PyTorchLightningPruningCallback(trial, monitor="val_loss")
            early_stopper = EarlyStopping(
                "val_loss", min_delta=0.001, patience=3, verbose=True
            )

            # optionally also add the (scaled) year value as a past covariate
            if include_year:
                encoders = {
                    "datetime_attribute": {"past": ["year"]},
                    "transformer": Scaler(),
                }
            else:
                encoders = None

            # reproducibility
            torch.manual_seed(42)

            # build the TCN model
            model = TCNModel(
                input_chunk_length=in_len,
                output_chunk_length=out_len,
                batch_size=8,
                n_epochs=2,
                nr_epochs_val_period=1,
                kernel_size=kernel_size,
                num_filters=num_filters,
                optimizer_kwargs={"lr": lr},
                add_encoders=encoders,
                likelihood=GaussianLikelihood(),
                pl_trainer_kwargs={
                    **tfm_kwargs["pl_trainer_kwargs"],
                    "callbacks": [pruner, early_stopper],
                },
                model_name="tcn_model",
                force_reset=True,
                save_checkpoints=True,
                work_dir=os.getcwd(),
            )

            # when validating during training, we can use a slightly longer validation
            # set which also contains the first input_chunk_length time steps
            model_val_set = self.scaler.transform(
                self.series[-(self.val_length + in_len) :]
            )

            # train the model
            model.fit(
                series=self.train,
                val_series=model_val_set,
            )

            # reload best model over course of training
            model = TCNModel.load_from_checkpoint(
                model_name="tcn_model", work_dir=os.getcwd()
            )

            # Evaluate how good it is on the validation set, using sMAPE
            preds = model.predict(series=self.train, n=self.val_length)
            smapes = smape(self.val, preds)
            smape_val = np.mean(smapes)

            return smape_val if smape_val != np.nan else float("inf")

        # optimize hyperparameters by minimizing the sMAPE on the validation set
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=3)

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="requires torch")
    def test_pruning_callback_inherits_from_pl_callback(self):
        """The Darts callback must satisfy PyTorch Lightning's `Callback`
        type so that it can be registered via ``pl_trainer_kwargs`` without
        the multiple-inheritance workaround used historically."""
        from pytorch_lightning.callbacks import Callback as PLCallback

        study = optuna.create_study(direction="minimize")
        trial = study.ask()
        callback = PyTorchLightningPruningCallback(trial, monitor="val_loss")
        assert isinstance(callback, PLCallback)
        assert callback.monitor == "val_loss"
        assert callback.is_ddp_backend is False

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="requires torch")
    def test_pruning_callback_warns_on_missing_metric(self):
        """When the monitored metric is absent from ``callback_metrics``,
        the callback must warn (not crash) and skip reporting."""

        class _StubTrainer:
            sanity_checking = False
            callback_metrics: dict = {}

        class _StubModule:
            current_epoch = 0

        study = optuna.create_study(direction="minimize")
        trial = study.ask()
        callback = PyTorchLightningPruningCallback(trial, monitor="missing_metric")

        with pytest.warns(UserWarning, match="missing_metric"):
            callback.on_validation_end(_StubTrainer(), _StubModule())  # type: ignore[arg-type]

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="requires torch")
    def test_pruning_callback_raises_when_trial_should_prune(self):
        """When the trial's pruner decides to prune, the callback must
        raise :class:`optuna.TrialPruned`."""

        class _AlwaysPruningTrial:
            def __init__(self, real_trial):
                self._real = real_trial

            def report(self, value, step):
                self._real.report(value, step)

            def should_prune(self):
                return True

        class _StubTrainer:
            sanity_checking = False
            callback_metrics = {"val_loss": torch.tensor(1.0)}

        class _StubModule:
            current_epoch = 0

        study = optuna.create_study(direction="minimize")
        real_trial = study.ask()
        callback = PyTorchLightningPruningCallback(
            _AlwaysPruningTrial(real_trial), monitor="val_loss"
        )

        with pytest.raises(optuna.TrialPruned):
            callback.on_validation_end(_StubTrainer(), _StubModule())  # type: ignore[arg-type]

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="requires torch")
    def test_pruning_callback_skips_during_sanity_check(self):
        """``Trainer`` invokes ``on_validation_end`` for the sanity check at
        epoch 0. The callback must short-circuit in that case so it does not
        call ``trial.report`` twice."""

        class _ReportRecorder:
            def __init__(self):
                self.reports: list = []

            def report(self, value, step):
                self.reports.append((value, step))

            def should_prune(self):
                return False

        class _StubTrainer:
            sanity_checking = True
            callback_metrics = {"val_loss": torch.tensor(1.0)}

        class _StubModule:
            current_epoch = 0

        recorder = _ReportRecorder()
        callback = PyTorchLightningPruningCallback(recorder, monitor="val_loss")
        callback.on_validation_end(_StubTrainer(), _StubModule())  # type: ignore[arg-type]
        assert recorder.reports == []

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="requires torch")
    def test_pruning_callback_check_pruned_is_noop_outside_ddp(self):
        """Outside of a DDP + RDB-storage context, ``check_pruned`` must be
        a no-op so callers can invoke it unconditionally after fit()."""
        study = optuna.create_study(direction="minimize")
        trial = study.ask()
        callback = PyTorchLightningPruningCallback(trial, monitor="val_loss")
        # No exception expected; the method returns silently when the
        # study storage is not the DDP-aware CachedStorage.
        callback.check_pruned()

    @pytest.mark.parametrize(
        "params",
        product(
            [True, False],  # multi_models
            [1, 3],  # ocl
        ),
    )
    def test_optuna_regression_model(self, params):
        """Check that optuna works as expected with a regression model"""

        multi_models, ocl = params

        # define objective function
        def objective(trial):
            # select input and encoder usage
            target_lags = trial.suggest_int("lags", 1, 12)
            include_year = trial.suggest_categorical("year", [False, True])

            # optionally also add the (scaled) year value as a past covariate
            if include_year:
                encoders = {
                    "datetime_attribute": {"past": ["year"]},
                    "transformer": Scaler(),
                }
                past_lags = trial.suggest_int("lags_past_covariates", 1, 12)
            else:
                encoders = None
                past_lags = None

            # build the model
            model = LinearRegressionModel(
                lags=target_lags,
                lags_past_covariates=past_lags,
                output_chunk_length=ocl,
                multi_models=multi_models,
                add_encoders=encoders,
            )
            model.fit(
                series=self.train,
            )

            # Evaluate how good it is on the validation set, using sMAPE
            preds = model.predict(series=self.train, n=self.val_length)
            smapes = smape(self.val, preds)
            smape_val = np.mean(smapes)

            return smape_val if smape_val != np.nan else float("inf")

        # optimize hyperparameters by minimizing the sMAPE on the validation set
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=3)
