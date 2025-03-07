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
    from pytorch_lightning.callbacks import Callback, EarlyStopping

    # hacky workaround found in https://github.com/Lightning-AI/pytorch-lightning/issues/17485
    # to avoid import of both lightning and pytorch_lightning
    class PatchedPruningCallback(
        optuna.integration.PyTorchLightningPruningCallback, Callback
    ):
        pass

    from darts.models import TCNModel
    from darts.utils.likelihood_models import GaussianLikelihood


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
            pruner = PatchedPruningCallback(trial, monitor="val_loss")
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
            smapes = smape(self.val, preds, n_jobs=-1)
            smape_val = np.mean(smapes)

            return smape_val if smape_val != np.nan else float("inf")

        # optimize hyperparameters by minimizing the sMAPE on the validation set
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=3)

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
            smapes = smape(self.val, preds, n_jobs=-1)
            smape_val = np.mean(smapes)

            return smape_val if smape_val != np.nan else float("inf")

        # optimize hyperparameters by minimizing the sMAPE on the validation set
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=3)
