import numpy as np
import pytest
from sklearn.preprocessing import MaxAbsScaler

from darts.dataprocessing.transformers import Scaler
from darts.datasets import AirPassengersDataset
from darts.metrics import smape
from darts.models import LinearRegressionModel
from darts.tests.conftest import RAY_AVAILABLE, TORCH_AVAILABLE, tfm_kwargs

if not RAY_AVAILABLE:
    pytest.skip(
        f"Ray not available. {__name__} tests will be skipped.",
        allow_module_level=True,
    )

from ray import tune
from ray.tune.tuner import Tuner

if TORCH_AVAILABLE:
    from pytorch_lightning.callbacks import Callback, EarlyStopping
    from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
    from torchmetrics import (
        MeanAbsoluteError,
        MeanAbsolutePercentageError,
        MetricCollection,
    )

    from darts.models import NBEATSModel


class TestRay:
    series = AirPassengersDataset().load().astype(np.float32)

    val_length = 36
    train, val = series.split_after(val_length)

    # scale
    scaler = Scaler(MaxAbsScaler())
    train = scaler.fit_transform(train)
    val = scaler.transform(val)

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="requires torch")
    def test_ray_torch_model(self, tmpdir_fn):
        """Check that ray works as expected with a torch-based model"""

        def train_model(model_args, callbacks, train, val):
            torch_metrics = MetricCollection([
                MeanAbsolutePercentageError(),
                MeanAbsoluteError(),
            ])

            # Create the model using model_args from Ray Tune
            model = NBEATSModel(
                input_chunk_length=4,
                output_chunk_length=3,
                n_epochs=2,
                torch_metrics=torch_metrics,
                pl_trainer_kwargs={
                    **tfm_kwargs["pl_trainer_kwargs"],
                    "callbacks": callbacks,
                },
                **model_args,
            )

            model.fit(
                series=train,
                val_series=val,
            )

        # Early stop callback
        my_stopper = EarlyStopping(
            monitor="val_MeanAbsolutePercentageError",
            patience=5,
            min_delta=0.05,
            mode="min",
        )

        # set up ray tune callback
        class TuneReportCallback(TuneReportCheckpointCallback, Callback):
            pass

        tune_callback = TuneReportCallback(
            {
                "loss": "val_loss",
                "MAPE": "val_MeanAbsolutePercentageError",
            },
            on="validation_end",
        )

        # Define the trainable function that will be tuned by Ray Tune
        train_fn_with_parameters = tune.with_parameters(
            train_model,
            callbacks=[tune_callback, my_stopper],
            train=self.train,
            val=self.val,
        )

        # define the hyperparameter space
        param_space = {
            "batch_size": tune.choice([8, 16]),
            "num_blocks": tune.choice([1, 2]),
            "num_stacks": tune.choice([2, 4]),
        }

        # the number of combinations to try
        num_samples = 2

        # Create the Tuner object and run the hyperparameter search
        tuner = Tuner(
            trainable=train_fn_with_parameters,
            param_space=param_space,
            tune_config=tune.TuneConfig(
                metric="MAPE", mode="min", num_samples=num_samples
            ),
            run_config=tune.RunConfig(name="tune_darts"),
        )
        tuner.fit()

    def test_ray_regression_model(self, tmpdir_fn):
        """Check that ray works as expected with a regression model"""

        # define objective function
        def objective(config):
            # optionally also add the (scaled) year value as a past covariate
            if config["include_year"]:
                encoders = {
                    "datetime_attribute": {"past": ["year"]},
                    "transformer": Scaler(),
                }
                past_lags = 1
            else:
                encoders = None
                past_lags = None

            # build the model
            model = LinearRegressionModel(
                lags=config["lags"],
                lags_past_covariates=past_lags,
                output_chunk_length=1,
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

        search_space = {
            "lags": tune.choice([1, 3, 6, 12]),
            "include_years": tune.choice([True, False]),
        }

        tuner = tune.Tuner(objective, param_space=search_space)
        tuner.fit()
