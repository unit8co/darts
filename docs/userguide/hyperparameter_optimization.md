# Hyperparameter Optimization in Darts

There is nothing special in Darts when it comes to hyperparameter optimization.
The main thing to be aware of is probably the existence of PyTorch Lightning callbacks for early stopping and pruning of experiments with Darts' deep learning based TorchForecastingModels.
Below, we show examples of hyperparameter optimization done with [Optuna](https://optuna.org/) and
[Ray Tune](https://docs.ray.io/en/latest/tune/examples/tune-pytorch-lightning.html).

## Hyperparameter optimization with Optuna

[Optuna](https://optuna.org/) is a great option for hyperparameter optimization with Darts. Below, we show a minimal example
using PyTorch Lightning callbacks for pruning experiments.
For the sake of the example, we train a `TCNModel` on a single series, and optimize (probably overfitting) its hyperparameters by minimizing the prediction error on a validation set.
You can also have a look at [this notebook](https://github.com/unit8co/darts/blob/master/examples/17-hyperparameter-optimization.ipynb)
for a more complete example.

> **NOTE** (2023-19-02): Optuna's `PyTorchLightningPruningCallback` raises an error with pytorch-lightning>=1.8. Until this fixed, a workaround is proposed [here](https://github.com/optuna/optuna-examples/issues/166#issuecomment-1403112861).

```python
import numpy as np
import optuna
import torch
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.callbacks import Callback, EarlyStopping
from sklearn.preprocessing import MaxAbsScaler

from darts.dataprocessing.transformers import Scaler
from darts.datasets import AirPassengersDataset
from darts.metrics import smape
from darts.models import TCNModel
from darts.utils.likelihood_models.torch import GaussianLikelihood

# load data
series = AirPassengersDataset().load().astype(np.float32)

# split in train / validation (note: in practice we would also need a test set)
VAL_LEN = 36
train, val = series[:-VAL_LEN], series[-VAL_LEN:]

# scale
scaler = Scaler(MaxAbsScaler())
train = scaler.fit_transform(train)
val = scaler.transform(val)


#  workaround found in https://github.com/Lightning-AI/pytorch-lightning/issues/17485
# to avoid import of both lightning and pytorch_lightning
class PatchedPruningCallback(optuna.integration.PyTorchLightningPruningCallback, Callback):
    pass


# define objective function
def objective(trial):
    # select input and output chunk lengths
    in_len = trial.suggest_int("in_len", 12, 36)
    out_len = trial.suggest_int("out_len", 1, in_len - 1)

    # Other hyperparameters
    kernel_size = trial.suggest_int("kernel_size", 2, 5)
    num_filters = trial.suggest_int("num_filters", 1, 5)
    weight_norm = trial.suggest_categorical("weight_norm", [False, True])
    dilation_base = trial.suggest_int("dilation_base", 2, 4)
    dropout = trial.suggest_float("dropout", 0.0, 0.4)
    lr = trial.suggest_float("lr", 5e-5, 1e-3, log=True)
    include_year = trial.suggest_categorical("year", [False, True])

    # throughout training we'll monitor the validation loss for both pruning and early stopping
    pruner = PatchedPruningCallback(trial, monitor="val_loss")
    early_stopper = EarlyStopping("val_loss", min_delta=0.001, patience=3, verbose=True)
    callbacks = [pruner, early_stopper]

    # detect if a GPU is available
    if torch.cuda.is_available():
        num_workers = 4
    else:
        num_workers = 0

    pl_trainer_kwargs = {
        "accelerator": "auto",
        "callbacks": callbacks,
    }

    # optionally also add the (scaled) year value as a past covariate
    if include_year:
        encoders = {"datetime_attribute": {"past": ["year"]},
                    "transformer": Scaler()}
    else:
        encoders = None

    # reproducibility
    torch.manual_seed(42)

    # build the TCN model
    model = TCNModel(
        input_chunk_length=in_len,
        output_chunk_length=out_len,
        batch_size=32,
        n_epochs=100,
        nr_epochs_val_period=1,
        kernel_size=kernel_size,
        num_filters=num_filters,
        weight_norm=weight_norm,
        dilation_base=dilation_base,
        dropout=dropout,
        optimizer_kwargs={"lr": lr},
        add_encoders=encoders,
        likelihood=GaussianLikelihood(),
        pl_trainer_kwargs=pl_trainer_kwargs,
        model_name="tcn_model",
        force_reset=True,
        save_checkpoints=True,
    )

    # when validating during training, we can use a slightly longer validation
    # set which also contains the first input_chunk_length time steps
    model_val_set = scaler.transform(series[-(VAL_LEN + in_len):])

    # train the model
    model.fit(
        series=train,
        val_series=model_val_set,
    )

    # reload best model over course of training
    model = TCNModel.load_from_checkpoint("tcn_model")

    # Evaluate how good it is on the validation set, using sMAPE
    preds = model.predict(series=train, n=VAL_LEN)
    smapes = smape(val, preds, n_jobs=-1, verbose=True)
    smape_val = np.mean(smapes)

    return smape_val if smape_val != np.nan else float("inf")


# for convenience, print some optimization trials information
def print_callback(study, trial):
    print(f"Current value: {trial.value}, Current params: {trial.params}")
    print(f"Best value: {study.best_value}, Best params: {study.best_trial.params}")


# optimize hyperparameters by minimizing the sMAPE on the validation set
if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100, callbacks=[print_callback])
```

## Hyperparameter optimization with Ray Tune

[Ray Tune](https://docs.ray.io/en/latest/tune/examples/tune-pytorch-lightning.html) is another option for hyperparameter optimization with automatic pruning.

Here is an example of how to use Ray Tune to with the `NBEATSModel` model using the [Asynchronous Hyperband scheduler](https://blog.ml.cmu.edu/2018/12/12/massively-parallel-hyperparameter-optimization/). The example was tested with ray version `ray==2.32.0`.

```python
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from ray import tune
from ray.train import RunConfig
from ray.tune import CLIReporter
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
from ray.tune.schedulers import ASHAScheduler
from ray.tune.tuner import Tuner
from torchmetrics import (
    MeanAbsoluteError,
    MeanAbsolutePercentageError,
    MetricCollection,
)

from darts.dataprocessing.transformers import Scaler
from darts.datasets import AirPassengersDataset
from darts.models import NBEATSModel


def train_model(model_args, callbacks, train, val):
    torch_metrics = MetricCollection(
        [MeanAbsolutePercentageError(), MeanAbsoluteError()]
    )
    # Create the model using model_args from Ray Tune
    model = NBEATSModel(
        input_chunk_length=24,
        output_chunk_length=12,
        n_epochs=100,
        torch_metrics=torch_metrics,
        pl_trainer_kwargs={"callbacks": callbacks, "enable_progress_bar": False},
        **model_args,
    )

    model.fit(
        series=train,
        val_series=val,
    )


# Read data:
series = AirPassengersDataset().load().astype(np.float32)

# Create training and validation sets:
train, val = series.split_after(pd.Timestamp(year=1957, month=12, day=1))

# Normalize the time series (note: we avoid fitting the transformer on the validation set)
transformer = Scaler()
transformer.fit(train)
train = transformer.transform(train)
val = transformer.transform(val)

# Early stop callback
my_stopper = EarlyStopping(
    monitor="val_MeanAbsolutePercentageError",
    patience=5,
    min_delta=0.05,
    mode="min",
)


# set up ray tune callback
class TuneReportCallback(TuneReportCheckpointCallback, pl.Callback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


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
    train=train,
    val=val,
)

# Set the resources to be used for each trial (disable GPU, if you don't have one)
resources_per_trial = {"cpu": 8, "gpu": 1}

# define the hyperparameter space
config = {
    "batch_size": tune.choice([16, 32, 64, 128]),
    "num_blocks": tune.choice([1, 2, 3, 4, 5]),
    "num_stacks": tune.choice([32, 64, 128]),
    "dropout": tune.uniform(0, 0.2),
}

# the number of combinations to try
num_samples = 10

# Configure the ASHA scheduler
scheduler = ASHAScheduler(max_t=1000, grace_period=3, reduction_factor=2)

# Configure the CLI reporter to display the progress
reporter = CLIReporter(
    parameter_columns=list(config.keys()),
    metric_columns=["loss", "MAPE", "training_iteration"],
)

# Create the Tuner object and run the hyperparameter search
tuner = Tuner(
    trainable=tune.with_resources(
        train_fn_with_parameters, resources=resources_per_trial
    ),
    param_space=config,
    tune_config=tune.TuneConfig(
        metric="MAPE", mode="min", num_samples=num_samples, scheduler=scheduler
    ),
    run_config=RunConfig(name="tune_darts", progress_reporter=reporter),
)
results = tuner.fit()

# Print the best hyperparameters found
print("Best hyperparameters found were: ", results.get_best_result().config)
```

## Hyperparameter optimization using `gridsearch()`

Each forecasting models in Darts offer a `gridsearch()` method for basic hyperparameter search.
This method is limited to very simple cases, with very few hyperparameters, and working with a single time series only.
