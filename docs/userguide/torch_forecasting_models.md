# Torch Forecasting Models
This document was written for darts version 0.15.0 and later.

We assume that you already know about covariates in Darts. If you're new to the topic we recommend you to read our [guide on covariates](https://unit8co.github.io/darts/userguide/covariates.html) first.

## Content of this section

[The first subsection](#introduction) covers the most important points about Torch Forecasting Models (TFMs):
- How to use TFMs
- Top-level look at chunks
- TFM covariates support
- Time span requirements for target and covariate series

[The second subsection](#in-depth-look-at-how-input-data-is-used-when-training-and-predicting-with-tfms) gives
an in-depth guide of how input data is used when training and predicting with TFMs.

## Introduction
In Darts, **Torch Forecasting Models (TFMs)** are broadly speaking "machine learning based" models, which denote PyTorch-based (deep learning) models.

TFMs train and predict on fixed-length chunks (sub-samples) of your input `target` and `*_covariates` series (if supported). `Target` is the series for which we want to predict the future, `*_covariates` are the past and / or future covariates.

Each chunk contains an input chunk - representing the sample's past - and an output chunk - the sample's future. The sample's prediction point lies at the end of the input chunk. The length of these chunks has to be specified at model creation with parameters `input_chunk_length` and `output_chunk_length` (more on chunks in [the next subsection](#top-level-look-at-training-and-predicting-with-chunks)).

```
# model that looks 7 time steps back (past) and 1 time step ahead (future)
model = SomeTorchForecastingModel(input_chunk_length=7,
                                  output_chunk_length=1,
                                  **model_kwargs)
```

All TFMs can be trained on single or multiple `target` series and, depending on their covariate support (covered in [this subsection](#torch-forecasting-model-covariates-support)), `past_covariates` and / or `future_covariates`. When using covariates you have to supply one dedicated past and / or future covariates series for each target series.

Optionally, you can use a validation set with dedicated covariates during training. If the covariates have the required time spans, you can use the same for training, validation and prediction. (covered in [this subsection](#required-target-time-spans-for-training-validation-and-prediction))

```python
# fit the model on a single target series with optional past and / or future covariates
model.fit(target,
          past_covariates=past_covariates,
          future_covariates=future_covariates,
          val_series=target_val,  # optionally, use a validation set
          val_past_covariates=past_covariates_val,
          val_future_covariates=future_covariates_val)

# fit the model on multiple target series
model.fit([target, target2, ...],
          past_covariates=[past_covariates, past_covariates2, ...],
          ...
          )
```

You can produce forecasts for any input `target` TimeSeries or for several targets given as a sequence of TimeSeries. This will also work on series that have not been seen during training, as long as each series contains at least `input_chunk_length` time steps.

```python
# predict the next n=3 time steps for any input series with `series`
prediction = model.predict(n=3,
                           series=target,
                           past_covariates=past_covariates,
                           future_covariates=future_covariates)
```

If you want to know more about the training and prediction process of our Torch Forecasting Models and how they work with covariates, read on.

## Top level look at training and predicting with chunks

In Figure 1 you can see how your data is distributed to the input and output chunks for each sample when calling `fit()` or `predict()`. For this example we look at data with daily frequency. The input chunk extracts values from `target` and optionally from `past_covariates` and / or `future_covariates` that fall into the input chunk time span. These "past" values of `future_covariates` are called "historic future covariates".

The output chunk only takes optional `future_covariates` values that fall into the output chunk time span. The future values of our `past_covariates` - "future past covariates" - are only used to provide the input chunk of upcoming samples with new data.

All this information is used to predict the "future target" - the next `output_chunk_length` points after the end of "past target".

![figure0](./images/covariates/tfm.png)

**Figure 1: Top level look at training / predicting on chunks with Torch Forecasting Models**


When calling `predict()` and depending on your forecast horizon `n`, the model can either predict in one go (if `n <= output_chunk_length`), or auto-regressively, by predicting on multiple chunks in the future (if `n > output_chunk_length`). That is the reason why when predicting with `past_covariates` you might have to supply additional "future values of your `past_covariates`".

## Torch Forecasting Model Covariates Support

Under the hood, Darts has 5 types of `{X}CovariatesModel` classes implemented to cover different combinations of the covariate types mentioned before:

| Class                   | past covariates | future past covariates | future covariates | historic future covariates |
|-------------------------|-----------------|------------------------|-------------------|----------------------------|
| `PastCovariatesModel`   | ✅               | ✅                      |                   |                            |
| `FutureCovariatesModel` |                 |                        | ✅                 |                            |
| `DualCovariatesModel`   |                 |                        | ✅                 | ✅                          |
| `MixedCovariatesModel`  | ✅               | ✅                      | ✅                 | ✅                          |
| `SplitCovariatesModel`  | ✅               | ✅                      | ✅                 |                            |

**Table 1: Darts' "{X}CovariatesModels" covariate support**

Each Torch Forecasting Model inherits from one `{X}CovariatesModel` (covariate class names are abbreviated by the `X`-part):

| TFM                | `Past` | `Future` | `Dual` | `Mixed` | `Split` |
|--------------------|--------|----------|--------|---------|---------|
| `RNNModel`         |        |          | ✅      |         |         |
| `BlockRNNModel`    | ✅      |          |        |         |         |
| `NBEATSModel`      | ✅      |          |        |         |         |
| `TCNModel`         | ✅      |          |        |         |         |
| `TransformerModel` | ✅      |          |        |         |         |
| `TFTModel`         |        |          |        | ✅       |         |

**Table 2: Darts' Torch Forecasting Model covariate support**

## Required target time spans for training, validation and prediction
The relevant data is extracted automatically by the models, based on the time axes of the series.
You can use the same covariates series for both `fit()` and `predict()` if they meet the requirements below.

**Training** only works if at least one sample with an input and output chunk can be extracted from the data you passed to `fit()`. This applies both to training and validation data. In terms of minimum required time spans, this means:
- `target` series of minimum length `input_chunk_length + output_chunk_length`
- `*_covariates` time span requirements for `fit()` from [covariates guide section 2.3.](https://unit8co.github.io/darts/userguide/covariates.html#id6)

For **prediction** you have to supply the `target` series that you wish to forecast. For any forecast horizon `n` the minimum time span requirements are:
- `target` series of minimum length `input_chunk_length`
- `*_covariates` time span requirements for `predict()` also from from [covariates guide section 2.3.](https://unit8co.github.io/darts/userguide/covariates.html#id6)

Side note: Our `*RNNModels` accept a `training_length` parameter at model creation instead of `output_chunk_length`. Internally the `output_chunk_length` for these models is automatically set to `1`. For training, past `target` must have a minimum length of `training_length + 1` and for prediction, a length of `input_chunk_length`.

## In-depth look at how input data is used when training and predicting with TFMs
### Training

Let's have a look at how the models work under the hood.

Let's assume we run an ice-cream shop and we want to predict sales for the next day.
We have one year (365 days) past data of our end-of-day ice-cream sales and of the average measured daily ambient temperature.
We also noticed that our ice-cream sales depend on the day of the week so we want to include this in our model.

- past target: actual past ice-cream sales `ice_cream_sales`
- future target: predict the ice-cream sales for the next day
- past covariates: measured average daily temperatures in the past `temperature`
- future covariates: day of the week for past and future `weekday`

Checking Table 1, a model that would accomodate this kind of covariates would be a
`SplitCovariatesModel` (if we don't use historic values of future covariates), or
`MixedCovariatesModel` (if we do). We choose a `MixedCovariatesModel` - the `TFTModel`.

Imagine that we saw a pattern in our past ice cream sales that repeated week after week.
So we set `input_chunk_length = 7` days to let the model look back an entire week into the past.
The `output_chunk_length` can be set to `1` day to predict the next day.

Now we can create a model and train it! Figure 2 shows you how `TFTModel` will use our data.

```
from darts.models import TFTModel

model = TFTModel(input_chunk_length=7, output_chunk_length=1)
model.fit(series=ice_cream_sales,
          past_covariates=temperature,
          future_covariates=weekday)
```

![figure4](./images/covariates/seq_covs_single.png)

**Figure 2: Overview of a single sequence from our ice-cream sales example**; Mon1 - Sun1 stand for the first 7 days from our training dataset (week 1 of the year). Mon2 is the Monday of week 2.

When calling `fit()`, the models will build an appropriate `darts.utils.data.TrainingDataset`, which specifies how to slice the data to obtain training samples. If you want to control this slicing yourself, you can instantiate your own `TrainingDataset` and call `model.fit_from_dataset()` instead of `fit()`. By default, most models (though not all) will build *sequential* datasets, which basically means that all sub-slices of length `input_chunk_length + output_chunk_length` in the provided series will be used for training.

So during training, the torch models will go through the training data in sequences (see Figure 3). Using information from the **input chunk** and **output chunk**, the model predicts the future target on the output chunk. The training loss is evaluated between the predicted future target and the actual target value on the output chunk. The model trains itself by minimizing the loss over all sequences.

![figure5](./images/covariates/seq_covs_1.png)

**Figure 3: Prediction and loss evaluation in a single sequence**

After having completed computations on the first sequence, the model moves to the next one and performs the same training steps. *The starting point of each sequence is selected randomly from the sequential dataset*. Figure 4 shows how this would look like if by pure chance the second sequence started one time step (day) after the first.

This sequence-to-sequence process is repeated until all 365 days were covered.

Side note: Having "long" `target` series can result in a very large number of training sequences / samples. You can set an upper bound for the number of sequences / samples per `target` that the model should be trained on with `fit()`-parameter `max_samples_per_ts`. This will take the most recent sequences for every `target` series (sequences closest to `target` end).

```python
# fit only on the 10 "most recent" sequences
model.fit(target, max_samples_per_ts=10)
```

![figure6](./images/covariates/sequential_training.png)

**Figure 4: Sequence-to-sequence: Move to next sequence and repeat training steps**

### Training with a validation dataset

You can also train your models with a validation dataset:

```python
# create train and validation sets
ice_cream_sales_train, ice_cream_sales_val = ice_cream_sales.split_after(training_cutoff)

# train with validation set
model.fit(series=ice_cream_sales_train,
          past_covariates=temperature,
          future_covariates=weekday,
          val_series=ice_cream_sales_val,
          val_past_covariates=temperature,
          val_future_covariates=weekday)
```

If you split your data, you have to define a `training_cutoff` (a date or fraction at which to split the dataset) so that both the train and validation datasets satisfy the minimum length requirements
from [this subsection](#required-target-time-spans-for-training-validation-and-prediction)

Instead of splitting by time, you can also use another subset of time series as validation set.

The model trains itself the same way as before but additionally evaluates the loss on the validation dataset. If you want to keep track of the best performing model on the validation set, you have to enable checkpoint saving as shown next.

### Saving and Loading Model States
Per default, the models don't automatically save any checkpoints. If you want to keep track of the best performing model on the validation set and the latest 5 epochs, you have to enable checkpoint saving at model creation:

```python
model = SomeTorchForecastingModel(..., model_name='MyModel', save_checkpoints=True)

# checkpoints are saved automatically
model.fit(...)

# load the model state that performed best on validation set
best_model = model.load_from_checkpoint(model_name='MyModel', best=True)
```

You can also save or load manually:
```python
model.save_model(model_path)
loaded_model = model.load_model(model_path)
```
/!\ Warning /!\ At this stage of Darts development, we are not (yet) ensuring backward compatibility, so it might not always be possible to load a model saved by an older version of the library.

## Forecast/Prediction

After having trained the model, we want to predict the future ice-cream sales for any number of days after our 365 days
training data.

The actual prediction works very similar to how we trained the data on sequences. Depending on the number of days we
want to predict - the forecast horizon `n` - we distinguish between two cases:

- If `n <= output_chunk_length`: we can predict `n` in one go (using one "internal model call")
    - in our example: predict the next day's ice-cream sales (`n = 1`)
- If `n > output_chunk_length`: we must predict `n` by calling the internal model multiple times. Each call outputs `output_chunk_length` prediction points. We go through as many calls as needed until we get to the final `n` prediction points, in an auto-regressive fashion.
    - in our example: predict ice-cream sales for the next 3 days at once (`n = 3`)

  To do this we have to supply additional `past_covariates` for the next `n - output_chunk_length = 2` time steps (days) after the end of our 365 days training data. Unfortunately, we do not have measured `temperture` for the future. But let's assume we have access to temperature forecasts for the next 2 days. We can just append them to `temperature` and the prediction will work!

  ```python
  temperature = temperature.concatenate(temperature_forecast, axis=0)
  ```

```python
prediction = model.predict(n=n,
                           series=ice_cream_sales_train,
                           past_covariates=temperature,
                           future_covariates=weekday)
```

![figure7](./images/covariates/prediction_once.png)

**Figure 5: Forecast with a single sequence for `n <= output_chunk_length`**


![figure8](./images/covariates/prediction_multi.png)

**Figure 6: Auto-regressive forecast for `n > output_chunk_length`**


## Performance Recommendations
This section recaps the main factors impacting the performance when
training and using torch-based models.

### Build your `TimeSeries` using 32-bits data
The models in Darts will dynamically cast themselves (to 64 or 32-bits)
to follow the dtype in the `TimeSeries`. Large performance and memory gains
can often be obtained when everything (data and model) is in float32.
To achieve this, it is enough to build your `TimeSeries` from arrays (or Dataframe-backing array) having dtype `np.float32`, or simply call `my_series32 = my_series.astype(np.float32)`. Calling `my_series.dtype` gives you the dtype of your `TimeSeries`.

### Use a GPU
In many cases using a GPU will provide a drastic speedup compared to CPU.
It can also incur some overheads (for transferring data to/from the GPU),
so some testing and tuning is often necessary.
We refer to our [GPU/TPU guide](https://unit8co.github.io/darts/userguide/gpu_and_tpu_usage.html)
for more informations on how to setup a GPU (or a TPU) via PyTorch Lightning.

### Tune the batch size
A larger batch size tends to speed up the training because it reduces the number
of backward passes per epoch and has the potential to better parallelize computation. However it also changes the training dynamics (e.g. you might need more epochs, and the convergence dynamics is affected). Furthermore larger batch sizes increase memory consumption. So here too some testing is required.

### Tune `num_loader_workers`
All deep learning models in Darts have a parameter `num_loader_workers` in their `fit()` and `predict()` functions, which
configures the `num_workers` parameter in the PyTorch `DataLoaders`. By default
it is set to 0, which means that the main process will also take care of loading the data. Setting `num_workers > 0` will use additional workers to load the data. This typically incurs some overhead (notably increasing memory consumption), but in some cases it can also substantially improve performance.
The ideal value depends on many factors such as the batch size, whether you are using a GPU, the number of CPU cores available, and whether
loading the data involved I/O operations (if the series are stored on disk).

### Small models first
Of course one of the main factors affecting performance is the model size
(number of parameters) and the number of operations required by forward/backward passes. Models in Darts can be tuned (e.g. number of layers, attention heads, widths etc), and these hyper-parameters tend to have a large impact on performance. When starting out, it is a good idea to build models of modest size first.

### Data in Memory and I/O bottlenecks
It's helpful to load all your `TimeSeries` in memory upfront if you can.
Darts offers the possibility to train models on any `Sequence[TimeSeries]`,
which means that for big datasets, you can write your own `Sequence` implementation, and read the time series lazily from disk. This will typically incur a high I/O cost, though. So when training on multiple series, first try to build a simple `List[TimeSeries]` upfront, and see if it holds in the computer memory.

### Do not use *all* possible sub-series for training
By default, when calling `fit()`, the models in Darts will build a `TrainingDataset` instance that is
suitable for the model that you are using (e.g., `PastCovariatesTorchModel`, `FutureCovariatesTorchModel`, etc.).
By default, these training datasets will often contain *all* possible consecutive (input, output) subseries present
in each `TimeSeries`. If your `TimeSeries` are long, this can result in a large amount of training samples, which directly (linearly)
impacts the time required to train the model for one epoch. You have two options to limit this:

* Specify some `max_samples_per_ts` argument to the `fit()` function. This will use only the most recent `max_samples_per_ts` samples
per `TimeSeries` for training.
* If this option does not do what you want, you can implement your own `TrainingDataset` instance, and define
how to slice your `TimeSeries` for training yourself. We suggest to have a look at [this submodule](https://github.com/unit8co/darts/tree/master/darts/utils/data)
to see examples of how to do it.


-------------

## Early Stop
Early stopping can be an efficient way to cut down on the training time.  Using early stopping trains the model for fewer epochs by stopping the training when the validation loss does not improve for a set number of `nr_epochs_val_period`.

Early stopping can be easily implemented on all Darts neural networks based models, by leveraging PyTorch Lightning's `EarlyStopping` callback, as shown in the example code below.
```python
    from darts.models import NBEATSModel
    from darts.datasets import AirPassengersDataset
    from pytorch_lightning.callbacks import EarlyStopping
    from torchmetrics import MeanAbsolutePercentageError
    import pandas as pd
    from darts.dataprocessing.transformers import Scaler

    # Read data:
    series = AirPassengersDataset().load()

    # Create training and validation sets:
    train, val = series.split_after(pd.Timestamp(year=1957, month=12, day=1))

    # Normalize the time series (note: we avoid fitting the transformer on the validation set)
    transformer = Scaler()
    transformer.fit(train)
    train = transformer.transform(train)
    val = transformer.transform(val)

    # A TorchMetric or val_loss can be used as the monitor
    torch_metrics = MeanAbsolutePercentageError()

    # Early stop callback
    my_stopper = EarlyStopping(
        monitor="val_MeanAbsolutePercentageError",  # "val_loss",
        patience=5,
        min_delta=0.05,
        mode='min',
    )
    pl_trainer_kwargs = {"callbacks": [my_stopper]}

    # Create the model
    model = NBEATSModel(
        input_chunk_length=24,
        output_chunk_length=12,
        n_epochs=500,
        torch_metrics=torch_metrics,
        pl_trainer_kwargs=pl_trainer_kwargs)

    model.fit(
        series=train,
        val_series=val,
    )
```

-------------

### Hyperparameter Tuning with Ray Tune
Hyperparameter tuning is a great way to find the best parameters for your model. [Ray Tune](https://docs.ray.io/en/latest/tune/examples/tune-pytorch-lightning.html) provides a convenient way to speed this process up with automatic pruning.

Here is an example of how to use Ray Tune to with the `NBEATSModel` model using the [Asynchronous Hyperband scheduler](https://blog.ml.cmu.edu/2018/12/12/massively-parallel-hyperparameter-optimization/). 

```python
from darts.models import NBEATSModel
from darts.datasets import AirPassengersDataset
from pytorch_lightning.callbacks import EarlyStopping
import pandas as pd
from darts.dataprocessing.transformers import Scaler
from torchmetrics import MetricCollection, MeanAbsolutePercentageError, MeanAbsoluteError
from ray import tune
from ray.tune import CLIReporter
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.tune.schedulers import ASHAScheduler

def train_model(model_args, callbacks, train, val):
    torch_metrics = MetricCollection([MeanAbsolutePercentageError(), MeanAbsoluteError()])
    # Create the model using model_args from Ray Tune
    model = NBEATSModel(
        input_chunk_length=24,
        output_chunk_length=12,
        n_epochs=500,
        torch_metrics=torch_metrics,
        pl_trainer_kwargs={"callbacks": callbacks, "enable_progress_bar": False},
        **model_args)

    model.fit(
        series=train,
        val_series=val,
    )

# Read data:
series = AirPassengersDataset().load()

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
    mode='min',
)

# set up ray tune callback
tune_callback = TuneReportCallback(
    {
        "loss": "val_Loss",
        "MAPE": "val_MeanAbsolutePercentageError",
    },
    on="validation_end",
)

# define the hyperparameter space
config = {
    "batch_size": tune.choice([16, 32, 64, 128]),
    "num_blocks": tune.choice([1, 2, 3, 4, 5]),
    "num_stacks": tune.choice([32, 64, 128]),
    "dropout": tune.uniform(0, 0.2),
}

reporter = CLIReporter(
    parameter_columns=list(config.keys()),
    metric_columns=["loss", "MAPE", "training_iteration"],
)

resources_per_trial = {"cpu": 8, "gpu": 1}

# the number of combinations to try
num_samples = 10

scheduler = ASHAScheduler(max_t=1000, grace_period=3, reduction_factor=2)

train_fn_with_parameters = tune.with_parameters(
    train_model, callbacks=[my_stopper, tune_callback], train=train, val=val,
)

analysis = tune.run(
    train_fn_with_parameters,
    resources_per_trial=resources_per_trial,
    # Using a metric instead of loss allows for 
    # comparison between different likelihood or loss functions.
    metric="MAPE",  # any value in TuneReportCallback. 
    mode="min",
    config=config,
    num_samples=num_samples,
    scheduler=scheduler,
    progress_reporter=reporter,
    name="tune_darts",
)

print("Best hyperparameters found were: ", analysis.best_config)
```

-------------

### Example Benchmark
As an example, we show here the time required to train one epoch on the first 80% of the energy dataset (`darts.datasets.EnergyDataset`), which consists of one multivariate series that is 28050 timesteps long and has 28 dimensions.
We train two models; `NBEATSModel` and `TFTModel`, with default parameters and `input_chunk_length=48` and `output_chunk_length=12` (which results in 27991 training samples with default sequential training datasets). For the TFT model, we also set the parameter `add_cyclic_encoder='hour'`. The tests are made on a Intel CPU i9-10900K CPU @ 3.70GHz, with an Nvidia RTX 2080s GPU, 32 GB of RAM. All `TimeSeries` are pre-loaded in memory and given to the models as a list.

| Model         | Dataset | dtype | CUDA | Batch size | num workers | time per epoch |
|---------------|---------|-------|------|------------|-------------|----------------|
| `NBEATSModel` | Energy  | 64    | no   | 32         | 0           | 283s           |
| `NBEATSModel` | Energy  | 64    | no   | 32         | 2           | 285s           |
| `NBEATSModel` | Energy  | 64    | no   | 32         | 4           | 282s           |
| `NBEATSModel` | Energy  | 64    | no   | 1024       | 0           | 58s            |
| `NBEATSModel` | Energy  | 64    | no   | 1024       | 2           | 57s            |
| `NBEATSModel` | Energy  | 64    | no   | 1024       | 4           | 58s            |
| `NBEATSModel` | Energy  | 64    | yes  | 32         | 0           | 63s            |
| `NBEATSModel` | Energy  | 64    | yes  | 32         | 2           | 62s            |
| `NBEATSModel` | Energy  | 64    | yes  | 1024       | 0           | 13.3s          |
| `NBEATSModel` | Energy  | 64    | yes  | 1024       | 2           | 12.1s          |
| `NBEATSModel` | Energy  | 64    | yes  | 1024       | 4           | 12.3s          |
|               |         |       |      |            |             |                |
| `NBEATSModel` | Energy  | 32    | no   | 32         | 0           | 117s           |
| `NBEATSModel` | Energy  | 32    | no   | 32         | 2           | 115s           |
| `NBEATSModel` | Energy  | 32    | no   | 32         | 4           | 117s           |
| `NBEATSModel` | Energy  | 32    | no   | 1024       | 0           | 28.4s          |
| `NBEATSModel` | Energy  | 32    | no   | 1024       | 2           | 27.4s          |
| `NBEATSModel` | Energy  | 32    | no   | 1024       | 4           | 27.5s          |
| `NBEATSModel` | Energy  | 32    | yes  | 32         | 0           | 41.5s          |
| `NBEATSModel` | Energy  | 32    | yes  | 32         | 2           | 40.6s          |
| `NBEATSModel` | Energy  | 32    | yes  | 1024       | 0           | 2.8s           |
| `NBEATSModel` | Energy  | 32    | yes  | 1024       | 2           | 1.65           |
| `NBEATSModel` | Energy  | 32    | yes  | 1024       | 4           | 1.8s           |
|               |         |       |      |            |             |                |
| `TFTModel`    | Energy  | 64    | no   | 32         | 0           | 78s            |
| `TFTModel`    | Energy  | 64    | no   | 32         | 2           | 72s            |
| `TFTModel`    | Energy  | 64    | no   | 32         | 4           | 72s            |
| `TFTModel`    | Energy  | 64    | no   | 1024       | 0           | 46s            |
| `TFTModel`    | Energy  | 64    | no   | 1024       | 2           | 38s            |
| `TFTModel`    | Energy  | 64    | no   | 1024       | 4           | 39s            |
| `TFTModel`    | Energy  | 64    | yes  | 32         | 0           | 125s           |
| `TFTModel`    | Energy  | 64    | yes  | 32         | 2           | 115s           |
| `TFTModel`    | Energy  | 64    | yes  | 1024       | 0           | 59s            |
| `TFTModel`    | Energy  | 64    | yes  | 1024       | 2           | 50s            |
| `TFTModel`    | Energy  | 64    | yes  | 1024       | 4           | 50s            |
|               |         |       |      |            |             |                |
| `TFTModel`    | Energy  | 32    | no   | 32         | 0           | 70s            |
| `TFTModel`    | Energy  | 32    | no   | 32         | 2           | 62.6s          |
| `TFTModel`    | Energy  | 32    | no   | 32         | 4           | 63.6           |
| `TFTModel`    | Energy  | 32    | no   | 1024       | 0           | 31.9s          |
| `TFTModel`    | Energy  | 32    | no   | 1024       | 2           | 45s            |
| `TFTModel`    | Energy  | 32    | no   | 1024       | 4           | 44s            |
| `TFTModel`    | Energy  | 32    | yes  | 32         | 0           | 73s            |
| `TFTModel`    | Energy  | 32    | yes  | 32         | 2           | 58s            |
| `TFTModel`    | Energy  | 32    | yes  | 1024       | 0           | 41s            |
| `TFTModel`    | Energy  | 32    | yes  | 1024       | 2           | 31s            |
| `TFTModel`    | Energy  | 32    | yes  | 1024       | 4           | 31s            |


