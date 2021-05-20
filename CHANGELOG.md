
# Changelog

Darts is still in an early development phase and we cannot always guarantee backwards compatibility. Changes that may **break code which uses a previous release of Darts** are marked with a "&#x1F534;".

## [Unreleased](https://github.com/unit8co/darts/tree/develop)
[Full Changelog](https://github.com/unit8co/darts/compare/0.7.0...develop)


## [0.8.0](https://github.com/unit8co/darts/tree/0.8.0) (2021-05-21)

### For users of the library:
**Added:**
- `RandomForest` algorithm implemented. Uses the scikit-learn `RandomForestRegressor` to predict future values from (lagged) exogenous
variables and lagged values of the target.
- `darts.datasets` is a new submodule allowing to easily download, cache and import some commonly used time series. 
E.g. doing
```
from darts.datasets import AirPassengers
ts: TimeSeries = AirPassengers.load()
```
- Better support for processing sequences of `TimeSeries`. 
  * The Transformers, Pipelines and metrics have been adapted to be used on sequences of `TimeSeries` 
  (rather than isolated series).
  * The inference of neural networks on sequences of series has been improved
- There is a new utils function `darts.utils.model_selection.train_test_split` which allows to split a `TimeSeries`
or a sequence of `TimeSeries` into train and test sets; either along the sample axis or along the time axis.
It also optionally allows to do "model-aware" splitting, where the split reclaims as much data as possible for the
training set.
- Our implementation of N-BEATS, `NBEATSModel`, now supports multivariate time series, as well as covariates.

**Changed**
- `RegressionModel` is now a user exposed class. It acts as a wrapper around any regression model with a `fit()` and `predict()`
method. It enables the flexible usage of lagged values of the target variable as well as lagged values of multiple exogenous
variables. Allowed values for the `lags` argument are positive integers or a list of positive integers indicating which lags
should be used during training and prediction, e.g. `lags=12` translates to training with the last 12 lagged values of the target variable. 
`lags=[1, 4, 8, 12]` translates to training with the previous value, the value at lag 4, lag 8 and lag 12.
- &#x1F534; `StandardRegressionModel` is now called `LinearRegressionModel`. It implements a linear regression model 
from `sklearn.linear_model.LinearRegression`. Users who still need to use the former `StandardRegressionModel` with
another sklearn model should use the `RegressionModel` now.

**Fixed**
- We have fixed a bug arising when multiple scalers were used.
- We have fixed a small issue in the TCN architecture, which makes our implementation follow the original paper
more closely.

### For developers of the library:
**Added:**
- We have added some [contribution guidelines](https://github.com/unit8co/darts/blob/develop/CONTRIBUTE.md).

## [0.7.0](https://github.com/unit8co/darts/tree/0.7.0) (2021-04-14)

[Full Changelog](https://github.com/unit8co/darts/compare/0.6.0...0.7.0)
### For users of the library:

**Added:**
- `darts` Pypi package. It is now possible to `pip install darts`. The older name `u8darts` is still maintained
and provides the different flavours for lighter installs.
- New forecasting model available: VARIMA (Vector Autoregressive moving average).
- Support for exogeneous variables in ARIMA, AutoARIMA and VARIMA (optional `exog` parameter in `fit()` and `predict()`
methods).
- New argument `dummy_index` for `TimeSeries` creation. If a series is just composed of a sequence of numbers
without timestamps, setting this flag will allow to create a `TimeSeries` which uses a "dummy time index" behind the
scenes. This simplifies the creation of `TimeSeries` in such cases, and makes it possible to use all forecasting models,
except those that explicitly rely on dates.
- New method `TimeSeries.diff()` returning differenced `TimeSeries`.
- Added an example of `RegressionEnsembleModel` in intro notebook.

**Changed:**
- Improved N-BEATS example notebook.
- Methods `TimeSeries.split_before()` and `split_after()` now also accept integer or float arguments (in addition to
timestamp) for the breaking point (e.g. specify 0.8 in order to obtain a 80%/20% split).
- Argument `value_cols` no longer has to be provided if not necessary when creating a `TimeSeries` from a `DataFrame`.
- Update of dependency requirements to more recent versions.

**Fixed:**
- Fix issue with MAX_TORCH_SEED_VALUE on 32-bit architectures (https://github.com/unit8co/darts/issues/235).
- Corrected a bug in TCN inference, which should improve accuracy.
- Fix historical forecasts not returning last point.
- Fixed bug when calling the `TimeSeries.gaps()` function for non-regular time frequencies.
- Many small bug fixes.


## [0.6.0](https://github.com/unit8co/darts/tree/0.6.0) (2021-02-02)

[Full Changelog](https://github.com/unit8co/darts/compare/0.5.0...0.6.0)
### For users of the library:
**Added:**
- `Pipeline.invertible()` a getter which returns whether the pipeline is invertible or not.
- `TimeSeries.to_json()` and `TimeSeries.from_json()` methods to convert `TimeSeries` to/from a `JSON` string.
- New base class `GlobalForecastingModel` for all models supporting training on multiple time series, as well
as covariates. All PyTorch models are now `GlobalForecastingModel`s.
- As a consequence of the above, the `fit()` function of PyTorch models (all neural networks) can optionally be called
with a sequence of time series (instead of a single time series).
- Similarly, the `predict()` function of these models also accepts a specification of which series should be forecasted
- A new `TrainingDataset` base class.
- Some implementations of `TrainingDataset` containing some slicing logic for the training of neural networks on
several time series.
- A new `TimeSeriesInferenceDataset` base class.
- An implementation `SimpleInferenceDataset` of `TimeSeriesInferenceDataset`.
- All PyTorch models have a new `fit_from_dataset()` method which allows to directly fit the model from a specified
`TrainingDataset` instance (instead of using a default instance when going via the `fit()` method).
- A new explanatory notebooks for global models:
https://github.com/unit8co/darts/blob/master/examples/02-multi-time-series-and-covariates.ipynb

**Changed:**
- &#x1F534; removed the arguments `training_series` and `target_series` in `ForecastingModel`s. Please consult
the API documentation of forecasting models to see the new signatures.
- &#x1F534; removed `UnivariateForecastingModel` and `MultivariateForecastingModel` base classes. This distinction does
not exist anymore. Instead, now some models are "global" (can be trained on multiple series) or "local" (they cannot).
All implementations of `GlobalForecastingModel`s support multivariate time series out of the box, except N-BEATS.
- Improved the documentation and README.
- Re-ordered the example notebooks to improve the flow of examples.

**Fixed:**
- Many small bug fixes.
- Unit test speedup by about 15x.

## [0.5.0](https://github.com/unit8co/darts/tree/0.5.0) (2020-11-09)

[Full Changelog](https://github.com/unit8co/darts/compare/0.4.0...0.5.0)
### For users of the library:
**Added:**
- Ensemble models, a new kind of `ForecastingModel` which allows to ensemble multiple models to make predictions:
  - `EnsembleModel` is the abstract base class for ensemble models. Classes deriving from `EnsembleModel` must implement the `ensemble()` method, which takes in a `List[TimeSeries]` of predictions from the constituent models, and returns the ensembled prediction (a single `TimeSeries` object)
  - `RegressionEnsembleModel`, a concrete implementation of `EnsembleModel `which allows to specify any regression model (providing `fit()` and `predict()` methods) to use to ensemble the constituent models' predictions.
- A new method to `TorchForecastingModel`: `untrained_model()` returns the model as it was initally created, allowing to retrain the exact same model from scratch. Works both when specifying a `random_state` or not.
- New `ForecastingModel.backtest()` and `RegressionModel.backtest()` functions which by default compute a single error score from the historical forecasts the model would have produced.
  - A new `reduction` parameter allows to specify whether to compute the mean/median/… of errors or (when `reduction` is set to `None`) to return a list of historical errors.
  - The previous `backtest()` functionality still exists but has been renamed `historical_forecasts()`
- Added a new `last_points_only` parameter to `historical_forecasts()`, `backtest()` and `gridsearch()`

**Changed:**
- &#x1F534; Renamed `backtest()` into `historical_forecasts()`
- `fill_missing_values()` and `MissingValuesFiller` used to remove the variable names when used with `fill='auto'` – not anymore.
- Modified the default plotting style to increase contrast and make plots lighter.

**Fixed:**
- Small mistake in the `NaiveDrift` model implementation which caused the first predicted value to repeat the last training value.

### For developers of the library:
**Changed:**
- `@random_method` decorator now always assigns a `_random_instance` field to decorated methods (seeded with a random seed). This doesn't change the observed behavior, but allows to deterministically "reset" `TorchForecastingModel` by saving `_random_instance` along with the other parameters of the model upon creation.

## [0.4.0](https://github.com/unit8co/darts/tree/0.4.0) (2020-10-28)

[Full Changelog](https://github.com/unit8co/darts/compare/0.3.0...0.4.0)

### For users of the library:
**Added:**
- Data (pre) processing abilities using `DataTransformer`, `Pipeline`:
  - `DataTransformer` provide a unified interface to apply transformations on `TimeSeries`, using their `transform()` method
  - `Pipeline`:
    - allow chaining of `DataTransformers`
    - provide `fit()`, `transform()`, `fit_transform()` and `inverse_transform()` methods.
  - Implementing your own data transformers:
    - Data transformers which need to be fitted first should derive from the `FittableDataTransformer` base class and implement a `fit()` method. Fittable transformers also provide a `fit_transform()` method, which fits the transformer and then transforms the data with a single call.
    - Data transformers which perform an invertible transformation should derive from the `InvertibleDataTransformer` base class and implement a `inverse_transform()` method.
    - Data transformers wich are neither fittable nor invertible should derive from the `BaseDataTransformer` base class
    - All data transformers must implement a `transform()` method.
- Concrete `DataTransformer` implementations:
  - `MissingValuesFiller` wraps around `fill_missing_value()` and allows to fill missing values using either a constant value or the `pd.interpolate()` method.
  - `Mapper` and `InvertibleMapper` allow to easily perform the equivalent of a `map()` function on a TimeSeries, and can be made part of a `Pipeline`
  - `BoxCox` allows to apply a BoxCox transformation to the data
- Extended `map()` on `TimeSeries` to accept functions which use both a value and its timestamp to compute a new value e.g.`f(timestamp, datapoint) = new_datapoint`
- Two new forecasting models:
  - `TransformerModel`, an implementation based on the architecture described in [Attention Is All You Need](https://arxiv.org/abs/1706.03762) by Vaswani et al. (2017)
  - `NBEATSModel`, an implementation based on the N-BEATS architecture described in [N-BEATS: Neural basis expansion analysis for interpretable time series forecasting](https://openreview.net/forum?id=r1ecqn4YwB) by Boris N. Oreshkin et al. (2019)

**Changed:**
- &#x1F534; Removed `cols` parameter from `map()`. Using indexing on `TimeSeries` is preferred.
  ```python
  # Assuming a multivariate TimeSeries named series with 3 columns or variables.
  # To apply fn to columns with names '0' and '2':

  #old syntax
  series.map(fn, cols=['0', '2']) # returned a time series with 3 columns
  #new syntax
  series[['0', '2']].map(fn) # returns a time series with only 2 columns
  ```
- &#x1F534; Renamed `ScalerWrapper` into `Scaler`
- &#x1F534; Renamed the `preprocessing` module into `dataprocessing`
- &#x1F534; Unified `auto_fillna()` and `fillna()` into a single `fill_missing_value()` function
  ```python
  #old syntax
  fillna(series, fill=0)

  #new syntax
  fill_missing_values(series, fill=0)

  #old syntax
  auto_fillna(series, **interpolate_kwargs)

  #new syntax
  fill_missing_values(series, fill='auto', **interpolate_kwargs)
  fill_missing_values(series, **interpolate_kwargs) # fill='auto' by default
  ```

### For developers of the library
**Changed:**
- GitHub release workflow is now triggered manually from the GitHub "Actions" tab in the repository, providing a `#major`, `#minor`, or `#patch` argument. [\#211](https://github.com/unit8co/darts/pull/211)
- (A limited number of) notebook examples are now run as part of the GitHub develop workflow.

## [0.3.0](https://github.com/unit8co/darts/tree/0.3.0) (2020-10-05)

[Full Changelog](https://github.com/unit8co/darts/compare/0.2.3...0.3.0)

### For users of the library:
**Added:**

- Better indexing on TimeSeries (support for column/component indexing) [\#150](https://github.com/unit8co/darts/pull/150)
- New `FourTheta` forecasting model [\#123](https://github.com/unit8co/darts/pull/123), [\#156](https://github.com/unit8co/darts/pull/156)
- `map()` method for TimeSeries [\#121](https://github.com/unit8co/darts/issues/121), [\#166](https://github.com/unit8co/darts/pull/166)
- Further improved the backtesting functions [\#111](https://github.com/unit8co/darts/pull/111):
  - Added support for multivariate TimeSeries and models
  - Added `retrain` and `stride` parameters
- Custom style for matplotlib plots [\#191](https://github.com/unit8co/darts/pull/191)
- sMAPE metric [\#129](https://github.com/unit8co/darts/pull/129)
- Option to specify a `random_state` at model creation using the `@random_method` decorator on models using neural networks to allow reproducibility of results [\#118](https://github.com/unit8co/darts/pull/118)

**Changed:**

- &#x1F534; **Refactored backtesting** [\#184](https://github.com/unit8co/darts/pull/184)
  - Moved backtesting functionalities inside `ForecastingModel` and `RegressionModel`
    ```python
    # old syntax:
    backtest_forecasting(forecasting_model, *args, **kwargs)

    # new syntax:
    forecasting_model.backtest(*args, **kwargs)

    # old syntax:
    backtest_regression(regression_model, *args, **kwargs)

    # new syntax:
    regression_model.backtest(*args, **kwargs)
    ```
  - Consequently removed the `backtesting` module
- &#x1F534; `ForecastingModel` `fit()` **method syntax** using TimeSeries indexing instead of additional parameters [\#161](https://github.com/unit8co/darts/pull/161)
  ```python
  # old syntax:
  multivariate_model.fit(multivariate_series, target_indices=[0, 1])

  # new syntax:
  multivariate_model.fit(multivariate_series, multivariate_series[["0", "1"]])

  # old syntax:
  univariate_model.fit(multivariate_series, component_index=2)

  # new syntax:
  univariate_model.fit(multivariate_series["2"])
  ```

**Fixed:**
- Solved issue of TorchForecastingModel.predict(n) throwing an error at n=1. [\#108](https://github.com/unit8co/darts/pull/108)
- Fixed MASE metrics [\#129](https://github.com/unit8co/darts/pull/129)
- \[BUG\] ForecastingModel.backtest: Can bypass sanity checks [\#188](https://github.com/unit8co/darts/issues/188)
- ForecastingModel.backtest\(\) fails if forecast\_horizon isn't provided [\#186](https://github.com/unit8co/darts/issues/186)

### For developers of the library

**Added:**
- Gradle to build docs, docker image, run tests, … [\#112](https://github.com/unit8co/darts/pull/112), [\#127](https://github.com/unit8co/darts/pull/127), [\#159](https://github.com/unit8co/darts/pull/159)
- M4 competition benchmark and notebook to the examples [\#138](https://github.com/unit8co/darts/pull/138)
- Check of test coverage [\#141](https://github.com/unit8co/darts/pull/141)

**Changed:**
- Dependencies' versions are now fixed [\#173](https://github.com/unit8co/darts/pull/173)
- Workflow: tests trigger on Pull Request [\#165](https://github.com/unit8co/darts/pull/165)

**Fixed:**
- Passed the `freq` parameter to the `TimeSeries` constructor in all TimeSeries generating functions [\#157](https://github.com/unit8co/darts/pull/157)

## Older releases

[Full Changelog](https://github.com/unit8co/darts/compare/f618c4536bf7ed6e3b6a2239fbca4e3089736426...0.2.3)
