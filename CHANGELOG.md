# Changelog

We do our best to avoid the introduction of breaking changes,
but cannot always guarantee backwards compatibility. Changes that may **break code which uses a previous release of Darts** are marked with a "🔴". Changes that deprecate functionalities are marked with a "🟠".

## [Unreleased](https://github.com/unit8co/darts/tree/master)

[Full Changelog](https://github.com/unit8co/darts/compare/0.34.0...master)

### For users of the library:

**Improved**

- Added support for categorical covariate to  `CatBoostModel`. You can now define categorical components at model construction with parameters `categorical_*_covariates: List[str]` for past, future, and static covariates. [#2733](https://github.com/unit8co/darts/pull/2750) by [Jonas Blanc](https://github.com/jonasblanc).
- Added new forecasting model: `AutoMFLES`, a simple time series method based on gradient boosting time series decomposition as proposed in [this repository](https://github.com/tblume1992/MFLES). This implementation is based on [AutoMFLES](https://nixtlaverse.nixtla.io/statsforecast/docs/models/mfles.html) from Nixtla's `statsforecasts` library. [#2747](https://github.com/unit8co/darts/pull/2747) by [Che Hang Ng](https://github.com/CheHangNg).
- 🔴 Simplified all `StatsForecast*` model names by removing the `StatsForecast` part. [#2762](https://github.com/unit8co/darts/pull/2762) by [Dennis Bader](https://github.com/dennisbader).
  - Renamed `StatsForecastAutoARIMA` to `AutoARIMA`
  - Renamed `StatsForecastAutoCES` to `AutoCES`
  - Renamed `StatsForecastAutoETS` to `AutoETS`
  - Renamed `StatsForecastAutoTBATS` to `AutoTBATS`
  - Renamed `StatsForecastAutoTheta` to `AutoTheta`

**Removed / moved**

- 🔴 Removed model `AutoARIMA`. To support `numpy>=2.0.0`, we unfortunately had to remove the `pmdarima` dependency. Use `StatsForecastAutoARIMA` instead. [#2734](https://github.com/unit8co/darts/pull/2734) by [Dennis Bader](https://github.com/dennisbader).
- 🔴 Removed deprecated method `TimeSeries.pd_dataframe()`. Use `TimeSeries.to_dataframe()` instead. [#2733](https://github.com/unit8co/darts/pull/2733) by [Dennis Bader](https://github.com/dennisbader).
- 🔴 Removed deprecated method `TimeSeries.pd_serise()`. Use `TimeSeries.to_series()` instead. [#2733](https://github.com/unit8co/darts/pull/2733) by [Dennis Bader](https://github.com/dennisbader).

**Fixed**

- Fixed a bug in `CatBoostModel` with `likelihood="gaussian"`, where predicting with `predict_likelihood_parameters=True` resulted in wrong ordering of the predicted parameters. [#2742](https://github.com/unit8co/darts/pull/2742) by [Dennis Bader](https://github.com/dennisbader).

**Dependencies**

### For developers of the library:

- Refactored likelihoods: [#2742](https://github.com/unit8co/darts/pull/2742) by [Dennis Bader](https://github.com/dennisbader).
  - Moved all likelihood related files into `darts/utils/likelihood_models/`.
  - Added `BaseLikelihood` as a base class for all likelihood models to `darts.utils.likelihood_models.base`.
  - Added `RegressionModel` likelihoods to `darts.utils.likelihood_models.sklearn`.
  - Moved `TorchForecastingModel` likelihoods to `darts.utils.likelihood_models.torch`. They can still be imported through `darts.utils.likelihood_models`.
  - Removed `darts.models.forecasting.regression_model._LikelihoodMixin`. Use dedicated Likelihood models instead.
- Made  `RegressionModelWithCategoricalCovariates` abstract, rewrote logic for categorical component names matching and refactored categorical covariate logic to make it easier to add categorical cov support to a model. [#2733](https://github.com/unit8co/darts/pull/2750) by [Jonas Blanc](https://github.com/jonasblanc).


## [0.34.0](https://github.com/unit8co/darts/tree/0.34.0) (2025-03-09)

### For users of the library:

**Improved**

- Improvements to `TimeSeries`:
  - `from_dataframe()` and `from_series()` now support creating `TimeSeries` from **additional DataFrame backends** (Polars, PyArrow, ...). We leverage `narwhals` as the compatibility layer between DataFrame libraries. See their [documentation](https://narwhals-dev.github.io/narwhals/) for all supported backends. [#2661](https://github.com/unit8co/darts/pull/2661) by [Jules Authier](https://github.com/authierj)
  - Added **new export methods**: `to_dataframe()` and `to_series()`. These methods support exporting `TimeSeries` to DataFrames and Series for all backends supported by `narwhals`. See their [documentation](https://narwhals-dev.github.io/narwhals/extending/) for all supported backends. [#2701](https://github.com/unit8co/darts/pull/2701) by [Jules Authier](https://github.com/authierj)
  - Added support for **embedding metadata** in `TimeSeries`. You can now attach any type of metadata as a dictionary to your time series, either at series creation using parameter `metadata` (or `metadata_cols` with `from_group_dataframe()`), or add it to an existing series via the `with_metadata()` method. Unlike static covariates, metadata is not utilized by Darts' models during the forecasting process. However, the metadata is propagated through all downstream tasks, preserving the integrity and continuity of the information throughout the data pipeline. [#2656](https://github.com/unit8co/darts/pull/2656) by [Yoav Matzkevich](https://github.com/ymatzkevich), [Jules Authier](https://github.com/authierj) and [Dennis Bader](https://github.com/dennisbader).
  - Method `plot()` now supports **setting the color for each component** in the series. Simply pass a list / sequence of colors with length matching the number of components as parameters "c" or "colors". [#2680](https://github.com/unit8co/darts/pull/2680) by [Jules Authier](https://github.com/authierj)
  - Improved documentation: Added examples to `TimeSeries` factory methods and updated quickstart notebook with the extended backend support for creating and exporting `TimeSeries`. [#2725](https://github.com/unit8co/darts/pull/2725) by [Dennis Bader](https://github.com/dennisbader).
  - 🟠 Methods `pd_dataframe()` and `pd_series()` are deprecated and will be removed in Darts version 0.35.0. Use `to_dataframe()` and `to_series()` instead. [#2701](https://github.com/unit8co/darts/pull/2701) by [Jules Authier](https://github.com/authierj)
- Improvements to `TorchForecastingModel`:
  - Added ONNX support for torch-based models with method `to_onnx()`. Check out [this example](https://unit8co.github.io/darts/userguide/gpu_and_tpu_usage.html#exporting-model-to-onnx-format-for-inference) from the user guide on how to export and load a model for inference. [#2620](https://github.com/unit8co/darts/pull/2620) by [Antoine Madrona](https://github.com/madtoinou)
- Improvements to `RegressionModel`:
  - Added `quantile` parameter to method `get_estimator()` to get the specific quantile estimator for probabilistic regression models using the `quantile` likelihood. [#2716](https://github.com/unit8co/darts/pull/2716) by [Antoine Madrona](https://github.com/madtoinou)
  - Improved `CatBoostModel` documentation by describing how to use native multi-output regression. [#2659](https://github.com/unit8co/darts/pull/2659) by [Jonas Blanc](https://github.com/jonasblanc)
  - 🔴 Removed method `get_multioutput_estimator()`. Use `get_estimator()` instead. [#2716](https://github.com/unit8co/darts/pull/2716) by [Antoine Madrona](https://github.com/madtoinou)
- Other improvements:
  - Made method `ForecastingModel.untrained_model()` public. Use this method to get a new (untrained) model instance created with the same parameters. [#2684](https://github.com/unit8co/darts/pull/2684) by [Timon Erhart](https://github.com/turbotimon)
  - Made it possible to run the quickstart notebook `00-quickstart.ipynb` in local dev mode without installation. [#2691](https://github.com/unit8co/darts/pull/2691) by [Jules Authier](https://github.com/authierj)

**Fixed**

- 🔴 / 🟢 Fixed a bug which raised an error when loading a `TorchForecastingModel` that was saved with Darts versions < 0.33.0. This is a breaking change and models saved with version 0.33.0 will not be loadable anymore. [#2692](https://github.com/unit8co/darts/pull/2692) by [Dennis Bader](https://github.com/dennisbader).
- Fixed a bug in `NLinearModel` with `normalize=True` where the normalization was not applied properly. The predictive power is now greatly improved. [#2724](https://github.com/unit8co/darts/pull/2724) by [Dennis Bader](https://github.com/dennisbader).
- Fixed a bug in `StaticCovariatesTransformer` which raised an error when trying to inverse transform one-hot encoded categorical static covariates with identical values across time-series. Each categorical static covariates is now referred to by `{covariate_name}_{category_name}`, regardless of the number of categories. [#2710](https://github.com/unit8co/darts/pull/2710) by [Antoine Madrona](https://github.com/madtoinou)
- Fixed a bug in `RegressionModel` where performing optimized historical forecasts with `max(lags) < -1` resulted in forecasts that extended too far into the future. [#2715](https://github.com/unit8co/darts/pull/2715) by [Jules Authier](https://github.com/authierj)

**Dependencies**

- Bumped minimum scikit-learn version from `1.0.1` to `1.6.0`. This was required due to sklearn deprecating `_get_tags` in favor of `BaseEstimator.__sklearn_tags__` in version 1.7. This leads to increasing both sklearn and XGBoost minimum supported version to 1.6 and 2.1.4 respectively. [#2659](https://github.com/unit8co/darts/pull/2659) by [Jonas Blanc](https://github.com/jonasblanc)
- Bumped minimum xgboost version from `1.6.0` to `2.1.4` for the same reason as bumping the minimum sklearn version. [#2659](https://github.com/unit8co/darts/pull/2659) by [Jonas Blanc](https://github.com/jonasblanc)
- Various code changes to reflect the updated dependencies versions (sklearn, statsmodel) and eliminate various warnings. [#2722](https://github.com/unit8co/darts/pull/2722) by [Antoine Madrona](https://github.com/madtoinou)

### For developers of the library:

**Improved**

- Refactored and improved the multi-output support handling for `RegressionModel`. [#2659](https://github.com/unit8co/darts/pull/2659) by [Jonas Blanc](https://github.com/jonasblanc)

## [0.33.0](https://github.com/unit8co/darts/tree/0.33.0) (2025-02-14)

### For users of the library:

**Improved**

- Improvements to `TimeSeries`:
  - Added more resampling methods to `TimeSeries.resample()`. This allows to aggregate values when down-sampling and to fill or keep the holes when up-sampling. [#2654](https://github.com/unit8co/darts/pull/2654) by [Jonas Blanc](https://github.com/jonasblanc)
  - Added the `title` attribute to `TimeSeries.plot()`. This allows to set a title for the plot. [#2639](https://github.com/unit8co/darts/pull/2639) by [Jonathan Koch](https://github.com/jonathankoch99).
  - Added general function `darts.slice_intersect()` to intersect a sequence of `TimeSeries` along the time index. [#2592](https://github.com/unit8co/darts/pull/2592) by [Yoav Matzkevich](https://github.com/ymatzkevich).
- Improvements to Model Saving & Loading:
  - Added parameter `clean: bool` to `ForecastingModel.save()` to store a cleaned version of the model (removes training data from global models, and Lightning Trainer-related parameters from torch models). [#2649](https://github.com/unit8co/darts/pull/2649) by [Jonas Blanc](https://github.com/jonasblanc).
  - Added parameter `pl_trainer_kwargs` to `TorchForecastingModel.load()` to setup a new Lightning Trainer used to configure the model for downstream tasks (e.g. prediction). [#2649](https://github.com/unit8co/darts/pull/2649) by [Jonas Blanc](https://github.com/jonasblanc).
- Improvements to Anomaly Detection:
  - Added parameter `component_wise` to `show_anomalies()` to separately plot each component in multivariate series. [#2544](https://github.com/unit8co/darts/pull/2544) by [He Weilin](https://github.com/cnhwl).
  - Improved the documentation of how `WindowedAnomalyScorer` extract the training data from the input series. [#2674](https://github.com/unit8co/darts/pull/2674) by [Dennis Bader](https://github.com/dennisbader).
- Other improvemets:
  - Added new forecasting model: `StatsForecastAutoTBATS`, wrapping [AutoTBATS](https://nixtlaverse.nixtla.io/statsforecast/src/core/models.html#autotbats) from Nixtla's `statsforecasts` library. [#2611](https://github.com/unit8co/darts/pull/2611) by [He Weilin](https://github.com/cnhwl).
  - Added new time aggregated metric `wmape()` (Weighted Mean Absolute Percentage Error). [#2544](https://github.com/unit8co/darts/pull/2648) by [He Weilin](https://github.com/cnhwl).

**Fixed**

- Fixed a bug which raised an error when loading a torch model with `torch>=2.6.0`. [#2658](https://github.com/unit8co/darts/pull/2658) by [Dennis Bader](https://github.com/dennisbader).
- Fixed a bug when performing optimized historical forecasts with `stride=1` using a `RegressionModel` with `output_chunk_shift>=1` and `output_chunk_length=1`, where the forecast time index was not properly shifted. [#2634](https://github.com/unit8co/darts/pull/2634) by [Mattias De Charleroy](https://github.com/MattiasDC).
- Fixed a bug where global naive models could not be used in ensemble models. [#2666](https://github.com/unit8co/darts/pull/2666) by [Dennis Bader](https://github.com/dennisbader).
- Fixed a bug with global naive models where some `supports_*` properties were wrongly defined as methods. [#2666](https://github.com/unit8co/darts/pull/2666) by [Dennis Bader](https://github.com/dennisbader).
- Fixed a bug in `LightGBMModel`, `XGBModel`, and `CatBoostModel` which raised an error when calling `fit()` with `val_sample_weight`. [#2626](https://github.com/unit8co/darts/pull/2626) by [Kylin Schmidt](https://github.com/kylinschmidt).
- Fixed the `ShapExplainer` `summary_plot` title where Horizon does not include `output_chunk_shift`. [#2647](https://github.com/unit8co/darts/pull/2647) by [He Weilin](https://github.com/cnhwl).

**Dependencies**

- Removed the upper version cap on `sklearn<1.6.0` since `xboost` added support in version `2.1.4`. [#2665](https://github.com/unit8co/darts/pull/2665) by [Dennis Bader](https://github.com/dennisbader).

### For developers of the library:

- Bumped `jinja2` from 3.1.4 to 3.1.5 (release requirement). [#2672](https://github.com/unit8co/darts/pull/2672) by dapendabot.

## [0.32.0](https://github.com/unit8co/darts/tree/0.32.0) (2024-12-21)

### For users of the library:

**Improved**

- 🚀🚀 Introducing Conformal Prediction: You can now add calibrated prediction intervals to any pre-trained global forecasting model with our first two conformal prediction models : [#2552](https://github.com/unit8co/darts/pull/2552) by [Dennis Bader](https://github.com/dennisbader).
  - `ConformalNaiveModel`: Uses past point forecast errors to produce calibrated forecast intervals with a specified coverage probability.
  - `ConformalQRModel`: Combines quantile regression (or any probabilistic model) with conformal prediction techniques. It adjusts the quantile estimates to generate calibrated prediction intervals with a specified coverage probability.
  - Both models offer the following support:
    - identical API as forecasting models
    - use any pre-trained global forecasting model as the base forecaster
    - uni and multivariate forecasts
    - single and multiple series forecasts
    - single and multi-horizon forecasts
    - generate a single or multiple calibrated prediction intervals
    - direct quantile value predictions (interval bounds) or sampled predictions from these quantile values
    - covariates based on the underlying forecasting model
  - Check out our [conformal prediction notebook](https://unit8co.github.io/darts/examples/23-Conformal-Prediction-examples.html) for detailed information and usage examples!
- Improvements to backtesting with `ForecastingModel` (`historical_forecasts()`, `backtest()`, `residuals()`, and `gridsearch()`):
  - 🚀🚀 Added support for data transformers and pipelines. Use argument `data_transformers` to automatically apply any `DataTransformer` and/or `Pipeline` to the input series without data-leakage (optional fit on historic window of input series, transform the input series, and inverse transform the forecasts). [#2529](https://github.com/unit8co/darts/pull/2529) by [Antoine Madrona](https://github.com/madtoinou) and [Jan Fidor](https://github.com/JanFidor)
  - Improved `start` handling. If `start` is not within the trainable / forecastable points, uses the closest valid start point that is a round multiple of `stride` ahead of start. Raises a ValueError, if no valid start point exists. This guarantees that all historical forecasts are `n * stride` points away from start, and will simplify many downstream tasks. [#2560](https://github.com/unit8co/darts/issues/2560) by [Dennis Bader](https://github.com/dennisbader).
  - Added support for `overlap_end=True` to `residuals()`. This computes historical forecasts and residuals that can extend further than the end of the target series. Guarantees that all returned residual values have the same length per forecast (the last residuals will contain missing values, if the forecasts extended further into the future than the end of the target series). [#2552](https://github.com/unit8co/darts/pull/2552) by [Dennis Bader](https://github.com/dennisbader).
- Improvements to `metrics`: Added three new quantile interval metrics (plus their aggregated versions) : [#2552](https://github.com/unit8co/darts/pull/2552) by [Dennis Bader](https://github.com/dennisbader).
  - Interval Winkler Score `iws()`, and Mean Interval Winkler Scores `miws()` (time-aggregated) ([source](https://otexts.com/fpp3/distaccuracy.html))
  - Interval Coverage `ic()` (binary if observation is within the quantile interval), and Mean Interval Coverage `mic()` (time-aggregated)
  - Interval Non-Conformity Score for Quantile Regression `incs_qr()`, and Mean ... `mincs_qr()` (time-aggregated) ([source](https://arxiv.org/pdf/1905.03222))
- Added `series_idx` argument to `DataTransformer` that allows users to use only a subset of the transformers when `global_fit=False` and severals series are used. [#2529](https://github.com/unit8co/darts/pull/2529) by [Antoine Madrona](https://github.com/madtoinou)
- Updated the Documentation URL of `Statsforecast` models. [#2610](https://github.com/unit8co/darts/pull/2610) by [He Weilin](https://github.com/cnhwl).

**Fixed**

- Fixed a bug when initiating a `RegressionModel` with `lags_past_covariates` as dict and `lags_future_covariates` as some other type (not dict) and `output_chunk_shift>0`, [#2652](https://github.com/unit8co/darts/issues/2652) by [Jules Authier](https://github.com/authierj).
- Fixed a bug which raised an error when computing residuals (or backtest with "per time step" metrics) on multiple series with corresponding historical forecasts of different lengths. [#2604](https://github.com/unit8co/darts/pull/2604) by [Dennis Bader](https://github.com/dennisbader).
- Fixed a bug when using `darts.utils.data.tabularization.create_lagged_component_names()` with target `lags=None`, that did not return any lagged target label component names. [#2576](https://github.com/unit8co/darts/pull/2576) by [Dennis Bader](https://github.com/dennisbader).
- Fixed a bug when using `num_samples > 1` with a deterministic regression model and the optimized `historical_forecasts()` method, which did not raise an exception. [#2576](https://github.com/unit8co/darts/pull/2588) by [Antoine Madrona](https://github.com/madtoinou).
- Fixed a bug when performing optimized historical forecasts with `RegressionModel` and `last_points_only=False`, where the forecast index generation could result in out-of-bound dates. [#2623](https://github.com/unit8co/darts/pull/2623) by [Dennis Bader](https://github.com/dennisbader).
- Fixed the failing docker image deployment. [#2583](https://github.com/unit8co/darts/pull/2583) by [Dennis Bader](https://github.com/dennisbader).

**Dependencies**

- 🔴 Removed support for Python 3.8. The new minimum Python version is 3.9. [#2586](https://github.com/unit8co/darts/pull/2586) by [Dennis Bader](https://github.com/dennisbader).
- We set an upper version cap on `sklkearn<=1.5.0` until `xgboost` officially supports version `1.6.0`.  [#2618](https://github.com/unit8co/darts/pull/2618) by [Dennis Bader](https://github.com/dennisbader).

### For developers of the library:

**Improved**

- Improvements to CI/CD : [#2584](https://github.com/unit8co/darts/pull/2584) by [Dennis Bader](https://github.com/dennisbader).
  - updated all workflows with most recent GitHub actions versions
  - improved caching across `master` branch and its children
  - fixed failing docker deployment
  - removed `gradle` dependency in favor of native GitHub actions plugins.
- Updated ruff to v0.7.2 and target-version to python39, also fixed various typos. [#2589](https://github.com/unit8co/darts/pull/2589) by [Greg DeVosNouri](https://github.com/gdevos010) and [Antoine Madrona](https://github.com/madtoinou).
- Replaced the deprecated `torch.nn.utils.weight_norm` function with `torch.nn.utils.parametrizations.weight_norm`. [#2593](https://github.com/unit8co/darts/pull/2593) by [Saeed Foroutan](https://github.com/SaeedForoutan).

## [0.31.0](https://github.com/unit8co/darts/tree/0.31.0) (2024-10-13)

### For users of the library:

**Improved**

- Improvements to `metrics`:
  - Added support for computing metrics on one or multiple quantiles `q`, either from probabilistic or quantile forecasts. [#2530](https://github.com/unit8co/darts/pull/2530) by [Dennis Bader](https://github.com/dennisbader).
  - Added quantile interval metrics `miw` (Mean Interval Width, time aggregated) and `iw` (Interval Width, per time step / non-aggregated) which compute the width of quantile intervals `q_intervals` (expected to be a tuple or sequence of tuples with (lower quantile, upper quantile)). [#2530](https://github.com/unit8co/darts/pull/2530) by [Dennis Bader](https://github.com/dennisbader).
- Improvements to `backtest()` and `residuals()`:
  - Added support for computing backtest and residuals on one or multiple quantiles `q` in the `metric_kwargs`, either from probabilistic or quantile forecasts. [#2530](https://github.com/unit8co/darts/pull/2530) by [Dennis Bader](https://github.com/dennisbader).
  - Added support for parameters `enable_optimization` and `predict_likelihood_parameters`. [#2530](https://github.com/unit8co/darts/pull/2530) by [Dennis Bader](https://github.com/dennisbader).
- Improvements to `TimeSeries`:
  - Added support for broadcasting TimeSeries on component and sample level for arithmetic operations. [#2476](https://github.com/unit8co/darts/pull/2476) by [Joel L.](https://github.com/Joelius300).
  - Added property `TimeSeries.shape` to get the shape of the time series. [#2530](https://github.com/unit8co/darts/pull/2530) by [Dennis Bader](https://github.com/dennisbader).
- Other improvements:
  - Added a new anomaly detector `IQRDetector`, that allows to detect anomalies using the Interquartile Range algorithm. [#2441](https://github.com/unit8co/darts/pull/2441) by [Igor Urbanik](https://github.com/u8-igor).
  - Added hyperparameters `temporal_hidden_size_past/future` controlling the hidden layer sizes for the feature encoders in `TiDEModel`. [#2408](https://github.com/unit8co/darts/pull/2408) by [eschibli](https://github.com/eschibli).
  - Added hyperparameter `activation` to `BlockRNNModel` to specify the activation function in case of a multi-layer output network. [#2504](https://github.com/unit8co/darts/pull/2504) by [Szymon Cogiel](https://github.com/SzymonCogiel).
  - Helper function `darts.utils.utils.generate_index()` now accepts datetime strings as `start` and `end` parameters to generate the pandas DatetimeIndex. [#2522](https://github.com/unit8co/darts/pull/2522) by [Dennis Bader](https://github.com/dennisbader).
- Improvements to the documentation:
  - Made README's forecasting model support table more colorblind-friendly. [#2433](https://github.com/unit8co/darts/pull/2433) by [Jatin Shridhar](https://github.com/jatins).
  - Updated the Ray Tune Hyperparameter Optimization example in the [user guide](https://unit8co.github.io/darts/userguide/hyperparameter_optimization.html) to work with the latest `ray` versions (`>=2.31.0`). [#2459](https://github.com/unit8co/darts/pull/2459) by [He Weilin](https://github.com/cnhwl).
  - Indicate that `multi_models=False` induce a lags shift for each step in `output_chunk_length` in `RegressionModel` and `LinearRegressionModel`. [#2511](https://github.com/unit8co/darts/pull/2511) by [Antoine Madrona](https://github.com/madtoinou).
  - Added reference to `timeseries_generation.datetime_attribute_timeseries` in `TimeSeries.add_datetime_attribute` (0-indexing of encoding is enforced). [#2511](https://github.com/unit8co/darts/pull/2511) by [Antoine Madrona](https://github.com/madtoinou).

**Fixed**

- Fixes to `RegressionModel`:
  - Fixed a bug when performing probabilistic optimized historical forecasts (`num_samples>1, retrain=False, enable_optimization=True`) with regression models, where reshaping the array resulted in a wrong order of samples across components and forecasts. [#2534](https://github.com/unit8co/darts/pull/2534) by [Dennis Bader](https://github.com/dennisbader).
  - Fixed a bug when predicting with `predict_likelihood_parameters=True`, `n > 1` and a `RegressionModel` with `multi_models=False` that uses a `likelihood`. The prediction now works without raising an exception. [#2545](https://github.com/unit8co/darts/pull/2545) by [Dennis Bader](https://github.com/dennisbader).
  - Fixed a bug when using `historical_forecasts()` with a pre-trained `RegressionModel` that has no target lags `lags=None` but uses static covariates. [#2426](https://github.com/unit8co/darts/pull/2426) by [Dennis Bader](https://github.com/dennisbader).
  - Fixed a bug when using `fit()` with a `RegressionModel` that uses an underlying `model` which does not support `sample_weight`. [#2445](https://github.com/unit8co/darts/pull/2445) by [He Weilin](https://github.com/cnhwl).
  - Fixed a bug when using `save()` and `load()` with a `RegressionEnsembleModel` that ensembles any `TorchForecastingModel`. [#2437](https://github.com/unit8co/darts/pull/2437) by [GeorgeXiaojie](https://github.com/GeorgeXiaojie).
  - Fixed a bug with `xgboost>=2.1.0`, where multi output regression was not properly handled. [#2426](https://github.com/unit8co/darts/pull/2426) by [Dennis Bader](https://github.com/dennisbader).
- Fixes to `TimeSeries`:
  - Fixed a bug when plotting a probabilistic multivariate series with `TimeSeries.plot()`, where all confidence intervals (starting from 2nd component) had the same color as the median line. [#2532](https://github.com/unit8co/darts/pull/2532) by [Dennis Bader](https://github.com/dennisbader).
  - Fixed a bug when using `TimeSeries.from_group_dataframe()` with a `time_col` of type integer, where the resulting time index was wrongly converted to a DatetimeIndex. [#2512](https://github.com/unit8co/darts/pull/2512) by [Alessio Pellegrini](https://github.com/AlessiopSymplectic)
  - Fixed a bug where passing an empty array to `TimeSeries.prepend/append_values()` raised an error. [#2522](https://github.com/unit8co/darts/pull/2522) by [Alessio Pellegrini](https://github.com/AlessiopSymplectic)
  - Fixed a bug with `TimeSeries.prepend/append_values()`, where the name of the (time) index was lost. [#2522](https://github.com/unit8co/darts/pull/2522) by [Alessio Pellegrini](https://github.com/AlessiopSymplectic)
- Other fixes:
  - Fixed a bug when using `ShapExplainer.explain()` with some selected `target_components` and a regression model that natively supports multi output regression: The target components were not properly mapped. [#2428](https://github.com/unit8co/darts/pull/2428) by [Dennis Bader](https://github.com/dennisbader).
  - Fixed a bug with `CrostonModel`, which actually does not support future covariates. [#2511](https://github.com/unit8co/darts/pull/2511) by [Antoine Madrona](https://github.com/madtoinou).
  - Fixed the comment of `scorers_are_univariate` in class `AnomalyModel`. [#2452](https://github.com/unit8co/darts/pull/2542) by [He Weilin](https://github.com/cnhwl).

**Dependencies**

- Bumped release requirements versions for jupyterlab and dependencies : [#2515](https://github.com/unit8co/darts/pull/2515) by [Dennis Bader](https://github.com/dennisbader).
  - Bumped `ipython` from 8.10.0 to 8.18.1
  - Bumped `ipykernel` from 5.3.4 to 6.29.5
  - Bumped `ipywidgets` from 7.5.1 to 8.1.5
  - Bumped `jupyterlab` from 4.0.11 to 4.2.5

### For developers of the library:

## [0.30.0](https://github.com/unit8co/darts/tree/0.30.0) (2024-06-19)

### For users of the library:

**Improved**

- 🚀🚀 All `GlobalForecastingModel` now support being trained with sample weights (regression-, ensemble-, and neural network models) : [#2404](https://github.com/unit8co/darts/pull/2404), [#2410](https://github.com/unit8co/darts/pull/2410), [#2417](https://github.com/unit8co/darts/pull/2417) and [#2418](https://github.com/unit8co/darts/pull/2418) by [Anton Ragot](https://github.com/AntonRagot) and [Dennis Bader](https://github.com/dennisbader).
  - Added parameters `sample_weight` and `val_sample_weight` to `fit()`, `historical_forecasts()`, `backtest()`, `residuals`, and `gridsearch()` to apply weights to each observation, label (each step in the output chunk), and target component in the training and evaluation set. Supported by both deterministic and probabilistic models. The sample weight can either be `TimeSeries` themselves or built-in weight generators "linear" and "exponential" decay. In case of a `TimeSeries` it is handled identically as the covariates (e.g. pass multiple weight series with multiple target series, relevant time frame extraction is handled automatically for you, ...). You can find an example [here](https://unit8co.github.io/darts/quickstart/00-quickstart.html#Sample-Weights).
- 🚀🚀 Improvements to the Anomaly Detection Module through major refactor. The refactor includes major performance optimization for the majority of processes and improvements to the API, consistency, reliability, and the documentation. Some of these necessary changes come at the cost of breaking changes : [#1477](https://github.com/unit8co/darts/pull/1477) by [Dennis Bader](https://github.com/dennisbader), [Samuele Giuliano Piazzetta](https://github.com/piaz97), [Antoine Madrona](https://github.com/madtoinou), [Julien Herzen](https://github.com/hrzn), [Julien Adda](https://github.com/julien12234).
  - Added an [example notebook](https://unit8co.github.io/darts/examples/22-anomaly-detection-examples.html) that showcases how to use Darts for Time Series Anomaly Detection.
  - Added a new dataset `TaxiNewYorkDataset` for anomaly detection with the number of taxi passengers in New York from the years 2014 and 2015.
  - `FittableWindowScorer` (KMeans, PyOD, and Wasserstein Scorers) now accept any of darts [&#34;per-time&#34; step metrics](https://unit8co.github.io/darts/generated_api/darts.metrics.html) as difference function `diff_fn`.
  - `ForecastingAnomalyModel` is now much faster in generating forecasts (input for the scorers) thanks to optimized historical forecasts. We also added more control over the historical forecasts generation through additional parameters in all model methods.
  - 🔴 Breaking changes:
    - `FittableWindowScorer` (KMeans, PyOD, and Wasserstein Scorers) now expects `diff_fn` to be one of Darts "per-time" step metrics
    - `ForecastingAnomalyModel` : `model` is now enforced to be a `GlobalForecastingModel`
    - `*.eval_accuracy()` : (Aggregators, Detectors, Filtering/Forecasting Anomaly Models, Scorers)
      - renamed method to `eval_metric()` :
      - renamed params `actual_anomalies` to `anomalies`, and `anomaly_score` to `pred_scores`
    - `*.show_anomalies()` : (Filtering/Forecasting Anomaly Models, Scorers)
      - renamed params `actual_anomalies` to `anomalies`
    - `*.fit()` (Filtering/Forecasting Anomaly Models)
      - renamed params `actual_anomalies` to `anomalies`
    - `Scorer.*_from_prediction()` (Scorers)
      - renamed method `eval_accuracy_from_prediction()` to `eval_metric_from_prediction()`
      - renamed params `actual_series` to `series`, and `actual_anomalies` to `anomalies`
    - `darts.ad.utils.eval_accuracy_from_scores` :
      - renamed function to `eval_metric_from_scores`
      - renamed params `actual_anoamlies` to `anomalies`, and `anomaly_score` to `pred_scores`
    - `darts.ad.utils.eval_accuracy_from_binary_prediction` :
      - renamed function to `eval_metric_from_binary_prediction`
      - renamed params `actual_anoamlies` to `anomalies`, and `binary_pred_anomalies` to `pred_anomalies`
    - `darts.ad.utils.show_anomalies_from_scores` :
      - renamed params `series` to `actual_series`, `actual_anomalies` to `anomalies`, `model_output` to `pred_series`, and `anomaly_scores` to `pred_scores`
- Improvements to `TorchForecastingModel` : [#2295](https://github.com/unit8co/darts/pull/2295) by [Bohdan Bilonoh](https://github.com/BohdanBilonoh).
  - Added `dataloader_kwargs` parameters to `fit*()`, `predict*()`, and `find_lr()` for more control over the PyTorch `DataLoader` setup.
  - 🔴 Removed parameter `num_loader_workers` from `fit*()`, `predict*()`, `find_lr()`. You can now set the parameter through the `dataloader_kwargs` dict.
- Improvements to `DataTransformers` :
  - Significant speed up when using `fit`, `fit_transform`, `transform`, and `inverse_transform` on a large number of series. The component masking logic was moved into the parallelized transform methods. [#2401](https://github.com/unit8co/darts/pull/2401) by [Dennis Bader](https://github.com/dennisbader).
- Improvements to `TimeSeries` : [#1477](https://github.com/unit8co/darts/pull/1477) by [Dennis Bader](https://github.com/dennisbader).
  - New method `with_times_and_values()`, which returns a new series with a new time index and new values but with identical columns and metadata as the series called from (static covariates, hierarchy).
  - New method `slice_intersect_times()`, which returns the sliced time index of a series, where the index has been intersected with another series.
  - Method `with_values()` now also acts on array-like `values` rather than only on numpy arrays.
- Improvements to quick start notebook : [#2418](https://github.com/unit8co/darts/pull/2418) by [Dennis Bader](https://github.com/dennisbader).
  - Added examples for using sample weights, forecast start shifting, direct likelihood parameter predictions.
  - Enhanced examples for historical forecasts, backtest and residuals.

**Fixed**

- Fixed a bug when using a `RegressionModel` (that supports validation series) with a validation set: encoders, static covariates, and component-specific lags are now correctly applied to the validation set. [#2383](https://github.com/unit8co/darts/pull/2383) by [Dennis Bader](https://github.com/dennisbader).
- Fixed a bug where `darts.utils.utils.n_steps_between()` did not work properly with custom business frequencies. This affected metrics computation. [#2357](https://github.com/unit8co/darts/pull/2357) by [Dennis Bader](https://github.com/dennisbader).
- Fixed a bug when calling `predict()` with a `MixedCovariatesTorchModel` (e.g. TiDE, N/DLinear, ...), `n<output_chunk_length` and a list of series with length `len(series) < n`, where the predictions did not return the correct number of series. [#2374](https://github.com/unit8co/darts/pull/2374) by [Dennis Bader](https://github.com/dennisbader).
- Fixed a bug when using a `TorchForecastingModel` with stateful torch metrics, where the metrics were incorrectly computed as non-stateful. [#2391](https://github.com/unit8co/darts/pull/2391) by [Tim Rosenflanz](https://github.com/tRosenflanz)

**Dependencies**

- We set an upper version cap on `numpy<2.0.0` until all dependencies have migrated. [#2413](https://github.com/unit8co/darts/pull/2413) by [Dennis Bader](https://github.com/dennisbader).

### For developers of the library:

**Dependencies**

- Improvements to linting via updated pre-commit configurations: [#2324](https://github.com/unit8co/darts/pull/2324) by [Jirka Borovec](https://github.com/borda).
- Improvements to unified linting by switch `isort` to Ruff's rule I. [#2339](https://github.com/unit8co/darts/pull/2339) by [Jirka Borovec](https://github.com/borda)
- Improvements to unified linting by switch `pyupgrade` to Ruff's rule UP. [#2340](https://github.com/unit8co/darts/pull/2340) by [Jirka Borovec](https://github.com/borda)
- Improvements to CI, running lint locally via pre-commit instead of particular tools. [#2327](https://github.com/unit8co/darts/pull/2327) by [Jirka Borovec](https://github.com/borda)

## [0.29.0](https://github.com/unit8co/darts/tree/0.29.0) (2024-04-17)

### For users of the library:

**Improved**

- 🚀🚀 New forecasting model: `TSMixerModel`  as proposed in [this paper](https://arxiv.org/abs/2303.06053). An MLP based model that combines temporal, static and cross-sectional feature information using stacked mixing layers. [#2293](https://github.com/unit8co/darts/pull/2293), by [Dennis Bader](https://github.com/dennisbader) and [Cristof Rojas](https://github.com/cristof-r).
- 🚀🚀 Improvements to metrics, historical forecasts, backtest, and residuals through major refactor. The refactor includes optimization of multiple process and improvements to consistency, reliability, and the documentation. Some of these necessary changes come at the cost of breaking changes. [#2284](https://github.com/unit8co/darts/pull/2284) by [Dennis Bader](https://github.com/dennisbader).
  - **Metrics**:
    - Optimized all metrics, which now run **> n * 20 times faster** than before for series with `n` components/columns. This boosts direct metric computations as well as backtest and residuals computation!
    - Added new metrics:
      - Time aggregated metric `merr()` (Mean Error)
      - Time aggregated scaled metrics  `rmsse()`, and `msse()` : The (Root) Mean Squared Scaled Error.
      - "Per time step" metrics that return a metric score per time step: `err()` (Error), `ae()` (Absolute Error), `se()` (Squared Error), `sle()` (Squared Log Error), `ase()` (Absolute Scaled Error), `sse` (Squared Scaled Error), `ape()` (Absolute Percentage Error), `sape()` (symmetric Absolute Percentage Error), `arre()` (Absolute Ranged Relative Error), `ql` (Quantile Loss)
    - All scaled metrics (`mase()`, ...) now accept `insample` series that can be overlapping into `pred_series` (before they had to end exactly one step before `pred_series`).  Darts will handle the correct time extraction for you.
    - Improvements to the documentation:
      - Added a summary list of all metrics to the [metrics documentation page](https://unit8co.github.io/darts/generated_api/darts.metrics.html)
      - Standardized the documentation of each metric (added formula, improved return documentation, ...)
    - 🔴 Breaking changes:
      - Improved metric output consistency based on the type of input `series`, and the applied reductions. For some scenarios, the output type changed compared to previous Darts versions. You can find a detailed description in the [metric API documentation](https://unit8co.github.io/darts/generated_api/darts.metrics.metrics.html#darts.metrics.metrics.mae).
      - Renamed metric parameter `reduction` to `component_reduction`.
      - Renamed metric parameter `inter_reduction` to `series_reduction`.
      - `quantile_loss()` :
        - Renamed to `mql()` (Mean Quantile Loss).
        - Renamed quantile parameter `tau` to `q`.
        - The metric is now multiplied by a factor `2` to make the loss more interpretable (e.g. for `q=0.5` it is identical to the `MAE`)
      - `rho_risk()` :
        - Renamed to `qr()` (Quantile Risk).
        - Renamed quantile parameter `rho` to `q`.
      - Scaled metrics do not allow seasonality inference anymore with `m=None`.
      - Custom metrics using decorators `multi_ts_support` and `multivariate_support` must now act on multivariate series (possibly containing missing values) instead of univariate series.
  - **Historical Forecasts**:
    - 🔴 Improved historical forecasts output consistency based on the type of input `series` : If `series` is a sequence, historical forecasts will now always return a sequence/list of the same length (instead of trying to reduce to a `TimeSeries` object). You can find a detailed description in the [historical forecasts API documentation](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.linear_regression_model.html#darts.models.forecasting.linear_regression_model.LinearRegressionModel.historical_forecasts).
  - **Backtest**:
    - Metrics are now computed only once on all `series` and `historical_forecasts`, significantly speeding things up when using a large number of `series`.
    - Added support for scaled metrics as `metric` (such as `ase`, `mase`, ...). No extra code required, backtest extracts the correct `insample` series for you.
    - Added support for passing additional metric (-specific) arguments with parameter `metric_kwargs`. This allows for example to parallelize the metric computation with `n_jobs`, customize the metric reduction with `*_reduction`, specify seasonality `m` for scaled metrics, etc.
    - 🔴 Breaking changes:
      - Improved backtest output consistency based on the type of input `series`, `historical_forecast`, and the applied backtest reduction. For some scenarios, the output type changed compared to previous Darts versions. You can find a detailed description in the [backtest API documentation](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.linear_regression_model.html#darts.models.forecasting.linear_regression_model.LinearRegressionModel.backtest).
      - `reduction` callable now acts on `axis=1` rather than `axis=0` to aggregate the metrics per series.
      - Backtest will now raise an error when user supplied `historical_forecasts` don't have the expected format based on input `series` and the `last_points_only` value.
  - **Residuals**: While the default behavior of `residuals()` remains identical, the method is now very similar to `backtest()` but that it computes any "per time step" `metric` on `historical_forecasts` :
    - Added support for multivariate `series`.
    - Added support for all `historical_forecasts()` parameters to generate the historical forecasts for the residuals computation.
    - Added support for pre-computed historical forecasts with parameter `historical_forecasts`.
    - Added support for computing the residuals with any of Darts' "per time step" metric with parameter `metric` (e.g. `err()`, `ae()`, `ape()`, ...). By default, uses `err()` (Error).
    - Added support for passing additional metric arguments with parameter `metric_kwargs`. This allows for example to parallelize the metric computation with `n_jobs`, specify seasonality `m` for scaled metrics, etc.
    - 🔴 Improved residuals output and consistency based on the type of input `series` and `historical_forecast`. For some scenarios, the output type changed compared to previous Darts versions. You can find a detailed description in the [residuals API documentation](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.linear_regression_model.html#darts.models.forecasting.linear_regression_model.LinearRegressionModel.residuals).
- Improvements to `TimeSeries` :
  - `from_group_dataframe()` now supports parallelized creation over the `pandas.DataFrame` groups. This can be enabled with parameter `n_jobs`. [#2292](https://github.com/unit8co/darts/pull/2292) by [Bohdan Bilonoha](https://github.com/BohdanBilonoh).
  - New method `slice_intersect_values()`, which returns the sliced values of a series, where the time index has been intersected with another series. [#2284](https://github.com/unit8co/darts/pull/2284) by [Dennis Bader](https://github.com/dennisbader).
  - Performance boost for methods: `slice_intersect()`, `has_same_time_as()`. [#2284](https://github.com/unit8co/darts/pull/2284) by [Dennis Bader](https://github.com/dennisbader).
- Improvements to forecasting models:
  - Improvements to `RNNModel`, [#2329](https://github.com/unit8co/darts/pull/2329) by [Dennis Bader](https://github.com/dennisbader):
    - 🔴 Enforce `training_length>input_chunk_length` since otherwise, during training the model is never run for as many iterations as it will during prediction.
    - Historical forecasts now correctly infer all possible prediction start points for untrained and pre-trained `RNNModel`.
  - Added a progress bar to `RegressionModel` when performing optimized historical forecasts (`retrain=False` and no autoregression) to display the series-level progress. [#2320](https://github.com/unit8co/darts/pull/2320) by [Dennis Bader](https://github.com/dennisbader).
  - Renamed private `ForecastingModel._is_probabilistic` property to public `supports_probabilistic_prediction`. [#2269](https://github.com/unit8co/darts/pull/2269) by [Felix Divo](https://github.com/felixdivo).
- Other improvements:
  - All `InvertibleDataTransformer` now supports parallelized inverse transformation for `series` being a list of lists of `TimeSeries` (`Sequence[Sequence[TimeSeries]]`). This type represents the output of `historical_forecasts()` when using multiple series with `last_points_only=False`. [#2267](https://github.com/unit8co/darts/pull/2267) by [Alicja Krzeminska-Sciga](https://github.com/alicjakrzeminska).
  - Added [release notes](https://unit8co.github.io/darts/release_notes/RELEASE_NOTES.html) to the Darts Documentation. [#2333](https://github.com/unit8co/darts/pull/2333) by [Dennis Bader](https://github.com/dennisbader).
  - 🔴 Moved around utils functions to clearly separate Darts-specific from non-Darts-specific logic, [#2284](https://github.com/unit8co/darts/pull/2284) by [Dennis Bader](https://github.com/dennisbader):
    - Moved function `generate_index()` from `darts.utils.timeseries_generation` to `darts.utils.utils`
    - Moved functions `retain_period_common_to_all()`, `series2seq()`, `seq2series()`, `get_single_series()` from `darts.utils.utils` to `darts.utils.ts_utils`.

**Fixed**

- Fixed the order of the features when using component-specific lags so that they are grouped by values, then by components (before, they were grouped by components, then by values). [#2272](https://github.com/unit8co/darts/pull/2272) by [Antoine Madrona](https://github.com/madtoinou).
- Fixed a bug when using a dropout with a `TorchForecastingModel` and pytorch lightning versions >= 2.2.0, where the dropout was not properly activated during training. [#2312](https://github.com/unit8co/darts/pull/2312) by [Dennis Bader](https://github.com/dennisbader).
- Fixed a bug when performing historical forecasts with an untrained `TorchForecastingModel` and using covariates, where the historical forecastable time index generation did not take the covariates into account. [#2329](https://github.com/unit8co/darts/pull/2329) by [Dennis Bader](https://github.com/dennisbader).
- Fixed a bug in `quantile_loss`, where the loss was computed on all samples rather than only on the predicted quantiles. [#2284](https://github.com/unit8co/darts/pull/2284) by [Dennis Bader](https://github.com/dennisbader).
- Fixed a segmentation fault that some users were facing when importing a `LightGBMModel`. [#2304](https://github.com/unit8co/darts/pull/2304) by [Dennis Bader](https://github.com/dennisbader).
- Fixed type hint warning "Unexpected argument" when calling `historical_forecasts()` caused by the `_with_sanity_checks` decorator. The type hinting is now properly configured to expect any input arguments and return the output type of the method for which the sanity checks are performed for. [#2286](https://github.com/unit8co/darts/pull/2286) by [Dennis Bader](https://github.com/dennisbader).

### For developers of the library:

- Fixed failing docs build by adding new dependency `lxml_html_clean` for `nbsphinx`. [#2303](https://github.com/unit8co/darts/pull/2303) by [Dennis Bader](https://github.com/dennisbader).
- Bumped `black` from 24.1.1 to 24.3.0. [#2308](https://github.com/unit8co/darts/pull/2308) by [Dennis Bader](https://github.com/dennisbader).
- Bumped `codecov-action` from v2 to v4 and added codecov token as repository secret for codecov upload authentication in CI pipelines. [#2309](https://github.com/unit8co/darts/pull/2309) and [#2312](https://github.com/unit8co/darts/pull/2312) by [Dennis Bader](https://github.com/dennisbader).
- Improvements to linting, switch from `flake8` to Ruff. [#2323](https://github.com/unit8co/darts/pull/2323) by [Jirka Borovec](https://github.com/borda).

## [0.28.0](https://github.com/unit8co/darts/tree/0.28.0) (2024-03-05)

### For users of the library:

**Improved**

- Improvements to `GlobalForecastingModel` :
  - 🚀🚀🚀 All global models (regression and torch models) now support shifted predictions with model creation parameter `output_chunk_shift`. This will shift the output chunk for training and prediction by `output_chunk_shift` steps into the future. [#2176](https://github.com/unit8co/darts/pull/2176) by [Dennis Bader](https://github.com/dennisbader).
- Improvements to `TimeSeries`, [#2196](https://github.com/unit8co/darts/pull/2196) by [Dennis Bader](https://github.com/dennisbader):
  - 🚀🚀🚀 Significant performance boosts for several `TimeSeries` methods resulting increased efficiency across the entire `Darts` library. Up to 2x faster creation times for series indexed with "regular" frequencies (e.g. Daily, hourly, ...), and >100x for series indexed with "special" frequencies (e.g. "W-MON", ...). Affects:
    - All `TimeSeries` creation methods
    - Additional boosts for slicing with integers and Timestamps
    - Additional boosts for `from_group_dataframe()` by performing some of the heavy-duty computations on the entire DataFrame, rather than iteratively on the group level.
  - Added option to exclude some `group_cols` from being added as static covariates when using `TimeSeries.from_group_dataframe()` with parameter `drop_group_cols`.
- 🚀 New global baseline models that use fixed input and output chunks for prediction. This offers support for univariate, multivariate, single and multiple target series prediction, one-shot- or autoregressive/moving forecasts, optimized historical forecasts, batch prediction, prediction from datasets, and more. [#2261](https://github.com/unit8co/darts/pull/2261) by [Dennis Bader](https://github.com/dennisbader).
  - `GlobalNaiveAggregate` : Computes an aggregate (using a custom or built-in `torch` function) for each target component over the last `input_chunk_length` points, and repeats the values `output_chunk_length` times for prediction. Depending on the parameters, this model can be equivalent to `NaiveMean` and `NaiveMovingAverage`.
  - `GlobalNaiveDrift` : Takes the slope of each target component over the last `input_chunk_length` points and projects the trend over the next `output_chunk_length` points for prediction. Depending on the parameters, this model can be equivalent to `NaiveDrift`.
  - `GlobalNaiveSeasonal` : Takes the target component value at the `input_chunk_length`th point before the end of the target `series`, and repeats the values `output_chunk_length` times for prediction. Depending on the parameters, this model can be equivalent to `NaiveSeasonal`.
- Improvements to `TorchForecastingModel` :
  - Added support for additional lr scheduler configuration parameters for more control ("interval", "frequency", "monitor", "strict", "name"). [#2218](https://github.com/unit8co/darts/pull/2218) by [Dennis Bader](https://github.com/dennisbader).
- Improvements to `RegressionModel`, [#2246](https://github.com/unit8co/darts/pull/2246) by [Antoine Madrona](https://github.com/madtoinou):
  - Added a `get_estimator()` method to access the underlying estimator
  - Added attribute `lagged_label_names` to identify the forecasted step and component of each estimator
  - Updated the docstring of `get_multioutout_estimator()`
- Other improvements:
  - Added argument `keep_names` to `WindowTransformer` and `window_transform` to indicate whether the original component names should be kept. [#2207](https://github.com/unit8co/darts/pull/2207) by [Antoine Madrona](https://github.com/madtoinou).
  - Added new helper function `darts.utils.utils.n_steps_between()` to efficiently compute the number of time steps (periods) between two points with a given frequency. Improves efficiency for regression model tabularization by avoiding `pd.date_range()`. [#2176](https://github.com/unit8co/darts/pull/2176) by [Dennis Bader](https://github.com/dennisbader).
  - 🔴 Changed the default `start` value in `ForecastingModel.gridsearch()` from `0.5` to `None`, to make it consistent with `historical_forecasts` and other methods. [#2243](https://github.com/unit8co/darts/pull/2243) by [Thomas Kientz](https://github.com/thomktz).
  - Improvements to `ARIMA` documentation: Specified possible `p`, `d`, `P`, `D`, `trend` advanced options that are available in statsmodels. More explanations on the behaviour of the parameters were added. [#2142](https://github.com/unit8co/darts/pull/2142) by [MarcBresson](https://github.com/MarcBresson).

**Fixed**

- Fixed a bug when using `RegressionModel` with `lags=None`, some `lags_*covariates`, and the covariates starting after or at the same time as the first predictable time step; the lags were not extracted from the correct indices. [#2176](https://github.com/unit8co/darts/pull/2176) by [Dennis Bader](https://github.com/dennisbader).
- Fixed a bug when calling `window_transform` on a `TimeSeries` with a hierarchy. The hierarchy is now only preserved for single transformations applied to all components, or removed otherwise. [#2207](https://github.com/unit8co/darts/pull/2207) by [Antoine Madrona](https://github.com/madtoinou).
- Fixed a bug in probabilistic `LinearRegressionModel.fit()`, where the `model` attribute was not pointing to all underlying estimators. [#2205](https://github.com/unit8co/darts/pull/2205) by [Antoine Madrona](https://github.com/madtoinou).
- Raise an error in `RegressionEsembleModel` when the `regression_model` was created with `multi_models=False` (not supported). [#2205](https://github.com/unit8co/darts/pull/2205) by [Antoine Madrona](https://github.com/madtoinou).
- Fixed a bug in `coefficient_of_variation()` with `intersect=True`, where the coefficient was not computed on the intersection. [#2202](https://github.com/unit8co/darts/pull/2202) by [Antoine Madrona](https://github.com/madtoinou).
- Fixed a bug in `gridsearch()` with `use_fitted_values=True`, where the model was not properly instantiated for sanity checks. [#2222](https://github.com/unit8co/darts/pull/2222) by [Antoine Madrona](https://github.com/madtoinou).
- Fixed a bug in `TimeSeries.append/prepend_values()`, where the component names and the hierarchy were dropped. [#2237](https://github.com/unit8co/darts/pull/2237) by [Antoine Madrona](https://github.com/madtoinou).
- Fixed a bug in `get_multioutput_estimator()`, where the index of the estimator was incorrectly calculated. [#2246](https://github.com/unit8co/darts/pull/2246) by [Antoine Madrona](https://github.com/madtoinou).
- 🔴 Fixed a bug in `datetime_attribute_timeseries()`, where 1-indexed attributes were not properly handled. Also, 0-indexing is now enforced for all the generated encodings. [#2242](https://github.com/unit8co/darts/pull/2242) by [Antoine Madrona](https://github.com/madtoinou).

**Dependencies**

- Removed upper version cap (<=v2.1.2) for PyTorch Lightning. [#2251](https://github.com/unit8co/darts/pull/2251) by [Dennis Bader](https://github.com/dennisbader).

### For developers of the library:

- Updated pre-commit hooks to the latest version using `pre-commit autoupdate`. Change `pyupgrade` pre-commit hook argument to `--py38-plus`. [#2228](https://github.com/unit8co/darts/pull/2228)  by [MarcBresson](https://github.com/MarcBresson).
- Bumped dev dependencies to newest versions, [#2248](https://github.com/unit8co/darts/pull/2248) by [Dennis Bader](https://github.com/dennisbader):
  - black[jupyter]: from 22.3.0 to 24.1.1
  - flake8: from 4.0.1 to 7.0.0
  - isort: from 5.11.5 to 5.13.2
  - pyupgrade: 2.31.0 from to v3.15.0

## [0.27.2](https://github.com/unit8co/darts/tree/0.27.2) (2024-01-21)

### For users of the library:

**Improved**

- Added `darts.utils.statistics.plot_ccf` that can be used to plot the cross correlation between a time series (e.g. target series) and the lagged values of another time series (e.g. covariates series). [#2122](https://github.com/unit8co/darts/pull/2122) by [Dennis Bader](https://github.com/dennisbader).
- Improvements to `TimeSeries` : Improved the time series frequency inference when using slices or pandas DatetimeIndex as keys for `__getitem__`. [#2152](https://github.com/unit8co/darts/pull/2152) by [DavidKleindienst](https://github.com/DavidKleindienst).

**Fixed**

- Fixed a bug when using a `TorchForecastingModel` with `use_reversible_instance_norm=True` and predicting with `n > output_chunk_length`. The input normalized multiple times. [#2160](https://github.com/unit8co/darts/pull/2160) by [FourierMourier](https://github.com/FourierMourier).

### For developers of the library:

## [0.27.1](https://github.com/unit8co/darts/tree/0.27.1) (2023-12-10)

### For users of the library:

**Improved**

- 🔴 Added `CustomRNNModule` and `CustomBlockRNNModule` for defining custom RNN modules that can be used with `RNNModel` and `BlockRNNModel`. The custom `model` must now be a subclass of the custom modules. [#2088](https://github.com/unit8co/darts/pull/2088) by [Dennis Bader](https://github.com/dennisbader).

**Fixed**

- Fixed a bug in historical forecasts, where some `fit/predict_kwargs` were not passed to the underlying model's fit/predict methods. [#2103](https://github.com/unit8co/darts/pull/2103) by [Dennis Bader](https://github.com/dennisbader).
- Fixed an import error when trying to create a `TorchForecastingModel` with PyTorch Lightning v<2.0.0. [#2087](https://github.com/unit8co/darts/pull/2087) by [Eschibli](https://github.com/eschibli).
- Fixed a bug when creating a `RNNModel` with a custom `model`. [#2088](https://github.com/unit8co/darts/pull/2088) by [Dennis Bader](https://github.com/dennisbader).

### For developers of the library:

- Added a folder `docs/generated_api` to define custom .rst files for generating the documentation. [#2115](https://github.com/unit8co/darts/pull/2115) by [Dennis Bader](https://github.com/dennisbader).

## [0.27.0](https://github.com/unit8co/darts/tree/0.27.0) (2023-11-18)

### For users of the library:

**Improved**

- Improvements to `TorchForecastingModel` :
  - 🚀🚀 We optimized `historical_forecasts()` for pre-trained `TorchForecastingModel` running up to 20 times faster than before (and even more when tuning the batch size)!. [#2013](https://github.com/unit8co/darts/pull/2013) by [Dennis Bader](https://github.com/dennisbader).
  - Added callback `darts.utils.callbacks.TFMProgressBar` to customize at which model stages to display the progress bar. [#2020](https://github.com/unit8co/darts/pull/2020) by [Dennis Bader](https://github.com/dennisbader).
  - All `InferenceDataset`s now support strided forecasts with parameters `stride`, `bounds`. These datasets can be used with `TorchForecastingModel.predict_from_dataset()`. [#2013](https://github.com/unit8co/darts/pull/2013) by [Dennis Bader](https://github.com/dennisbader).
- Improvements to `RegressionModel` :
  - New example notebook for the `RegressionModels` explaining features such as (component-specific) lags, `output_chunk_length` in relation with `multi_models`, multivariate support, and more. [#2039](https://github.com/unit8co/darts/pull/2039) by [Antoine Madrona](https://github.com/madtoinou).
  - `XGBModel` now leverages XGBoost's native Quantile Regression support that was released in version 2.0.0 for improved probabilistic forecasts. [#2051](https://github.com/unit8co/darts/pull/2051) by [Dennis Bader](https://github.com/dennisbader).
- Improvements to `LocalForecastingModel`
  - Added optional keyword arguments dict `kwargs` to `ExponentialSmoothing` that will be passed to the constructor of the underlying `statsmodels.tsa.holtwinters.ExponentialSmoothing` model. [#2059](https://github.com/unit8co/darts/pull/2059) by [Antoine Madrona](https://github.com/madtoinou).
- General model improvements:
  - Added new arguments `fit_kwargs` and `predict_kwargs` to `historical_forecasts()`, `backtest()` and `gridsearch()` that will be passed to the model's `fit()` and / or `predict` methods. E.g., you can now set a batch size, static validation series, ... depending on the model support. [#2050](https://github.com/unit8co/darts/pull/2050) by [Antoine Madrona](https://github.com/madtoinou)
  - For transparency, we issue a (removable) warning when performing auto-regressive forecasts with past covariates (with `n >= output_chunk_length`) to inform users that future values of past covariates will be accessed. [#2049](https://github.com/unit8co/darts/pull/2049) by [Antoine Madrona](https://github.com/madtoinou)
- Other improvements:
  - Added support for time index time zone conversion with parameter `tz` before generating/computing holidays and datetime attributes. Support was added to all Time Axis Encoders, standalone encoders and forecasting models' `add_encoders`, time series generation utils functions `holidays_timeseries()` and `datetime_attribute_timeseries()`, and `TimeSeries` methods `add_datetime_attribute()` and `add_holidays()`. [#2054](https://github.com/unit8co/darts/pull/2054) by [Dennis Bader](https://github.com/dennisbader).
  - Added new data transformer: `MIDAS`, which uses mixed-data sampling to convert `TimeSeries` from high frequency to low frequency (and back). [#1820](https://github.com/unit8co/darts/pull/1820) by [Boyd Biersteker](https://github.com/Beerstabr), [Antoine Madrona](https://github.com/madtoinou) and [Dennis Bader](https://github.com/dennisbader).
  - Added new dataset `ElectricityConsumptionZurichDataset` : The dataset contains the electricity consumption of households in Zurich, Switzerland from 2015-2022 on different grid levels. We also added weather measurements for Zurich which can be used as covariates for modelling. [#2039](https://github.com/unit8co/darts/pull/2039) by [Antoine Madrona](https://github.com/madtoinou) and [Dennis Bader](https://github.com/dennisbader).
  - Adapted the example notebooks to properly apply data transformers and avoid look-ahead bias. [#2020](https://github.com/unit8co/darts/pull/2020) by [Samriddhi Singh](https://github.com/SimTheGreat).

**Fixed**

- Fixed a bug when calling `historical_forecasts()` and `overlap_end=False` that did not generate the last possible forecast. [#2013](https://github.com/unit8co/darts/pull/2013) by [Dennis Bader](https://github.com/dennisbader).
- Fixed a bug when calling optimized `historical_forecasts()` for a `RegressionModel` trained with varying component-specific lags. [#2040](https://github.com/unit8co/darts/pull/2040) by [Antoine Madrona](https://github.com/madtoinou).
- Fixed a bug when using encoders with `RegressionModel` and series with a non-evenly spaced frequency (e.g. Month Begin). This raised an error during lagged data creation when trying to divide a pd.Timedelta by the ambiguous frequency. [#2034](https://github.com/unit8co/darts/pull/2034) by [Antoine Madrona](https://github.com/madtoinou).
- Fixed a bug when loading the weights of a `TorchForecastingModel` that was trained with a precision other than `float64`. [#2046](https://github.com/unit8co/darts/pull/2046) by [Freddie Hsin-Fu Huang](https://github.com/Hsinfu).
- Fixed broken links in the `Transfer learning` example notebook with publicly hosted version of the three datasets. [#2067](https://github.com/unit8co/darts/pull/2067) by [Antoine Madrona](https://github.com/madtoinou).
- Fixed a bug when using `NLinearModel` on multivariate series with covariates and `normalize=True`. [#2072](https://github.com/unit8co/darts/pull/2072) by [Antoine Madrona](https://github.com/madtoinou).
- Fixed a bug when using `DLinearModel` and `NLinearModel` on multivariate series with static covariates shared across components and `use_static_covariates=True`. [#2070](https://github.com/unit8co/darts/pull/2070) by [Antoine Madrona](https://github.com/madtoinou).

### For developers of the library:

No changes.

## [0.26.0](https://github.com/unit8co/darts/tree/0.26.0) (2023-09-16)

### For users of the library:

**Improved**

- Improvements to `RegressionModel`, [#1962](https://github.com/unit8co/darts/pull/1962) by [Antoine Madrona](https://github.com/madtoinou):
  - 🚀🚀 All models now support component/column-specific lags for target, past, and future covariates series.
- Improvements to `TorchForecastingModel` :
  - 🚀 Added `RINorm` (Reversible Instance Norm) as an input normalization option for all models except `RNNModel`. Activate it with model creation parameter `use_reversible_instance_norm`. [#1969](https://github.com/unit8co/darts/pull/1969) by [Dennis Bader](https://github.com/dennisbader).
  - 🔴 Added past covariates feature projection to `TiDEModel` with parameter `temporal_width_past` following the advice of the model architect. Parameter `temporal_width` was renamed to `temporal_width_future`. Additionally, added the option to bypass the feature projection with `temporal_width_past/future=0`. [#1993](https://github.com/unit8co/darts/pull/1993) by [Dennis Bader](https://github.com/dennisbader).
- Improvements to `EnsembleModel`, [#1815](https://github.com/unit8co/darts/pull/#1815) by [Antoine Madrona](https://github.com/madtoinou) and [Dennis Bader](https://github.com/dennisbader):
  - 🔴 Renamed model constructor argument `models` to `forecasting_models`.
  - 🚀🚀 Added support for pre-trained `GlobalForecastingModel` as `forecasting_models` to avoid re-training when ensembling. This requires all models to be pre-trained global models.
  - 🚀 Added support for generating the `forecasting_model` forecasts (used to train the ensemble model) with historical forecasts rather than direct (auto-regressive) predictions. Enable it with `train_using_historical_forecasts=True` at model creation.
  - Added an example notebook for ensemble models.
- Improvements to historical forecasts, backtest and gridsearch, [#1866](https://github.com/unit8co/darts/pull/1866) by [Antoine Madrona](https://github.com/madtoinou):
  - Added support for negative `start` values to start historical forecasts relative to the end of the target series.
  - Added a new argument `start_format` that allows to use an integer `start` either as the index position or index value/label for `series` indexed with a `pd.RangeIndex`.
  - Added support for `TimeSeries` with a `RangeIndex` starting at a negative integer.
- Other improvements:
  - Reduced the size of the Darts docker image `unit8/darts:latest`, and included all optional models as well as dev requirements. [#1878](https://github.com/unit8co/darts/pull/1878) by [Alex Colpitts](https://github.com/alexcolpitts96).
  - Added short examples in the docstring of all the models, including covariates usage and some model-specific parameters. [#1956](https://github.com/unit8co/darts/pull/1956) by [Antoine Madrona](https://github.com/madtoinou).
  - Added method `TimeSeries.cumsum()` to get the cumulative sum of the time series along the time axis. [#1988](https://github.com/unit8co/darts/pull/1988) by [Eliot Zubkoff](https://github.com/Eliotdoesprogramming).

**Fixed**

- Fixed a bug in `TimeSeries.from_dataframe()` when using a pandas.DataFrame with `df.columns.name != None`. [#1938](https://github.com/unit8co/darts/pull/1938) by [Antoine Madrona](https://github.com/madtoinou).
- Fixed a bug in `RegressionEnsembleModel.extreme_lags` when the forecasting models have only covariates lags. [#1942](https://github.com/unit8co/darts/pull/1942) by [Antoine Madrona](https://github.com/madtoinou).
- Fixed a bug when using `TFTExplainer` with a `TFTModel` running on GPU. [#1949](https://github.com/unit8co/darts/pull/1949) by [Dennis Bader](https://github.com/dennisbader).
- Fixed a bug in `TorchForecastingModel.load_weights()` that raised an error when loading the weights from a valid architecture. [#1952](https://github.com/unit8co/darts/pull/1952) by [Antoine Madrona](https://github.com/madtoinou).
- Fixed a bug in `NLinearModel` where `normalize=True` and past covariates could not be used at the same time. [#1873](https://github.com/unit8co/darts/pull/1873) by [Eliot Zubkoff](https://github.com/Eliotdoesprogramming).
- Raise an error when an `EnsembleModel` containing at least one `LocalForecastingModel` is calling `historical_forecasts` with `retrain=False`.  [#1815](https://github.com/unit8co/darts/pull/#1815) by [Antoine Madrona](https://github.com/madtoinou).
- 🔴 Dropped support for lambda functions in `add_encoders`’s “custom” encoder in favor of named functions to ensure that models can be exported. [#1957](https://github.com/unit8co/darts/pull/1957) by [Antoine Madrona](https://github.com/madtoinou).

### For developers of the library:

**Improved**

- Refactored all tests to use pytest instead of unittest. [#1950](https://github.com/unit8co/darts/pull/1950) by [Dennis Bader](https://github.com/dennisbader).

## [0.25.0](https://github.com/unit8co/darts/tree/0.25.0) (2023-08-04)

### For users of the library:

**Installation**

- 🔴 Removed Prophet, LightGBM, and CatBoost dependencies from PyPI packages (`darts`, `u8darts`, `u8darts[torch]`), and conda-forge packages (`u8darts`, `u8darts-torch`)  to avoid installation issues that some users were facing (installation on Apple M1/M2 devices, ...). [#1589](https://github.com/unit8co/darts/pull/1589) by [Julien Herzen](https://github.com/hrzn) and [Dennis Bader](https://github.com/dennisbader).
  - The models are still supported by installing the required packages as described in our [installation guide](https://github.com/unit8co/darts/blob/master/INSTALL.md#enabling-optional-dependencies).
  - The Darts package including all dependencies can still be installed with PyPI package `u8darts[all]` or conda-forge package `u8darts-all`.
  - Added new PyPI flavor `u8darts[notorch]`, and conda-forge flavor `u8darts-notorch` which are equivalent to the old `u8darts` installation (all dependencies except neural networks).
- 🔴 Removed support for Python 3.7 [#1864](https://github.com/unit8co/darts/pull/1864) by [Dennis Bader](https://github.com/dennisbader).

**Improved**

- General model improvements:
  - 🚀🚀 Optimized `historical_forecasts()` for `RegressionModel` when `retrain=False` and `forecast_horizon <= output_chunk_length` by vectorizing the prediction. This can run up to 700 times faster than before! [#1885](https://github.com/unit8co/darts/pull/1885) by [Antoine Madrona](https://github.com/madtoinou).
  - Improved efficiency of `historical_forecasts()` and `backtest()` for all models giving significant process time reduction for larger number of predict iterations and series. [#1801](https://github.com/unit8co/darts/pull/1801) by [Dennis Bader](https://github.com/dennisbader).
  - 🚀🚀 Added support for direct prediction of the likelihood parameters to probabilistic models using a likelihood (regression and torch models). Set `predict_likelihood_parameters=True` when calling `predict()`. [#1811](https://github.com/unit8co/darts/pull/1811) by [Antoine Madrona](https://github.com/madtoinou).
  - 🚀🚀 New forecasting model: `TiDEModel`  as proposed in [this paper](https://arxiv.org/abs/2304.08424). An MLP based encoder-decoder model that is said to outperform many Transformer-based architectures. [#1727](https://github.com/unit8co/darts/pull/1727) by [Alex Colpitts](https://github.com/alexcolpitts96).
  - `Prophet` now supports conditional seasonalities, and properly handles all parameters passed to `Prophet.add_seasonality()` and model creation parameter `add_seasonalities` [#1829](https://github.com/unit8co/darts/pull/1829) by [Idan Shilon](https://github.com/id5h).
  - Added method `generate_fit_predict_encodings()` to generate the encodings (from `add_encoders` at model creation) required for training and prediction. [#1925](https://github.com/unit8co/darts/pull/1925) by [Dennis Bader](https://github.com/dennisbader).
  - Added support for `PathLike` to the `save()` and `load()` functions of all non-deep learning based models. [#1754](https://github.com/unit8co/darts/pull/1754) by [Simon Sudrich](https://github.com/sudrich).
  - Added model property `ForecastingModel.supports_multivariate` to indicate whether the model supports multivariate forecasting. [#1848](https://github.com/unit8co/darts/pull/1848) by [Felix Divo](https://github.com/felixdivo).
- Improvements to `EnsembleModel` :
  - Model creation parameter `forecasting_models` now supports a mix of `LocalForecastingModel` and `GlobalForecastingModel` (single `TimeSeries` training/inference only, due to the local models). [#1745](https://github.com/unit8co/darts/pull/1745) by [Antoine Madrona](https://github.com/madtoinou).
  - Future and past covariates can now be used even if `forecasting_models` have different covariates support. The covariates passed to `fit()`/`predict()` are used only by models that support it. [#1745](https://github.com/unit8co/darts/pull/1745) by [Antoine Madrona](https://github.com/madtoinou).
  - `RegressionEnsembleModel` and `NaiveEnsembleModel` can generate probabilistic forecasts, probabilistics `forecasting_models` can be sampled to train the `regression_model`, updated the documentation (stacking technique). [#1692](https://github.com/unit8co/darts/pull/1692) by [Antoine Madrona](https://github.com/madtoinou).
- Improvements to `Explainability` module:
  - 🚀🚀 New forecasting model explainer: `TFTExplainer` for `TFTModel`. You can now access and visualize the trained model's feature importances and self attention. [#1392](https://github.com/unit8co/darts/pull/1392) by [Sebastian Cattes](https://github.com/Cattes) and [Dennis Bader](https://github.com/dennisbader).
  - Added static covariates support to `ShapeExplainer`. [#1803](https://github.com/unit8co/darts/pull/1803) by [Anne de Vries](https://github.com/anne-devries) and [Dennis Bader](https://github.com/dennisbader).
- Improvements to documentation [#1904](https://github.com/unit8co/darts/pull/1904) by [Dennis Bader](https://github.com/dennisbader):
  - made model sections in README.md, covariates user guide and forecasting model API Reference more user friendly by adding model links and reorganizing them into model categories.
  - added the Dynamic Time Warping (DTW) module and improved its appearance.
- Other improvements:
  - Improved static covariates column naming when using `StaticCovariatesTransformer` with a `sklearn.preprocessing.OneHotEncoder`. [#1863](https://github.com/unit8co/darts/pull/1863) by [Anne de Vries](https://github.com/anne-devries).
  - Added `MSTL` (Season-Trend decomposition using LOESS for multiple seasonalities) as a `method` option for `extract_trend_and_seasonality()`. [#1879](https://github.com/unit8co/darts/pull/1879) by [Alex Colpitts](https://github.com/alexcolpitts96).
  - Added `RINorm` (Reversible Instance Norm) as a new input normalization option for `TorchForecastingModel`. So far only `TiDEModel` supports it with model creation parameter `use_reversible_instance_norm`. [#1865](https://github.com/unit8co/darts/pull/1856) by [Alex Colpitts](https://github.com/alexcolpitts96).
  - Improvements to `TimeSeries.plot()` : custom axes are now properly supported with parameter `ax`. Axis is now returned for downstream tasks. [#1916](https://github.com/unit8co/darts/pull/1916) by [Dennis Bader](https://github.com/dennisbader).

**Fixed**

- Fixed an issue not considering original component names for `TimeSeries.plot()` when providing a label prefix. [#1783](https://github.com/unit8co/darts/pull/1783) by [Simon Sudrich](https://github.com/sudrich).
- Fixed an issue with the string representation of `ForecastingModel` when using array-likes at model creation. [#1749](https://github.com/unit8co/darts/pull/1749) by [Antoine Madrona](https://github.com/madtoinou).
- Fixed an issue with `TorchForecastingModel.load_from_checkpoint()` not properly loading the loss function and metrics. [#1759](https://github.com/unit8co/darts/pull/1759) by [Antoine Madrona](https://github.com/madtoinou).
- Fixed a bug when loading the weights of a `TorchForecastingModel` trained with encoders or a Likelihood. [#1744](https://github.com/unit8co/darts/pull/1744) by [Antoine Madrona](https://github.com/madtoinou).
- Fixed a bug when using selected `target_components` with `ShapExplainer`. [#1803](https://github.com/unit8co/darts/pull/1803) by [Dennis Bader](https://github.com/dennisbader).
- Fixed `TimeSeries.__getitem__()` for series with a RangeIndex with start != 0 and freq != 1. [#1868](https://github.com/unit8co/darts/pull/1868) by [Dennis Bader](https://github.com/dennisbader).
- Fixed an issue where `DTWAlignment.plot_alignment()` was not plotting the alignment plot of series with a RangeIndex correctly. [#1880](https://github.com/unit8co/darts/pull/1880) by [Ahmet Zamanis](https://github.com/AhmetZamanis) and [Dennis Bader](https://github.com/dennisbader).
- Fixed an issue when calling `ARIMA.predict()` and `num_samples > 1` (probabilistic forecasting), where the start point of the simulation was not anchored to the end of the target series. [#1893](https://github.com/unit8co/darts/pull/1893) by [Dennis Bader](https://github.com/dennisbader).
- Fixed an issue when using `TFTModel.predict()` with `full_attention=True` where the attention mask was not applied properly. [#1392](https://github.com/unit8co/darts/pull/1392) by [Dennis Bader](https://github.com/dennisbader).

### For developers of the library:

**Improvements**

- Refactored the `ForecastingModelExplainer` and `ExplainabilityResult` to simplify implementation of new explainers. [#1392](https://github.com/unit8co/darts/pull/1392) by [Dennis Bader](https://github.com/dennisbader).
- Adapted all unit tests to run successfully on M1 devices. [#1933](https://github.com/unit8co/darts/pull/1933) by [Dennis Bader](https://github.com/dennisbader).

## [0.24.0](https://github.com/unit8co/darts/tree/0.24.0) (2023-04-12)

### For users of the library:

**Improved**

- General model improvements:
  - New baseline forecasting model `NaiveMovingAverage`. [#1557](https://github.com/unit8co/darts/pull/1557) by [Janek Fidor](https://github.com/JanFidor).
  - New models `StatsForecastAutoCES`, and `StatsForecastAutoTheta` from Nixtla's statsforecasts library as local forecasting models without covariates support. AutoTheta supports probabilistic forecasts. [#1476](https://github.com/unit8co/darts/pull/1476) by [Boyd Biersteker](https://github.com/Beerstabr).
  - Added support for future covariates, and probabilistic forecasts to `StatsForecastAutoETS`. [#1476](https://github.com/unit8co/darts/pull/1476) by [Boyd Biersteker](https://github.com/Beerstabr).
  - Added support for logistic growth to `Prophet` with parameters `growth`, `cap`, `floor`. [#1419](https://github.com/unit8co/darts/pull/1419) by [David Kleindienst](https://github.com/DavidKleindienst).
  - Improved the model string / object representation style similar to scikit-learn models. [#1590](https://github.com/unit8co/darts/pull/1590) by [Janek Fidor](https://github.com/JanFidor).
  - 🔴 Renamed `MovingAverage` to `MovingAverageFilter` to avoid confusion with new `NaiveMovingAverage` model. [#1557](https://github.com/unit8co/darts/pull/1557) by [Janek Fidor](https://github.com/JanFidor).
- Improvements to `RegressionModel` :
  - Optimized lagged data creation for fit/predict sets achieving a drastic speed-up. [#1399](https://github.com/unit8co/darts/pull/1399) by [Matt Bilton](https://github.com/mabilton).
  - Added support for categorical past/future/static covariates to `LightGBMModel` with model creation parameters `categorical_*_covariates`. [#1585](https://github.com/unit8co/darts/pull/1585) by [Rijk van der Meulen](https://github.com/rijkvandermeulen).
  - Added lagged feature names for better interpretability; accessible with model property `lagged_feature_names`. [#1679](https://github.com/unit8co/darts/pull/1679) by [Antoine Madrona](https://github.com/madtoinou).
  - 🔴 New `use_static_covariates` option for all models: When True (default), models use static covariates if available at fitting time and enforce identical static covariate shapes across all target `series` used for training or prediction; when False, models ignore static covariates. [#1700](https://github.com/unit8co/darts/pull/1700) by [Dennis Bader](https://github.com/dennisbader).
- Improvements to `TorchForecastingModel` :
  - New methods `load_weights()` and `load_weights_from_checkpoint()` for loading only the weights from a manually saved model or checkpoint. This allows to fine-tune the pre-trained models with different optimizers or learning rate schedulers. [#1501](https://github.com/unit8co/darts/pull/1501) by [Antoine Madrona](https://github.com/madtoinou).
  - New method `lr_find()` that helps to find a good initial learning rate for your forecasting problem. [#1609](https://github.com/unit8co/darts/pull/1609) by [Levente Szabados](https://github.com/solalatus) and [Dennis Bader](https://github.com/dennisbader).
  - Improved the [user guide](https://unit8co.github.io/darts/userguide/torch_forecasting_models.html) and added new sections about saving/loading (checkpoints, manual save/load, loading weights only), and callbacks. [#1661](https://github.com/unit8co/darts/pull/1661) by [Antoine Madrona](https://github.com/madtoinou).
  - 🔴 Replaced `":"` in save file names with `"_"` to avoid issues on some operating systems. For loading models saved on earlier Darts versions, try to rename the file names by replacing `":"` with `"_"`. [#1501](https://github.com/unit8co/darts/pull/1501) by [Antoine Madrona](https://github.com/madtoinou).
  - 🔴 New `use_static_covariates` option for `TFTModel`, `DLinearModel` and `NLinearModel` : When True (default), models use static covariates if available at fitting time and enforce identical static covariate shapes across all target `series` used for training or prediction; when False, models ignore static covariates. [#1700](https://github.com/unit8co/darts/pull/1700) by [Dennis Bader](https://github.com/dennisbader).
- Improvements to `TimeSeries` :
  - Added support for integer indexed input to `from_*` factory methods, if index can be converted to a pandas.RangeIndex. [#1527](https://github.com/unit8co/darts/pull/1527) by [Dennis Bader](https://github.com/dennisbader).
  - Added support for integer indexed input with step sizes (freq) other than 1. [#1527](https://github.com/unit8co/darts/pull/1527) by [Dennis Bader](https://github.com/dennisbader).
  - Optimized time series creation with `fill_missing_dates=True` achieving a drastic speed-up . [#1527](https://github.com/unit8co/darts/pull/1527) by [Dennis Bader](https://github.com/dennisbader).
  - `from_group_dataframe()` now warns the user if there is suspicion of a "bad" time index (monotonically increasing). [#1628](https://github.com/unit8co/darts/pull/1628) by [Dennis Bader](https://github.com/dennisbader).
- Added a parameter to give a custom function name to the transformed output of `WindowTransformer`; improved the explanation of the `window` parameter. [#1676](https://github.com/unit8co/darts/pull/1676) and [#1666](https://github.com/unit8co/darts/pull/1666) by [Jing Qiang Goh](https://github.com/JQGoh).
- Added `historical_forecasts` parameter to `backtest()` that allows to use precomputed historical forecasts from `historical_forecasts()`. [#1597](https://github.com/unit8co/darts/pull/1597) by [Janek Fidor](https://github.com/JanFidor).
- Added feature values and SHAP object to `ShapExplainabilityResult`, giving easy user access to all SHAP-specific explainability results. [#1545](https://github.com/unit8co/darts/pull/1545) by [Rijk van der Meulen](https://github.com/rijkvandermeulen).
- New `quantile_loss()` (pinball loss) metric for probabilistic forecasts. [#1559](https://github.com/unit8co/darts/pull/1559) by [Janek Fidor](https://github.com/JanFidor).

**Fixed**

- Fixed an issue in `BottomUp/TopDownReconciliator` where the order of the series components was not taken into account. [#1592](https://github.com/unit8co/darts/pull/1592) by [David Kleindienst](https://github.com/DavidKleindienst).
- Fixed an issue with `DLinearModel` not supporting even numbered `kernel_size`. [#1695](https://github.com/unit8co/darts/pull/1695) by [Antoine Madrona](https://github.com/madtoinou).
- Fixed an issue with `RegressionEnsembleModel` not using future covariates during training. [#1660](https://github.com/unit8co/darts/pull/1660) by [Rajesh Balakrishnan](https://github.com/Rajesh4AI).
- Fixed an issue where `NaiveEnsembleModel` prediction did not transfer the series' component name. [#1602](https://github.com/unit8co/darts/pull/1602) by [David Kleindienst](https://github.com/DavidKleindienst).
- Fixed an issue in `TorchForecastingModel` that prevented from using multi GPU training. [#1509](https://github.com/unit8co/darts/pull/1509) by [Levente Szabados](https://github.com/solalatus).
- Fixed a bug when saving a `FFT` model with `trend=None`. [#1594](https://github.com/unit8co/darts/pull/1594) by [Antoine Madrona](https://github.com/madtoinou).
- Fixed some issues with PyTorch-Lightning version 2.0.0. [#1651](https://github.com/unit8co/darts/pull/1651) by [Dennis Bader](https://github.com/dennisbader).
- Fixed a bug in `QuantileDetector` which raised an error when low and high quantiles had identical values. [#1553](https://github.com/unit8co/darts/pull/1553) by [Julien Adda](https://github.com/julien12234).
- Fixed an issue preventing `TimeSeries` from being empty. [#1359](https://github.com/unit8co/darts/pull/1359) by [Antoine Madrona](https://github.com/madtoinou).
- Fixed an issue when using `backtest()` on multiple series. [#1517](https://github.com/unit8co/darts/pull/1517) by [Julien Herzen](https://github.com/hrzn).
- General fixes to `historical_forecasts()`
  - Fixed issue where `retrain` functions were not handled properly; Improved handling of `start`, and `train_length` parameters; better interpretability with warnings and improved error messages (warnings can be turned of with `show_warnings=False`). By [#1675](https://github.com/unit8co/darts/pull/1675) by [Antoine Madrona](https://github.com/madtoinou) and [Dennis Bader](https://github.com/dennisbader).
  - Fixed an issue for several models (mainly ensemble and local models) where automatic `start` did not respect the minimum required training lengths. [#1616](https://github.com/unit8co/darts/pull/1616) by [Janek Fidor](https://github.com/JanFidor) and [Dennis Bader](https://github.com/dennisbader).
  - Fixed an issue when using a `RegressionModel` with future covariates lags only. [#1685](https://github.com/unit8co/darts/pull/1685) by [Maxime Dumonal](https://github.com/dumjax).

### For developers of the library:

**Improvements**

- Option to skip slow tests locally with `pytest . --no-cov -m "not slow"`. [#1625](https://github.com/unit8co/darts/pull/1625) by [Blazej Nowicki](https://github.com/BlazejNowicki).
- Major refactor of data transformers which simplifies implementation of new transformers. [#1409](https://github.com/unit8co/darts/pull/1409) by [Matt Bilton](https://github.com/mabilton).

## [0.23.1](https://github.com/unit8co/darts/tree/0.23.1) (2023-01-12)

Patch release

**Fixed**

- Fix an issue in `TimeSeries` which made it incompatible with Python 3.7.
  [#1449](https://github.com/unit8co/darts/pull/1449) by [Dennis Bader](https://github.com/dennisbader).
- Fix an issue with static covariates when series have variable lengths with `RegressionModel`s.
  [#1469](https://github.com/unit8co/darts/pull/1469) by [Eliane Maalouf](https://github.com/eliane-maalouf).
- Fix an issue with PyTorch Lightning trainer handling.
  [#1459](https://github.com/unit8co/darts/pull/1459) by [Dennis Bader](https://github.com/dennisbader).
- Fix an issue with `historical_forecasts()` retraining PyTorch models iteratively instead of from scratch.
  [#1465](https://github.com/unit8co/darts/pull/1465) by [Dennis Bader](https://github.com/dennisbader).
- Fix an issue with `historical_forecasts()` not working in some cases when `future_covariates`
  are provided and `start` is not specified. [#1481](https://github.com/unit8co/darts/pull/1481)
  by [Maxime Dumonal](https://github.com/dumjax).
- Fix an issue with `slice_n_points` functions on integer indexes.
  [#1482](https://github.com/unit8co/darts/pull/1482) by [Julien Herzen](https://github.com/hrzn).

## [0.23.0](https://github.com/unit8co/darts/tree/0.23.0) (2022-12-23)

### For users of the library:

**Improved**

- 🚀🚀🚀 Brand new Darts module dedicated to anomaly detection on time series: `darts.ad`.
  More info on the API doc page: https://unit8co.github.io/darts/generated_api/darts.ad.html.
  [#1256](https://github.com/unit8co/darts/pull/1256) by [Julien Adda](https://github.com/julien12234)
  and [Julien Herzen](https://github.com/hrzn).
- New forecasting models: `DLinearModel` and `NLinearModel` as proposed in [this paper](https://arxiv.org/pdf/2205.13504.pdf).
  [#1139](https://github.com/unit8co/darts/pull/1139)  by [Julien Herzen](https://github.com/hrzn) and [Greg DeVos](https://github.com/gdevos010).
- New forecasting model: `XGBModel` implementing XGBoost.
  [#1405](https://github.com/unit8co/darts/pull/1405) by [Julien Herzen](https://github.com/hrzn).
- New `multi_models` option for all `RegressionModel`s: when set to False, uses only a single underlying
  estimator for multi-step forecasting, which can drastically increase computational efficiency.
  [#1291](https://github.com/unit8co/darts/pull/1291) by [Eliane Maalouf](https://github.com/eliane-maalouf).
- All `RegressionModel`s (incl. LightGBM, Catboost, XGBoost, Random Forest, ...)
  now support static covariates.
  [#1412](https://github.com/unit8co/darts/pull/1412) by [Eliane Maalouf](https://github.com/eliane-maalouf).
- `historical_forecasts()` and `backtest()` now work on multiple series, too.
  [#1318](https://github.com/unit8co/darts/pull/1318) by [Maxime Dumonal](https://github.com/dumjax).
- New window transformation capabilities: `TimeSeries.window_transform()` and
  a new `WindowTransformer` which allow to easily create window features.
  [#1269](https://github.com/unit8co/darts/pull/1269) by [Eliane Maalouf](https://github.com/eliane-maalouf).
- 🔴 Improvements to `TorchForecastingModels` : Load models directly to CPU that were trained on GPU. Save file size reduced.
  Improved PyTorch Lightning Trainer handling fixing several minor issues.
  Removed deprecated methods `load_model` and `save_model`
  [#1371](https://github.com/unit8co/darts/pull/1371) by [Dennis Bader](https://github.com/dennisbader).
- Improvements to encoders: Added support for encoders to all models with covariate support through `add_encoders` at model creation.
  Encoders now generate the correct minimum required covariate time spans for all models.
  [#1338](https://github.com/unit8co/darts/pull/1338) by [Dennis Bader](https://github.com/dennisbader).
- New datasets available in `darts.datasets` (`ILINetDataset`, `ExchangeRateDataset`, `TrafficDataset`, `WeatherDataset`)
  [#1298](https://github.com/unit8co/darts/pull/1298) by [Kamil Wierciak](https://github.com/FEJTWOW).
  [#1291](https://github.com/unit8co/darts/pull/1291) by [Eliane Maalouf](https://github.com/eliane-maalouf).
- New `Diff` transformer, which can difference and "undifference" series
  [#1380](https://github.com/unit8co/darts/pull/1380) by [Matt Bilton](https://github.com/mabilton).
- Improvements to KalmanForecaster: The model now accepts different TimeSeries for prediction than the ones used to fit the model.
  [#1338](https://github.com/unit8co/darts/pull/1338) by [Dennis Bader](https://github.com/dennisbader).
- Backtest functions can now accept a list of metric functions [#1333](https://github.com/unit8co/darts/pull/1333)
  by [Antoine Madrona](https://github.com/madtoinou).
- Extension of baseline models to work on multivariate series
  [#1373](https://github.com/unit8co/darts/pull/1373) by [Błażej Nowicki](https://github.com/BlazejNowicki).
- Improvement to `TimeSeries.gaps()` [#1265](https://github.com/unit8co/darts/pull/1265) by
  [Antoine Madrona](https://github.com/madtoinou).
- Speedup of `TimeSeries.quantile_timeseries()` method
  [#1351](https://github.com/unit8co/darts/pull/1351) by [@tranquilitysmile](https://github.com/tranquilitysmile).
- Some dependencies which can be hard to install (LightGBM, Catboost, XGBoost, Prophet, Statsforecast)
  are not required anymore (if not installed the corresponding models will not be available)
  [#1360](https://github.com/unit8co/darts/pull/1360) by [Antoine Madrona](https://github.com/madtoinou).
- Removed `IPython` as a dependency. [#1331](https://github.com/unit8co/darts/pull/1331) by [Erik Hasse](https://github.com/erik-hasse)
- Allow the creation of empty `TimeSeries` [#1359](https://github.com/unit8co/darts/pull/1359)
  by [Antoine Madrona](https://github.com/madtoinou).

**Fixed**

- Fixed edge case in ShapExplainer for regression models where covariates series > target series
  [#1310](https://github.com/unit8co/darts/pull/1310) by [Rijk van der Meulen](https://github.com/rijkvandermeulen)
- Fixed a bug in `TimeSeries.resample()` [#1350](https://github.com/unit8co/darts/pull/1350)
  by [Antoine Madrona](https://github.com/madtoinou).
- Fixed splitting methods when split point is not in the series
  [#1415](https://github.com/unit8co/darts/pull/1415) by [@DavidKleindienst](https://github.com/DavidKleindienst)
- Fixed issues with `append_values()` and `prepend_values()` not correctly extending `RangeIndex`es
  [#1435](https://github.com/unit8co/darts/pull/1435) by [Matt Bilton](https://github.com/mabilton).
- Fixed some issues with time zones [#1343](https://github.com/unit8co/darts/pull/1343)
  by [Antoine Madrona](https://github.com/madtoinou).
- Fixed some issues when using a single target series with `RegressionEnsembleModel`
  [#1357](https://github.com/unit8co/darts/pull/1357) by [Dennis Bader](https://github.com/dennisbader).
- Fixed treatment of stochastic models in ensemble models
  [#1423](https://github.com/unit8co/darts/pull/1423) by [Eliane Maalouf](https://github.com/eliane-maalouf).

## [0.22.0](https://github.com/unit8co/darts/tree/0.22.0) (2022-10-04)

### For users of the library:

**Improved**

- New explainability feature. The class `ShapExplainer` in `darts.explainability` can provide Shap-values explanations of the importance of each lag and each dimension in producing each forecasting lag for `RegressionModel`s. [#909](https://github.com/unit8co/darts/pull/909) by [Maxime Dumonal](https://github.com/dumjax).
- New model: `StatsForecastsETS`. Similarly to `StatsForecastsAutoARIMA`, this model offers the ETS model from Nixtla's `statsforecasts` library as a local forecasting model supporting future covariates. [#1171](https://github.com/unit8co/darts/pull/1171) by [Julien Herzen](https://github.com/hrzn).
- Added support for past and future covariates to `residuals()` function. [#1223](https://github.com/unit8co/darts/pull/1223) by [Eliane Maalouf](https://github.com/eliane-maalouf).
- Added support for retraining model(s) every `n` iteration and on custom conditions in `historical_forecasts` method of `ForecastingModel`s. [#1139](https://github.com/unit8co/darts/pull/1139) by [Francesco Bruzzesi](https://github.com/fbruzzesi).
- Added support for beta-NLL in `GaussianLikelihood`s, as proposed in [this paper](https://arxiv.org/abs/2203.09168). [#1162](https://github.com/unit8co/darts/pull/1162) by [Julien Herzen](https://github.com/hrzn).
- New LayerNorm alternatives, RMSNorm and LayerNormNoBias [#1113](https://github.com/unit8co/darts/pull/1113) by [Greg DeVos](https://github.com/gdevos010).
- 🔴 Improvements to encoders: improve fitting behavior of encoders' transformers and solve a couple of issues. Remove support for absolute index encoding. [#1257](https://github.com/unit8co/darts/pull/1257) by [Dennis Bader](https://github.com/dennisbader).
- Overwrite min_train_series_length for Catboost and LightGBM [#1214](https://github.com/unit8co/darts/pull/1214) by [Anne de Vries](https://github.com/anne-devries).
- New example notebook showcasing and end-to-end example of hyperparameter optimization with Optuna [#1242](https://github.com/unit8co/darts/pull/1242) by [Julien Herzen](https://github.com/hrzn).
- New user guide section on hyperparameter optimization with Optuna and Ray Tune [#1242](https://github.com/unit8co/darts/pull/1242) by [Julien Herzen](https://github.com/hrzn).
- Documentation on model saving and loading. [#1210](https://github.com/unit8co/darts/pull/1210) by [Amadej Kocbek](https://github.com/amadejkocbek).
- 🔴 `torch_device_str` has been removed from all torch models in favor of Pytorch Lightning's `pl_trainer_kwargs` method [#1244](https://github.com/unit8co/darts/pull/1244) by [Greg DeVos](https://github.com/gdevos010).

**Fixed**

- An issue with `add_encoders` in `RegressionModel`s when fit/predict were called with a single target series. [#1193](https://github.com/unit8co/darts/pull/1193) by [Dennis Bader](https://github.com/dennisbader).
- Some issues with integer-indexed series. [#1191](https://github.com/unit8co/darts/pull/1191) by [Julien Herzen](https://github.com/hrzn).
- A bug when using the latest versions (>=1.1.1) of Prophet. [#1208](https://github.com/unit8co/darts/pull/1208) by [Julien Herzen](https://github.com/hrzn).
- An issue with calling `fit_transform()` on reconciliators. [#1165](https://github.com/unit8co/darts/pull/1165) by [Julien Herzen](https://github.com/hrzn).
- A bug in `GaussianLikelihood` object causing issues with confidence intervals. [#1162](https://github.com/unit8co/darts/pull/1162) by [Julien Herzen](https://github.com/hrzn).
- An issue which prevented plotting `TimeSeries` of length 1. [#1206](https://github.com/unit8co/darts/issues/1206) by [Julien Herzen](https://github.com/hrzn).
- Type hinting for ExponentialSmoothing model [#1185](https://github.com/unit8co/darts/pull/1185) by [Rijk van der Meulen](https://github.com/rijkvandermeulen)

## [0.21.0](https://github.com/unit8co/darts/tree/0.21.0) (2022-08-12)

### For users of the library:

**Improved**

- New model: Catboost, incl `quantile`, `poisson` and `gaussian` likelihoods support. [#1007](https://github.com/unit8co/darts/pull/1007), [#1044](https://github.com/unit8co/darts/pull/1044) by [Jonas Racine](https://github.com/jonasracine).
- Extension of the `add_encoders` option to `RegressionModel`s. It is now straightforward to add calendar based or custom past or future covariates to these models, similar to torch models. [#1093](https://github.com/unit8co/darts/pull/1093) by [Dennis Bader](https://github.com/dennisbader).
- Introduction of `StaticCovariatesTransformer`, categorical static covariate support for `TFTModel`, example and user-guide updates on static covariates. [#1081](https://github.com/unit8co/darts/pull/1081) by [Dennis Bader](https://github.com/dennisbader).
- ARIMA and VARIMA models now support being applied to a new series, different than the one used for training. [#1036](https://github.com/unit8co/darts/pull/1036) by [Samuele Giuliano Piazzetta](https://github.com/piaz97).
- All Darts forecasting models now have unified `save()` and `load()` methods. [#1070](https://github.com/unit8co/darts/pull/1070) by [Dustin Brunner](https://github.com/brunnedu).
- Improvements in logging. [#1034](https://github.com/unit8co/darts/pull/1034) by [Dustin Brunner](https://github.com/brunnedu).
- Re-integrating Prophet >= 1.1 in core dependencies (as it does not depend on PyStan anymore). [#1054](https://github.com/unit8co/darts/pull/1054) by [Julien Herzen](https://github.com/hrzn).
- Added a new `AustralianTourismDataset`. [#1141](https://github.com/unit8co/darts/pull/1141) by [Julien Herzen](https://github.com/hrzn).
- Added a new notebook demonstrating hierarchical reconciliation. [#1147](https://github.com/unit8co/darts/pull/1147) by [Julien Herzen](https://github.com/hrzn).
- Added `drop_columns()` method to `TimeSeries`. [#1040](https://github.com/unit8co/darts/pull/1040) by [@shaido987](https://github.com/shaido987)
- Speedup static covariates when no casting is needed. [#1053](https://github.com/unit8co/darts/pull/1053) by [Julien Herzen](https://github.com/hrzn).
- Implemented the min_train_series_length method for the FourTheta and Theta models that overwrites the minimum default of 3 training samples by 2*seasonal_period when appropriate. [#1101](https://github.com/unit8co/darts/pull/1101) by [Rijk van der Meulen](https://github.com/rijkvandermeulen).
- Make default formatting optional in plots. [#1056](https://github.com/unit8co/darts/pull/1056) by [Colin Delahunty](https://github.com/colin99d)
- Introduce `retrain` option in `residuals()` method. [#1066](https://github.com/unit8co/darts/pull/1066) by [Julien Herzen](https://github.com/hrzn).
- Improved error messages. [#1066](https://github.com/unit8co/darts/pull/1066) by [Julien Herzen](https://github.com/hrzn).
- Small readability improvements to user guide. [#1039](https://github.com/unit8co/darts/pull/1039), [#1046](https://github.com/unit8co/darts/pull/1046/files) by [Ryan Russell](https://github.com/ryanrussell)

**Fixed**

- Fixed an error when loading torch forecasting models. [#1124](https://github.com/unit8co/darts/pull/1124) by [Dennis Bader](https://github.com/dennisbader).
- 🔴 renamed `ignore_time_axes` into `ignore_time_axis` in `TimeSeries.concatenate()`. [#1073](https://github.com/unit8co/darts/pull/1073/files) by [Thomas KIENTZ](https://github.com/thomktz)
- Propagate static covs and hierarchy in missing value filler. [#1076](https://github.com/unit8co/darts/pull/1076) by [Julien Herzen](https://github.com/hrzn).
- Fixed an issue where num_stacks is used instead of self.num_stacks in the NBEATSModel. Also, a few mistakes in API reference docs. [#1103](https://github.com/unit8co/darts/pull/1103) by [Rijk van der Meulen](https://github.com/rijkvandermeulen).
- Fixed `univariate_component()` method to propagate static covariates and drop hierarchy. [#1128](https://github.com/unit8co/darts/pull/1128) by [Julien Herzen](https://github.com/hrzn).
- Fixed various issues. [#1106](https://github.com/unit8co/darts/pull/1106) by [Julien Herzen](https://github.com/hrzn).
- Fixed an issue with `residuals` on `RNNModel`. [#1066](https://github.com/unit8co/darts/pull/1066) by [Julien Herzen](https://github.com/hrzn).

## [0.20.0](https://github.com/unit8co/darts/tree/0.20.0) (2022-06-22)

### For users of the library:

**Improved**

- Added support for static covariates in `TimeSeries` class. [#966](https://github.com/unit8co/darts/pull/966) by [Dennis Bader](https://github.com/dennisbader).
- Added support for static covariates in TFT model. [#966](https://github.com/unit8co/darts/pull/966) by [Dennis Bader](https://github.com/dennisbader).
- Support for storing hierarchy of components in `TimeSeries` (in view of hierarchical reconciliation) [#1012](https://github.com/unit8co/darts/pull/1012) by [Julien Herzen](https://github.com/hrzn).
- New Reconciliation transformers for forecast reconciliation: bottom up, top down and MinT. [#1012](https://github.com/unit8co/darts/pull/1012) by [Julien Herzen](https://github.com/hrzn).
- Added support for Monte Carlo Dropout, as a way to capture model uncertainty with torch models at inference time. [#1013](https://github.com/unit8co/darts/pull/1013) by [Julien Herzen](https://github.com/hrzn).
- New datasets: ETT and Electricity. [#617](https://github.com/unit8co/darts/pull/617)
  by [Greg DeVos](https://github.com/gdevos010)
- New dataset, [Uber TLC](https://github.com/fivethirtyeight/uber-tlc-foil-response). [#1003](https://github.com/unit8co/darts/pull/1003) by [Greg DeVos](https://github.com/gdevos010).
- Model Improvements: Option for changing activation function for NHiTs and NBEATS. NBEATS support for dropout. NHiTs Support for AvgPooling1d. [#955](https://github.com/unit8co/darts/pull/955) by [Greg DeVos](https://github.com/gdevos010).
- Implemented [&#34;GLU Variants Improve Transformer&#34;](https://arxiv.org/abs/2002.05202) for transformer based models (transformer and TFT). [#968](https://github.com/unit8co/darts/pull/968) by [Greg DeVos](https://github.com/gdevos010).
- Added support for torch metrics during training and validation. [#996](https://github.com/unit8co/darts/pull/996) by [Greg DeVos](https://github.com/gdevos010).
- Better handling of logging [#1010](https://github.com/unit8co/darts/pull/1010) by [Dustin Brunner](https://github.com/brunnedu).
- Better support for Python 3.10, and dropping `prophet` as a dependency (`Prophet` model still works if `prophet` package is installed separately) [#1023](https://github.com/unit8co/darts/pull/1023) by [Julien Herzen](https://github.com/hrzn).
- Option to avoid global matplotlib configuration changes.
  [#924](https://github.com/unit8co/darts/pull/924) by [Mike Richman](https://github.com/zgana).
- 🔴 `HNiTSModel` renamed to `HNiTS` [#1000](https://github.com/unit8co/darts/pull/1000) by [Greg DeVos](https://github.com/gdevos010).

**Fixed**

- A bug with `tail()` and `head()` [#942](https://github.com/unit8co/darts/pull/942) by [Julien Herzen](https://github.com/hrzn).
- An issue with arguments being reverted for the `metric` function of gridsearch and backtest [#989](https://github.com/unit8co/darts/pull/989) by [Clara Grotehans](https://github.com/ClaraGrthns).
- An error checking whether `fit()` has been called in global models [#944](https://github.com/unit8co/darts/pull/944) by [Julien Herzen](https://github.com/hrzn).
- An error in Gaussian Process filter happening with newer versions of sklearn [#963](https://github.com/unit8co/darts/pull/963) by [Julien Herzen](https://github.com/hrzn).

### For developers of the library:

**Fixed**

- An issue with LinearLR scheduler in tests. [#928](https://github.com/unit8co/darts/pull/928) by [Dennis Bader](https://github.com/dennisbader).

## [0.19.0](https://github.com/unit8co/darts/tree/0.19.0) (2022-04-13)

### For users of the library:

**Improved**

- New model: `NHiTS` implementing the N-HiTS model.
  [#898](https://github.com/unit8co/darts/pull/898) by [Julien Herzen](https://github.com/hrzn).
- New model: `StatsForecastAutoARIMA` implementing the (faster) AutoARIMA version of
  [statsforecast](https://github.com/Nixtla/statsforecast).
  [#893](https://github.com/unit8co/darts/pull/893) by [Julien Herzen](https://github.com/hrzn).
- New model: `Croston` method.
  [#893](https://github.com/unit8co/darts/pull/893) by [Julien Herzen](https://github.com/hrzn).
- Better way to represent stochastic `TimeSeries` from distributions specified by quantiles.
  [#899](https://github.com/unit8co/darts/pull/899) by [Gian Wiher](https://github.com/gnwhr).
- Better sampling of trajectories for stochastic `RegressionModel`s.
  [#899](https://github.com/unit8co/darts/pull/899) by [Gian Wiher](https://github.com/gnwhr).
- Improved user guide with more sections. [#905](https://github.com/unit8co/darts/pull/905)
  by [Julien Herzen](https://github.com/hrzn).
- New notebook showcasing transfer learning and training forecasting models on large time
  series datasets. [#885](https://github.com/unit8co/darts/pull/885)
  by [Julien Herzen](https://github.com/hrzn).

**Fixed**

- Some issues with PyTorch Lightning >= 1.6.0 [#888](https://github.com/unit8co/darts/pull/888)
  by [Julien Herzen](https://github.com/hrzn).

## [0.18.0](https://github.com/unit8co/darts/tree/0.18.0) (2022-03-22)

### For users of the library:

**Improved**

- `LinearRegressionModel` and `LightGBMModel` can now be probabilistic, supporting quantile
  and poisson regression. [#831](https://github.com/unit8co/darts/pull/831),
  [#853](https://github.com/unit8co/darts/pull/853) by [Gian Wiher](https://github.com/gnwhr).
- New models: `BATS` and `TBATS`, based on [tbats](https://github.com/intive-DataScience/tbats).
  [#816](https://github.com/unit8co/darts/pull/816) by [Julien Herzen](https://github.com/hrzn).
- Handling of stochastic inputs in PyTorch based models. [#833](https://github.com/unit8co/darts/pull/833)
  by [Julien Herzen](https://github.com/hrzn).
- GPU and TPU user guide. [#826](https://github.com/unit8co/darts/pull/826)
  by [@gsamaras](https://github.com/gsamaras).
- Added train and validation loss to PyTorch Lightning progress bar.
  [#825](https://github.com/unit8co/darts/pull/825) by [Dennis Bader](https://github.com/dennisbader).
- More losses available in `darts.utils.losses` for PyTorch-based models:
  `SmapeLoss`, `MapeLoss` and `MAELoss`. [#845](https://github.com/unit8co/darts/pull/845)
  by [Julien Herzen](https://github.com/hrzn).
- Improvement to the seasonal decomposition [#862](https://github.com/unit8co/darts/pull/862).
  by [Gian Wiher](https://github.com/gnwhr).
- The `gridsearch()` method can now return best metric score.
  [#822](https://github.com/unit8co/darts/pull/822) by [@nlhkh](https://github.com/nlhkh).
- Removed needless checkpoint loading when predicting. [#821](https://github.com/unit8co/darts/pull/821)
  by [Dennis Bader](https://github.com/dennisbader).
- Changed default number of epochs for validation from 10 to 1.
  [#825](https://github.com/unit8co/darts/pull/825) by [Dennis Bader](https://github.com/dennisbader).

**Fixed**

- Fixed some issues with encoders in `fit_from_dataset()`.
  [#829](https://github.com/unit8co/darts/pull/829) by [Julien Herzen](https://github.com/hrzn).
- Fixed an issue with covariates slicing for `DualCovariatesForecastingModels`.
  [#858](https://github.com/unit8co/darts/pull/858) by [Dennis Bader](https://github.com/dennisbader).

## [0.17.1](https://github.com/unit8co/darts/tree/0.17.1) (2022-02-17)

Patch release

### For users of the library:

**Fixed**

- Fixed issues with (now deprecated) `torch_device_str` parameter, and improved documentation
  related to using devices with PyTorch Lightning. [#806](https://github.com/unit8co/darts/pull/806)
  by [Dennis Bader](https://github.com/dennisbader).
- Fixed an issue with `ReduceLROnPlateau`. [#806](https://github.com/unit8co/darts/pull/806)
  by [Dennis Bader](https://github.com/dennisbader).
- Fixed an issue with the periodic basis functions of N-BEATS. [#804](https://github.com/unit8co/darts/pull/804)
  by [Vladimir Chernykh](https://github.com/vladimir-chernykh).
- Relaxed requirements for `pandas`; from `pandas>=1.1.0` to `pandas>=1.0.5`.
  [#800](https://github.com/unit8co/darts/pull/800) by [@adelnick](https://github.com/adelnick).

## [0.17.0](https://github.com/unit8co/darts/tree/0.17.0) (2022-02-15)

### For users of the library:

**Improved**

- 🚀 Support for [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning): All deep learning
  models are now implemented using PyTorch Lightning. This means that many more features are now available
  via PyTorch Lightning trainers functionalities; such as tailored callbacks, or multi-GPU training.
  [#702](https://github.com/unit8co/darts/pull/702) by [Dennis Bader](https://github.com/dennisbader).
- The `RegressionModel`s now accept an `output_chunk_length` parameter; meaning that they can be trained to
  predict more than one time step in advance (and used auto-regressively to predict on longer horizons).
  [#761](https://github.com/unit8co/darts/pull/761) by [Dustin Brunner](https://github.com/brunnedu).
- 🔴 `TimeSeries` "simple statistics" methods (such as `mean()`, `max()`, `min()` etc, ...) have been refactored
  to work natively on stochastic `TimeSeries`, and over configurable axes. [#773](https://github.com/unit8co/darts/pull/773)
  by [Gian Wiher](https://github.com/gnwhr).
- 🔴 `TimeSeries` now support only pandas `RangeIndex` as an integer index, and does not support `Int64Index` anymore,
  as it became deprecated with pandas 1.4.0. This also now brings the guarantee that `TimeSeries` do not have missing
  "dates" even when indexed with integers. [#777](https://github.com/unit8co/darts/pull/777)
  by [Julien Herzen](https://github.com/hrzn).
- New model: `KalmanForecaster` is a new probabilistic model, working on multivariate series, accepting future covariates,
  and which works by running the state-space model of a given Kalman filter into the future. The `fit()` function uses the
  N4SID algorithm for system identification. [#743](https://github.com/unit8co/darts/pull/743)
  by [Julien Herzen](https://github.com/hrzn).
- The `KalmanFilter` now also works on `TimeSeries` containing missing values. [#743](https://github.com/unit8co/darts/pull/743)
  by [Julien Herzen](https://github.com/hrzn).
- The estimators (forecasting and filtering models) now also return their own instance when calling `fit()`,
  which allows chaining calls. [#741](https://github.com/unit8co/darts/pull/741)
  by [Julien Herzen](https://github.com/hrzn).

**Fixed**

- Fixed an issue with tensorboard and gridsearch when `model_name` is provided.
  [#760](https://github.com/unit8co/darts/pull/760) by [@gdevos010](https://github.com/gdevos010).
- Fixed issues with pip-tools. [#762](https://github.com/unit8co/darts/pull/762)
  by [Tomas Van Pottelbergh](https://github.com/tomasvanpottelbergh).

### For developers of the library:

- Some linting checks have been added to the CI pipeline. [#749](https://github.com/unit8co/darts/pull/749)
  by [Tomas Van Pottelbergh](https://github.com/tomasvanpottelbergh).

## [0.16.1](https://github.com/unit8co/darts/tree/0.16.1) (2022-01-24)

Patch release

### For users of the library:

- Fixed an incompatibility with latest version of Pandas ([#752](https://github.com/unit8co/darts/pull/752))
  by [Julien Herzen](https://github.com/hrzn).
- Fixed non contiguous error when using lstm_layers > 1 on GPU. ([#740](https://github.com/unit8co/darts/pull/740))
  by [Dennis Bader](https://github.com/dennisbader).
- Small improvement in type annotations in API documentation ([#744](https://github.com/unit8co/darts/pull/744))
  by [Dustin Brunner](https://github.com/brunnedu).

### For developers of the library:

- Added flake8 tests to CI pipelines ([#749](https://github.com/unit8co/darts/pull/749),
  [#748](https://github.com/unit8co/darts/pull/748), [#745](https://github.com/unit8co/darts/pull/745))
  by [Tomas Van Pottelbergh](https://github.com/tomasvanpottelbergh)
  and [Dennis Bader](https://github.com/dennisbader).

## [0.16.0](https://github.com/unit8co/darts/tree/0.16.0) (2022-01-13)

### For users of the library:

**Improved**

- The [documentation page](https://unit8co.github.io/darts/index.html) has been revamped and now contains
  a brand new Quickstart guide, as well as a User Guide section, which will be populated over time.
- The [API documentation](https://unit8co.github.io/darts/generated_api/darts.html) has been revamped and improved,
  notably using `numpydoc`.
- The datasets building procedure has been improved in `RegressionModel`, which yields dramatic speed improvements.

**Added**

- The `KalmanFilter` can now do system identification using `fit()` (using [nfoursid](https://github.com/spmvg/nfoursid)).

**Fixed**

- Catch a [potentially problematic case](https://github.com/unit8co/darts/issues/724) in ensemble models.
- Fixed support for `ReduceLROnPlateau` scheduler.

### For developers of the library:

- We have switched to [black](https://black.readthedocs.io/en/stable/) for code formatting (this is checked
  by the CI pipeline).

## [0.15.0](https://github.com/unit8co/darts/tree/0.15.0) (2021-12-24)

### For users of the library:

**Added**:

- On-the-fly encoding of position and calendar information in Torch-based models.
  Torch-based models now accept an option `add_encoders` parameter, specifying how to
  use certain calendar and position information as past and/or future covariates on the-fly.

  Example:

  ```
  from darts.dataprocessing.transformers import Scaler
  add_encoders={
      'cyclic': {'future': ['month']},
      'datetime_attribute': {'past': ['hour', 'dayofweek']},
      'position': {'past': ['absolute'], 'future': ['relative']},
      'custom': {'past': [lambda idx: (idx.year - 1950) / 50]},
      'transformer': Scaler()
  }
  ```

  This will add a cyclic encoding of the month as future covariates, add some datetime
  attributes as past and future covariates, an absolute/relative position (index), and
  even some custom mapping of the index (such as a function of the year). A `Scaler` will
  be applied to fit/transform all of these covariates both during training and inference.
- The scalers can now also be applied on stochastic `TimeSeries`.
- There is now a new argument `max_samples_per_ts` to the :func:`fit()` method of Torch-based
  models, which can be used to limit the number of samples contained in the underlying
  training dataset, by taking (at most) the most recent `max_samples_per_ts` training samples
  per time series.
- All local forecasting models that support covariates (Prophet, ARIMA, VARIMA, AutoARIMA)
  now handle covariate slicing themselves; this means that you don't need to make sure your
  covariates have the exact right time span. As long as they contain the right time span, the
  models will slice them for you.
- `TimeSeries.map()` and mappers data transformers now work on stochastic `TimeSeries`.
- Granger causality function: `utils.statistics.granger_causality_tests` can test if one
  univariate `TimeSeries` "granger causes" another.
- New stationarity tests for univariate `TimeSeries` : `darts.utils.statistics.stationarity_tests`,
  `darts.utils.statistics.stationarity_test_adf` and `darts.utils.statistics.stationarity_test_kpss`.
- New test coverage badge 🦄

**Fixed**:

- Fixed various issues in different notebooks.
- Fixed a bug handling frequencies in Prophet model.
- Fixed an issue causing `PastCovariatesTorchModels` (such as `NBEATSModel`) prediction
  to fail when `n > output_chunk_length` AND `n` not being a multiple of `output_chunk_length`.
- Fixed an issue in backtesting which was causing untrained models
  not to be trained on the initial window when `retrain=False`.
- Fixed an issue causing `residuals()` to fail for Torch-based models.

### For developers of the library:

- Updated the [contribution guidelines](https://github.com/unit8co/darts/blob/master/CONTRIBUTING.md)
- The unit tests have been re-organised with submodules following that of the library.
- All relative import paths have been removed and replaced by absolute paths.
- pytest and pytest-cov are now used to run tests and compute coverage.

## [0.14.0](https://github.com/unit8co/darts/tree/0.14.0) (2021-11-28)

### For users of the library:

**Added**:

- Probabilistic N-BEATS: The `NBEATSModel` can now produce probabilistic forecasts,
  in a similar way as all the other deep learning models in Darts (specifying a `likelihood`
  and predicting with `num_samples` >> 1).
- We have improved the speed of the data loaing functionalities for PyTorch-based models.
  This should speedup training, typically by a few percents.
- Added `num_loader_workers` parameters to `fit()` and `predict()` methods of PyTorch-based models,
  in order to control the `num_workers` of PyTorch DataLoaders. This can sometimes result in drastic speedups.
- New method `TimeSeries.astype()` which allows to easily case (e.g. between `np.float64` and `np.float32`).
- Added `dtype` as an option to the time series generation modules.
- Added a small [performance guide](https://github.com/unit8co/darts/blob/master/guides/performance.md) for
  PyTorch-based models.
- Possibility to specify a (relative) time index to be used as future covariates in the TFT Model.
  Future covariates don't have to be specified when this is used.
- New TFT example notebook.
- Less strict dependencies: we have loosened the required dependencies versions.

**Fixed**:

- A small fix on the Temporal Fusion Transformer `TFTModel`, which should improve performance.
- A small fix in the random state of some unit tests.
- Fixed a typo in Transformer example notebook.

## [0.13.1](https://github.com/unit8co/darts/tree/0.13.1) (2021-11-08)

### For users of the library:

**Added**:

- Factory methods in `TimeSeries` are now `classmethods`, which makes inheritance of
  `TimeSeries` more convenient.

**Fixed**:

- An issue which was causing some of the flavours installations not to work

## [0.13.0](https://github.com/unit8co/darts/tree/0.13.0) (2021-11-07)

### For users of the library:

**Added**:

- New forecasting model, [Temporal Fusion Transformer](https://arxiv.org/abs/1912.09363) (`TFTModel`).
  A new deep learning model supporting both past and future covariates.
- Improved support for Facebook Prophet model (`Prophet`):
  - Added support for fit & predict with future covariates. For instance:
    `model.fit(train, future_covariates=train_covariates)` and
    `model.predict(n=len(test), num_sample=1, future_covariates=test_covariates)`
  - Added stochastic forecasting, for instance: `model.predict(n=len(test), num_samples=200)`
  - Added user-defined seasonalities either at model creation with kwarg
    `add_seasonality` (`Prophet(add_seasonality=kwargs_dict)`) or pre-fit with
    `model.add_seasonality(kwargs)`. For more information on how to add seasonalities,
    see the [Prophet docs](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.prophet.html).
  - Added possibility to predict and return the base model's raw output with `model.predict_raw()`.
    Note that this returns a pd.DataFrame `pred_df`, which will not be supported for further
    processing with the Darts API. But it is possible to access Prophet's methods such as
    plots with `model.model.plot_compenents(pred_df)`.
- New `n_random_samples` in `gridsearch()` method, which allows to specify a number of (random)
  hyper parameters combinations to be tried, in order mainly to limit the gridsearch time.
- Improvements in the checkpointing and saving of Torch models.
  - Now models don't save checkpoints by default anymore. Set `save_checkpoints=True` to enable them.
  - Models can be manually saved with `YourTorchModel.save_model(file_path)`
    (file_path pointing to the .pth.tar file).
  - Models can be manually loaded with `YourTorchModel.load_model(file_path)` or
    the original method `YourTorchModel.load_from_checkpoint()`.
- New `QuantileRegression` Likelihood class in `darts.utils.likelihood_models`.
  Allows to apply quantile regression loss, and get probabilistic forecasts on all deep
  learning models supporting likelihoods.
  Used by default in the Temporal Fusion Transformer.

**Fixed:**

- Some issues with `darts.concatenate()`.
- Fixed some bugs with `RegressionModel`s applied on multivariate series.
- An issue with the confidence bounds computation in ACF plot.
- Added a check for some models that do not support `retrain=False` for `historical_forecasts()`.
- Small fixes in install instructions.
- Some rendering issues with bullet points lists in examples.

## [0.12.0](https://github.com/unit8co/darts/tree/0.12.0) (2021-09-25)

### For users of the library:

**Added**:

- Improved probabilistic forecasting with neural networks
  - Now all neural networks based forecasting models (except `NBEATSModel`) support probabilistic forecasting,
    by providing the `likelihood` parameter to the model's constructor method.
  - `darts.utils.likelihood_models` now contains many more distributions. The complete list of likelihoods
    available to train neural networks based models is
    available here: https://unit8co.github.io/darts/generated_api/darts.utils.likelihood_models.html
  - Many of the available likelihood models now offer the possibility to specify "priors" on the distribution's
    parameters. Specifying such priors will regularize the training loss to make the output distribution
    more like the one specified by the prior parameters values.
- Performance improvements on `TimeSeries` creation. creating `TimeSeries` is now be significantly faster,
  especially for large series, and filling missing dates has also been significantly sped up.
- New rho-risk metric for probabilistic forecasts.
- New method `darts.utils.statistics.plot_hist()` to plot histograms of time series data (e.g. backtest errors).
- New argument `fillna_value` to `TimeSeries` factory methods, allowing to specify a value to fill missing dates
  (instead of `np.nan`).
- Synthetic `TimeSeries` generated with `darts.utils.timeseries_generation` methods can now be integer-index
  (just pass an integer instead of a timestamp for the `start` argument).
- Removed some deprecation warnings
- Updated conda installation instructions

**Fixed:**

- Removed [extra 1x1 convolutions](https://github.com/unit8co/darts/pull/471) in TCN Model.
- Fixed an issue with linewidth parameter when plotting `TimeSeries`.
- Fixed a column name issue in datetime attribute time series.

### For developers of the library:

- We have removed the `develop` branch.
- We force sklearn<1.0 has we have observed issues with pmdarima and sklearn==1.0

## [0.11.0](https://github.com/unit8co/darts/tree/0.11.0) (2021-09-04)

### For users of the library:

**Added:**

- New model: `LightGBMModel` is a new regression model. Regression models allow to predict future values
  of the target, given arbitrary lags of the target as well as past and/or future covariates. `RegressionModel`
  already works with any scikit-learn regression model, and now `LightGBMModel` does the same with LightGBM.
  If you want to activate LightGBM support in Darts, please read the detailed install notes on
  the [README](https://github.com/unit8co/darts/blob/master/README.md) carefully.
- Added stride support to gridsearch

**Fixed:**

- A bug which was causing issues when training on a GPU with a validation set
- Some issues with custom-provided RNN modules in `RNNModel`.
- Properly handle `kwargs` in the `fit` function of `RegressionModel`s.
- Fixed an issue which was causing problems with latest versions of Matplotlib.
- An issue causing errors in the FFT notebook

## [0.10.1](https://github.com/unit8co/darts/tree/0.10.1) (2021-08-19)

### For users of the library:

**Fixed:**

- A bug with memory pinning that was causing issues with training models on GPUs.

**Changed:**

- Clarified conda support on the README

## [0.10.0](https://github.com/unit8co/darts/tree/0.10.0) (2021-08-13)

### For users of the library:

**Added:**

- 🔴 Improvement of the covariates support. Before, some models were accepting a `covariates` (or `exog`)
  argument, but it wasn't always clear whether this represented "past-observed" or "future-known" covariates.
  We have made this clearer. Now all covariate-aware models support `past_covariates` and/or `future_covariates` argument
  in their `fit()` and `predict()` methods, which makes it clear what series is used as a past or future covariate.
  We recommend [this article](https://medium.com/unit8-machine-learning-publication/time-series-forecasting-using-past-and-future-external-data-with-darts-1f0539585993)
  for more information and examples.
- 🔴 Significant improvement of `RegressionModel` (incl. `LinearRegressionModel` and `RandomForest`).
  These models now support training on multiple (possibly multivariate) time series. They also support both
  `past_covariates` and `future_covariates`. It makes it easier than ever to fit arbitrary regression models (e.g. from
  scikit-learn) on multiple series, to predict the future of a target series based on arbitrary lags of the target and
  the past/future covariates. The signature of these models changed: It's not using "`exog`" keyword arguments, but
  `past_covariates` and `future_covariates` instead.
- Dynamic Time Warping. There is a brand new `darts.dataprocessing.dtw` submodule that
  implements Dynamic Time Warping between two `TimeSeries`. It's also coming with a new `dtw`
  metric in `darts.metrics`. We recommend going over the
  [new DTW example notebook](https://github.com/unit8co/darts/blob/master/examples/13-Dynamic-Time-Warping-example.ipynb)
  for a good overview of the new functionalities
- Conda forge installation support (fully supported with Python 3.7 only for now). You can now
  `conda install u8darts-all`.
- `TimeSeries.from_csv()` allows to obtain a `TimeSeries` from a CSV file directly.
- Optional cyclic encoding of the datetime attributes future covariates; for instance it's now possible to call
  `my_series.add_datetime_attribute('weekday', cyclic=True)`, which will add two columns containing a sin/cos
  encoding of the weekday.
- Default seasonality inference in `ExponentialSmoothing`. If left to `None`, the `seasonal_periods` is inferred
  from the `freq` of the provided series.
- Various documentation improvements.

**Fixed:**

- Now transformations and forecasting maintain the columns' names of the `TimeSeries`.
  The generation module `darts.utils.timeseries_generation` also comes with better default columns names.
- Some issues with our Docker build process
- A bug with GPU usage

**Changed:**

- For probabilistic PyTorch based models, the generation of multiple samples (and series) at prediction time is now
  vectorized, which improves inference performance.

## [0.9.1](https://github.com/unit8co/darts/tree/0.9.1) (2021-07-17)

### For users of the library:

**Added:**

- Improved `GaussianProcessFilter`, now handling missing values, and better handling
  time series indexed by datetimes.
- Improved Gaussian Process notebook.

**Fixed:**

- `TimeSeries` now supports indexing using `pandas.Int64Index` and not just `pandas.RangeIndex`,
  which solves some indexing issues.
- We have changed all factory methods of `TimeSeries` to have `fill_missing_dates=False` by
  default. This is because in some cases inferring the frequency for missing dates and
  resampling the series is causing significant performance overhead.
- Fixed backtesting to make it work with integer-indexed series.
- Fixed a bug that was causing inference to crash on GPUs for some models.
- Fixed the default folder name, which was causing issues on Windows systems.
- We have slightly improved the documentation rendering and fixed the titles
  of the documentation pages for `RNNModel` and `BlockRNNModel` to distinguish them.

**Changed:**

- The dependencies are not pinned to some exact versions anymore.

### For developers of the library:

- We have fixed the building process.

## [0.9.0](https://github.com/unit8co/darts/tree/0.9.0) (2021-07-09)

### For users of the library:

**Added:**

- Multiple forecasting models can now produce probabilistic forecasts by specifying a `num_samples` parameter when calling `predict()`. Stochastic forecasts are stored by utilizing the new `samples` dimension in the refactored `TimeSeries` class (see 'Changed' section). Models supporting probabilistic predictions so far are `ARIMA`, `ExponentialSmoothing`, `RNNModel` and `TCNModel`.
- Introduced `LikelihoodModel` class which is used by probabilistic `TorchForecastingModel` classes in order to make predictions in the form of parametrized distributions of different types.
- Added new abstract class `TorchParametricProbabilisticForecastingModel` to serve as parent class for probabilistic models.
- Introduced new `FilteringModel` abstract class alongside `MovingAverage`, `KalmanFilter` and `GaussianProcessFilter` as concrete implementations.
- Future covariates are now utilized by `TorchForecastingModels` when the forecasting horizon exceeds the `output_chunk_length` of the model. Before, `TorchForecastingModel` instances could only predict beyond their `output_chunk_length` if they were not trained on covariates, i.e. if they predicted all the data they need as input. This restriction has now been lifted by letting a model not only consume its own output when producing long predictions, but also utilizing the covariates known in the future, if available.
- Added a new `RNNModel` class which utilizes and rnn module as both encoder and decoder. This new class natively supports the use of the most recent future covariates when making a forecast. See documentation for more details.
- Introduced optional `epochs` parameter to the `TorchForecastingModel.predict()` method which, if provided, overrides the `n_epochs` attribute in that particular model instance and training session.
- Added support for `TimeSeries` with a `pandas.RangeIndex` instead of just allowing `pandas.DatetimeIndex`.
- `ForecastingModel.gridsearch` now makes use of parallel computation.
- Introduced a new `force_reset` parameter to `TorchForecastingModel.__init__()` which, if left to False, will prevent the user from overriding model data with the same name and directory.

**Fixed:**

- Solved bug occurring when training `NBEATSModel` on a GPU.
- Fixed crash when running `NBEATSModel` with `log_tensorboard=True`
- Solved bug occurring when training a `TorchForecastingModel` instance with a `batch_size` bigger than the available number of training samples.
- Some fixes in the documentation, including adding more details
- Other minor bug fixes

**Changed:**

- 🔴 The `TimeSeries` class has been refactored to support stochastic time series representation by adding an additional dimension to a time series, namely `samples`. A time series is now based on a 3-dimensional `xarray.DataArray` with shape `(n_timesteps, n_components, n_samples)`. This overhaul also includes a change of the constructor which is incompatible with the old one. However, factory methods have been added to create a `TimeSeries` instance from a variety of data types, including `pd.DataFrame`. Please refer to the documentation of `TimeSeries` for more information.
- 🔴 The old version of `RNNModel` has been renamed to `BlockRNNModel`.
- The `historical_forecast()` and `backtest()` methods of `ForecastingModel` have been reorganized a bit by making use of new wrapper methods to fit and predict models.
- Updated `README.md` to reflect the new additions to the library.

## [0.8.1](https://github.com/unit8co/darts/tree/0.8.1) (2021-05-22)

**Fixed:**

- Some fixes in the documentation

**Changed:**

- The way to instantiate Dataset classes; datasets should now be used like this

```
from darts.datasets import AirPassengers
ts: TimeSeries = AirPassengers().load()
```

## [0.8.0](https://github.com/unit8co/darts/tree/0.8.0) (2021-05-21)

### For users of the library:

**Added:**

- `RandomForest` algorithm implemented. Uses the scikit-learn `RandomForestRegressor` to predict future values from (lagged) exogenous
  variables and lagged values of the target.
- `darts.datasets` is a new submodule allowing to easily download, cache and import some commonly used time series.
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
- 🔴 `StandardRegressionModel` is now called `LinearRegressionModel`. It implements a linear regression model
  from `sklearn.linear_model.LinearRegression`. Users who still need to use the former `StandardRegressionModel` with
  another sklearn model should use the `RegressionModel` now.

**Fixed**

- We have fixed a bug arising when multiple scalers were used.
- We have fixed a small issue in the TCN architecture, which makes our implementation follow the original paper
  more closely.

### For developers of the library:

**Added:**

- We have added some [contribution guidelines](https://github.com/unit8co/darts/blob/master/CONTRIBUTE.md).

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
  `TrainingDataset` instance (instead of using a default instance when going via the :func:`fit()` method).
- A new explanatory notebooks for global models:
  https://github.com/unit8co/darts/blob/master/examples/02-multi-time-series-and-covariates.ipynb

**Changed:**

- 🔴 removed the arguments `training_series` and `target_series` in `ForecastingModel`s. Please consult
  the API documentation of forecasting models to see the new signatures.
- 🔴 removed `UnivariateForecastingModel` and `MultivariateForecastingModel` base classes. This distinction does
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
- A new method to `TorchForecastingModel` : `untrained_model()` returns the model as it was initially created, allowing to retrain the exact same model from scratch. Works both when specifying a `random_state` or not.
- New `ForecastingModel.backtest()` and `RegressionModel.backtest()` functions which by default compute a single error score from the historical forecasts the model would have produced.
  - A new `reduction` parameter allows to specify whether to compute the mean/median/… of errors or (when `reduction` is set to `None`) to return a list of historical errors.
  - The previous `backtest()` functionality still exists but has been renamed `historical_forecasts()`
- Added a new `last_points_only` parameter to `historical_forecasts()`, `backtest()` and `gridsearch()`

**Changed:**

- 🔴 Renamed `backtest()` into `historical_forecasts()`
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

- Data (pre) processing abilities using `DataTransformer`, `Pipeline` :
  - `DataTransformer` provide a unified interface to apply transformations on `TimeSeries`, using their `transform()` method
  - `Pipeline` :
    - allow chaining of `DataTransformers`
    - provide `fit()`, `transform()`, `fit_transform()` and `inverse_transform()` methods.
  - Implementing your own data transformers:
    - Data transformers which need to be fitted first should derive from the `FittableDataTransformer` base class and implement a `fit()` method. Fittable transformers also provide a `fit_transform()` method, which fits the transformer and then transforms the data with a single call.
    - Data transformers which perform an invertible transformation should derive from the `InvertibleDataTransformer` base class and implement a `inverse_transform()` method.
    - Data transformers which are neither fittable nor invertible should derive from the `BaseDataTransformer` base class
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

- 🔴 Removed `cols` parameter from `map()`. Using indexing on `TimeSeries` is preferred.
  ```python
  # Assuming a multivariate TimeSeries named series with 3 columns or variables.
  # To apply fn to columns with names '0' and '2':

  #old syntax
  series.map(fn, cols=['0', '2']) # returned a time series with 3 columns
  #new syntax
  series[['0', '2']].map(fn) # returns a time series with only 2 columns
  ```
- 🔴 Renamed `ScalerWrapper` into `Scaler`
- 🔴 Renamed the `preprocessing` module into `dataprocessing`
- 🔴 Unified `auto_fillna()` and `fillna()` into a single `fill_missing_value()` function
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

- GitHub release workflow is now triggered manually from the GitHub "Actions" tab in the repository, providing a `#major`, `#minor`, or `#patch` argument. [#211](https://github.com/unit8co/darts/pull/211)
- (A limited number of) notebook examples are now run as part of the GitHub PR workflow.

## [0.3.0](https://github.com/unit8co/darts/tree/0.3.0) (2020-10-05)

[Full Changelog](https://github.com/unit8co/darts/compare/0.2.3...0.3.0)

### For users of the library:

**Added:**

- Better indexing on TimeSeries (support for column/component indexing) [#150](https://github.com/unit8co/darts/pull/150)
- New `FourTheta` forecasting model [#123](https://github.com/unit8co/darts/pull/123), [#156](https://github.com/unit8co/darts/pull/156)
- `map()` method for TimeSeries [#163](https://github.com/unit8co/darts/pull/163), [#166](https://github.com/unit8co/darts/pull/166)
- Further improved the backtesting functions [#111](https://github.com/unit8co/darts/pull/111):
  - Added support for multivariate TimeSeries and models
  - Added `retrain` and `stride` parameters
- Custom style for matplotlib plots [#191](https://github.com/unit8co/darts/pull/191)
- sMAPE metric [#129](https://github.com/unit8co/darts/pull/129)
- Option to specify a `random_state` at model creation using the `@random_method` decorator on models using neural networks to allow reproducibility of results [#118](https://github.com/unit8co/darts/pull/118)

**Changed:**

- 🔴 **Refactored backtesting** [#184](https://github.com/unit8co/darts/pull/184)
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
- 🔴 `ForecastingModel` `fit()` **method syntax** using TimeSeries indexing instead of additional parameters [#161](https://github.com/unit8co/darts/pull/161)
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

- Solved issue of TorchForecastingModel.predict(n) throwing an error at n=1. [#108](https://github.com/unit8co/darts/pull/108)
- Fixed MASE metrics [#129](https://github.com/unit8co/darts/pull/129)
- BUG ForecastingModel.backtest: Can bypass sanity checks [#189](https://github.com/unit8co/darts/pull/189)
- `ForecastingModel.backtest()` fails if `forecast_horizon` isn't provided [#186](https://github.com/unit8co/darts/issues/186)

### For developers of the library

**Added:**

- Gradle to build docs, docker image, run tests, … [#112](https://github.com/unit8co/darts/pull/112), [#127](https://github.com/unit8co/darts/pull/127), [#159](https://github.com/unit8co/darts/pull/159)
- M4 competition benchmark and notebook to the examples [#138](https://github.com/unit8co/darts/pull/138)
- Check of test coverage [#141](https://github.com/unit8co/darts/pull/141)

**Changed:**

- Dependencies' versions are now fixed [#173](https://github.com/unit8co/darts/pull/173)
- Workflow: tests trigger on Pull Request [#165](https://github.com/unit8co/darts/pull/165)

**Fixed:**

- Passed the `freq` parameter to the `TimeSeries` constructor in all TimeSeries generating functions [#157](https://github.com/unit8co/darts/pull/157)

## Older releases

[Full Changelog](https://github.com/unit8co/darts/compare/f618c4536bf7ed6e3b6a2239fbca4e3089736426...0.2.3)
