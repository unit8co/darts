
# Changelog

We do our best to avoid the introduction of breaking changes,
but cannot always guarantee backwards compatibility. Changes that may **break code which uses a previous release of Darts** are marked with a "🔴".

## [Unreleased](https://github.com/unit8co/darts/tree/master)

[Full Changelog](https://github.com/unit8co/darts/compare/0.26.0...master)

### For users of the library:

**Improved**
- Improvements to `TorchForecastingModel`:
  - 🚀🚀 Optimized `historical_forecasts()` for pre-trained `TorchForecastingModel` running up to 20 times faster than before!. [#2013](https://github.com/unit8co/darts/pull/2013) by [Dennis Bader](https://github.com/dennisbader).
  - Added callback `darts.utils.callbacks.TFMProgressBar` to customize at which model stages to display the progress bar. [#2020](https://github.com/unit8co/darts/pull/2020) by [Dennis Bader](https://github.com/dennisbader).
- Improvements to documentation:
  - Adapted the example notebooks to properly apply data transformers and avoid look-ahead bias. [#2020](https://github.com/unit8co/darts/pull/2020) by [Samriddhi Singh](https://github.com/SimTheGreat).
- Improvements to Regression Models:
  - `XGBModel` now leverages XGBoost's native Quantile Regression support that was released in version 2.0.0 for improved probabilistic forecasts. [#2051](https://github.com/unit8co/darts/pull/2051) by [Dennis Bader](https://github.com/dennisbader).

**Fixed**
- Fixed a bug when calling optimized `historical_forecasts()` for a `RegressionModel` trained with unequal component-specific lags. [#2040](https://github.com/unit8co/darts/pull/2040) by [Antoine Madrona](https://github.com/madtoinou).
- Fixed a bug when using encoders with `RegressionModel` and series with a non-evenly spaced frequency (e.g. Month Begin). This raised an error during lagged data creation when trying to divide a pd.Timedelta by the ambiguous frequency. [#2034](https://github.com/unit8co/darts/pull/2034) by [Antoine Madrona](https://github.com/madtoinou).

### For developers of the library:

## [0.26.0](https://github.com/unit8co/darts/tree/0.26.0) (2023-09-16)
### For users of the library:

**Improved**
- Improvements to `RegressionModel`: [#1962](https://github.com/unit8co/darts/pull/1962) by [Antoine Madrona](https://github.com/madtoinou).
  - 🚀🚀 All models now support component/column-specific lags for target, past, and future covariates series.
- Improvements to `TorchForecastingModel`:
  - 🚀 Added `RINorm` (Reversible Instance Norm) as an input normalization option for all models except `RNNModel`. Activate it with model creation parameter `use_reversible_instance_norm`. [#1969](https://github.com/unit8co/darts/pull/1969) by [Dennis Bader](https://github.com/dennisbader).
  - 🔴 Added past covariates feature projection to `TiDEModel` with parameter `temporal_width_past` following the advice of the model architect. Parameter `temporal_width` was renamed to `temporal_width_future`. Additionally, added the option to bypass the feature projection with `temporal_width_past/future=0`. [#1993](https://github.com/unit8co/darts/pull/1993) by [Dennis Bader](https://github.com/dennisbader).
- Improvements to `EnsembleModel`:  [#1815](https://github.com/unit8co/darts/pull/#1815) by [Antoine Madrona](https://github.com/madtoinou) and [Dennis Bader](https://github.com/dennisbader).
  - 🔴 Renamed model constructor argument `models` to `forecasting_models`.
  - 🚀🚀 Added support for pre-trained `GlobalForecastingModel` as `forecasting_models` to avoid re-training when ensembling. This requires all models to be pre-trained global models.
  - 🚀 Added support for generating the `forecasting_model` forecasts (used to train the ensemble model) with historical forecasts rather than direct (auto-regressive) predictions. Enable it with `train_using_historical_forecasts=True` at model creation. 
  - Added an example notebook for ensemble models.
- Improvements to historical forecasts, backtest and gridsearch:  [#1866](https://github.com/unit8co/darts/pull/1866) by [Antoine Madrona](https://github.com/madtoinou).
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
- Improvements to `EnsembleModel`:
  - Model creation parameter `forecasting_models` now supports a mix of `LocalForecastingModel` and `GlobalForecastingModel` (single `TimeSeries` training/inference only, due to the local models). [#1745](https://github.com/unit8co/darts/pull/1745) by [Antoine Madrona](https://github.com/madtoinou).
  - Future and past covariates can now be used even if `forecasting_models` have different covariates support. The covariates passed to `fit()`/`predict()` are used only by models that support it. [#1745](https://github.com/unit8co/darts/pull/1745) by [Antoine Madrona](https://github.com/madtoinou).
  - `RegressionEnsembleModel` and `NaiveEnsembleModel` can generate probabilistic forecasts, probabilistics `forecasting_models` can be sampled to train the `regression_model`, updated the documentation (stacking technique). [#1692](https://github.com/unit8co/darts/pull/1692) by [Antoine Madrona](https://github.com/madtoinou).
- Improvements to `Explainability` module:
  - 🚀🚀 New forecasting model explainer: `TFTExplainer` for `TFTModel`. You can now access and visualize the trained model's feature importances and self attention. [#1392](https://github.com/unit8co/darts/issues/1392) by [Sebastian Cattes](https://github.com/Cattes) and [Dennis Bader](https://github.com/dennisbader).
  - Added static covariates support to `ShapeExplainer`. [#1803](https://github.com/unit8co/darts/pull/1803) by [Anne de Vries](https://github.com/anne-devries) and [Dennis Bader](https://github.com/dennisbader).
- Improvements to documentation [#1904](https://github.com/unit8co/darts/pull/1904) by [Dennis Bader](https://github.com/dennisbader):
  - made model sections in README.md, covariates user guide and forecasting model API Reference more user friendly by adding model links and reorganizing them into model categories.
  - added the Dynamic Time Warping (DTW) module and improved its appearance.
- Other improvements:
  - Improved static covariates column naming when using `StaticCovariatesTransformer` with a `sklearn.preprocessing.OneHotEncoder`. [#1863](https://github.com/unit8co/darts/pull/1863) by [Anne de Vries](https://github.com/anne-devries).
  - Added `MSTL` (Season-Trend decomposition using LOESS for multiple seasonalities) as a `method` option for `extract_trend_and_seasonality()`. [#1879](https://github.com/unit8co/darts/pull/1879) by [Alex Colpitts](https://github.com/alexcolpitts96).
  - Added `RINorm` (Reversible Instance Norm) as a new input normalization option for `TorchForecastingModel`. So far only `TiDEModel` supports it with model creation parameter `use_reversible_instance_norm`. [#1865](https://github.com/unit8co/darts/issues/1856) by [Alex Colpitts](https://github.com/alexcolpitts96).
  - Improvements to `TimeSeries.plot()`: custom axes are now properly supported with parameter `ax`. Axis is now returned for downstream tasks. [#1916](https://github.com/unit8co/darts/pull/1916) by [Dennis Bader](https://github.com/dennisbader).

**Fixed**
- Fixed an issue not considering original component names for `TimeSeries.plot()` when providing a label prefix. [#1783](https://github.com/unit8co/darts/pull/1783) by [Simon Sudrich](https://github.com/sudrich).
- Fixed an issue with the string representation of `ForecastingModel` when using array-likes at model creation. [#1749](https://github.com/unit8co/darts/pull/1749) by [Antoine Madrona](https://github.com/madtoinou).
- Fixed an issue with `TorchForecastingModel.load_from_checkpoint()` not properly loading the loss function and metrics. [#1759](https://github.com/unit8co/darts/pull/1759) by [Antoine Madrona](https://github.com/madtoinou).
- Fixed a bug when loading the weights of a `TorchForecastingModel` trained with encoders or a Likelihood. [#1744](https://github.com/unit8co/darts/pull/1744) by [Antoine Madrona](https://github.com/madtoinou).
- Fixed a bug when using selected `target_components` with `ShapExplainer`. [#1803](https://github.com/unit8co/darts/pull/1803) by [Dennis Bader](https://github.com/dennisbader).
- Fixed `TimeSeries.__getitem__()` for series with a RangeIndex with start != 0 and freq != 1. [#1868](https://github.com/unit8co/darts/pull/1868) by [Dennis Bader](https://github.com/dennisbader).
- Fixed an issue where `DTWAlignment.plot_alignment()` was not plotting the alignment plot of series with a RangeIndex correctly. [#1880](https://github.com/unit8co/darts/pull/1880) by [Ahmet Zamanis](https://github.com/AhmetZamanis) and [Dennis Bader](https://github.com/dennisbader).
- Fixed an issue when calling `ARIMA.predict()` and `num_samples > 1` (probabilistic forecasting), where the start point of the simulation was not anchored to the end of the target series. [#1893](https://github.com/unit8co/darts/pull/1893) by [Dennis Bader](https://github.com/dennisbader).
- Fixed an issue when using `TFTModel.predict()` with `full_attention=True` where the attention mask was not applied properly. [#1392](https://github.com/unit8co/darts/issues/1392) by [Dennis Bader](https://github.com/dennisbader).

### For developers of the library:

**Improvements**
- Refactored the `ForecastingModelExplainer` and `ExplainabilityResult` to simplify implementation of new explainers. [#1392](https://github.com/unit8co/darts/issues/1392) by [Dennis Bader](https://github.com/dennisbader).
- Adapted all unit tests to run successfully on M1 devices. [#1933](https://github.com/unit8co/darts/issues/1933) by [Dennis Bader](https://github.com/dennisbader).

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
- Improvements to `RegressionModel`:
    - Optimized lagged data creation for fit/predict sets achieving a drastic speed-up. [#1399](https://github.com/unit8co/darts/pull/1399) by [Matt Bilton](https://github.com/mabilton).
    - Added support for categorical past/future/static covariates to `LightGBMModel` with model creation parameters `categorical_*_covariates`. [#1585](https://github.com/unit8co/darts/pull/1585) by [Rijk van der Meulen](https://github.com/rijkvandermeulen).
    - Added lagged feature names for better interpretability; accessible with model property `lagged_feature_names`. [#1679](https://github.com/unit8co/darts/pull/1679) by [Antoine Madrona](https://github.com/madtoinou).
    - 🔴 New `use_static_covariates` option for all models: When True (default), models use static covariates if available at fitting time and enforce identical static covariate shapes across all target `series` used for training or prediction; when False, models ignore static covariates. [#1700](https://github.com/unit8co/darts/pull/1700) by [Dennis Bader](https://github.com/dennisbader).
- Improvements to `TorchForecastingModel`:
  - New methods `load_weights()` and `load_weights_from_checkpoint()` for loading only the weights from a manually saved model or checkpoint. This allows to fine-tune the pre-trained models with different optimizers or learning rate schedulers. [#1501](https://github.com/unit8co/darts/pull/1501) by [Antoine Madrona](https://github.com/madtoinou).
  - New method `lr_find()` that helps to find a good initial learning rate for your forecasting problem. [#1609](https://github.com/unit8co/darts/pull/1609) by [Levente Szabados](https://github.com/solalatus) and [Dennis Bader](https://github.com/dennisbader).
  - Improved the [user guide](https://unit8co.github.io/darts/userguide/torch_forecasting_models.html) and added new sections about saving/loading (checkpoints, manual save/load, loading weights only), and callbacks. [#1661](https://github.com/unit8co/darts/pull/1661) by [Antoine Madrona](https://github.com/madtoinou).
  - 🔴 Replaced `":"` in save file names with `"_"` to avoid issues on some operating systems. For loading models saved on earlier Darts versions, try to rename the file names by replacing `":"` with `"_"`. [#1501](https://github.com/unit8co/darts/pull/1501) by [Antoine Madrona](https://github.com/madtoinou).
  - 🔴 New `use_static_covariates` option for `TFTModel`, `DLinearModel` and `NLinearModel`: When True (default), models use static covariates if available at fitting time and enforce identical static covariate shapes across all target `series` used for training or prediction; when False, models ignore static covariates. [#1700](https://github.com/unit8co/darts/pull/1700) by [Dennis Bader](https://github.com/dennisbader).
- Improvements to `TimeSeries`:
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
- 🔴 Improvements to `TorchForecastingModels`: Load models directly to CPU that were trained on GPU. Save file size reduced.
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
  [#1310](https://https://github.com/unit8co/darts/pull/1310) by [Rijk van der Meulen](https://github.com/rijkvandermeulen)
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
- New LayerNorm alternatives, RMSNorm and LayerNormNoBias [#1113](https://github.com/unit8co/darts/issues/1113) by [Greg DeVos](https://github.com/gdevos010).
- 🔴 Improvements to encoders: improve fitting behavior of encoders' transformers and solve a couple of issues. Remove support for absolute index encoding. [#1257](https://github.com/unit8co/darts/pull/1257) by [Dennis Bader](https://github.com/dennisbader).
- Overwrite min_train_series_length for Catboost and LightGBM [#1214](https://https://github.com/unit8co/darts/pull/1214) by [Anne de Vries](https://github.com/anne-devries).
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
- Type hinting for ExponentialSmoothing model [#1185](https://https://github.com/unit8co/darts/pull/1185) by [Rijk van der Meulen](https://github.com/rijkvandermeulen)

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
- New dataset: [Uber TLC](https://github.com/fivethirtyeight/uber-tlc-foil-response). [#1003](https://github.com/unit8co/darts/pull/1003) by [Greg DeVos](https://github.com/gdevos010).
- Model Improvements: Option for changing activation function for NHiTs and NBEATS. NBEATS support for dropout. NHiTs Support for AvgPooling1d. [#955](https://github.com/unit8co/darts/pull/955) by [Greg DeVos](https://github.com/gdevos010).
- Implemented ["GLU Variants Improve Transformer"](https://arxiv.org/abs/2002.05202) for transformer based models (transformer and TFT). [#959](https://github.com/unit8co/darts/issues/959) by [Greg DeVos](https://github.com/gdevos010).
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
- &#x1F534; `TimeSeries` "simple statistics" methods (such as `mean()`, `max()`, `min()` etc, ...) have been refactored
  to work natively on stochastic `TimeSeries`, and over configurable axes. [#773](https://github.com/unit8co/darts/pull/773)
  by [Gian Wiher](https://github.com/gnwhr).
- &#x1F534; `TimeSeries` now support only pandas `RangeIndex` as an integer index, and does not support `Int64Index` anymore,
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
  [#759](https://github.com/unit8co/darts/issues/759) by [@gdevos010](https://github.com/gdevos010).
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
- New stationarity tests for univariate `TimeSeries`: `darts.utils.statistics.stationarity_tests`,
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
- New forecasting model: [Temporal Fusion Transformer](https://arxiv.org/abs/1912.09363) (`TFTModel`).
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
- Removed [extra 1x1 convolutions](https://github.com/unit8co/darts/issues/470) in TCN Model.
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
- &#x1F534; Improvement of the covariates support. Before, some models were accepting a `covariates` (or `exog`)
argument, but it wasn't always clear whether this represented "past-observed" or "future-known" covariates.
We have made this clearer. Now all covariate-aware models support `past_covariates` and/or `future_covariates` argument
in their `fit()` and `predict()` methods, which makes it clear what series is used as a past or future covariate.
We recommend [this article](https://medium.com/unit8-machine-learning-publication/time-series-forecasting-using-past-and-future-external-data-with-darts-1f0539585993)
for more information and examples.

- &#x1F534; Significant improvement of `RegressionModel` (incl. `LinearRegressionModel` and `RandomForest`).
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
- &#x1F534; The `TimeSeries` class has been refactored to support stochastic time series representation by adding an additional dimension to a time series, namely `samples`. A time series is now based on a 3-dimensional `xarray.DataArray` with shape `(n_timesteps, n_components, n_samples)`. This overhaul also includes a change of the constructor which is incompatible with the old one. However, factory methods have been added to create a `TimeSeries` instance from a variety of data types, including `pd.DataFrame`. Please refer to the documentation of `TimeSeries` for more information.
- &#x1F534; The old version of `RNNModel` has been renamed to `BlockRNNModel`.
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
- &#x1F534; `StandardRegressionModel` is now called `LinearRegressionModel`. It implements a linear regression model
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
- A new method to `TorchForecastingModel`: `untrained_model()` returns the model as it was initially created, allowing to retrain the exact same model from scratch. Works both when specifying a `random_state` or not.
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
- (A limited number of) notebook examples are now run as part of the GitHub PR workflow.

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
