# Changelog

Darts is still in an early development phase and we cannot always guarantee backwards compatibility. Changes that may **break code which uses a previous release of Darts** are marked with a "&#x1F534;".

## [Unreleased](https://github.com/unit8co/darts/tree/develop)

[Full Changelog](https://github.com/unit8co/darts/compare/0.3.0...develop)

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