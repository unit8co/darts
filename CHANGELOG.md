# Changelog

Darts is still in an early development phase and we cannot always guarantee backwards compatibility. Changes suceptible to **break existing code** working on a previous release are marked with a "&#x1F534;". Sorry for the inconvenience!

## [Unreleased](https://github.com/unit8co/darts/tree/develop)

[Full Changelog](https://github.com/unit8co/darts/compare/0.3.0...develop)

## [0.3.0](https://github.com/unit8co/darts/tree/0.3.0) (2020-10-dd)

### For users of the library:
**Added:**

- Indexing on TimeSeries [\#150](https://github.com/unit8co/darts/pull/150)
- New `FourTheta` forecasting model [\#123](https://github.com/unit8co/darts/pull/123), [\#156](https://github.com/unit8co/darts/pull/156)
- `map()` method for TimeSeries [\#121](https://github.com/unit8co/darts/issues/121), [\#166](https://github.com/unit8co/darts/pull/166)
- Further improved Multivariate TimeSeries capabilities:
  - Added multivariate support to the Neural Network based forecasting models, namely `TorchForecastingModel` and its subclasses (`TCNModel` and `RNNModel`)
  - Added multivariate support to backtesting functions [\#111](https://github.com/unit8co/darts/pull/111)
- Custom style for matplotlib plots [\#191](https://github.com/unit8co/darts/pull/191)
- Option to pass a single string as value column name to the `TimeSeries.from_dataframe()` function. [\#114](https://github.com/unit8co/darts/pull/114)
- sMAPE metric [\#129](https://github.com/unit8co/darts/pull/129)
- `freq` parameter for TimeSeries generating functions [\#157](https://github.com/unit8co/darts/pull/157)
- Option to specify a `random_state` at model creation using the `@random_method` decorator on models using neural networks to allow reproducibility of results [\#118](https://github.com/unit8co/darts/pull/118)

**Changed:**

- &#x1F534; **Refactored backtesting** [\#184](https://github.com/unit8co/darts/pull/184)
  - Moved backtesting functionalities inside `ForecastingModel` and `RegressionModel`
    - Old syntax: `backtest_forecasting()` and `backtest_regression()`
    - New syntax: `model.backtest()`
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

## [0.2.3](https://github.com/unit8co/darts/tree/0.2.3) (2020-07-20)

[Full Changelog](https://github.com/unit8co/darts/compare/0.2.2...0.2.3)

**Fixed bugs:**

- ModuleNotFoundError: No module named 'darts' [\#151](https://github.com/unit8co/darts/issues/151)
- \[BUG\] Inverse transform does not work on series of length \< 3 [\#142](https://github.com/unit8co/darts/issues/142)

## [0.2.2](https://github.com/unit8co/darts/tree/0.2.2) (2020-07-14)

[Full Changelog](https://github.com/unit8co/darts/compare/0.2.1...0.2.2)

**Merged pull requests:**

- Hotfix/preprocessing short timeseries [\#143](https://github.com/unit8co/darts/pull/143) ([pennfranc](https://github.com/pennfranc))

## [0.2.1](https://github.com/unit8co/darts/tree/0.2.1) (2020-07-13)

[Full Changelog](https://github.com/unit8co/darts/compare/0.2.0...0.2.1)

**Closed issues:**

- Multi-Quantile Recurrent Neural Network \(MQRNN\) [\#14](https://github.com/unit8co/darts/issues/14)
- Non-Parametric Time Series \(NPTS\) [\#13](https://github.com/unit8co/darts/issues/13)

**Merged pull requests:**

- hotfix/manual-frequency-override \(\#patch\) [\#139](https://github.com/unit8co/darts/pull/139) ([pennfranc](https://github.com/pennfranc))

## [0.2.0](https://github.com/unit8co/darts/tree/0.2.0) (2020-06-25)

[Full Changelog](https://github.com/unit8co/darts/compare/0.1.0...0.2.0)

**Implemented enhancements:**

- Fix dependency versions [\#9](https://github.com/unit8co/darts/issues/9)

**Closed issues:**

- seq2seq [\#19](https://github.com/unit8co/darts/issues/19)
- LSTM [\#18](https://github.com/unit8co/darts/issues/18)
- Fast Fourier Transform \(FFT\) [\#17](https://github.com/unit8co/darts/issues/17)
- Research reference dataset for comparison [\#11](https://github.com/unit8co/darts/issues/11)

**Merged pull requests:**

- Develop [\#115](https://github.com/unit8co/darts/pull/115) ([TheMP](https://github.com/TheMP))

## [0.1.0](https://github.com/unit8co/darts/tree/0.1.0) (2020-06-18)

[Full Changelog](https://github.com/unit8co/darts/compare/f618c4536bf7ed6e3b6a2239fbca4e3089736426...0.1.0)

**Implemented enhancements:**

- Dockerfile to ease up dev setup [\#10](https://github.com/unit8co/darts/issues/10)

**Closed issues:**

- Theta [\#16](https://github.com/unit8co/darts/issues/16)

**Merged pull requests:**

- Fix/release workflow fix [\#102](https://github.com/unit8co/darts/pull/102) ([hrzn](https://github.com/hrzn))
- fix/fft-notebook [\#101](https://github.com/unit8co/darts/pull/101) ([pennfranc](https://github.com/pennfranc))
- Fix/tcn-test [\#100](https://github.com/unit8co/darts/pull/100) ([pennfranc](https://github.com/pennfranc))
- fix/short-timeseries-support [\#99](https://github.com/unit8co/darts/pull/99) ([pennfranc](https://github.com/pennfranc))
- fix\(examples/notebooks\): fix issue with renaming u8timeseries to darts [\#98](https://github.com/unit8co/darts/pull/98) ([Droxef](https://github.com/Droxef))
- update logo on README [\#95](https://github.com/unit8co/darts/pull/95) ([hrzn](https://github.com/hrzn))
- Check workflows. [\#94](https://github.com/unit8co/darts/pull/94) ([endrjuskr](https://github.com/endrjuskr))
- Renamed u8timeseries to darts [\#90](https://github.com/unit8co/darts/pull/90) ([hrzn](https://github.com/hrzn))
- Update issue templates [\#89](https://github.com/unit8co/darts/pull/89) ([endrjuskr](https://github.com/endrjuskr))
- Add pull request template. [\#87](https://github.com/unit8co/darts/pull/87) ([endrjuskr](https://github.com/endrjuskr))
- Feature/gridsearch [\#86](https://github.com/unit8co/darts/pull/86) ([pennfranc](https://github.com/pennfranc))
- Feature/TCN [\#85](https://github.com/unit8co/darts/pull/85) ([pennfranc](https://github.com/pennfranc))
- Upsample timeseries [\#84](https://github.com/unit8co/darts/pull/84) ([endrjuskr](https://github.com/endrjuskr))
- readme updates [\#83](https://github.com/unit8co/darts/pull/83) ([TheMP](https://github.com/TheMP))
- Remove warnings from docs [\#82](https://github.com/unit8co/darts/pull/82) ([endrjuskr](https://github.com/endrjuskr))
- Holiday timeseries [\#81](https://github.com/unit8co/darts/pull/81) ([endrjuskr](https://github.com/endrjuskr))
- Feature/residuals [\#80](https://github.com/unit8co/darts/pull/80) ([pennfranc](https://github.com/pennfranc))
- Fix notebooks [\#79](https://github.com/unit8co/darts/pull/79) ([radujica](https://github.com/radujica))
- Add missing dates [\#78](https://github.com/unit8co/darts/pull/78) ([endrjuskr](https://github.com/endrjuskr))
- Update title [\#77](https://github.com/unit8co/darts/pull/77) ([endrjuskr](https://github.com/endrjuskr))
- Feature/refactor [\#76](https://github.com/unit8co/darts/pull/76) ([hrzn](https://github.com/hrzn))
- Badge round corners [\#75](https://github.com/unit8co/darts/pull/75) ([radujica](https://github.com/radujica))
- Add test coverage [\#74](https://github.com/unit8co/darts/pull/74) ([radujica](https://github.com/radujica))
- Add more info about lib [\#73](https://github.com/unit8co/darts/pull/73) ([endrjuskr](https://github.com/endrjuskr))
- Fix/suppress prophet logs [\#72](https://github.com/unit8co/darts/pull/72) ([pennfranc](https://github.com/pennfranc))
- Include readme in TOC [\#71](https://github.com/unit8co/darts/pull/71) ([endrjuskr](https://github.com/endrjuskr))
- add badges [\#70](https://github.com/unit8co/darts/pull/70) ([endrjuskr](https://github.com/endrjuskr))
- Clean & document dependencies [\#69](https://github.com/unit8co/darts/pull/69) ([radujica](https://github.com/radujica))
- Add example checks. [\#68](https://github.com/unit8co/darts/pull/68) ([endrjuskr](https://github.com/endrjuskr))
- docker hub [\#67](https://github.com/unit8co/darts/pull/67) ([endrjuskr](https://github.com/endrjuskr))
- improved README [\#66](https://github.com/unit8co/darts/pull/66) ([hrzn](https://github.com/hrzn))
- Feature/more unittests [\#65](https://github.com/unit8co/darts/pull/65) ([pennfranc](https://github.com/pennfranc))
- Merge back to develop part 2 [\#64](https://github.com/unit8co/darts/pull/64) ([endrjuskr](https://github.com/endrjuskr))
- New release process. [\#63](https://github.com/unit8co/darts/pull/63) ([endrjuskr](https://github.com/endrjuskr))
- Add flake8 linter [\#62](https://github.com/unit8co/darts/pull/62) ([radujica](https://github.com/radujica))
- fix\(logging\): removed all custom\_logging messages from unittest output [\#59](https://github.com/unit8co/darts/pull/59) ([pennfranc](https://github.com/pennfranc))
- Merge back to develop [\#58](https://github.com/unit8co/darts/pull/58) ([endrjuskr](https://github.com/endrjuskr))
- Add conda [\#55](https://github.com/unit8co/darts/pull/55) ([radujica](https://github.com/radujica))
- Feature/fft \[WIP\] [\#54](https://github.com/unit8co/darts/pull/54) ([pennfranc](https://github.com/pennfranc))
- docs headers [\#53](https://github.com/unit8co/darts/pull/53) ([endrjuskr](https://github.com/endrjuskr))
- Feature/new readme [\#52](https://github.com/unit8co/darts/pull/52) ([hrzn](https://github.com/hrzn))
- make plot kwargs work again [\#50](https://github.com/unit8co/darts/pull/50) ([hrzn](https://github.com/hrzn))
- Build image on push [\#49](https://github.com/unit8co/darts/pull/49) ([endrjuskr](https://github.com/endrjuskr))
- Merge workflows. [\#48](https://github.com/unit8co/darts/pull/48) ([endrjuskr](https://github.com/endrjuskr))
- Feature/rnns improvements [\#47](https://github.com/unit8co/darts/pull/47) ([hrzn](https://github.com/hrzn))
- Improve docs [\#46](https://github.com/unit8co/darts/pull/46) ([endrjuskr](https://github.com/endrjuskr))
- use action cache [\#45](https://github.com/unit8co/darts/pull/45) ([endrjuskr](https://github.com/endrjuskr))
- Test documentation on every branch [\#44](https://github.com/unit8co/darts/pull/44) ([endrjuskr](https://github.com/endrjuskr))
- Fix/timeseries.plot [\#43](https://github.com/unit8co/darts/pull/43) ([pennfranc](https://github.com/pennfranc))
- Add documentation [\#42](https://github.com/unit8co/darts/pull/42) ([endrjuskr](https://github.com/endrjuskr))
- Version bump script [\#41](https://github.com/unit8co/darts/pull/41) ([pennfranc](https://github.com/pennfranc))
- Feature/versioning [\#40](https://github.com/unit8co/darts/pull/40) ([pennfranc](https://github.com/pennfranc))
- Feature/logging [\#39](https://github.com/unit8co/darts/pull/39) ([pennfranc](https://github.com/pennfranc))
- Fix/timeseries generation [\#38](https://github.com/unit8co/darts/pull/38) ([pennfranc](https://github.com/pennfranc))
- Fix/dependencies [\#37](https://github.com/unit8co/darts/pull/37) ([hrzn](https://github.com/hrzn))
- small setup changes [\#36](https://github.com/unit8co/darts/pull/36) ([hrzn](https://github.com/hrzn))
- Timeseries generation [\#35](https://github.com/unit8co/darts/pull/35) ([pennfranc](https://github.com/pennfranc))
- GitHub action ci [\#34](https://github.com/unit8co/darts/pull/34) ([pennfranc](https://github.com/pennfranc))
- Dependency fix francesco [\#33](https://github.com/unit8co/darts/pull/33) ([pennfranc](https://github.com/pennfranc))
- fixing base image version + build/run Docker scripts [\#32](https://github.com/unit8co/darts/pull/32) ([TheMP](https://github.com/TheMP))
- correct issues and new features [\#30](https://github.com/unit8co/darts/pull/30) ([Droxef](https://github.com/Droxef))
- Create CODEOWNERS [\#29](https://github.com/unit8co/darts/pull/29) ([Bugdup](https://github.com/Bugdup))
- Feature/missing values [\#28](https://github.com/unit8co/darts/pull/28) ([MounBen](https://github.com/MounBen))
- Feature/theta method [\#27](https://github.com/unit8co/darts/pull/27) ([MounBen](https://github.com/MounBen))
- Feature/statistics [\#25](https://github.com/unit8co/darts/pull/25) ([MounBen](https://github.com/MounBen))
- Fix timeseries and dependencies [\#24](https://github.com/unit8co/darts/pull/24) ([MounBen](https://github.com/MounBen))
- Update timeseries.py [\#22](https://github.com/unit8co/darts/pull/22) ([MounBen](https://github.com/MounBen))
- Feature/add optional holidays in prophet [\#21](https://github.com/unit8co/darts/pull/21) ([hrzn](https://github.com/hrzn))
- Dockerfile for dev including jupyter-notebook and all dependencies [\#20](https://github.com/unit8co/darts/pull/20) ([brandtkilian](https://github.com/brandtkilian))
- Fixing dependency versions \#9 [\#15](https://github.com/unit8co/darts/pull/15) ([kstyrc](https://github.com/kstyrc))
- Feature/add intro notebook [\#8](https://github.com/unit8co/darts/pull/8) ([hrzn](https://github.com/hrzn))
- Feature/add drop after and before [\#7](https://github.com/unit8co/darts/pull/7) ([hrzn](https://github.com/hrzn))
- Feature/backtest ensembling [\#6](https://github.com/unit8co/darts/pull/6) ([hrzn](https://github.com/hrzn))
- Feature/ensembling [\#5](https://github.com/unit8co/darts/pull/5) ([hrzn](https://github.com/hrzn))
- Feature/refactoring [\#4](https://github.com/unit8co/darts/pull/4) ([hrzn](https://github.com/hrzn))
- Feature/backtesting supervised [\#3](https://github.com/unit8co/darts/pull/3) ([hrzn](https://github.com/hrzn))
- Feature/add backtesting and examples [\#2](https://github.com/unit8co/darts/pull/2) ([hrzn](https://github.com/hrzn))
- A first minimal version of timeseries lib [\#1](https://github.com/unit8co/darts/pull/1) ([hrzn](https://github.com/hrzn))



\* *This Changelog was automatically generated by [github_changelog_generator](https://github.com/github-changelog-generator/github-changelog-generator) and then manually edited*
