# Proposal for Refactoring Backtesting

Reference: #DARTS-133

---
## Progress

- [x] backtest_forcasting moved in model methods
- [ ] forcasting residuals directly returned from call to backtest
- [ ] explore_models
- [ ] separate plot

## Summary

The main idea behind this proposal is to refactor `backtesting.py` by implementing it directly in the model 
class. Here would be a quick list of changes proposed:

- `backtest_forecasting`, `backtest_regression`, `backtest_gridsearch` -> moved in corresping model methods
- `forecasting_residuals` -> function removed as `residuals` directly returned from call to backtest
- `explore_models` -> either removed or put in a file dedicated to model selection / model evaluation
- `plot_residuals` -> put in `darts.metrics` to follow sklearn practice.



## 1. functions acting on a single model

Currently backtesting is spread across multiple function in `darts/backtesting/backtesting.py`:

```python
def backtest_forecasting(series: TimeSeries,
                         model: ForecastingModel,
                         start: pd.Timestamp,
                         fcast_horizon_n: int,
                         trim_to_series: bool = True,
                         verbose: bool = False) -> TimeSeries:
    pass

def backtest_regression(feature_series: Iterable[TimeSeries],
                        target_series: TimeSeries,
                        model: RegressionModel,
                        start: pd.Timestamp,
                        fcast_horizon_n: int,
                        trim_to_series: bool = True,
                        verbose=False) -> TimeSeries:
    pass

...

```
Parameters of backtesting will always involve a `model` being backtested and a `TimeSeries` that act as ground truth.
Since backtesting is usually refering to a model I propose to add backtest as a method of the `ForecastingModel` and 
`RegressionModel`class.

This would lead to the following API for the end user:

```python
model = ExponentialSmoothing()
historical_forecast = model.backtest(series, pd.Timestamp('19550101'), fcast_horizon_n=3, verbose=True))
```

We could also add residuals computation for free and return it as computation seems unexpansive and require a backtest
forcasting anyway:

```python
historical_forecast, residuals = model.backtest(series, pd.Timestamp('19550101'), fcast_horizon_n=3, verbose=True))
```

We could also add the gridSearch logic directly on the model class as a `classmethod`:

```python
best_model = ExponentialSmoothing.gridsearch(series, fcast_horizon...)
```

## 2. functions acting on several model 

Only `explore_models` act on several model at once. This level of abstraction is usually left on the user ? 
Since several models will be involved I suggest to separate these functions and put them in a namespace similar to 
`sklearn.model_selection` which could be combined later with other method of selecting models.

```python
#darts/model_selection.py

def explore_models(models, series, ...)

```

This would lead to the following API for the end user:


```python
from darts.model_selection import explore_models

explore_models()
```

## 3. functions for ploting / visualisation

We could again follow `sklearn` example and implement all plotting functions in `darts.metrics`.

```python
# darts/metrics/visualisation.py
def plot_residuals_analysis(residuals: TimeSeries,
                            num_bins: int = 20,
                            fill_nan: bool = True):
    pass
```