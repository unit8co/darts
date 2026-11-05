# Safe model loading

Darts saves most forecasting models with Python `pickle` (`.pkl`) or, for PyTorch models, a Darts `.pt` wrapper plus a Lightning `.ckpt` checkpoint. Unpickling untrusted files can execute arbitrary code ([CWE-502](https://cwe.mitre.org/data/definitions/502.html)).

**Safe loading is the default** for all models, and related checkpoint/weight loaders. Restricted deserialization only reconstructs globals that pass Darts’ allow-list (known-safe libraries, class hierarchies, and checkpoint-driven registration for `.ckpt` files).

In most cases, safe loading should work out of the box.

## When loading fails

If a global is not allow-listed, loading raises an error that names the missing `{module}.{qualname}`. This can happen for example when using custom PyTorch Lightning callbacks, custom encoders, and others. If you trust that global (or file in general), you can still load using one of the steps below.

1. **Register missing globals as safe globals** — Same entry shapes as [PyTorch `add_safe_globals`](https://docs.pytorch.org/docs/stable/notes/serialization.html#torch.serialization.add_safe_globals) (a callable, or `(callable, qualname_str)` when the name in the file differs from the object’s module path). Registration binds each pickled `{module}.{qualname}` to the **exact** callable or class you provide, so loading works even when that name is not importable (for example functions defined in `__main__`, notebooks, or one-off scripts). Use the `(callable, qualname_str)` form when the string in the saved file does not match `{callable.__module__}.{callable.__qualname__}`:

   ```python
   import pandas as pd
   from darts.datasets import AirPassengersDataset
   from darts.models import LinearRegressionModel
   from darts.utils.serialization import add_safe_globals, safe_globals

   def my_encoder(idx: pd.DatetimeIndex):
       """Custom encoder example adding the series' month value."""
       return idx.month

   # add encoder as a future covariate, fit and save the model
   model = LinearRegressionModel(
       lags=12,
       lags_future_covariates=[0],
       add_encoders={"custom": {"future": [my_encoder]}}
   )
   series = AirPassengersDataset().load()
   model.fit(series)
   model.save("model.pkl")

   # by default, loading the model will fail since `my_encoder` is not an
   # allow-listed global. To fix this, either add the safe globals within a context
   with safe_globals([my_encoder]):
       model = LinearRegressionModel.load("model.pkl")

   # or add them globally via `add_safe_globals()`:
   add_safe_globals([my_encoder])
   model = LinearRegressionModel.load("model.pkl")
   ```

2. **Full trust opt-out** — Only if safe loading is not feasible and you fully trust the file source:

    ```python
    model = NBEATSModel.load("model.pt", trusted=True)
    ```

## Related API

- Find more information in [the API docs](https://unit8co.github.io/darts/generated_api/darts.utils.serialization.registry.html)
