Checklist before merging this PR:
- [x] Mentioned all issues that this PR fixes or addresses.
- [x] Summarized the updates of this PR under **Summary**.
- [ ] Added an entry under **Unreleased** in the [Changelog](../CHANGELOG.md).

<!-- Please mention an issue this pull request addresses. -->
Fixes #871.

### Summary

<!-- Provide a general description of the code changes in your pull
request. If your pull request is not ready to merge, please create
a draft and ask for comments. -->

This PR:
- adds `TorchExplainer` for explaining torch models with [SHAP](https://github.com/slundberg/shap),
- renames the existing SKLearn explainer to `SKLearnExplainer`,
- adds a new `explain_single()` method for explaining a single prediction instance in both explainers,
- adds **a new explainability notebook** with examples for both explainers,
- includes various bug fixes and improvements to the explainability module,
- misc updates to docs and tests.

# (NEW) Torch Explainer

`TorchExplainer` is introduced for `TorchForecastingModel` instances, with a feature set aligned with the SKLearn explainer:
- Batched explanations with `explain()`.
- Single-instance explanations with `explain_single()`.
- Visualization helpers with `summary_plot()` and `force_plot()`.

It supports target, past covariates, future covariates, and static covariates (including component-specific/global covariates), and returns SHAP values in `SHAPExplainabilityResult` / `SHAPSingleExplainabilityResult` objects.

## Motivation
An increasing number of models in Darts are torch-based (recently #3002, #2980, #2944) and users need a consistent way to explain their forecasts.

For scikit-learn models, the existing `SHAPExplainer` (now `SKLearnExplainer`) provides SHAP-based explanations with method selection based on model type.
For torch models, we need a new explainer that can handle the different model architectures, while conforming to existing explainability API patterns.

- **Why SHAP?** SHAP gives additive, model-agnostic feature attributions that are consistent across explainers.
- **Why Permutation Explainer?** For torch models, defaulting to `permutation` provides general applicability and faster explanations than `kernel` or `sampling`. Users can choose other SHAP methods if desired.
- **Why not DeepExplainer or GradientExplainer?** Both are designed for deep learning models and are faster than KernelSHAP. However, they have limitations (from my experiments):
  - DeepExplainer is incompatible with many torch models due to reused layers.
  - Both do not output base values, which are needed for consistent SHAP result objects and visualizations (e.g., waterfall, force plots).
- **Why not captum?** Meta's PyTorch native library supports various attribution methods (Integrated Gradients, DeepLIFT, etc.) and is efficient for torch models. However, as of now, it does not support multi-target explanations. Forecasts in Darts are multi-target in nature (multiple horizons x components x likelihood parameters), so using captum would incur for-loop overhead.
- **Future**: We can consider supporting DeepExplainer/GradientExplainer as additional SHAP methods in the future if they yield better efficiency for some torch models. This would require wrapping `PLForecastingModule` in a generic `nn.Module` that can be explained by these methods, in addition to the current numpy-based function wrapper.

## Design

- `TorchExplainer` mirrors the `SKLearnExplainer` API for consistency, with `explain()`, `summary_plot()`, and `force_plot()` methods.
- It builds SHAP inputs from torch inference datasets to stay consistent with Darts prediction semantics.
- It handles deterministic and probabilistic models (for probabilistic models, explanations are produced for likelihood parameter components).

## Implementation Details
- Internally, it flattens the SHAP inputs into a 2D numpy array expected by SHAP, while keeping track of the original feature structure (horizon/component/likelihood parameter) for constructing SHAP results.
- It wraps `PLForecastingModule` in a numpy-compatible function which:
    - recovers the spaghetti inputs (targets, past/future/static covariates) from the flattened 2D numpy array,
    - calls the module's `forward()` method to get predictions,
    - returns predictions also in a flattened 2D format, which is then passed to SHAP explainer.
- It constructs SHAP result objects with the same structure as `SKLearnExplainer` for consistency in querying and visualization.

## Differences to SKLearnExplainer

- Scope: `TorchForecastingModel` vs `SKLearnModel`.
- Supported SHAP methods differ (torch: `kernel`, `sampling`, `partition`, `permutation`; sklearn additionally supports tree/linear/additive where applicable).
- `TorchExplainer` can explain likelihood parameters of probabilistic forecasts, while `SKLearnExplainer` can only explain the median (quantile) or mean (poisson) predictions.
- `TorchExplainer` uses batched tensor to prevent OOM errors, while `SKLearnExplainer` uses full-size numpy arrays.

## Methods

- `explain()` for horizon/component-level explanations over forecastable timestamps.
- `explain_single()` for one forecast instance (equivalent prediction context to `predict(n=output_chunk_length)`).
- `summary_plot()` shows distributions of feature contributions.
- `force_plot()` shows feature contributions for a specific horizon/component.

## Use Cases

### Summary Plot

Feature-importance distribution analysis per horizon/component for torch models.

```py
import shap
shap.initjs()

from darts.datasets import WineDataset
from darts.explainability import TorchExplainer
from darts.models import TiDEModel

series = WineDataset().load().astype("float32")
model = TiDEModel(12, 12).fit(series[:36])
explainer = TorchExplainer(model)
explainer.summary_plot(horizons=[1])
```

### Force Plot

Local additive contribution view for a selected horizon and target component.

```py
explainer.force_plot(horizon=1)
```

### Explaining Multiple Instances

Batch explanations from foreground data with optional sampling controls for performance.

```py
result = explainer.explain(series[:36])
# return a `TimeSeries` of SHAP values where time index
# corresponds to the instance timestamps
result.get_explanation(horizon=1)
# return the raw SHAP explanation object for custom visualizations
shap_object = result.get_shap_explanation_object(horizon=1)
# plot waterfall for the first forecast instance
shap.plots.waterfall(shap_object[0])
```

### Explaining Single Instance

Per-instance explanation API (`explain_single()`) for local interpretability.

```py
single_result = explainer.explain_single(series[:36])
# return a `TimeSeries` of SHAP values where time index corresponds to the **prediction** timestamp
single_result.get_explanation()
# return the raw SHAP explanation object for custom visualizations
single_shap_object = single_result.get_shap_explanation_object()
# plot heatmap for the single instance explanation along the horizon
shap.plots.heatmap(single_shap_object, instance_order=np.arange(12))
```


### Explaining Probabilistic Forecasts

Probabilistic torch models are supported by explaining each likelihood parameter component, treating them as separate targets. This is useful for understanding how features contribute to uncertainty estimates.

```py
from darts.utils.likelihood_models import QuantileRegression
# fit a probabilistic model with quantile regression likelihood
prob_model = TiDEModel(12, 12, likelihood=QuantileRegression(quantiles=[0.1, 0.5, 0.9]))
prob_model.fit(series[:36])
# create an explainer for the probabilistic model
prob_explainer = TorchExplainer(prob_model)
# explain the probabilistic forecasts
# this will produce explanations for each likelihood parameter component
# (e.g., Y_q0.100, Y_q0.500, Y_q0.900)
prob_result = prob_explainer.explain(series[:36])
# get SHAP values as a `TimeSeries` for the 0.1 quantile at horizon 1
prob_result.get_explanation(horizon=1, component="Y_q0.100")
```

# (CHANGE) SKLearn Explainer

The previous `ShapExplainer` is renamed and aligned with the new naming/API style.

## Renaming

- `ShapExplainer` -> `SKLearnExplainer`.
- `ShapExplainabilityResult` -> `SHAPExplainabilityResult`.
- New `SHAPSingleExplainabilityResult` for `explain_single()` outputs.
- Public imports in `darts.explainability` now expose `SKLearnExplainer`, `TorchExplainer`, and SHAP result classes.

## Bug Fixes

- Improved input processing for explainers by using prediction-aware encoder generation for foreground data (`generate_fit_predict_encodings`), improving consistency with forecasting behavior.
- Better validation and clearer errors in explainability result querying (component/horizon checks).
- Improved stationarity warnings to indicate the specific component and series index.

## (NEW) Explaining Single Instance

`SKLearnExplainer.explain_single()` is added, returning SHAP and feature values for a single prediction instance in the same style as the torch explainer.

# (NEW) Explainability Notebook

Added `examples/28-Explainability-examples.ipynb` covering:
- Introduction to SHAP and explainability in Darts.
- Data and model setup for both sklearn and torch examples.
- Global explanations with `summary_plot()` and scatter dependence plots for both explainers (same below).
- Local batched explanations with `explain()` and `force_plot()` and common SHAP visualizations.
- Local single-instance explanations with `explain_single()` and corresponding visualizations.
- Explaining probabilistic forecasts with `TorchExplainer` and visualizing component-specific explanations.
- Migration note from `ShapExplainer` to `SKLearnExplainer`.
- Conclusion and references.

Notebook is wired into docs examples (`docs/source/examples.rst`) and referenced in docs indexing.

# Miscellaneous

- Reworked explainability module exports and docs text to consistently use `SHAP` capitalization.
- Added/expanded tests for both explainers:
  - `darts/tests/explainability/test_sklearn_explainer.py`
  - `darts/tests/explainability/test_torch_explainer.py`
- Added torch-side robustness fixes around dataset indexing and future-covariate length handling while creating SHAP arrays.

### Other Information

- This PR includes API renaming in explainability. Existing code using `ShapExplainer` should migrate to `SKLearnExplainer`.

<!--Thank you for contributing to darts! -->
