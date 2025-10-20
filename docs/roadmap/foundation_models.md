# Foundation Models Integration Roadmap

This document tracks the integration of time series foundation models into Darts, establishing it as the premier open-source library for foundation model forecasting.

## Completed: TimesFM (Google Research)

**Status**: ✅ Merged in PR #XXXX
**Model**: TimesFM 2.5 (200M parameters)
**Capabilities**: Univariate zero-shot forecasting
**Links**: [Paper](https://arxiv.org/abs/2310.10688) | [Blog](https://blog.research.google/2024/02/a-decoder-only-foundation-model-for.html) | [HuggingFace](https://huggingface.co/google/timesfm-2.5-200m-pytorch)

### What Works Today

- ✅ **TimesFMModel**: Zero-shot univariate forecasting
- ✅ **Device Support**: Automatic detection (CUDA, MPS, CPU)
- ✅ **Darts Integration**: Works with historical_forecasts, backtest
- ✅ **Documentation**: User guide, tutorial notebook

### Infrastructure for Future Development

The following infrastructure is included for **future** model integrations (not currently user-facing):

- **`FoundationForecastingModel` base class**: For models that extend beyond GlobalForecastingModel patterns
- **PEFT utilities** (`peft_utils.py`): Hugging Face-compatible LoRA support (raises NotImplementedError currently)
- **Device utilities** (`device_utils.py`): Shared across all foundation models

**Note**: TimesFM currently extends `GlobalForecastingModel` (not `FoundationForecastingModel`). The base class is architectural preparation for Chronos 2 and future models that may support fine-tuning.

---

## Next: Chronos 2 (Amazon Science)

**Status**: 🎯 High Priority
**Release**: [v2.0.0](https://github.com/amazon-science/chronos-forecasting/releases/tag/v2.0.0) (December 2024)
**Community Demand**: [Issue #2933](https://github.com/unit8co/darts/issues/2933) - Immediate user request
**Tracking**: [Issue #2359](https://github.com/unit8co/darts/issues/2359) - Foundation models roadmap

### Why Chronos 2?

Chronos 2 complements TimesFM with critical capabilities currently missing:
- **Multivariate forecasting**: TimesFM is univariate-only
- **Covariate support**: Past + future covariates (real + categorical)
- **Probabilistic forecasts**: Quantile-based uncertainty estimation
- **Extended context**: 8,192 tokens vs TimesFM's 512

This makes the pair comprehensive: TimesFM for efficient univariate zero-shot, Chronos 2 for multivariate probabilistic forecasting.

### Architecture Comparison

| Feature | TimesFM 2.5 | Chronos 2 |
|---------|-------------|-----------|
| **Backbone** | Decoder-only transformer | T5 (encoder-decoder) |
| **Parameters** | 200M | 120M |
| **Training Data** | 100B time points | 890K series (84B observations) |
| **Multivariate** | ❌ No | ✅ Yes |
| **Covariates** | ❌ No | ✅ Past + Future |
| **Output** | Point forecasts | Quantile forecasts (probabilistic) |
| **Context Length** | 512 tokens | 8,192 tokens |
| **API Style** | TimeSeries-based | DataFrame-based |

### Implementation Plan

**Phase 1: Base Integration** (2-3 weeks)

Create `ChronosModel` in the foundation package:

```python
from darts.models.foundation import ChronosModel

model = ChronosModel(
    model_name="chronos-2",  # Also support chronos-bolt variants
    device="auto"
)

# Multivariate + covariates + probabilistic
forecast = model.predict(
    n=24,
    series=multivariate_series,
    past_covariates=temperature,      # NEW: Covariate support
    future_covariates=holidays,
    num_samples=100,                   # NEW: Probabilistic sampling
    quantile_levels=[0.1, 0.5, 0.9]   # Quantile forecasts
)
```

**Key Tasks:**
- [ ] Implement `ChronosModel` class extending `FoundationForecastingModel`
- [ ] Add DataFrame → TimeSeries adapter layer
- [ ] Implement covariate handling (past + future)
- [ ] Add probabilistic forecast support (quantiles)
- [ ] Device compatibility (CUDA, MPS, CPU)

**Phase 2: Darts Integration** (1 week)

Ensure seamless integration with Darts ecosystem:

```python
# Should work with existing Darts utilities
from darts import TimeSeries
from darts.models.foundation import ChronosModel

model = ChronosModel()

# Historical forecasts for backtesting
historical_forecasts = model.historical_forecasts(
    series=train_series,
    past_covariates=past_covs,
    future_covariates=future_covs,
    num_samples=100
)

# Metrics work with probabilistic forecasts
from darts.metrics import mape, rmse
mape_score = mape(actual, forecast)
```

**Key Tasks:**
- [ ] Support `historical_forecasts()` with covariates
- [ ] Integrate with Darts metrics (MAPE, RMSE, etc.)
- [ ] Enable `backtest()` functionality
- [ ] Preserve Darts conventions (series slicing, concatenation)

### Technical Challenges

**1. API Translation: DataFrame → TimeSeries**

Chronos 2 uses DataFrames with explicit column specifications:
```python
# Chronos native API
pred_df = pipeline.predict_df(
    context_df,
    future_df=future_df,
    id_column="series_id",
    timestamp_column="timestamp",
    target="value"
)
```

**Solution**: Create adapter layer that:
- Converts Darts `TimeSeries` → Chronos `DataFrame` format
- Handles multivariate series (multiple components)
- Maps covariates to future_df columns
- Converts predictions back to `TimeSeries`

**2. Covariate Temporal Alignment**

Ensuring past/future covariates align correctly with forecast horizon:

**Solution**:
- Leverage existing Darts covariate validation infrastructure
- Add checks for future covariate length (must extend `n` steps)
- Clear error messages when alignment fails

**3. Probabilistic Output Mapping**

Chronos returns quantiles, Darts uses stochastic `TimeSeries`:

**Solution**:
- Use Darts' existing quantile forecast infrastructure
- Map `quantile_levels` → stochastic samples
- Ensure metrics can consume probabilistic forecasts

**4. Model Variants**

Chronos has multiple model families:
- Chronos-2 (multivariate, covariates)
- Chronos-Bolt (speed-optimized, 9M-205M parameters)
- Original Chronos T5 (8M-710M parameters)

**Solution**:
- Single `ChronosModel` class with `model_name` parameter
- Auto-detect variant capabilities
- Graceful degradation (e.g., Bolt doesn't support covariates)

### Success Criteria

**Functionality:**
- [ ] Zero-shot multivariate forecasting works
- [ ] Past/future covariates validated on multiple datasets
- [ ] Probabilistic forecasts integrate with Darts metrics
- [ ] Performance comparable to native Chronos implementation

**Testing:**
- [ ] 25+ unit tests covering edge cases
- [ ] Integration tests with Darts utilities (backtest, historical_forecasts)
- [ ] Covariate handling tests (past, future, categorical, real)
- [ ] Multi-series forecasting tests

**Documentation:**
- [ ] Tutorial notebook demonstrating all capabilities
- [ ] User guide section on Chronos 2
- [ ] API documentation (docstrings)
- [ ] Migration guide for users switching from TimesFM

**Timeline**: Target Q1 2026 (3-4 months from now)

---

## Community Engagement

**Active Issues:**
- [#2933](https://github.com/unit8co/darts/issues/2933) - Chronos 2 Model (October 2025) - Immediate demand
- [#2359](https://github.com/unit8co/darts/issues/2359) - Foundation Models (April 2024) - Broader tracking

**Contribution Opportunities:**
Contributors welcome! Follow established patterns in `darts/models/forecasting/foundation/`:
- Extend `FoundationForecastingModel` base class (future models)
- Leverage device utilities from `device_utils.py` (shared infrastructure)
- Add comprehensive tests following TimesFM test structure
- PEFT utilities available in `peft_utils.py` for future fine-tuning research

**Discussion Forums:**
- GitHub Issues for feature requests
- Darts community channels for implementation questions

---

## Measuring Success

**Adoption Metrics:**
- Downloads of `darts[chronos]` extra
- GitHub stars/forks after release
- User feedback on Issues/Discussions

**Technical Metrics:**
- Forecast accuracy vs native Chronos implementation
- Inference speed (batched vs single series)
- Memory efficiency for large-scale forecasting

**Community Impact:**
- Position Darts as **the** open-source library for foundation model forecasting
- Enable researchers to compare foundation models vs traditional approaches
- Accelerate adoption of foundation models in production time series applications

---

*This roadmap is living documentation. Updates reflect community priorities, technical learnings, and foundation model releases.*
