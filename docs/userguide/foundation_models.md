# Foundation Models for Time Series
This document was written for darts version 0.30.0 and later.

## What are Time Series Foundation Models?

**Time Series Foundation Models (TSFMs)** represent one of the most exciting paradigm shifts in forecasting, paralleling how large language models like [GPT-4](https://openai.com/research/gpt-4) and [BERT](https://arxiv.org/abs/1810.04805) transformed natural language processing. These models bring the power of foundation model pre-training—pioneered in [NLP](https://arxiv.org/abs/2108.07258) and [computer vision](https://arxiv.org/abs/2010.11929)—to the time series domain.

The breakthrough lies in massive-scale pre-training: TSFMs learn from datasets containing over [100 billion time points](https://arxiv.org/abs/2310.10688) spanning diverse domains—energy consumption, financial markets, weather patterns, retail sales, web traffic, and more. This extensive exposure enables them to internalize **universal temporal patterns**: seasonality structures, trend behaviors, regime transitions, and complex dependencies that transcend any single domain.

### The Zero-Shot Revolution

Unlike traditional Darts models that require training on your specific dataset, **TSFMs come ready to use immediately**. They generate forecasts through **zero-shot inference**—no training required, no hyperparameter tuning, no dataset-specific configuration. Just like GPT can answer questions about topics it wasn't explicitly trained on, TSFMs can forecast patterns they've never seen before by recognizing analogous structures from their pre-training corpus.

This is forecasting's "GPT moment"—the shift from domain-specific training to universal pattern recognition.

The spectrum of foundation model usage includes:
- **Zero-shot**: Direct prediction without any training

This pre-training paradigm fundamentally changes the forecasting workflow. Instead of the traditional "fit-then-predict" approach, you can now "predict immediately" with competitive accuracy, making foundation models ideal for cold-start scenarios, rapid prototyping, and situations with limited historical data.

## Model Comparison: TimesFM vs Chronos 2 vs Traditional

Choosing the right forecasting approach depends on your specific requirements. Here's a comprehensive comparison:

| **Feature** | **TimesFM 2.5** | **Chronos 2** | **Traditional Models** |
|------------|-----------------|---------------|------------------------|
| **Training Required** | ❌ Zero-shot only | ❌ Zero-shot only | ✅ Always required |
| **Probabilistic Forecasts** | ✅ Yes (100 samples) | ✅ Yes (unlimited samples) | ⚠️ Depends on model |
| **Uncertainty Quantification** | ✅ Confidence intervals | ✅ Full quantile forecasts | ⚠️ Limited |
| **Context Length** | 512-16K points (patch: 32) | 512-8K points (patch: 16) | Model-dependent |
| **Max Forecast Horizon** | 256 steps | 1024 steps | Unlimited |
| **Multivariate Support** | ❌ Univariate only | ✅ Yes (multivariate) | ✅ Many models |
| **Covariates Support** | ❌ Not supported | ✅ Past & Future covariates | ✅ Widely supported |
| **Model Size** | 200M parameters | 120M parameters | Varies widely |
| **Inference Speed** | ⚡ Fast (decoder-only) | 🐢 Moderate (encoder-decoder) | ⚡⚡ Very fast |
| **Memory Usage** | ~2GB GPU/RAM | ~1.5GB GPU/RAM | <100MB |
| **Best For** | Quick prototyping, hourly/daily data | Complex forecasting with external signals | Explainability, domain knowledge |
| **Architecture** | Decoder-only transformer | T5-based encoder-decoder | Statistical/ML |
| **Pre-training Data** | 100B+ time points | Diverse time series corpus | None (fit from scratch) |
| **Typical Use Case** | Cold-start, batch forecasting | Multi-factor forecasting, uncertainty | Production with domain expertise |

### Performance Characteristics

From comprehensive benchmarking across diverse datasets (Air Passengers, Energy Load, Taylor Electricity):

**TimesFM 2.5:**
- Excels on hourly/daily frequency data with clear patterns
- Fast inference makes it ideal for real-time applications
- Strong performance on structured seasonality
- Context window optimization critical for performance

**Chronos 2:**
- Superior uncertainty quantification with probabilistic forecasts
- Better handling of irregular patterns and spikes
- Robust across diverse data characteristics
- Slightly slower but more comprehensive uncertainty estimates

**Traditional Models (e.g., Exponential Smoothing):**
- Competitive on simple seasonal patterns
- Fastest inference (milliseconds vs seconds)
- Full explainability and interpretability
- Require careful hyperparameter tuning

### Decision Guide: Which Model Should I Use?

**Use TimesFM 2.5 when:**
- ⚡ Need fast inference (production real-time forecasting)
- 📊 Working with hourly/daily frequency data
- 🚀 Cold-start scenarios (new products/services)
- 💻 Limited computational resources
- 🎯 Point forecasts sufficient (though probabilistic available)

**Use Chronos 2 when:**
- 📈 Need uncertainty quantification (confidence intervals critical)
- ⚠️ Risk assessment and decision making under uncertainty
- 📉 Handling irregular patterns, spikes, stochastic behavior
- 🔮 Longer forecast horizons (up to 1024 steps)
- 🎲 Probabilistic forecasts drive business decisions
- 🔗 Need to incorporate external signals (past/future covariates)
- 📊 Working with multivariate time series

**Use Traditional Models when:**
- 🔍 Explainability is mandatory (regulatory/business requirements)
- 🧠 Domain expertise can be encoded (business rules, constraints)
- 📚 Extensive historical data with strong local patterns
- ⚙️ Custom seasonal patterns or business calendars
- 🏃 Extremely low latency required (<10ms)

**Use Ensemble Approaches when:**
- 🎯 Maximum accuracy required (combine foundation + traditional)
- 🔀 Diverse data characteristics across many series
- 💪 Robust to distribution shifts and outliers
- 🧪 Experimentation with multiple methodologies

**Quick Start Recommendation:**
1. **Start with foundation models** (TimesFM or Chronos 2) for immediate baselines - no tuning required
2. **Add traditional models** if domain knowledge available or explainability needed
3. **Ensemble both** for production if accuracy is critical

See **[Foundation Models Tutorial](../../examples/25-Foundation-Models.ipynb)** for hands-on examples across multiple datasets.

## API Design Philosophy

Foundation models break the fundamental contract of Darts' `ForecastingModel` base class: the requirement to call `fit()` before `predict()`. This isn't a limitation but a feature—zero-shot inference is the key innovation that makes these models immediately useful without training.

### Design Decision: Using GlobalForecastingModel

> **Why not create a separate `FoundationForecastingModel` base class?**
>
> TimesFM 2.5 intentionally extends `GlobalForecastingModel` to integrate seamlessly with existing Darts workflows. While a custom base class exists (`FoundationForecastingModel`) for future models that may deviate further from Darts conventions, TimesFM's zero-shot paradigm actually *enhances* rather than replaces the standard API:
>
> - ✅ Works with `historical_forecasts()`, `backtest()`, and other Darts utilities
> - ✅ Familiar API for existing Darts users
> - ✅ Can be dropped into ensemble models
> - ✅ Integrates with Darts metrics and evaluation frameworks
>
> This "compatibility-first" design means you can use TimesFM anywhere you'd use a traditional Darts model, but with the added superpower of zero-shot forecasting.

### The fit() Method: Required for API Consistency

> **Why must I call `fit()` even though zero-shot models don't train?**
>
> Darts enforces a **consistent API contract** across ALL models for safety and usability:
>
> **Purpose of `fit()` for zero-shot models:**
> - **Input validation**: Checks series are univariate (Chronos 2), lengths are sufficient
> - **Capability validation**: Ensures data is compatible (no covariates for models that don't support them)
> - **State initialization**: Sets internal `_fit_called` flag that `predict()` and `historical_forecasts()` check
> - **No training**: Pre-trained weights remain frozen—no gradient updates occur
>
> **Required workflow:**
> ```python
> model = ChronosModel()
> model.fit(train_series)  # Required: validates inputs and initializes state
> forecast = model.predict(n=24)  # Predicts continuation of train_series
> forecast = model.predict(n=12, series=other_series)  # Zero-shot on different series!
> ```
>
> After the initial `fit()` call, the model is ready for zero-shot predictions on ANY compatible series.
>
> The model downloads once, then caches for subsequent predictions. This is the foundation model paradigm—immediate utility with no configuration.

## Using TimesFM 2.5 (PyTorch Version)

[TimesFM 2.5](https://blog.research.google/2024/02/a-decoder-only-foundation-model-for.html) is Google's TimesFM 2.5 foundation model for time series, trained on [100 billion time points](https://arxiv.org/abs/2310.10688). Darts provides access to the [PyTorch implementation](https://huggingface.co/google/timesfm-2.5-200m-pytorch) (timesfm-2.5-torch), offering efficient inference on CPU, Apple Silicon (MPS), and NVIDIA GPUs.

**Note:** TimesFM 2.5 with 200M parameters is the only publicly available version from Google Research.

### Installation

```bash
# Install Darts with TimesFM support
pip install "darts[timesfm]"
```

This installs the lightweight PyTorch version of TimesFM, suitable for production inference workloads.

### Zero-Shot Forecasting Workflow

The beauty of foundation models is their simplicity. Here's a complete forecasting pipeline:

```python
from darts.models import TimesFMModel
from darts.datasets import AirPassengersDataset
from darts import concatenate

# Load your time series
series = AirPassengersDataset().load()

# Create the pre-trained model (no training needed!)
model = TimesFMModel(
    context_length=512,  # How much history to use (must be multiple of 32)
    device="auto"        # Automatically selects CPU/MPS/CUDA
)

# Generate zero-shot forecast - no fit() required!
forecast = model.predict(
    n=12,           # Forecast 12 steps ahead
    series=series   # Your time series
)

# Compare with traditional approach for reference
from darts.models import ExponentialSmoothing

traditional_model = ExponentialSmoothing()
traditional_model.fit(series)  # Traditional models require training
traditional_forecast = traditional_model.predict(12)

# Plot both forecasts
import matplotlib.pyplot as plt
series.plot(label="Historical")
forecast.plot(label="TimesFM (zero-shot)", low_quantile=0.1, high_quantile=0.9)
traditional_forecast.plot(label="Exponential Smoothing (trained)")
plt.legend()
plt.show()
```

### Device Selection and Performance

TimesFM 2.5 automatically selects the best available device, but you can specify it explicitly:

```python
# Automatic device selection (recommended)
model = TimesFMModel(device="auto")

# Explicit device selection
model_cpu = TimesFMModel(device="cpu")        # Force CPU
model_gpu = TimesFMModel(device="cuda:0")     # Specific GPU
model_mps = TimesFMModel(device="mps")        # Apple Silicon
```

### Context Length Optimization

Context length determines how much historical data the model uses. **It must be a multiple of 32** due to the model's architecture:

```python
# Short context for high-frequency data or quick inference
model_short = TimesFMModel(context_length=128)

# Medium context for daily/weekly patterns
model_medium = TimesFMModel(context_length=512)

# Long context for complex seasonality (maximum)
model_long = TimesFMModel(context_length=1536)
```

Longer context can capture more complex patterns but increases computation time. Start with 512 for most applications.

### Current Limitations

TimesFM 2.5's decoder-only architecture has specific constraints:
- **Univariate forecasting only**: Forecasts one time series at a time (no cross-series dependencies)
- **No covariate support**: Cannot incorporate external variables (weather, promotions, etc.)
- **Shorter context window**: 512 time steps (vs 8192 for Chronos 2)
- **Zero-shot only**: No training or fine-tuning (uses pre-trained weights as-is)

**When these limitations matter:**
- If you need **covariates** → Use **Chronos 2** instead
- If you need **multivariate forecasting** → Use **Chronos 2** instead
- If you need **long context** (>512 steps) → Use **Chronos 2** instead

Despite these constraints, TimesFM offers faster inference and often outperforms traditional models without any training, making it valuable for rapid prototyping and cold-start scenarios with univariate data.

## Using Chronos 2 (Amazon's Foundation Model)

[Chronos](https://arxiv.org/abs/2403.07815) is Amazon's foundation model that applies language model techniques to time series forecasting. Released in [v2.0.0](https://github.com/amazon-science/chronos-forecasting/releases/tag/v2.0.0), Chronos 2 uses transformer architectures to learn universal temporal patterns.

### Installation

```bash
# Install Darts with Chronos support
pip install "darts[chronos]"
```

This installs chronos-forecasting>=2.0.0 from PyPI for probabilistic forecasting.

### Zero-Shot Forecasting Workflow

Chronos 2 provides probabilistic forecasts out of the box:

```python
from darts.models import ChronosModel
from darts.datasets import AirPassengersDataset

# Load your time series
series = AirPassengersDataset().load()

# Create the pre-trained model (no training needed!)
model = ChronosModel()

# Generate zero-shot probabilistic forecast
forecast = model.predict(
    n=12,               # Forecast 12 steps ahead
    series=series,      # Your time series
    num_samples=100     # Number of samples for probabilistic forecast
)

# Plot with confidence intervals
import matplotlib.pyplot as plt
series.plot(label="Historical")
forecast.plot(
    label="Chronos 2 Forecast",
    low_quantile=0.1,
    high_quantile=0.9
)
plt.legend()
plt.show()
```

### Probabilistic Forecasting

Unlike deterministic models, Chronos 2 provides uncertainty quantification through quantile predictions:

```python
# Generate forecast with confidence intervals
prob_forecast = model.predict(
    n=24,
    series=series,
    num_samples=200
)

# Access different quantiles
median = prob_forecast.quantile(0.5)
lower_90 = prob_forecast.quantile(0.05)
upper_90 = prob_forecast.quantile(0.95)
```

The probabilistic nature makes Chronos 2 particularly valuable for:
- **Risk assessment**: Plan for best/worst case scenarios
- **Decision making**: Different actions for different probability levels
- **Anomaly detection**: Values outside intervals may be unusual
- **Uncertainty quantification**: Know when forecasts are less reliable

### Using Covariates with Chronos 2

Chronos 2 natively supports **covariates**—external variables that provide additional context for forecasting. This is a powerful capability that distinguishes Chronos 2 from TimesFM and many other foundation models.

#### Understanding Covariate Types

**Past Covariates (Historical Context):**
- Historical information known up to the forecast point
- Examples: temperature history, past promotions, historical events
- Used to understand how external factors influenced the target series in the past

**Future Covariates (Known Future Information):**
- Information known for future time steps
- Examples: planned promotions, holiday calendar, weather forecasts, scheduled events
- Used to inform predictions about future conditions

#### Covariate Usage Examples

**Using past covariates:**
```python
from darts.models import ChronosModel
from darts import TimeSeries
import pandas as pd

# Load your time series and historical context
series = TimeSeries.from_dataframe(sales_df, value_cols=['sales'])
weather_history = TimeSeries.from_dataframe(weather_df, value_cols=['temperature'])

# Create model and fit with past covariates
model = ChronosModel()  # Chronos 2 Base by default
model.fit(series, past_covariates=weather_history)

# Predict using past covariates
forecast = model.predict(
    n=24,
    series=series,
    past_covariates=weather_history
)
```

**Using future covariates:**
```python
# Future information like planned promotions or holiday calendar
future_promotions = TimeSeries.from_dataframe(
    promotions_df,
    value_cols=['promotion_intensity']
)

# Predict with future covariates
forecast = model.predict(
    n=24,
    series=series,
    future_covariates=future_promotions
)
```

**Using both past and future covariates:**
```python
# Combine historical context with known future information
forecast = model.predict(
    n=24,
    series=series,
    past_covariates=weather_history,
    future_covariates=future_promotions
)
```

**Multiple covariates:**
```python
# Stack multiple covariate series
from darts import concatenate

# Combine multiple past covariates
past_covs = concatenate([
    weather_history,
    economic_indicators,
    competitor_prices
], axis=1)  # Stack horizontally (multiple columns)

# Combine multiple future covariates
future_covs = concatenate([
    future_promotions,
    holiday_calendar,
    planned_events
], axis=1)

# Predict with multiple covariates
forecast = model.predict(
    n=24,
    series=series,
    past_covariates=past_covs,
    future_covariates=future_covs
)
```

#### Validation and Troubleshooting

**Covariate length requirements:**
- Past covariates must extend to the last point of the target series
- Future covariates must extend at least `n` steps beyond the target series

**Common errors:**

```python
# Error: future covariates too short
forecast = model.predict(n=24, series=series, future_covariates=short_cov)
# ValueError: future_covariates must extend at least 24 steps beyond series

# Error: past covariates don't reach series end
forecast = model.predict(n=24, series=series, past_covariates=incomplete_history)
# ValueError: past_covariates must extend to the last point of series
```

**Tip:** Use `len(series)` and `series.end_time()` to verify covariate alignment before prediction.

#### Covariate Best Practices

**1. Temporal Alignment:**
- Past covariates must align with the target series history
- Future covariates must extend at least `n` steps beyond the last target value

**2. Choosing Covariate Types:**
- Use **past covariates** for variables you can't predict (e.g., actual weather, competitor actions)
- Use **future covariates** for variables you know in advance (e.g., calendar events, your own plans)

**3. Feature Engineering:**
- Normalize covariates to similar scales as your target series
- Consider lagged features for past covariates
- Include cyclical encodings for periodic patterns (day of week, month of year)

**4. Validation:**
- Always verify covariate lengths match requirements
- Test forecasts with and without covariates to measure impact
- Monitor for data leakage (future information in past covariates)

## Foundation Model Capabilities Matrix

This comprehensive comparison shows the native capabilities of each foundation model family. All information is sourced from the model registry (see `darts/models/forecasting/foundation/registry.yaml`).

### Core Capabilities

| Model | Parameters | Context Window | Max Horizon | Univariate | Multivariate | Past Covariates | Future Covariates | Probabilistic | Quantiles |
|-------|------------|----------------|-------------|------------|--------------|-----------------|-------------------|---------------|-----------|
| **Chronos 2 Base** | 120M | 8192 | 1024 | ✅ | ✅ | ✅ | ✅ | ✅ | 21 |
| **TimesFM 2.5** | 200M | 512 | — | ✅ | ❌ | ❌ | ❌ | ✅ | 10 |

*Note: Only Chronos 2 Base (120M) and TimesFM 2.5 (200M) are currently supported in Darts.*

### Capability Legend

- ✅ **Natively Supported**: The model architecture includes native support for this capability
- ❌ **Not Supported**: The model cannot use this type of information
- **Context Window**: Maximum number of historical time steps the model can process
- **Max Horizon**: Maximum number of steps the model can forecast ahead (— = no documented limit)
- **Quantiles**: Number of quantile levels available for probabilistic forecasting

### Quantile Support Details

**Chronos 2 (21 quantiles):**
- Default: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
- Full range: [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
- Use case: Detailed uncertainty quantification with tail risk assessment

**TimesFM 2.5 (10 quantiles):**
- Available: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
- Use case: Standard confidence intervals with faster inference

### Architecture Implications

**Chronos 2 (Encoder-Decoder):**
- T5-based architecture enables processing of both target series and covariates
- Encoder processes historical context (series + past covariates)
- Decoder generates future predictions (conditioned on future covariates)
- Result: Native support for multivariate and covariate-based forecasting

**TimesFM 2.5 (Decoder-Only):**
- Streamlined architecture optimized for univariate patterns
- No encoder means no mechanism for processing auxiliary information
- Result: Faster inference but limited to univariate time series only

### Choosing Based on Capabilities

**Need covariates or multivariate?** → Use **Chronos 2** (only foundation model with native support)

**Univariate forecasting only?** → Consider **TimesFM 2.5** for faster inference or **Chronos 2** for better uncertainty quantification

**Maximum context needed?** → Use **Chronos 2** (8192 vs 512 time steps)

**Fastest inference?** → Use **TimesFM 2.5** (decoder-only is ~2-3x faster than encoder-decoder)

### Key Capabilities

Chronos 2 supports:
- **Univariate and multivariate forecasting**: Forecast single or multiple related time series
- **Zero-shot inference**: No training required (uses pre-trained weights)
- **Probabilistic forecasts**: Quantile-based uncertainty estimates with 21 quantiles
- **Covariate support**: Both past covariates (historical context) and future covariates (known future information)
- **Context length**: Up to 8192 time steps for historical context
- **Long forecast horizons**: Up to 1024 steps ahead

### When to Use Chronos 2

**Chronos 2 excels when:**
- You need uncertainty quantification (confidence intervals)
- Working with limited historical data
- Cold-start scenarios (new products/services)
- Batch forecasting thousands of diverse series
- Quick prototyping without hyperparameter tuning
- **Incorporating external signals** (covariates for weather, promotions, events)
- **Multivariate forecasting** with cross-series dependencies

**Consider traditional models when:**
- Domain expertise can be encoded (business rules)
- Explainability is critical
- You have extensive historical data with strong local patterns
- Extremely low latency required (<10ms inference time)

## Learn More

**Tutorial Notebooks:**
- **[Foundation Models Tutorial](../../examples/25-Foundation-Models.ipynb)** - TimesFM 2.5 and Chronos 2 examples with zero-shot forecasting and probabilistic predictions

**GitHub Issues & Tracking:**
- **[Issue #2359](https://github.com/unit8co/darts/issues/2359)** - Foundation models tracking (April 2024)
- **[Issue #2933](https://github.com/unit8co/darts/issues/2933)** - Chronos 2 integration request (October 2025)

**Academic Papers:**
- **[Foundation Models Survey](https://arxiv.org/abs/2108.07258)** - Comprehensive overview of pre-training paradigms
- **[TimesFM Paper](https://arxiv.org/abs/2310.10688)** - Technical details on decoder-only architecture
- **[Chronos Paper](https://arxiv.org/abs/2403.07815)** - "Chronos: Learning the Language of Time Series"

**External Resources:**
- **[TimesFM GitHub](https://github.com/google-research/timesfm)** - Google's official repository
- **[TimesFM HuggingFace](https://huggingface.co/google/timesfm-2.5-200m-pytorch)** - Pre-trained model
- **[Chronos GitHub](https://github.com/amazon-science/chronos-forecasting)** - Amazon's official repository (Chronos 2 loaded via chronos-forecasting package)