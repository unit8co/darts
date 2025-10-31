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
- **Few-shot**: Providing example time series to guide predictions (in-context learning)
- **Fine-tuning**: Adapting the pre-trained model to your specific domain (coming soon)

This pre-training paradigm fundamentally changes the forecasting workflow. Instead of the traditional "fit-then-predict" approach, you can now "predict immediately" with competitive accuracy, making foundation models ideal for cold-start scenarios, rapid prototyping, and situations with limited historical data.

## The Semantic Intelligence: Understanding Examples vs Covariates

One of the most powerful capabilities of TSFMs is their ability to distinguish between fundamentally different types of information: **temporal causality** (covariates) versus **pattern templates** (examples). This semantic intelligence mirrors how humans naturally separate "what affects my target" from "what my target resembles."

Foundation models introduce **few-shot examples**—a concept semantically distinct from traditional covariates. Understanding this distinction unlocks the full power of these models.

| **COVARIATES (TIME-DISTINGUISHED)** | **FEW-SHOT EXAMPLES (SHAPE TEMPLATES)** |
|--------------------------------------|------------------------------------------|
| **Question:** "What affects my target?" | **Question:** "What does my target resemble?" |
| **Semantic role:** Temporal causality<br>External influences at specific times | **Semantic role:** Pattern recognition<br>Behavioral templates from similar series (shape, cycles, seasonality) |
| **Key dimension:** TIME<br>"Temperature on July 15 affects sales on July 15" - temporal alignment | **Key dimension:** SHAPE<br>"Store A's weekly pattern teaches retail seasonality" - shape learning |
| **Structure:** TimeSeries objects (temperature, prices, holidays) | **Structure:** (context, future) pairs from analogous series |
| **Time alignment:** MUST align with target | **Time alignment:** Independent (unaligned) |
| **Persistence:** Used at EVERY time step throughout prediction horizon | **Persistence:** Ephemeral - used once to condition model, then discarded |
| **Mechanism:** Feature extraction (traditional ML pattern) | **Mechanism:** In-context learning (foundation model pattern) |
| **Example:** "Temperature influences ice cream sales moment-by-moment" | **Example:** "Store A's holiday spikes show how retail series behave—apply to B" |

### Why This Distinction Matters: TSFMs Understand Semantics

This distinction reveals a profound capability of foundation models: **they understand the semantic difference between influence and resemblance**.

**Covariates encode temporal causality**: "Temperature at time t affects sales at time t." The model processes these relationships at every prediction step because the causal mechanism persists through time. This is traditional machine learning—feature engineering where you tell the model "pay attention to this external variable."

**Examples encode pattern templates**: "Here's how similar series behave—recognize these shapes, cycles, and seasonality structures." The model consumes these demonstrations once to understand "what kind of pattern am I forecasting," then applies that understanding. This is **in-context learning**—the foundation model innovation that enables zero-shot transfer.

The robustness comes from TSFMs' training: by seeing billions of time points across domains, they've learned to distinguish:
- **When to look for external influences** (covariate-like patterns): "This series correlates with external factors"
- **When to apply learned templates** (example-like patterns): "This series resembles retail/weather/financial patterns I've seen"

Attempting to use few-shot examples as covariates is like using example sentences as grammar rules—semantically incorrect. Examples teach "how to forecast this TYPE of series," while covariates provide "what influences THIS specific series." Foundation models' power lies in understanding both, separately.

## API Design Philosophy

Foundation models break the fundamental contract of Darts' `ForecastingModel` base class: the requirement to call `fit()` before `predict()`. This isn't a limitation but a feature—zero-shot inference is the key innovation that makes these models immediately useful without training.

### Design Decision: Using GlobalForecastingModel

> **Why not create a separate `FoundationForecastingModel` base class?**
>
> TimesFM intentionally extends `GlobalForecastingModel` to integrate seamlessly with existing Darts workflows. While a custom base class exists (`FoundationForecastingModel`) for future models that may deviate further from Darts conventions, TimesFM's zero-shot paradigm actually *enhances* rather than replaces the standard API:
>
> - ✅ Works with `historical_forecasts()`, `backtest()`, and other Darts utilities
> - ✅ Familiar API for existing Darts users
> - ✅ Can be dropped into ensemble models
> - ✅ Integrates with Darts metrics and evaluation frameworks
>
> This "compatibility-first" design means you can use TimesFM anywhere you'd use a traditional Darts model, but with the added superpower of zero-shot forecasting.

### The fit() Method: Validation, Not Training

> **Why does TimesFM have a `fit()` method if it doesn't train?**
>
> For API compatibility and input validation:
> - **Validation**: Checks series are univariate, lengths are sufficient
> - **Model loading**: Loads the pre-trained checkpoint (if not already loaded)
> - **No training**: Pre-trained weights remain frozen—no gradient updates
> - **Darts utilities**: Some tools (like `historical_forecasts`) require calling `fit()` before `predict()`
>
> You can also use true zero-shot: call `predict()` directly without `fit()`, and the model will lazy-load automatically.

### Lazy Loading Pattern

> **How does zero-shot prediction without fit() work?**
>
> TimesFM implements **lazy model loading**: when you call `predict()` without first calling `fit()`, the model automatically loads the pre-trained checkpoint on first use. This enables the most direct forecasting workflow:
>
> ```python
> model = TimesFMModel()
> forecast = model.predict(n=12, series=my_series)  # No fit() needed!
> ```
>
> The model downloads once, then caches for subsequent predictions. This is the foundation model paradigm—immediate utility with no configuration.

## Using TimesFM (PyTorch Version)

[TimesFM](https://blog.research.google/2024/02/a-decoder-only-foundation-model-for.html) is Google's foundation model for time series, trained on [100 billion time points](https://arxiv.org/abs/2310.10688). Darts provides access to the [PyTorch implementation](https://huggingface.co/google/timesfm-2.5-200m-pytorch) (timesfm-2.5-torch), offering efficient inference on CPU, Apple Silicon (MPS), and NVIDIA GPUs.

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

TimesFM automatically selects the best available device, but you can specify it explicitly:

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

TimesFM currently supports:
- **Univariate forecasting only**: Forecasts one time series at a time (no cross-series dependencies)
- **Zero-shot inference**: No training or fine-tuning (uses pre-trained weights as-is)
- **200M parameter model**: TimesFM 2.5 from HuggingFace
- **No covariates**: External variables not supported

Despite these limitations, TimesFM often outperforms traditional models without any training, making it valuable for rapid prototyping and cold-start scenarios.

## Future: Chronos 2 Integration

The next foundation model integration is **Chronos 2** from Amazon Science ([Issue #2933](https://github.com/unit8co/darts/issues/2933)).

Released in [v2.0.0](https://github.com/amazon-science/chronos-forecasting/releases/tag/v2.0.0), [Chronos 2](https://arxiv.org/abs/2403.07815) will complement TimesFM with:
- **True multivariate forecasting**: Model cross-dependencies between multiple related series
- **Covariate support**: Past and future exogenous variables
- **Probabilistic forecasts**: Quantile-based uncertainty
- **Extended context**: 8,192 tokens vs TimesFM's 512

**Timeline**: Planned for Q1 2026.

## Learn More

- **[Tutorial Notebook](../../examples/25-TimesFM-foundation-model.ipynb)** - Hands-on examples with real datasets
- **[Issue #2359](https://github.com/unit8co/darts/issues/2359)** - Foundation models tracking (April 2024)
- **[Issue #2933](https://github.com/unit8co/darts/issues/2933)** - Chronos 2 integration request (October 2025)
- **[Foundation Models Survey](https://arxiv.org/abs/2108.07258)** - Comprehensive overview of pre-training paradigms
- **[TimesFM Paper](https://arxiv.org/abs/2310.10688)** - Technical details on decoder-only architecture
- **[Chronos Paper](https://arxiv.org/abs/2403.07815)** - Probabilistic forecasting with language model techniques