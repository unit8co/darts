# Foundation Model Integration Architecture

## Summary

This document proposes a new `FoundationForecastingModel` base class to properly integrate time series foundation models (TSFMs) into Darts. These models fundamentally break the traditional fit()-then-predict() paradigm, requiring a distinct architectural approach to prevent API confusion and maintain semantic clarity.

## 1. The Problem Statement

Foundation models for time series forecasting represent a paradigm shift in how we approach forecasting tasks. Unlike traditional models that require training on specific datasets, foundation models leverage massive pre-training to enable:

- **Zero-shot forecasting**: Direct prediction without any training
- **Few-shot learning**: Task adaptation through example pairs
- **Cross-domain transfer**: Knowledge from diverse time series domains

However, these capabilities **fundamentally violate** the core contract of Darts' existing `GlobalForecastingModel` class:

```python
# Current GlobalForecastingModel contract
model.fit(series)      # MANDATORY: Train on specific data
model.predict(n=12)    # Use fitted parameters for prediction

# Foundation model reality
model.predict(n=12)    # Direct prediction without fit()!
```

Without proper architecture, we risk:
- **API Confusion**: Users calling unnecessary fit() methods
- **Semantic Pollution**: Overloading existing parameters with new meanings
- **Limited Flexibility**: Inability to support emerging TSFM architectures

## 2. Why GlobalForecastingModel Won't Work

### 2.1 Contract Violation

The `GlobalForecastingModel` documentation explicitly states:
> "All implementations must implement the fit() and predict() methods. The fit() method is meant to train the model on one or several training time series."

Foundation models **cannot honor this contract**:
- Zero-shot models have no parameters to fit
- Pre-trained weights are frozen and task-agnostic
- fit() becomes a no-op, violating user expectations

### 2.2 API Clutter and Semantic Confusion

Attempting to shoehorn foundation model capabilities into existing APIs creates confusion:

```python
# Confusing: What does 'series' mean here?
model.predict(
    series=context_series,  # Historical context? Or training examples?
    future_covariates=...,   # Actual covariates? Or few-shot examples?
)
```

### 2.3 Inflexibility for Evolving Architectures

The TSFM landscape is rapidly evolving with distinct architectural patterns:
- **In-context learning** (TimesFM, Chronos)
- **Covariate-aware models** (Moirai)
- **Probabilistic architectures** (Lag-Llama)
- **Mixture of experts** (Time-MoE)

Each requires specific API affordances that don't fit the traditional mold.

## 3. The Solution: FoundationForecastingModel Base Class

### 3.1 Core Design Principles

```python
class FoundationForecastingModel(ForecastingModel):
    """Base class for foundation models with fit-less prediction as default."""

    # Establishes new contract
    def fit(self, series, ...):
        """Optional fine-tuning. Not required for zero-shot prediction."""
        pass  # Default: no-op for zero-shot models

    def predict(self, n, series=None, examples=None, ...):
        """Prediction with optional in-context learning examples."""
        pass
```

### 3.2 Key Architectural Benefits

1. **Clear Contract**: "Fit-less prediction" as the default behavior
2. **Semantic Clarity**: Distinct parameters for distinct concepts
3. **Future-Proof**: Extensible for emerging TSFM architectures
4. **Backward Compatible**: Inherits from ForecastingModel for ecosystem integration

## 4. Critical Distinction: Examples vs Covariates

One of the most important architectural decisions is clearly distinguishing between **covariates** and **few-shot examples**:

### 4.1 Detailed Comparison

| Aspect | Covariates | Few-Shot Examples |
|--------|------------|-------------------|
| **Purpose** | Provide external variables that influence the target | Demonstrate the task through input-output pairs |
| **Nature** | Time-aligned exogenous variables | Complete time series with known futures |
| **Examples** | Temperature, price, holidays | Similar series from same domain |
| **Structure** | `TimeSeries` objects aligned with target | `List[Tuple[TimeSeries, TimeSeries]]` pairs |
| **Time Alignment** | Must align with target series timestamps | Independent series, any time range |
| **Model Usage** | Fed at each time step during prediction | Used to condition model's task understanding |
| **Persistence** | Extracted features used throughout | Ephemeral, call-specific state |
| **Semantic Role** | "What external factors affect this series?" | "Here's how similar series behave" |

### 4.2 Why This Distinction Matters

```python
# WRONG: Conflating concepts in GlobalForecastingModel
model.predict(
    series=historical_context,
    future_covariates=few_shot_examples,  # Semantic confusion!
)

# RIGHT: Clear separation in FoundationForecastingModel
model.predict(
    series=historical_context,
    future_covariates=actual_covariates,  # Real external variables
    examples=few_shot_pairs,               # Task demonstration pairs
)
```

### 4.3 Implementation Examples

```python
# Covariates: Time-aligned external variables
temperature = TimeSeries.from_dataframe(df[['temperature']])
holidays = TimeSeries.from_dataframe(df[['is_holiday']])

# Few-shot examples: Complete series pairs for task demonstration
examples = [
    (store1_history, store1_future),  # Example 1: Input → Output
    (store2_history, store2_future),  # Example 2: Input → Output
    (store3_history, store3_future),  # Example 3: Input → Output
]

# Clear API usage
forecast = model.predict(
    n=28,
    series=target_store_history,
    future_covariates=temperature,  # Actual temperature predictions
    examples=examples,               # How similar stores performed
)
```

## 5. API Specification

### 5.1 FoundationForecastingModel Interface

```python
class FoundationForecastingModel(ForecastingModel):
    """
    Base class for foundation time series models.

    Foundation models are pre-trained on massive datasets and can perform
    zero-shot or few-shot forecasting without traditional training.
    """

    def predict(
        self,
        n: int,
        series: Optional[Union[TimeSeries, List[TimeSeries]]] = None,
        examples: Optional[List[Tuple[TimeSeries, TimeSeries]]] = None,
        past_covariates: Optional[Union[TimeSeries, List[TimeSeries]]] = None,
        future_covariates: Optional[Union[TimeSeries, List[TimeSeries]]] = None,
        num_samples: int = 1,
        verbose: bool = False,
        **kwargs
    ) -> Union[TimeSeries, List[TimeSeries]]:
        """
        Generate forecasts using the foundation model.

        Parameters
        ----------
        n : int
            Number of time steps to forecast.

        series : TimeSeries or List[TimeSeries], optional
            Historical context for prediction. If not provided, uses
            series from fit() if available.

        examples : List[Tuple[TimeSeries, TimeSeries]], optional
            Few-shot learning examples as (context, future) pairs.
            These demonstrate the forecasting task to the model through
            in-context learning. Each tuple contains:
            - context: Historical values of an example series
            - future: Known future values of that same series

            Example:
                examples = [
                    (series1[:-24], series1[-24:]),  # 24-step ahead example
                    (series2[:-24], series2[-24:]),  # Another example
                ]

        past_covariates : TimeSeries or List[TimeSeries], optional
            Exogenous series known for the input time range. Must be
            time-aligned with the target series.

        future_covariates : TimeSeries or List[TimeSeries], optional
            Exogenous series known for the forecast time range. Must
            extend n steps beyond the target series.

        num_samples : int, default=1
            Number of forecast samples to generate for probabilistic models.

        Returns
        -------
        TimeSeries or List[TimeSeries]
            Forecasted values for the next n time steps.
        """
        pass

    def fit(
        self,
        series: Union[TimeSeries, List[TimeSeries]],
        past_covariates: Optional[Union[TimeSeries, List[TimeSeries]]] = None,
        future_covariates: Optional[Union[TimeSeries, List[TimeSeries]]] = None,
        **kwargs
    ) -> 'FoundationForecastingModel':
        """
        Optional fine-tuning of the foundation model.

        Note: This method is OPTIONAL for foundation models. Zero-shot
        models can generate predictions without calling fit().

        Parameters
        ----------
        series : TimeSeries or List[TimeSeries]
            Target series for fine-tuning.

        Returns
        -------
        self
            Fitted model instance.
        """
        # Default: no-op for zero-shot models
        logger.info("Zero-shot model - fit() is optional")
        return self
```

## 6. TimesFM Implementation

### 6.1 Target Implementation

```python
class TimesFMModel(FoundationForecastingModel):
    """
    TimesFM: Google's decoder-only foundation model for time series.

    Pre-trained on 100B+ time points across diverse domains.
    Supports zero-shot and (upcoming) few-shot forecasting.
    """

    def __init__(
        self,
        model_version: str = "2.5",  # Target PyTorch version
        model_size: str = "200m",    # 200m or 500m parameters
        backend: str = "pytorch",     # Use PyTorch backend
        device: Optional[str] = None,  # Auto-detect CUDA/MPS
        **kwargs
    ):
        # Implementation targeting timesfm_2p5_torch
        pass
```

### 6.2 Key Implementation Details

- **PyTorch Backend**: Target `timesfm-pytorch` for better device support
- **Device Handling**: Automatic MPS (Apple Silicon) and CUDA detection
- **Zero-Shot Ready**: No fit() required for immediate use
- **Few-Shot Prepared**: API ready for Google's upcoming few-shot release

## 7. Roadmap for Other Foundation Models

### 7.1 Immediate Targets

| Model | Organization | Key Features | Implementation Priority |
|-------|--------------|--------------|------------------------|
| **Chronos** | Amazon | T5-based, multiple sizes (20M-710M) | High |
| **Moirai** | Salesforce | **Supports covariates**, multiple architectures | High |
| **Lag-Llama** | ServiceNow | Probabilistic, univariate specialist | Medium |
| **Time-MoE** | Google | Mixture of experts, scale efficiency | Medium |

### 7.2 Implementation Strategy

```python
# Chronos - Minimal changes needed
class ChronosModel(FoundationForecastingModel):
    """Amazon's T5-based foundation model."""
    pass

# Moirai - Leverages covariate support
class MoiraiModel(FoundationForecastingModel):
    """Salesforce's covariate-aware foundation model."""

    def predict(self, ..., past_covariates=None, future_covariates=None):
        # Native covariate support!
        pass

# Lag-Llama - Probabilistic focus
class LagLlamaModel(FoundationForecastingModel):
    """Probabilistic univariate specialist."""

    def predict(self, ..., num_samples=100):
        # Emphasizes probabilistic forecasting
        pass
```

## 8. Migration Path and Compatibility

### 8.1 Backward Compatibility

The `FoundationForecastingModel` maintains full backward compatibility:
- Inherits from `ForecastingModel` for ecosystem integration
- Works with existing utilities (backtest, ensemble, etc.)
- Supports traditional workflows when needed

### 8.2 Migration Strategy

```python
# Phase 1: Introduce FoundationForecastingModel
# - No breaking changes
# - TimesFM uses new base class

# Phase 2: Gradual adoption
# - New foundation models use the base class
# - Documentation emphasizes the distinction

# Phase 3: Ecosystem evolution
# - Utilities recognize foundation models
# - Optimized pipelines for zero-shot workflows
```

## 9. Industry Precedent

### 9.1 Hugging Face Transformers

The NLP community faced similar challenges with foundation models:

```python
# Hugging Face's solution: Task-specific pipelines
from transformers import pipeline

# Clear separation of concerns
generator = pipeline("text-generation", model="gpt2")
output = generator("Hello", max_length=50, num_return_sequences=3)
```

### 9.2 Lessons Learned

1. **Don't force-fit new paradigms into old APIs**
2. **Provide purpose-built abstractions**
3. **Maintain clear semantic boundaries**
4. **Enable gradual migration paths**

## 10. Conclusion

The `FoundationForecastingModel` base class is essential for properly integrating TSFMs into Darts. It:

1. **Prevents API confusion** through clear contracts
2. **Maintains semantic clarity** with distinct parameters
3. **Enables innovation** with flexible architecture
4. **Scales naturally** for the emerging TSFM landscape

Without this architecture, we risk creating a confusing API that conflates fundamentally different concepts and limits our ability to leverage the full potential of foundation models.

## References

### GitHub Issues
- [#2911: TimesFM Integration](https://github.com/unit8co/darts/pull/2911)
- [#2359: Foundation Model Support Discussion](https://github.com/unit8co/darts/issues/2359)
- [#2925: Covariate Handling in Global Models](https://github.com/unit8co/darts/issues/2925)

### Academic Papers
1. **TimesFM**: Das et al., "A decoder-only foundation model for time-series forecasting", ICML 2024. [arXiv:2310.10688](https://arxiv.org/abs/2310.10688)

2. **Chronos**: Ansari et al., "Chronos: Learning the Language of Time Series", 2024. [arXiv:2403.07815](https://arxiv.org/abs/2403.07815)

3. **Moirai**: Woo et al., "Unified Training of Universal Time Series Forecasting Transformers", 2024. [arXiv:2402.02592](https://arxiv.org/abs/2402.02592)

4. **Lag-Llama**: Rasul et al., "Lag-Llama: Towards Foundation Models for Time Series Forecasting", 2023. [arXiv:2310.08278](https://arxiv.org/abs/2310.08278)

5. **Time-MoE**: Zhou et al., "Time-MoE: Mixture of Experts for Time Series Foundation Models", 2024. [arXiv:2409.16040](https://arxiv.org/abs/2409.16040)

### Implementation References
- [Google TimesFM Repository](https://github.com/google-research/timesfm)
- [Hugging Face TimesFM Models](https://huggingface.co/google/timesfm-1.0-200m-pytorch)
- [Amazon Chronos](https://github.com/amazon-science/chronos-forecasting)
- [Salesforce Moirai](https://github.com/SalesforceAIResearch/uni2ts)

---

*This architecture document is intended for maintainer review and discussion. It represents a forward-looking approach to integrating foundation models while maintaining the integrity and usability of the Darts ecosystem.*