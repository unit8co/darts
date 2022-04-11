# Overview of Forecasting Models

Forecasting models are models that can produce predictions about future values of some time series, given the history of this series.
The forecasting models in Darts are [listed on the README](https://github.com/unit8co/darts#forecasting-models). They have different capabilities and features. For example, some models work on multidimensional series, return probabilistic forecasts, or accept other kinds of external *covariates* data in input.

Below, we give an overview of what these features mean.

## Generalities

All forecasting models work in the same way: first they are built (taking some hyper-paramers in argument), then they are fit on one or several series
by calling the `fit()` function, and finally they are used to obtain one or several forecasts by calling the `predict()` function.

Example:
```python
from darts.models import NaiveSeasonal

naive_model = NaiveSeasonal(K=1)            # init
naive_model.fit(train)                      # fit  
naive_forecast = naive_model.predict(n=36)  # predict
```

The argument `n` of `predict()` indicates the number of time stamps to predict.
When `fit()` is provided with only one training `TimeSeries`, this series is stored, and `predict()` will return forecasts for this series.
On the other hand, some models support calling `fit()` on multiple time series (a `Sequence[TimeSeries]`). In such cases, one or several series must
be provided to `predict()`, and the model will produce forecasts for this/these time series.

Example:
```python
from darts.models import NBEATSModel

model = NBEATSModel(input_chunk_length=24, 
                    output_chunk_length=12)

model.fit([train_air_scaled, train_milk_scaled], epochs=50)  # fit on two series
forecast = model.predict(series=some_other_series, n=36)     # predict another series
```

Furthermore, we define the following types of time series consummed by the models:

* **target series:** the series that we are interested in forecasting
* **covariate series:** some other series that we are not interested in forecasting, but that can provide valuable inputs to the forecasting model.


## Support for multivariate series

Some models support multivariate time series. This means that the target (and potential covariates) series provided to the model
during fit and predict stage can have multiple dimensions. The model will then produce multi-dimensional forecasts `TimeSeries`.

These models are shown with a ✅ under the `Multivariate` column on the [model list](https://github.com/unit8co/darts#forecasting-models).

## Handling multiple series

Some models support being fit on multiple time series. To do this, it is enough to simply provide a Python `Sequence` of `TimeSeries` (for instance a list of `TimeSeries`) to `fit()`. When a model is fit this way, the `predict()` function will expect the argument `series` to be set, containing
one or several `TimeSeries` (i.e., a single or a `Sequence` of `TimeSeries`) that need to be forecasted. 
The advantage of training on multiple series is that a single model can be exposed to more patterns occuring across all series in the training dataset. That can often be beneficial, especially for more expre based models.

In turn, the advantage of having `predict()` providing forecasts for potentially several series at once is that the computation can often be batched and vectorized across the multiple series, which is computationally faster than calling `predict()` multiple times on isolated series.

These models are shown with a ✅ under the `Multiple-series training` column on the [model list](https://github.com/unit8co/darts#forecasting-models).

[This article](https://medium.com/unit8-machine-learning-publication/training-forecasting-models-on-multiple-time-series-with-darts-dc4be70b1844) provides more explanations about training models on multiple series.

## Support for Covariates

Some models support *covariate series*. Covariate series are time series that the models can take as inputs, but will not forecast.
We distinguish between *past covariates* and *future covariates*:

* Past covariates are covariate time series whose values are **not** known into the future at prediction time. Those can for instance represent signals that have to be measured and are not known upfront. Models do not use the future values of `past_covariates` when making forecasts.
* Future covariates are covariate time series whose values are known into the future at prediction time (up until the forecast horizon). These can represent signals such as calendar information, holidays, weather forecasts, etc. Models that accept `future_covariates` will consume the future values (up to the forecast horizon) when making forecasts.

![covariates](./images/covariates/covariates-highlevel.png)

Past and future covariates can be used by providing respectively `past_covariates` and `future_covariates` arguments to `fit()` and `predict()`.
When a model is trained on multiple target series, one covariate has to be provided per target series. The covariate series themselves can be multivariate
and contain multiple "covariate dimensions"; see the [TimeSeries guide](https://unit8co.github.io/darts/userguide/timeseries.html) for how to build multivariate series.

Models supporting past (resp. future) covariates are indicated with a ✅ under the `Past-observed covariates support` (resp. `Future-known covariates support`) columns on the [model list](https://github.com/unit8co/darts#forecasting-models),

Have a look at [this article](https://medium.com/unit8-machine-learning-publication/time-series-forecasting-using-past-and-future-external-data-with-darts-1f0539585993) for some examples of how to use past and future covariates.

## Probabilistic forecasts

Some of the models in Darts can produce probabilistic forecasts. For these models, the `TimeSeries` returned by `predict()` will be probabilistic, and contain a certain number of Monte Carlo samples describing the joint distribution over time and components. The number of samples can be directly determined by the argument `num_samples` of the `predict()` function (leaving `num_samples=1` will return a deterministic `TimeSeries`).

The distribution of the forecasts depend on the model.

Some models such as ARIMA, Exponential Smoothing or (T)BATS make normality assumptions and the resulting distribution is a Gaussian with time-dependent parameters. For example:
```python
from darts.datasets import AirPassengersDataset
from darts import TimeSeries
from darts.models import ExponentialSmoothing

series = AirPassengersDataset().load()
train, val = series[:-36], series[-36:]

model = ExponentialSmoothing()
model.fit(train)
pred = model.predict(n=36, num_samples=500)

series.plot()
pred.plot(label='forecast')
```

![Exponential Smoothing](./images/probabilistic/example_ets.png)

### Probabilistic neural networks
All neural networks (torch-based models) in Darts have a rich support to fit different kinds of distribution. When creating the model, it is possible to provide one of the *likelihood models* available in `darts.utils.likelihood_models`, which determine the distribution that will be fit by the model. In such cases, the model will output the parameters of the distribution, and it will be trained by minimising the negative log-likelihood of the training samples. Most of the likelihood models also support prior values for the distribution's parameters, in which case the training loss is regularized by a Kullback-Leibler divergence term pushing the resulting distribution in the direction of the distribution specified by the prior parameters. Finally, it is also possible to perform quantile regression (using arbitrary quantiles) with neural networks, by using `darts.utils.likelihood_models.QuantileRegression`; in which case the network will be trained with the pinball (or quantile regression) loss. 

For example, the code below trains a TCNModel to fit a Laplace distribution. So the neural network outputs 2 parameters (location and scale) of the Laplace distribution. We also specify a prior value of 0.1 on the scale parameter.

```python
from darts.datasets import AirPassengersDataset
from darts import TimeSeries
from darts.models import TCNModel
from darts.dataprocessing.transformers import Scaler
from darts.utils.likelihood_models import LaplaceLikelihood

series = AirPassengersDataset().load()
train, val = series[:-36], series[-36:]

scaler = Scaler()
train = scaler.fit_transform(train)
val = scaler.transform(val)
series = scaler.transform(series)

model = TCNModel(input_chunk_length=30, 
                 output_chunk_length=12,
                 likelihood=LaplaceLikelihood(prior_b=0.1))
model.fit(train, epochs=400)
pred = model.predict(n=36, num_samples=500)

series.plot()
pred.plot(label='forecast')
```

![TCN Laplace regression](./images/probabilistic/example_tcn_laplace.png)


### Probabilistic regression models
Some regression models can be configured to produce probabilistic forecasts too. At the time of writing, [LinearRegressionModel](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.linear_regression_model.html) and [LightGBMModel](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.gradient_boosted_model.html) support a `likelihood` argument. When set to `"poisson"` the model will fit a Poisson distribution, and when set to `"quantile"` the model will use the pinball loss to perform quantile regression (the quantiles themselves can be specified using the `quantiles` argument).

Example:
```python
from darts.datasets import AirPassengersDataset
from darts import TimeSeries
from darts.models import LinearRegressionModel

series = AirPassengersDataset().load()
train, val = series[:-36], series[-36:]

model = LinearRegressionModel(lags=30, 
                              likelihood="quantile", 
                              quantiles=[0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])
model.fit(train)
pred = model.predict(n=36, num_samples=500)

series.plot()
pred.plot(label='forecast')
```

![quantile linear regression](./images/probabilistic/example_linreg_quantile.png)