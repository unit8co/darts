import numpy as np

from darts import TimeSeries
from darts.datasets import AirPassengersDataset
from darts.models import RNNModel, TFTModel
from darts.utils.likelihood_models import QuantileRegression
from darts.utils.timeseries_generation import datetime_attribute_timeseries

series = AirPassengersDataset().load()
train = series[:100]
test = series[100:]

covariates = datetime_attribute_timeseries(series, attribute="year", one_hot=False)
covariates = covariates.stack(
    datetime_attribute_timeseries(series, attribute="month", one_hot=False)
)
covariates = covariates.stack(
    TimeSeries.from_times_and_values(
        times=series.time_index,
        values=np.arange(len(series)),
        columns=["linear_increase"],
    )
)
covariates = covariates.astype(np.float32)

covariates_train = covariates[:100]

input_chunk_length = 24
forecast_horizon = 12
TFT = True
if TFT:
    tftmodel = TFTModel(
        input_chunk_length=input_chunk_length,
        output_chunk_length=forecast_horizon,
        hidden_size=64,
        lstm_layers=1,
        num_attention_heads=4,
        dropout=0.1,
        batch_size=16,
        n_epochs=300,
        add_relative_index=False,
        add_encoders=None,
        likelihood=QuantileRegression(),
        random_state=42,
    )

    tftmodel.fit(train, future_covariates=covariates_train, verbose=False, epochs=1)

    print(f"test time series index {test.time_index}")
    forecast_instances = tftmodel.historical_forecasts(
        series,
        future_covariates=covariates,
        start=test.time_index[0],
        forecast_horizon=forecast_horizon,
        retrain=False,
        verbose=False,
        last_points_only=False,
        overlap_end=False,
    )

    print(len(forecast_instances))
    print(forecast_instances[0].time_index)
    print(forecast_instances[-1].time_index)

else:
    rnnmodel = RNNModel(
        input_chunk_length=input_chunk_length,
        output_chunk_length=forecast_horizon,
        add_encoders=None,
        likelihood=QuantileRegression(),
        random_state=42,
    )
    rnnmodel.fit(train, future_covariates=covariates_train, verbose=False, epochs=1)

    rnn_forecast_instances = rnnmodel.historical_forecasts(
        series,
        future_covariates=covariates,
        start=test.time_index[0],
        forecast_horizon=forecast_horizon,
        retrain=False,
        verbose=False,
        last_points_only=False,
        overlap_end=False,
    )

    print(len(rnn_forecast_instances))
    print(rnn_forecast_instances[0].time_index)
    print(rnn_forecast_instances[-1].time_index)
