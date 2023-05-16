"""
This is the main file for the benchmarking experiment.
"""
import os

from benchmark_tools import Dataset, convert_to_ts, experiment
from sklearn.preprocessing import StandardScaler

from darts.dataprocessing.transformers import Scaler
from darts.datasets import (
    AirPassengersDataset,
    ETTh1Dataset,
    ExchangeRateDataset,
    GasRateCO2Dataset,
    SunspotsDataset,
    USGasolineDataset,
    WeatherDataset,
)
from darts.models import (
    FFT,
    AutoARIMA,
    LightGBMModel,
    LinearRegressionModel,
    NaiveSeasonal,
    NBEATSModel,
    NHiTSModel,
    NLinearModel,
    Prophet,
    TCNModel,
)
from darts.utils import missing_values

# Loading the models to benchmark


models = [
    NaiveSeasonal,
    FFT,
    Prophet,
    NLinearModel,
    LightGBMModel,  # some warnings for boosting and num_iterations overriding some parameters but it works
    TCNModel,
    NHiTSModel,
    NBEATSModel,
    LinearRegressionModel,
    # ARIMA,  # Raytune gets stuck on this one
    AutoARIMA,
]

# loading the datasets to use for the benchmark
max_ts_length = 7000
scaler = Scaler(StandardScaler())

datasets = []
ds = scaler.fit_transform(convert_to_ts(GasRateCO2Dataset().load()["CO2%"]))[
    :max_ts_length
]
datasets += [Dataset(series=ds, name="GasRateCO2")]

ds = scaler.fit_transform(
    missing_values.fill_missing_values(WeatherDataset().load().resample("1h"))
)[:max_ts_length]
datasets += [
    Dataset(
        name="Weather",
        series=ds["T (degC)"],  # type: ignore
        past_covariates=ds[  # type: ignore
            [
                "p (mbar)",
                "rh (%)",
                "VPmax (mbar)",
                "VPact (mbar)",
                "VPdef (mbar)",
                "H2OC (mmol/mol)",
                "rho (g/m**3)",
                "wv (m/s)",
                "wd (deg)",
                "rain (mm)",
                "raining (s)",
                "SWDR (W/m²)",
            ]
        ],
    )
]

ds = scaler.fit_transform(missing_values.fill_missing_values(ETTh1Dataset().load()))[
    :max_ts_length
]
datasets += [
    Dataset(
        name="ETTh1",
        series=ds["OT"],  # type: ignore
        future_covariates=ds[["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"]],  # type: ignore
    )
]

ds = scaler.fit_transform(
    missing_values.fill_missing_values(
        convert_to_ts(ExchangeRateDataset().load()["0"])
    )[:max_ts_length]
)
datasets += [Dataset(series=ds, name="ExchangeRate")]

ds = scaler.fit_transform(
    missing_values.fill_missing_values(SunspotsDataset().load()["Sunspots"])
)[:max_ts_length]
datasets += [Dataset(series=ds, name="Sunspots")]

ds = scaler.fit_transform(
    missing_values.fill_missing_values(AirPassengersDataset().load()["#Passengers"])
)[:max_ts_length]
datasets += [Dataset(series=ds, name="Air passengers")]


ds = scaler.fit_transform(
    missing_values.fill_missing_values(USGasolineDataset().load()["Gasoline"])
)[:max_ts_length]
datasets += [Dataset(series=ds, name="USGasoline")]

if __name__ == "__main__":
    experiment(
        list_datasets=datasets,
        models=models,
        grid_search=True,
        forecast_horizon=0.05,
        time_budget=600,
        repeat=5,
        silent_search=True,
        experiment_dir=os.path.join(os.getcwd(), "results_long"),
    )
