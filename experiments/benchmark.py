"""
This is the main file for the benchmarking experiment.
"""
import pandas as pd
from model_evaluation import evaluate_model
from optuna_search import optuna_search
from param_space import FIXED_PARAMS

from darts import TimeSeries
from darts.datasets import (
    ETTh1Dataset,
    ExchangeRateDataset,
    GasRateCO2Dataset,
    SunspotsDataset,
    WeatherDataset,
)
from darts.metrics import mae
from darts.models import (
    ARIMA,
    FFT,
    CatBoostModel,
    LinearRegressionModel,
    NaiveSeasonal,
    NBEATSModel,
    NHiTSModel,
    Prophet,
    TCNModel,
)
from darts.utils import missing_values


def convert_to_ts(ds: TimeSeries):
    return TimeSeries.from_times_and_values(
        pd.to_datetime(ds.time_index), ds.all_values(), columns=ds.components
    )


metric = mae
models = [
    NaiveSeasonal,
    FFT,
    Prophet,
    ARIMA,
    TCNModel,
    NHiTSModel,
    NBEATSModel,
    LinearRegressionModel,
    CatBoostModel,
]

datasets = []

ds = convert_to_ts(GasRateCO2Dataset().load()["CO2%"])
datasets += [{"target_dataset": ds, "dataset_name": "GasRateCO2"}]
ds = missing_values.fill_missing_values(WeatherDataset().load().resample("1h"))
datasets += [
    {
        "dataset_name": "Weather",
        "target_dataset": ds["T (degC)"],
        "future_cov": ds[
            [
                "p (mbar)",
                "rh (%)",
                "VPmax (mbar)",
                "VPact (mbar)",
                "VPdef (mbar)",
                "sh (g/kg)",
                "H2OC (mmol/mol)",
                "rho (g/m**3)",
                "wv (m/s)",
                "max. wv (m/s)",
                "wd (deg)",
                "rain (mm)",
                "raining (s)",
                "SWDR (W/m²)",
                "PAR (µmol/m²/s)",
                "max. PAR (µmol/m²/s)",
                "CO2 (ppm)",
            ]
        ],
        "has_future_cov": True,
    }
]
ds = missing_values.fill_missing_values(ETTh1Dataset().load())
datasets += [
    {
        "dataset_name": "ETTh1",
        "target_dataset": ds["OT"],
        "future_cov": ds[["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"]],
        "has_future_cov": True,
    }
]

ds = missing_values.fill_missing_values(
    convert_to_ts(ExchangeRateDataset().load()["0"])
)
datasets += [{"target_dataset": ds, "dataset_name": "ExchangeRate"}]
ds = SunspotsDataset().load()["Sunspots"]
datasets += [{"target_dataset": ds, "dataset_name": "Sunspots"}]

results = []
for dataset in datasets:
    print("\n\n", dataset["dataset_name"])
    for model_class in models:
        fixed_params = FIXED_PARAMS[model_class.__name__](**dataset)
        config = optuna_search(
            model_class, fixed_params=fixed_params, **dataset, time_budget=60
        )
        output = evaluate_model(
            model_class, **dataset, model_params=fixed_params, split=0.8
        )
        print(model_class.__name__, output)
        results.append((dataset["dataset_name"], model_class.__name__, output))

print(results)
