"""
Datasets
--------

A few popular time series datasets.

Overall usage of this package:

.. highlight:: python
.. code-block:: python

    from darts.datasets import AirPassengersDataset
    ts: TimeSeries = AirPassengersDataset.load()
..
"""

from darts.datasets.datasets import (
    AirPassengersDataset,
    AusBeerDataset,
    AustralianTourismDataset,
    ElectricityConsumptionZurichDataset,
    ElectricityDataset,
    EnergyDataset,
    ETTh1Dataset,
    ETTh2Dataset,
    ETTm1Dataset,
    ETTm2Dataset,
    ExchangeRateDataset,
    GasRateCO2Dataset,
    HeartRateDataset,
    IceCreamHeaterDataset,
    ILINetDataset,
    MonthlyMilkDataset,
    MonthlyMilkIncompleteDataset,
    SunspotsDataset,
    TaxiNewYorkDataset,
    TaylorDataset,
    TemperatureDataset,
    TrafficDataset,
    UberTLCDataset,
    USGasolineDataset,
    WeatherDataset,
    WineDataset,
    WoolyDataset,
)

__all__ = [
    "AirPassengersDataset",
    "AusBeerDataset",
    "AustralianTourismDataset",
    "EnergyDataset",
    "GasRateCO2Dataset",
    "HeartRateDataset",
    "IceCreamHeaterDataset",
    "MonthlyMilkDataset",
    "MonthlyMilkIncompleteDataset",
    "SunspotsDataset",
    "TaylorDataset",
    "TemperatureDataset",
    "USGasolineDataset",
    "WineDataset",
    "WoolyDataset",
    "ETTh1Dataset",
    "ETTh2Dataset",
    "ETTm1Dataset",
    "ETTm2Dataset",
    "TaxiNewYorkDataset",
    "ElectricityDataset",
    "UberTLCDataset",
    "ILINetDataset",
    "ExchangeRateDataset",
    "TrafficDataset",
    "WeatherDataset",
    "ElectricityConsumptionZurichDataset",
]
