"""
Datasets
========

A few popular time series datasets.

Overall usage of this package:

.. highlight:: python
.. code-block:: python

    from darts.datasets import AirPassengersDataset
    ts: TimeSeries = AirPassengersDataset.load()
..

Univariate Datasets
-------------------

- :class:`~darts.datasets.datasets.AirPassengersDataset` - Monthly air passengers (1949-1960)
- :class:`~darts.datasets.datasets.AusBeerDataset` - Quarterly beer production in Australia (1956-2008)
- :class:`~darts.datasets.datasets.HeartRateDataset` - Heart rate measurements (1800 evenly-spaced points)
- :class:`~darts.datasets.datasets.MonthlyMilkDataset` - Monthly milk production (1962-1975)
- :class:`~darts.datasets.datasets.MonthlyMilkIncompleteDataset` - Monthly milk production with missing values
  (1962-1975)
- :class:`~darts.datasets.datasets.SunspotsDataset` - Monthly sunspot numbers (1749-1983)
- :class:`~darts.datasets.datasets.TaylorDataset` - Half-hourly electricity demand in England and Wales (2000)
- :class:`~darts.datasets.datasets.TaxiNewYorkDataset` - Taxi passengers in New York (2014-2015)
- :class:`~darts.datasets.datasets.TemperatureDataset` - Daily temperature in Melbourne (1981-1990)
- :class:`~darts.datasets.datasets.USGasolineDataset` - Weekly U.S. gasoline product supply (1991-2021)
- :class:`~darts.datasets.datasets.WineDataset` - Monthly wine sales in Australia (1980-1994)
- :class:`~darts.datasets.datasets.WoolyDataset` - Quarterly woollen yarn production in Australia (1965-1994)

Multivariate Datasets
---------------------

- :class:`~darts.datasets.datasets.AustralianTourismDataset` - Monthly tourism numbers by region/reason in Australia
- :class:`~darts.datasets.datasets.ElectricityDataset` - Electric power consumption (370 households, 15-min sampling)
- :class:`~darts.datasets.datasets.ElectricityConsumptionZurichDataset` - Electricity consumption in Zurich with
  weather (2015-2022)
- :class:`~darts.datasets.datasets.EnergyDataset` - Hourly energy consumption/generation/prices (2014-2018)
- :class:`~darts.datasets.datasets.ETTh1Dataset` - Electricity transformer temperature (hourly, 2016-2018)
- :class:`~darts.datasets.datasets.ETTh2Dataset` - Electricity transformer temperature (hourly, 2016-2018)
- :class:`~darts.datasets.datasets.ETTm1Dataset` - Electricity transformer temperature (15-min, 2016-2018)
- :class:`~darts.datasets.datasets.ETTm2Dataset` - Electricity transformer temperature (15-min, 2016-2018)
- :class:`~darts.datasets.datasets.ExchangeRateDataset` - Daily exchange rates (8 countries, 1990-2016)
- :class:`~darts.datasets.datasets.GasRateCO2Dataset` - Gas rate and CO2 measurements
- :class:`~darts.datasets.datasets.IceCreamHeaterDataset` - Monthly sales of heaters and ice cream (2004-2020)
- :class:`~darts.datasets.datasets.ILINetDataset` - Influenza-like illness patients (weekly, 1997-2022)
- :class:`~darts.datasets.datasets.TrafficDataset` - Hourly road occupancy rates (862 sensors, 2015-2016)
- :class:`~darts.datasets.datasets.UberTLCDataset` - Uber pickups by location (14.3M records, 2015)
- :class:`~darts.datasets.datasets.WeatherDataset` - Weather indicators (21 components, 10-min, 2020)
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
