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

from typing import TYPE_CHECKING

from darts.utils._lazy import setup_lazy_imports

if TYPE_CHECKING:
    from darts.datasets.datasets import AirPassengersDataset as AirPassengersDataset
    from darts.datasets.datasets import AusBeerDataset as AusBeerDataset
    from darts.datasets.datasets import (
        AustralianTourismDataset as AustralianTourismDataset,
    )
    from darts.datasets.datasets import (
        ElectricityConsumptionZurichDataset as ElectricityConsumptionZurichDataset,
    )
    from darts.datasets.datasets import ElectricityDataset as ElectricityDataset
    from darts.datasets.datasets import EnergyDataset as EnergyDataset
    from darts.datasets.datasets import ETTh1Dataset as ETTh1Dataset
    from darts.datasets.datasets import ETTh2Dataset as ETTh2Dataset
    from darts.datasets.datasets import ETTm1Dataset as ETTm1Dataset
    from darts.datasets.datasets import ETTm2Dataset as ETTm2Dataset
    from darts.datasets.datasets import ExchangeRateDataset as ExchangeRateDataset
    from darts.datasets.datasets import GasRateCO2Dataset as GasRateCO2Dataset
    from darts.datasets.datasets import HeartRateDataset as HeartRateDataset
    from darts.datasets.datasets import IceCreamHeaterDataset as IceCreamHeaterDataset
    from darts.datasets.datasets import ILINetDataset as ILINetDataset
    from darts.datasets.datasets import MonthlyMilkDataset as MonthlyMilkDataset
    from darts.datasets.datasets import (
        MonthlyMilkIncompleteDataset as MonthlyMilkIncompleteDataset,
    )
    from darts.datasets.datasets import SunspotsDataset as SunspotsDataset
    from darts.datasets.datasets import TaxiNewYorkDataset as TaxiNewYorkDataset
    from darts.datasets.datasets import TaylorDataset as TaylorDataset
    from darts.datasets.datasets import TemperatureDataset as TemperatureDataset
    from darts.datasets.datasets import TrafficDataset as TrafficDataset
    from darts.datasets.datasets import UberTLCDataset as UberTLCDataset
    from darts.datasets.datasets import USGasolineDataset as USGasolineDataset
    from darts.datasets.datasets import WeatherDataset as WeatherDataset
    from darts.datasets.datasets import WineDataset as WineDataset
    from darts.datasets.datasets import WoolyDataset as WoolyDataset

_LAZY_IMPORTS: dict[str, str] = {
    "AirPassengersDataset": "darts.datasets.datasets",
    "AusBeerDataset": "darts.datasets.datasets",
    "AustralianTourismDataset": "darts.datasets.datasets",
    "ElectricityConsumptionZurichDataset": "darts.datasets.datasets",
    "ElectricityDataset": "darts.datasets.datasets",
    "EnergyDataset": "darts.datasets.datasets",
    "ETTh1Dataset": "darts.datasets.datasets",
    "ETTh2Dataset": "darts.datasets.datasets",
    "ETTm1Dataset": "darts.datasets.datasets",
    "ETTm2Dataset": "darts.datasets.datasets",
    "ExchangeRateDataset": "darts.datasets.datasets",
    "GasRateCO2Dataset": "darts.datasets.datasets",
    "HeartRateDataset": "darts.datasets.datasets",
    "IceCreamHeaterDataset": "darts.datasets.datasets",
    "ILINetDataset": "darts.datasets.datasets",
    "MonthlyMilkDataset": "darts.datasets.datasets",
    "MonthlyMilkIncompleteDataset": "darts.datasets.datasets",
    "SunspotsDataset": "darts.datasets.datasets",
    "TaxiNewYorkDataset": "darts.datasets.datasets",
    "TaylorDataset": "darts.datasets.datasets",
    "TemperatureDataset": "darts.datasets.datasets",
    "TrafficDataset": "darts.datasets.datasets",
    "UberTLCDataset": "darts.datasets.datasets",
    "USGasolineDataset": "darts.datasets.datasets",
    "WeatherDataset": "darts.datasets.datasets",
    "WineDataset": "darts.datasets.datasets",
    "WoolyDataset": "darts.datasets.datasets",
}

__all__, __getattr__, __dir__ = setup_lazy_imports(_LAZY_IMPORTS, __name__, globals())
