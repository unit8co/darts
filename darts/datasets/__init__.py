"""
Datasets
--------

A few popular time series datasets
"""

from .dataset_loaders import DatasetLoaderCSV, DatasetLoaderMetadata

"""
    Overall usage of this package:
    from darts.datasets import AirPassengersDataset
    ts: TimeSeries = AirPassengersDataset.load()
"""

_DEFAULT_PATH = "https://raw.githubusercontent.com/unit8co/darts/master/datasets"


class AirPassengersDataset(DatasetLoaderCSV):
    """
    Monthly Air Passengers Dataset, from 1949 to 1960.
    """
    def __init__(self):
        super().__init__(metadata=DatasetLoaderMetadata(
            "air_passengers.csv",
            uri=_DEFAULT_PATH+"/AirPassengers.csv",
            hash="167ffa96204a2b47339c21eea25baf32",
            header_time="Month"
        ))


class AusBeerDataset(DatasetLoaderCSV):
    """
    Total quarterly beer production in Australia (in megalitres) from 1956:Q1 to 2008:Q3.

    https://rdrr.io/cran/fpp/man/ausbeer.html
    """
    def __init__(self):
        super().__init__(metadata=DatasetLoaderMetadata(
            "ausbeer.csv",
            uri=_DEFAULT_PATH+"/ausbeer.csv",
            hash="1f4028a570a20939411cc04de7364bbd",
            header_time="date",
            format_time="%Y-%m-%d"
        ))


class EnergyDataset(DatasetLoaderCSV):
    """
    Hourly energy dataset coming from
    https://www.kaggle.com/nicholasjhana/energy-consumption-generation-prices-and-weather

    Contains a time series with 28 hourly components between 2014-12-31 23:00:00 and 2018-12-31 22:00:00
    """
    def __init__(self):
        super().__init__(metadata=DatasetLoaderMetadata(
            "energy.csv",
            uri=_DEFAULT_PATH+"/energy_dataset.csv",
            hash="f564ef18e01574734a0fa20806d1c7ee",
            header_time="time",
            format_time="%Y-%m-%d %H:%M:%S"
        ))


class GasRateCO2Dataset(DatasetLoaderCSV):
    """
    Gas Rate CO2 dataset
    Two components, length 296 (integer time index)
    """

    def __init__(self):
        super().__init__(metadata=DatasetLoaderMetadata(
            "gasrate_co2.csv",
            uri=_DEFAULT_PATH+"/gasrate_co2.csv",
            hash="77bf383715a9cf81459f81fe17baf3b0",
            header_time=None,
            format_time=None
        ))


class HeartRateDataset(DatasetLoaderCSV):
    """
    The series contains 1800 evenly-spaced measurements of instantaneous heart rate from a single subject.
    The measurements (in units of beats per minute) occur at 0.5 second intervals, so that the length of
    each series is exactly 15 minutes.

    This is the series1 here: http://ecg.mit.edu/time-series/
    Using an integer time index.
    """

    def __init__(self):
        super().__init__(metadata=DatasetLoaderMetadata(
            "heart_rate.csv",
            uri=_DEFAULT_PATH+"/heart_rate.csv",
            hash="3c4a108e1116867cf056dc5be2c95386",
            header_time=None,
            format_time=None
        ))


class IceCreamHeaterDataset(DatasetLoaderCSV):
    """
    Monthly sales of heaters and ice cream between January 2004 and June 2020.
    """

    def __init__(self):
        super().__init__(metadata=DatasetLoaderMetadata(
            "ice_cream_heater.csv",
            uri=_DEFAULT_PATH+"/ice_cream_heater.csv",
            hash="62031c7b5cdc9339fe7cf389173ef1c3",
            header_time="Month",
            format_time="%Y-%m"
        ))


class MonthlyMilkDataset(DatasetLoaderCSV):
    """
    Monthly production of milk (in pounds per cow) between January 1962 and December 1975
    """

    def __init__(self):
        super().__init__(metadata=DatasetLoaderMetadata(
            "monthly_milk.csv",
            uri=_DEFAULT_PATH+"/monthly-milk.csv",
            hash="4784443e696da45d7082e76a67687b93",
            header_time="Month",
            format_time="%Y-%m"
        ))


class MonthlyMilkIncompleteDataset(DatasetLoaderCSV):
    """
    Monthly production of milk (in pounds per cow) between January 1962 and December 1975.
    Has some missing values.
    """

    def __init__(self):
        super().__init__(metadata=DatasetLoaderMetadata(
            "monthly_milk-incomplete.csv",
            uri=_DEFAULT_PATH+"/monthly-milk-incomplete.csv",
            hash="49b275c7e2f8f28a6a05224be1a049a4",
            header_time="Month",
            format_time="%Y-%m",
            freq='MS'
        ))


class SunspotsDataset(DatasetLoaderCSV):
    """
    Monthly Sunspot Numbers, 1749 - 1983

    Monthly mean relative sunspot numbers from 1749 to 1983.
    Collected at Swiss Federal Observatory, Zurich until 1960, then Tokyo Astronomical Observatory.

    https://www.rdocumentation.org/packages/datasets/versions/3.6.1/topics/sunspots
    """

    def __init__(self):
        super().__init__(metadata=DatasetLoaderMetadata(
            "sunspots.csv",
            uri=_DEFAULT_PATH+"/monthly-sunspots.csv",
            hash="4d27019c43d9c256d528f1bd6c5f40e0",
            header_time="Month",
            format_time="%Y-%m"
        ))


class TaylorDataset(DatasetLoaderCSV):
    """
    Half-hourly electricity demand in England and Wales from Monday 5 June 2000 to Sunday 27 August 2000.
    Discussed in Taylor (2003) [1], and kindly provided by James W Taylor. Units: Megawatts
    (Uses an integer time index).

    https://www.rdocumentation.org/packages/forecast/versions/8.13/topics/taylor

    [1] Taylor, J.W. (2003) Short-term electricity demand forecasting using double seasonal exponential smoothing.
    Journal of the Operational Research Society, 54, 799-805.
    """

    def __init__(self):
        super().__init__(metadata=DatasetLoaderMetadata(
            "taylor.csv",
            uri=_DEFAULT_PATH+"/taylor.csv",
            hash="1ea355c90e8214cb177788a674801a22",
            header_time=None,
            format_time=None
        ))


class TemperatureDataset(DatasetLoaderCSV):
    """
    Daily temperature in Melbourne between 1981 and 1990
    """

    def __init__(self):
        super().__init__(metadata=DatasetLoaderMetadata(
            "temperatures.csv",
            uri=_DEFAULT_PATH+"/temps.csv",
            hash="ce5b5e4929793ec8b6a54711110acebf",
            header_time="Date",
            format_time="%m/%d/%Y",
            freq='D'
        ))


class USGasolineDataset(DatasetLoaderCSV):
    """
    Weekly U.S. Product Supplied of Finished Motor Gasoline between 1991-02-08 and 2021-04-30

    Obtained from:
    https://www.eia.gov/dnav/pet/hist/LeafHandler.ashx?n=PET&s=wgfupus2&f=W
    """

    def __init__(self):
        super().__init__(metadata=DatasetLoaderMetadata(
            "us_gasoline.csv",
            uri=_DEFAULT_PATH+"/us_gasoline.csv",
            hash="25d440337a06cbf83423e81d0337a1ce",
            header_time="Week",
            format_time="%m/%d/%Y"
        ))


class WineDataset(DatasetLoaderCSV):
    """
    Australian total wine sales by wine makers in bottles <= 1 litre. Monthly between Jan 1980 and Aug 1994.

    https://www.rdocumentation.org/packages/forecast/versions/8.1/topics/wineind
    """

    def __init__(self):
        super().__init__(metadata=DatasetLoaderMetadata(
            "wine.csv",
            uri=_DEFAULT_PATH+"/wineind.csv",
            hash="b68971d7e709ad0b7e6300cab977e3cd",
            header_time="date",
            format_time="%Y-%m-%d"
        ))


class WoolyDataset(DatasetLoaderCSV):
    """
    Quarterly production of woollen yarn in Australia: tonnes. Mar 1965 -- Sep 1994.

    https://www.rdocumentation.org/packages/forecast/versions/8.1/topics/woolyrnq
    """

    def __init__(self):
        super().__init__(metadata=DatasetLoaderMetadata(
            "wooly.csv",
            uri=_DEFAULT_PATH+"/woolyrnq.csv",
            hash="4be8b12314db94c8fd76f5c674454bf0",
            header_time="date",
            format_time="%Y-%m-%d"
        ))
