from .dataset_loaders import DatasetLoaderCSV, DatasetLoaderMetadata

"""
    Overall usage of this package:
    from darts.datasets import AirPassengersDataset
    ts: TimeSeries = AirPassengersDataset.load()
"""

_DEFAULT_PATH = "https://raw.githubusercontent.com/unit8co/darts/develop/datasets"


AirPassengersDataset = DatasetLoaderCSV(
    metadata=DatasetLoaderMetadata(
        "air_passengers.csv",
        uri=_DEFAULT_PATH+"/AirPassengers.csv",
        hash="167ffa96204a2b47339c21eea25baf32",
        header_time="Month"
    )
)

AusBeerDataset = DatasetLoaderCSV(
    metadata=DatasetLoaderMetadata(
        "ausbeer.csv",
        uri=_DEFAULT_PATH+"/ausbeer.csv",
        hash="1f4028a570a20939411cc04de7364bbd",
        header_time="date",
        format_time="%Y-%m-%d"
    )
)

EnergyDataset = DatasetLoaderCSV(
    metadata=DatasetLoaderMetadata(
        "energy.csv",
        uri=_DEFAULT_PATH+"/energy_dataset.csv",
        hash="f564ef18e01574734a0fa20806d1c7ee",
        header_time="time",
        format_time="%Y-%m-%d %H:%M:%S"
    )
)

GasRateCO2Dataset = DatasetLoaderCSV(
    metadata=DatasetLoaderMetadata(
        "gasrate_co2.csv",
        uri=_DEFAULT_PATH+"/gasrate_co2.csv",
        hash="77bf383715a9cf81459f81fe17baf3b0",
        header_time=None,
        format_time=None
    )
)

HeartRateDataset = DatasetLoaderCSV(
    metadata=DatasetLoaderMetadata(
        "heart_rate.csv",
        uri=_DEFAULT_PATH+"/heart_rate.csv",
        hash="3c4a108e1116867cf056dc5be2c95386",
        header_time=None,
        format_time=None
    )
)

IceCreamHeaterDataset = DatasetLoaderCSV(
    metadata=DatasetLoaderMetadata(
        "ice_cream_heater.csv",
        uri=_DEFAULT_PATH+"/ice_cream_heater.csv",
        hash="62031c7b5cdc9339fe7cf389173ef1c3",
        header_time="Month",
        format_time="%Y-%m"
    )
)

MonthlyMilkDataset = DatasetLoaderCSV(
    metadata=DatasetLoaderMetadata(
        "monthly_milk.csv",
        uri=_DEFAULT_PATH+"/monthly-milk.csv",
        hash="4784443e696da45d7082e76a67687b93",
        header_time="Month",
        format_time="%Y-%m"
    )
)

MonthlyMilkIncompleteDataset = DatasetLoaderCSV(
    metadata=DatasetLoaderMetadata(
        "monthly_milk-incomplete.csv",
        uri=_DEFAULT_PATH+"/monthly-milk-incomplete.csv",
        hash="49b275c7e2f8f28a6a05224be1a049a4",
        header_time="Month",
        format_time="%Y-%m"
    )
)

SunspotsDataset = DatasetLoaderCSV(
    metadata=DatasetLoaderMetadata(
        "sunspots.csv",
        uri=_DEFAULT_PATH+"/monthly-sunspots.csv",
        hash="4d27019c43d9c256d528f1bd6c5f40e0",
        header_time="Month",
        format_time="%Y-%m"
    )
)

TaylorDataset = DatasetLoaderCSV(
    metadata=DatasetLoaderMetadata(
        "taylor.csv",
        uri=_DEFAULT_PATH+"/taylor.csv",
        hash="1ea355c90e8214cb177788a674801a22",
        header_time=None,
        format_time=None
    )
)

TemperatureDataset = DatasetLoaderCSV(
    metadata=DatasetLoaderMetadata(
        "temperatures.csv",
        uri=_DEFAULT_PATH+"/temps.csv",
        hash="ce5b5e4929793ec8b6a54711110acebf",
        header_time="Date",
        format_time="%m/%d/%Y"
    )
)

USGasolineDataset = DatasetLoaderCSV(
    metadata=DatasetLoaderMetadata(
        "us_gasoline.csv",
        uri=_DEFAULT_PATH+"/us_gasoline.csv",
        hash="25d440337a06cbf83423e81d0337a1ce",
        header_time="Week",
        format_time="%m/%d/%Y"
    )
)

WineDataset = DatasetLoaderCSV(
    metadata=DatasetLoaderMetadata(
        "wine.csv",
        uri=_DEFAULT_PATH+"/wineind.csv",
        hash="b68971d7e709ad0b7e6300cab977e3cd",
        header_time="date",
        format_time="%Y-%m-%d"
    )
)

WoolyDataset = DatasetLoaderCSV(
    metadata=DatasetLoaderMetadata(
        "wooly.csv",
        uri=_DEFAULT_PATH+"/woolyrnq.csv",
        hash="4be8b12314db94c8fd76f5c674454bf0",
        header_time="date",
        format_time="%Y-%m-%d"
    )
)
