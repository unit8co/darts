from .dataset_loaders import DatasetLoaderCSV, DatasetLoaderMetadata

"""
    Overall usage of this package:
    from darts.datasets import AirPassengersDataset
    ts: TimeSeries = AirPassengersDataset.load()
"""
AirPassengersDataset = DatasetLoaderCSV(
    metadata=DatasetLoaderMetadata(
        "air_passengers.csv",
        uri="https://raw.githubusercontent.com/unit8co/darts/feat/datasets/darts/datasets/data/AirPassengers.csv",
        hash="167ffa96204a2b47339c21eea25baf32",
        header_time="Month"
    )
)

AusBeerDataset = DatasetLoaderCSV(
    metadata=DatasetLoaderMetadata(
        "ausbeer.csv",
        uri="https://raw.githubusercontent.com/unit8co/darts/feat/datasets/darts/datasets/data/ausbeer.csv",
        hash="1f4028a570a20939411cc04de7364bbd",
        header_time="date",
        format_time="%Y-%m-%d"
    )
)

EnergyDataset = DatasetLoaderCSV(
    metadata=DatasetLoaderMetadata(
        "energy.csv",
        uri="https://raw.githubusercontent.com/unit8co/darts/feat/datasets/darts/datasets/data/energy_dataset.csv",
        hash="a772a9d714a8c3def41002226bf80fc8",
        header_time="time",
        format_time="%Y-%m-%d %H:%M:%S"
    )
)

GasRateCO2Dataset = DatasetLoaderCSV(
    metadata=DatasetLoaderMetadata(
        "gasrate_co2.csv",
        uri="https://raw.githubusercontent.com/unit8co/darts/feat/datasets/darts/datasets/data/gasrate_co2.csv",
        hash="1f2232c83660033c4af71c7f3b6c0c1a",
        header_time=None,
        format_time=None
    )
)

HeartRateDataset = DatasetLoaderCSV(
    metadata=DatasetLoaderMetadata(
        "heart_rate.csv",
        uri="https://raw.githubusercontent.com/unit8co/darts/feat/datasets/darts/datasets/data/heart_rate.csv",
        hash="3c4a108e1116867cf056dc5be2c95386",
        header_time=None,
        format_time=None
    )
)

IceCreamHeaterDataset = DatasetLoaderCSV(
    metadata=DatasetLoaderMetadata(
        "ice_cream_heater.csv",
        uri="https://raw.githubusercontent.com/unit8co/darts/feat/datasets/darts/datasets/data/ice_cream_heater.csv",
        hash="62031c7b5cdc9339fe7cf389173ef1c3",
        header_time="Month",
        format_time="%Y-%m"
    )
)

MonthlyMilkDataset = DatasetLoaderCSV(
    metadata=DatasetLoaderMetadata(
        "monthly_milk.csv",
        uri="https://raw.githubusercontent.com/unit8co/darts/feat/datasets/darts/datasets/data/monthly-milk.csv",
        hash="4784443e696da45d7082e76a67687b93",
        header_time="Month",
        format_time="%Y-%m"
    )
)

SunspotsDataset = DatasetLoaderCSV(
    metadata=DatasetLoaderMetadata(
        "sunspots.csv",
        uri="https://raw.githubusercontent.com/unit8co/darts/feat/datasets/darts/datasets/data/monthly-sunspots.csv",
        hash="4d27019c43d9c256d528f1bd6c5f40e0",
        header_time="Month",
        format_time="%Y-%m"
    )
)

TaylorDataset = DatasetLoaderCSV(
    metadata=DatasetLoaderMetadata(
        "taylor.csv",
        uri="https://raw.githubusercontent.com/unit8co/darts/feat/datasets/darts/datasets/data/taylor.csv",
        hash="1ea355c90e8214cb177788a674801a22",
        header_time=None,
        format_time=None
    )
)

TemperatureDataset = DatasetLoaderCSV(
    metadata=DatasetLoaderMetadata(
        "temperatures.csv",
        uri="https://raw.githubusercontent.com/unit8co/darts/feat/datasets/darts/datasets/data/temps.csv",
        hash="ce5b5e4929793ec8b6a54711110acebf",
        header_time="Date",
        format_time="%m/%d/%Y"
    )
)

USGasolineDataset = DatasetLoaderCSV(
    metadata=DatasetLoaderMetadata(
        "us_gasoline.csv",
        uri="https://raw.githubusercontent.com/unit8co/darts/feat/datasets/darts/datasets/data/us_gasoline.csv",
        hash="25d440337a06cbf83423e81d0337a1ce",
        header_time="Week",
        format_time="%m/%d/%Y"
    )
)

WineDataset = DatasetLoaderCSV(
    metadata=DatasetLoaderMetadata(
        "wine.csv",
        uri="https://raw.githubusercontent.com/unit8co/darts/feat/datasets/darts/datasets/data/wineind.csv",
        hash="b68971d7e709ad0b7e6300cab977e3cd",
        header_time="date",
        format_time="%Y-%m-%d"
    )
)

WoolyDataset = DatasetLoaderCSV(
    metadata=DatasetLoaderMetadata(
        "wooly.csv",
        uri="https://raw.githubusercontent.com/unit8co/darts/feat/datasets/darts/datasets/data/woolyrnq.csv",
        hash="4be8b12314db94c8fd76f5c674454bf0",
        header_time="date",
        format_time="%Y-%m-%d"
    )
)
