from .dataset_loaders import DatasetLoaderCSV, DatasetLoaderMetadata

"""
    Overall usage of this package:
    from darts.datasets import AirPassengers
    ts: TimeSeries = AirPassengers.load()
"""
AirPassengersDataset = DatasetLoaderCSV(
    metadata=DatasetLoaderMetadata(
        "air_passengers.csv",
        uri="https://raw.githubusercontent.com/unit8co/darts/master/examples/AirPassengers.csv",
        hash="167ffa96204a2b47339c21eea25baf32",
        header_time="Month"
    )
)

### Fails due to summer- winter time duplicates
# EnergyDataset = DatasetLoaderCSV(
#     metadata=DatasetLoaderMetadata(
#         "energy.csv",
#         uri="https://raw.githubusercontent.com/unit8co/darts/develop/examples/energy_dataset.csv",
#         hash="63afe36eed077c06fe342a7274d0e2e3",
#         header_time="time",
#         format_time="%Y-%m-%d %H:%M:%S"
#     )
# )

IceCreamHeaterDataset = DatasetLoaderCSV(
    metadata=DatasetLoaderMetadata(
        "ice_cream_heater.csv",
        uri="https://raw.githubusercontent.com/unit8co/darts/develop/examples/ice_cream_heater.csv",
        hash="62031c7b5cdc9339fe7cf389173ef1c3",
        header_time="Month",
        format_time="%Y-%m"
    )
)

MonthlyMilkDataset = DatasetLoaderCSV(
    metadata=DatasetLoaderMetadata(
        "monthly_milk.csv",
        uri="https://raw.githubusercontent.com/unit8co/darts/develop/examples/monthly-milk.csv",
        hash="4784443e696da45d7082e76a67687b93",
        header_time="Month",
        format_time="%Y-%m"
    )
)

SunspotsDataset = DatasetLoaderCSV(
    metadata=DatasetLoaderMetadata(
        "sunspots.csv",
        uri="https://raw.githubusercontent.com/unit8co/darts/develop/examples/monthly-sunspots.csv",
        hash="4d27019c43d9c256d528f1bd6c5f40e0",
        header_time="Month",
        format_time="%Y-%m"
    )
)

### Fails because some values start with "?"
# TemperatureDataset = DatasetLoaderCSV(
#     metadata=DatasetLoaderMetadata(
#         "temperatures.csv",
#         uri="https://raw.githubusercontent.com/unit8co/darts/develop/examples/temps.csv",
#         hash="bd3831f19147b09c41243eae8c3cd172",
#         header_time="Date",
#         format_time="%m/%d/%Y"
#     )
# )

USGasolineDataset = DatasetLoaderCSV(
    metadata=DatasetLoaderMetadata(
        "us_gasoline.csv",
        uri="https://raw.githubusercontent.com/unit8co/darts/feat/datasets/examples/us_gasoline.csv",
        hash="25d440337a06cbf83423e81d0337a1ce",
        header_time="Week",
        format_time="%m/%d/%Y"
    )
)

WoolyDataset = DatasetLoaderCSV(
    metadata=DatasetLoaderMetadata(
        "wooly.csv",
        uri="",
        hash="4be8b12314db94c8fd76f5c674454bf0",
        header_time="date",
        format_time="%Y-%m-%d"
    )
)

AusBeerDataset = DatasetLoaderCSV(
    metadata=DatasetLoaderMetadata(
        "ausbeer.csv",
        uri="",
        hash="1f4028a570a20939411cc04de7364bbd",
        header_time="date",
        format_time="%Y-%m-%d"
    )
)

WineDataset = DatasetLoaderCSV(
    metadata=DatasetLoaderMetadata(
        "wine.csv",
        uri="",
        hash="b68971d7e709ad0b7e6300cab977e3cd",
        header_time="date",
        format_time="%Y-%m-%d"
    )
)

TaylorDataset = DatasetLoaderCSV(
    metadata=DatasetLoaderMetadata(
        "taylor.csv",
        uri="",
        hash="1ea355c90e8214cb177788a674801a22",
        header_time="",
        format_time=None
    )
)

HeartRateDataset = DatasetLoaderCSV(
    metadata=DatasetLoaderMetadata(
        "heart_rate.csv",
        uri="",
        hash="3c4a108e1116867cf056dc5be2c95386",
        header_time="",
        format_time=None
    )
)