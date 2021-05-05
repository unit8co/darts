from .dataset_loaders import DatasetLoader, DatasetLoaderMetadata

"""
    Overall usage of this package:
    from darts.datasets import AirPassengers
    ts: TimeSeries = AirPassengers.load()
"""
AirPassengers = DatasetLoader(
    metadata=DatasetLoaderMetadata(
        "air_passengers",
        uri="https://raw.githubusercontent.com/unit8co/darts/master/examples/AirPassengers.csv",
        hash="167ffa96204a2b47339c21eea25baf32",
        header_time="Month",
    )
)
