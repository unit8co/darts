from .datasets import Dataset, DatasetMetadata

"""
    Overall usage of this package:
    from darts.datasets import AirPassengers
    ts: TimeSeries = AirPassengers.load()
"""
AirPassengers = Dataset(
    metadata=DatasetMetadata(
        "air_passengers",
        uri="https://raw.githubusercontent.com/unit8co/darts/master/examples/AirPassengers.csv",
        hash="167ffa96204a2b47339c21eea25baf32",
        header_time="Month",
        header_value="#Passengers"
    )
)
