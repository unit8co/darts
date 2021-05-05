from darts import TimeSeries
from darts.datasets import Dataset, DatasetMetadata
from darts.datasets import AirPassengers
from darts.datasets.datasets import DatasetLoadingException
from darts.tests.base_test_class import DartsBaseTestClass

wrong_hash_dataset = Dataset(
    metadata=DatasetMetadata(
        "wrong_hash",
        uri="https://raw.githubusercontent.com/unit8co/darts/master/examples/AirPassengers.csv",
        hash="will fail",
        header_time="Month",
        header_value="#Passengers"
    )
)

wrong_url_dataset = Dataset(
    metadata=DatasetMetadata(
        "wrong_hash",
        uri="https://this.url.does.not.exist.stuff/AirPassengers.csv",
        hash="will fail",
        header_time="Month",
        header_value="#Passengers"
    )
)


class DatasetTestCase(DartsBaseTestClass):
    def test_ok_dataset(self):
        ts: TimeSeries = AirPassengers.load()
        self.assertGreaterEqual(ts.width, 1)

    def test_hash(self):
        with self.assertRaises(DatasetLoadingException):
            wrong_hash_dataset.load()

    def test_uri(self):
        with self.assertRaises(DatasetLoadingException):
            wrong_url_dataset.load()
