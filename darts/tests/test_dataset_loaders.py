import os

from darts import TimeSeries
from darts.datasets import DatasetLoader, DatasetLoaderMetadata
from darts.datasets import AirPassengers
from darts.datasets.dataset_loaders import DatasetLoadingException
from darts.tests.base_test_class import DartsBaseTestClass

wrong_hash_dataset = DatasetLoader(
    metadata=DatasetLoaderMetadata(
        "wrong_hash",
        uri="https://raw.githubusercontent.com/unit8co/darts/master/examples/AirPassengers.csv",
        hash="will fail",
        header_time="Month",
    )
)

wrong_url_dataset = DatasetLoader(
    metadata=DatasetLoaderMetadata(
        "wrong_url",
        uri="https://AirPassengers.csv",
        hash="167ffa96204a2b47339c21eea25baf32",
        header_time="Month",
    )
)


class DatasetLoaderTestCase(DartsBaseTestClass):
    def tearDown(self):
        # we need to remove the cached datasets between each test
        default_directory = DatasetLoader._DEFAULT_DIRECTORY
        for f in os.listdir(default_directory):
            os.remove(os.path.join(default_directory, f))
        os.rmdir(DatasetLoader._DEFAULT_DIRECTORY)

    def test_ok_dataset(self):
        ts: TimeSeries = AirPassengers.load()
        self.assertGreaterEqual(ts.width, 1)

    def test_hash(self):
        with self.assertRaises(DatasetLoadingException):
            wrong_hash_dataset.load()

    def test_uri(self):
        with self.assertRaises(DatasetLoadingException):
            wrong_url_dataset.load()
