import os

from darts import TimeSeries
from darts.datasets import (
    _DEFAULT_PATH,
    AirPassengersDataset,
    AusBeerDataset,
    ElectricityDataset,
    EnergyDataset,
    ETTh1Dataset,
    ETTh2Dataset,
    ETTm1Dataset,
    ETTm2Dataset,
    GasRateCO2Dataset,
    HeartRateDataset,
    IceCreamHeaterDataset,
    MonthlyMilkDataset,
    MonthlyMilkIncompleteDataset,
    SunspotsDataset,
    TaylorDataset,
    TemperatureDataset,
    USGasolineDataset,
    WineDataset,
    WoolyDataset,
)
from darts.datasets.dataset_loaders import (
    DatasetLoader,
    DatasetLoaderCSV,
    DatasetLoaderMetadata,
    DatasetLoadingException,
)
from darts.tests.base_test_class import DartsBaseTestClass

datasets = [
    AirPassengersDataset,
    AusBeerDataset,
    EnergyDataset,
    HeartRateDataset,
    IceCreamHeaterDataset,
    MonthlyMilkDataset,
    SunspotsDataset,
    TaylorDataset,
    TemperatureDataset,
    USGasolineDataset,
    WineDataset,
    WoolyDataset,
    GasRateCO2Dataset,
    MonthlyMilkIncompleteDataset,
    ETTh1Dataset,
    ETTh2Dataset,
    ETTm1Dataset,
    ETTm2Dataset,
    ElectricityDataset,
]

width_datasets = [1, 1, 28, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 7, 7, 7, 7, 370]

wrong_hash_dataset = DatasetLoaderCSV(
    metadata=DatasetLoaderMetadata(
        "wrong_hash",
        uri="https://raw.githubusercontent.com/unit8co/darts/master/datasets/data/AirPassengers.csv",
        hash="will fail",
        header_time="Month",
        format_time="%Y-%m",
    )
)

wrong_url_dataset = DatasetLoaderCSV(
    metadata=DatasetLoaderMetadata(
        "wrong_url",
        uri="https://AirPassengers.csv",
        hash="167ffa96204a2b47339c21eea25baf32",
        header_time="Month",
        format_time="%Y-%m",
    )
)

wrong_zip_url_dataset = DatasetLoaderCSV(
    metadata=DatasetLoaderMetadata(
        "wrong_zip_url",
        uri="https://Electricity.zip",
        hash="d17748042ea98fc9c5fb4db0946d5fa4",
        header_time="Unnamed: 0",
        format_time="%Y-%m-%d %H:%M:%S",
        pre_process_zipped_csv_fn=lambda x: x,
    )
)

no_pre_process_fn_dataset = DatasetLoaderCSV(
    metadata=DatasetLoaderMetadata(
        "no_pre_process_fn",
        uri=_DEFAULT_PATH + "/test.zip",
        hash="167ffa96204a2b47339c21eea25baf32",
        header_time="Month",
        pre_process_zipped_csv_fn=None,
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
        for width, dataset_cls in zip(width_datasets, datasets):
            dataset = dataset_cls()
            ts: TimeSeries = dataset.load()
            self.assertEqual(ts.width, width)

    def test_hash(self):
        with self.assertRaises(DatasetLoadingException):
            wrong_hash_dataset.load()

    def test_uri(self):
        with self.assertRaises(DatasetLoadingException):
            wrong_url_dataset.load()

    def test_zip_uri(self):
        with self.assertRaises(DatasetLoadingException):
            wrong_zip_url_dataset.load()

    def test_pre_process_fn(self):
        with self.assertRaises(DatasetLoadingException):
            no_pre_process_fn_dataset.load()
