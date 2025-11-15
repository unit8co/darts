import os
import shutil
import tempfile

import pytest

from darts import TimeSeries
from darts.datasets import (
    _DEFAULT_PATH,
    AirPassengersDataset,
    AusBeerDataset,
    AustralianTourismDataset,
    ElectricityConsumptionZurichDataset,
    ElectricityDataset,
    EnergyDataset,
    ETTh1Dataset,
    ETTh2Dataset,
    ETTm1Dataset,
    ETTm2Dataset,
    ExchangeRateDataset,
    GasRateCO2Dataset,
    HeartRateDataset,
    IceCreamHeaterDataset,
    ILINetDataset,
    MonthlyMilkDataset,
    MonthlyMilkIncompleteDataset,
    SunspotsDataset,
    TaxiNewYorkDataset,
    TaylorDataset,
    TemperatureDataset,
    TrafficDataset,
    UberTLCDataset,
    USGasolineDataset,
    WeatherDataset,
    WineDataset,
    WoolyDataset,
)
from darts.datasets.dataset_loaders import (
    DatasetLoader,
    DatasetLoaderCSV,
    DatasetLoaderMetadata,
    DatasetLoadingException,
)

_DEFAULT_PATH_TEST = _DEFAULT_PATH + "/tests"

datasets_with_width = [
    (AirPassengersDataset, 1),
    (AusBeerDataset, 1),
    (AustralianTourismDataset, 96),
    (EnergyDataset, 28),
    (HeartRateDataset, 1),
    (IceCreamHeaterDataset, 2),
    (MonthlyMilkDataset, 1),
    (SunspotsDataset, 1),
    (TaylorDataset, 1),
    (TemperatureDataset, 1),
    (USGasolineDataset, 1),
    (WineDataset, 1),
    (WoolyDataset, 1),
    (GasRateCO2Dataset, 2),
    (MonthlyMilkIncompleteDataset, 1),
    (ETTh1Dataset, 7),
    (ETTh2Dataset, 7),
    (ETTm1Dataset, 7),
    (ETTm2Dataset, 7),
    (ElectricityDataset, 370),
    (UberTLCDataset, 262),
    (ILINetDataset, 11),
    (ExchangeRateDataset, 8),
    (TrafficDataset, 862),
    (WeatherDataset, 21),
    (ElectricityConsumptionZurichDataset, 10),
    (TaxiNewYorkDataset, 1),
]

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
        uri=_DEFAULT_PATH_TEST + "/test.zip",
        hash="167ffa96204a2b47339c21eea25baf32",
        header_time="Month",
        pre_process_zipped_csv_fn=None,
    )
)
ele_multi_series_dataset = DatasetLoaderCSV(
    metadata=DatasetLoaderMetadata(
        "Electricity_test.csv",
        uri=_DEFAULT_PATH_TEST + "/Electricity_test.csv",
        hash="e036be148b06dacf2bb78b4647e6ea2b",
        header_time="Time",
        pre_process_zipped_csv_fn=None,
    )
)


@pytest.fixture(scope="module", autouse=True)
def tmp_dir_dataset():
    """Configures the DataLoaders to use a temporary directory for storing the datasets,
    and removes the path at the end of all tests in this module."""
    temp_work_dir = tempfile.mkdtemp(prefix="darts")
    DatasetLoader._DEFAULT_DIRECTORY = temp_work_dir
    yield temp_work_dir
    shutil.rmtree(temp_work_dir)


class TestDatasetLoader:
    @pytest.mark.slow
    @pytest.mark.parametrize("dataset_config", datasets_with_width)
    def test_ok_dataset(self, dataset_config, tmp_dir_dataset):
        dataset_cls, width = dataset_config
        dataset = dataset_cls()
        assert dataset._DEFAULT_DIRECTORY == tmp_dir_dataset
        ts: TimeSeries = dataset.load()
        assert ts.width == width
        assert os.path.exists(os.path.join(tmp_dir_dataset, dataset._metadata.name))

    def test_hash(self, tmp_dir_dataset):
        with pytest.raises(DatasetLoadingException):
            wrong_hash_dataset.load()
        assert not os.path.exists(
            os.path.join(tmp_dir_dataset, wrong_hash_dataset._metadata.name)
        )

    def test_uri(self, tmp_dir_dataset):
        with pytest.raises(DatasetLoadingException):
            wrong_url_dataset.load()
        assert not os.path.exists(
            os.path.join(tmp_dir_dataset, wrong_hash_dataset._metadata.name)
        )

    def test_zip_uri(self, tmp_dir_dataset):
        with pytest.raises(DatasetLoadingException):
            wrong_zip_url_dataset.load()
        assert not os.path.exists(
            os.path.join(tmp_dir_dataset, wrong_hash_dataset._metadata.name)
        )

    def test_pre_process_fn(self, tmp_dir_dataset):
        with pytest.raises(DatasetLoadingException):
            no_pre_process_fn_dataset.load()
        assert not os.path.exists(
            os.path.join(tmp_dir_dataset, wrong_hash_dataset._metadata.name)
        )

    def test_multi_series_dataset(self):
        # processing _to_multi_series takes a long time. Test function with 5 cols.
        ts = ele_multi_series_dataset.load().to_dataframe()

        ms = ElectricityDataset()._to_multi_series(ts)
        assert len(ms) == 5
        assert len(ms[0]) == 105216

        multi_series_datasets = [
            UberTLCDataset,
            ILINetDataset,
            ExchangeRateDataset,
            TrafficDataset,
            WeatherDataset,
        ]
        for dataset in multi_series_datasets:
            ms = dataset()._to_multi_series(ts)
            assert len(ms) == 5
            assert len(ms[0]) == len(ts.index)
