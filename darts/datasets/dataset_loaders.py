import hashlib
import os
import tempfile
import zipfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Union

import pandas as pd
import requests

from darts import TimeSeries
from darts.logging import get_logger

logger = get_logger(__name__)


@dataclass
class DatasetLoaderMetadata:
    # name of the dataset file, including extension
    name: str
    # uri of the dataset, expects a publicly available file
    uri: str
    # md5 hash of the file to be downloaded
    hash: str
    # used to parse the dataset file
    header_time: Optional[str]
    # used to convert the string date to pd.Datetime
    # https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior
    format_time: Optional[str] = None
    # used to indicate the freq when we already know it
    freq: Optional[str] = None
    # a custom function to handling non-csv based datasets
    pre_process_zipped_csv_fn: Optional[Callable] = None
    # multivariate
    multivariate: Optional[bool] = None


class DatasetLoadingException(BaseException):
    pass


class DatasetLoader(ABC):
    """
    Class that downloads a dataset and caches it locally.
    Assumes that the file can be downloaded (i.e. publicly available via a URI)
    """

    _DEFAULT_DIRECTORY = Path(os.path.join(Path.home(), Path(".darts/datasets/")))

    def __init__(self, metadata: DatasetLoaderMetadata, root_path: Path = None):
        self._metadata: DatasetLoaderMetadata = metadata
        if root_path is None:
            self._root_path: Path = DatasetLoader._DEFAULT_DIRECTORY
        else:
            self._root_path: Path = root_path

    def load(self) -> TimeSeries:
        """
        Load the dataset in memory, as a TimeSeries.
        Downloads the dataset if it is not present already

        Raises
        -------
        DatasetLoadingException
            If loading fails (MD5 Checksum is invalid, Download failed, Reading from disk failed)

        Returns
        -------
        time_series: TimeSeries
            A TimeSeries object that contains the dataset
        """
        if not self._is_already_downloaded():
            if self._metadata.uri.endswith(".zip"):
                self._download_zip_dataset()
            else:
                self._download_dataset()
        self._check_dataset_integrity_or_raise()

        return self._load_from_disk(self._get_path_dataset(), self._metadata)

    def _check_dataset_integrity_or_raise(self):
        """
        Ensures that the dataset exists and its MD5 checksum matches the expected hash.

        Raises
        -------
        DatasetLoadingException
            if checks fail

        Returns
        -------
        """
        if not self._is_already_downloaded():
            raise DatasetLoadingException(
                f"Checking md5 checksum of a absent file: {self._get_path_dataset()}"
            )

        with open(self._get_path_dataset(), "rb") as f:
            md5_hash = hashlib.md5(f.read()).hexdigest()
            if md5_hash != self._metadata.hash:
                raise DatasetLoadingException(
                    f"Expected hash for {self._get_path_dataset()}: {self._metadata.hash}"
                    f", got: {md5_hash}"
                )

    def _download_dataset(self):
        """
        Downloads the dataset in the root_path directory

        Raises
        -------
        DatasetLoadingException
            if downloading or writing the file to disk fails

        Returns
        -------
        """
        if self._metadata.pre_process_zipped_csv_fn:
            logger.warning(
                "Loading a CSV file does not use the pre_process_zipped_csv_fn"
            )
        os.makedirs(self._root_path, exist_ok=True)
        try:
            request = requests.get(self._metadata.uri)
            with open(self._get_path_dataset(), "wb") as f:
                f.write(request.content)
        except Exception as e:
            raise DatasetLoadingException(
                "Could not download the dataset. Reason:" + e.__repr__()
            ) from None

    def _download_zip_dataset(self):
        os.makedirs(self._root_path, exist_ok=True)
        try:
            request = requests.get(self._metadata.uri)
            with tempfile.TemporaryFile() as tf:
                tf.write(request.content)
                with tempfile.TemporaryDirectory() as td:
                    with zipfile.ZipFile(tf, "r") as zip_ref:
                        zip_ref.extractall(td)
                        self._metadata.pre_process_zipped_csv_fn(
                            td, self._get_path_dataset()
                        )
        except Exception as e:
            raise DatasetLoadingException(
                "Could not download the dataset. Reason:" + e.__repr__()
            ) from None

    @abstractmethod
    def _load_from_disk(
        self, path_to_file: Path, metadata: DatasetLoaderMetadata
    ) -> TimeSeries:
        """
        Given a Path to the file and a DataLoaderMetadata object, return a TimeSeries
        One can assume that the file exists and its MD5 checksum has been verified before this function is called

        Parameters
        ----------
        path_to_file: Path
            A Path object where the dataset is located
        metadata: Metadata
            The dataset's metadata

        Returns
        -------
        time_series: TimeSeries
            a TimeSeries object that contains the whole dataset
        """
        pass

    def _get_path_dataset(self) -> Path:
        return Path(os.path.join(self._root_path, self._metadata.name))

    def _is_already_downloaded(self) -> bool:
        return os.path.isfile(self._get_path_dataset())

    def _format_time_column(self, df):
        df[self._metadata.header_time] = pd.to_datetime(
            df[self._metadata.header_time],
            format=self._metadata.format_time,
            errors="raise",
        )
        return df


class DatasetLoaderCSV(DatasetLoader):
    def __init__(self, metadata: DatasetLoaderMetadata, root_path: Path = None):
        super().__init__(metadata, root_path)

    def _load_from_disk(
        self, path_to_file: Path, metadata: DatasetLoaderMetadata
    ) -> Union[TimeSeries, List[TimeSeries]]:

        df = pd.read_csv(path_to_file)
        if metadata.header_time is not None:
            df = self._format_time_column(df)
            series = TimeSeries.from_dataframe(
                df=df, time_col=metadata.header_time, freq=metadata.freq
            )
        else:
            df.sort_index(inplace=True)
            series = TimeSeries.from_dataframe(df)

        if (
            self._metadata.multivariate is not None
            and self._metadata.multivariate is False
        ):
            try:
                series = self._to_multi_series(series.pd_dataframe())
            except Exception as e:
                raise DatasetLoadingException(
                    "Could not convert to multi-series. Reason:" + e.__repr__()
                ) from None

        return series
