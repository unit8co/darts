from dataclasses import dataclass
import os
from pathlib import Path
import hashlib
from typing import Callable, Optional

import pandas as pd
import requests

from ..timeseries import TimeSeries


@dataclass
class DatasetLoaderMetadata:
    # name of the dataset
    name: str
    # uri of the dataset, expects a publicly available CSV file
    uri: str
    # md5 hash of the file to be downloaded
    hash: str
    # used to parse CSV file
    header_time: str


class DatasetLoadingException(BaseException):
    pass


class DatasetLoader:
    """
    Class that downloads a dataset and caches it locally.
    Assumes that the file can be downloaded (i.e. publicly available via an URI)
    """
    _DEFAULT_DIRECTORY = Path(os.path.join(Path.home(), Path('.darts/datasets/')))

    def __init__(self, metadata: DatasetLoaderMetadata, root_path: Path = None, post_processing_function: Callable = None):
        self._metadata: DatasetLoaderMetadata = metadata
        if root_path is None:
            self._root_path: Path = DatasetLoader._DEFAULT_DIRECTORY
        else:
            self._root_path: Path = root_path
        self._post_processing_function: Optional[Callable] = post_processing_function

    def load(self) -> TimeSeries:
        """
        Load the dataset in memory, as a TimeSeries.
        Downloads the dataset if it is not present already
        :raise DatasetLoadingException
        :return: A TimeSeries object that contains the dataset
        """
        if not self._is_already_downloaded():
            self._download_dataset()
        self._check_dataset_integrity_or_raise()
        return self._load_from_disk()

    def _check_dataset_integrity_or_raise(self):
        """
        Ensures that the dataset exists and its MD5 checksum matches the expected hash.
        :raise DatasetLoadingException if checks fail
        :return: Nothing
        """
        if not self._is_already_downloaded():
            raise DatasetLoadingException(f"Checking md5 checksum of a absent file: {self._get_path_dataset()}")

        with open(self._get_path_dataset(), "rb") as f:
            md5_hash = hashlib.md5(f.read()).hexdigest()
            if md5_hash != self._metadata.hash:
                raise DatasetLoadingException(f"Expected hash for {self._get_path_dataset()}: {self._metadata.hash}"
                                              f", got: {md5_hash}")

    def _download_dataset(self):
        """
        Downloads the dataset in the root_path directory
        :raise DatasetLoadingException if downloading or writing the file to disk fails
        :return: Nothing
        """
        os.makedirs(self._root_path, exist_ok=True)
        try:
            request = requests.get(self._metadata.uri)
            with open(self._get_path_dataset(), "wb") as f:
                f.write(request.content)
        except Exception as e:
            raise DatasetLoadingException("Could not download the dataset. Reason:" + e.__repr__()) from None

    def _load_from_disk(self) -> TimeSeries:
        """
        Reads CSV and puts it in a TimeSeries object.
        Assumes the file exists and has the right format and hash.
        :return: A TimeSeries object containing the dataset
        """
        df = pd.read_csv(self._get_path_dataset())
        return TimeSeries.from_dataframe(df, self._metadata.header_time)

    def _get_path_dataset(self):
        return os.path.join(self._root_path, f"{self._metadata.name}.csv")

    def _is_already_downloaded(self) -> bool:
        return os.path.isfile(self._get_path_dataset())
