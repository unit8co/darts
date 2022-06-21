"""
Datasets
--------

A few popular time series datasets
"""
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from darts import TimeSeries
from darts.logging import get_logger, raise_if_not

from .dataset_loaders import DatasetLoaderCSV, DatasetLoaderMetadata

"""
    Overall usage of this package:
    from darts.datasets import AirPassengersDataset
    ts: TimeSeries = AirPassengersDataset.load()
"""

logger = get_logger(__name__)

_DEFAULT_PATH = "https://raw.githubusercontent.com/unit8co/darts/master/datasets"


class AirPassengersDataset(DatasetLoaderCSV):
    """
    Monthly Air Passengers Dataset, from 1949 to 1960.
    """

    def __init__(self):
        super().__init__(
            metadata=DatasetLoaderMetadata(
                "air_passengers.csv",
                uri=_DEFAULT_PATH + "/AirPassengers.csv",
                hash="167ffa96204a2b47339c21eea25baf32",
                header_time="Month",
            )
        )


class AusBeerDataset(DatasetLoaderCSV):
    """
    Total quarterly beer production in Australia (in megalitres) from 1956:Q1 to 2008:Q3 [1]_.

    References
    ----------
    .. [1] https://rdrr.io/cran/fpp/man/ausbeer.html
    """

    def __init__(self):
        super().__init__(
            metadata=DatasetLoaderMetadata(
                "ausbeer.csv",
                uri=_DEFAULT_PATH + "/ausbeer.csv",
                hash="1f4028a570a20939411cc04de7364bbd",
                header_time="date",
                format_time="%Y-%m-%d",
            )
        )


class EnergyDataset(DatasetLoaderCSV):
    """
    Hourly energy dataset coming from [1]_.

    Contains a time series with 28 hourly components between 2014-12-31 23:00:00 and 2018-12-31 22:00:00

    References
    ----------
    .. [1] https://www.kaggle.com/nicholasjhana/energy-consumption-generation-prices-and-weather
    """

    def __init__(self):
        super().__init__(
            metadata=DatasetLoaderMetadata(
                "energy.csv",
                uri=_DEFAULT_PATH + "/energy_dataset.csv",
                hash="f564ef18e01574734a0fa20806d1c7ee",
                header_time="time",
                format_time="%Y-%m-%d %H:%M:%S",
            )
        )


class GasRateCO2Dataset(DatasetLoaderCSV):
    """
    Gas Rate CO2 dataset
    Two components, length 296 (integer time index)
    """

    def __init__(self):
        super().__init__(
            metadata=DatasetLoaderMetadata(
                "gasrate_co2.csv",
                uri=_DEFAULT_PATH + "/gasrate_co2.csv",
                hash="77bf383715a9cf81459f81fe17baf3b0",
                header_time=None,
                format_time=None,
            )
        )


class HeartRateDataset(DatasetLoaderCSV):
    """
    The series contains 1800 evenly-spaced measurements of instantaneous heart rate from a single subject.
    The measurements (in units of beats per minute) occur at 0.5 second intervals, so that the length of
    each series is exactly 15 minutes.

    This is the series1 in [1]_.
    It uses an integer time index.

    References
    ----------
    .. [1] http://ecg.mit.edu/time-series/
    """

    def __init__(self):
        super().__init__(
            metadata=DatasetLoaderMetadata(
                "heart_rate.csv",
                uri=_DEFAULT_PATH + "/heart_rate.csv",
                hash="3c4a108e1116867cf056dc5be2c95386",
                header_time=None,
                format_time=None,
            )
        )


class IceCreamHeaterDataset(DatasetLoaderCSV):
    """
    Monthly sales of heaters and ice cream between January 2004 and June 2020.
    """

    def __init__(self):
        super().__init__(
            metadata=DatasetLoaderMetadata(
                "ice_cream_heater.csv",
                uri=_DEFAULT_PATH + "/ice_cream_heater.csv",
                hash="62031c7b5cdc9339fe7cf389173ef1c3",
                header_time="Month",
                format_time="%Y-%m",
            )
        )


class MonthlyMilkDataset(DatasetLoaderCSV):
    """
    Monthly production of milk (in pounds per cow) between January 1962 and December 1975
    """

    def __init__(self):
        super().__init__(
            metadata=DatasetLoaderMetadata(
                "monthly_milk.csv",
                uri=_DEFAULT_PATH + "/monthly-milk.csv",
                hash="4784443e696da45d7082e76a67687b93",
                header_time="Month",
                format_time="%Y-%m",
            )
        )


class MonthlyMilkIncompleteDataset(DatasetLoaderCSV):
    """
    Monthly production of milk (in pounds per cow) between January 1962 and December 1975.
    Has some missing values.
    """

    def __init__(self):
        super().__init__(
            metadata=DatasetLoaderMetadata(
                "monthly_milk-incomplete.csv",
                uri=_DEFAULT_PATH + "/monthly-milk-incomplete.csv",
                hash="49b275c7e2f8f28a6a05224be1a049a4",
                header_time="Month",
                format_time="%Y-%m",
                freq="MS",
            )
        )


class SunspotsDataset(DatasetLoaderCSV):
    """
    Monthly Sunspot Numbers, 1749 - 1983

    Monthly mean relative sunspot numbers from 1749 to 1983.
    Collected at Swiss Federal Observatory, Zurich until 1960, then Tokyo Astronomical Observatory.

    Source: [1]_

    References
    ----------
    .. [1] https://www.rdocumentation.org/packages/datasets/versions/3.6.1/topics/sunspots
    """

    def __init__(self):
        super().__init__(
            metadata=DatasetLoaderMetadata(
                "sunspots.csv",
                uri=_DEFAULT_PATH + "/monthly-sunspots.csv",
                hash="4d27019c43d9c256d528f1bd6c5f40e0",
                header_time="Month",
                format_time="%Y-%m",
            )
        )


class TaylorDataset(DatasetLoaderCSV):
    """
    Half-hourly electricity demand in England and Wales from Monday 5 June 2000 to Sunday 27 August 2000.
    Discussed in Taylor (2003) [1]_, and kindly provided by James W Taylor [2]_. Units: Megawatts
    (Uses an integer time index).

    References
    ----------
    .. [1] Taylor, J.W. (2003) Short-term electricity demand forecasting using double seasonal exponential smoothing.
           Journal of the Operational Research Society, 54, 799-805.

    .. [2] https://www.rdocumentation.org/packages/forecast/versions/8.13/topics/taylor
    """

    def __init__(self):
        super().__init__(
            metadata=DatasetLoaderMetadata(
                "taylor.csv",
                uri=_DEFAULT_PATH + "/taylor.csv",
                hash="1ea355c90e8214cb177788a674801a22",
                header_time=None,
                format_time=None,
            )
        )


class TemperatureDataset(DatasetLoaderCSV):
    """
    Daily temperature in Melbourne between 1981 and 1990
    """

    def __init__(self):
        super().__init__(
            metadata=DatasetLoaderMetadata(
                "temperatures.csv",
                uri=_DEFAULT_PATH + "/temps.csv",
                hash="ce5b5e4929793ec8b6a54711110acebf",
                header_time="Date",
                format_time="%m/%d/%Y",
                freq="D",
            )
        )


class USGasolineDataset(DatasetLoaderCSV):
    """
    Weekly U.S. Product Supplied of Finished Motor Gasoline between 1991-02-08 and 2021-04-30

    Obtained from [1]_.

    References
    ----------
    .. [1] https://www.eia.gov/dnav/pet/hist/LeafHandler.ashx?n=PET&s=wgfupus2&f=W
    """

    def __init__(self):
        super().__init__(
            metadata=DatasetLoaderMetadata(
                "us_gasoline.csv",
                uri=_DEFAULT_PATH + "/us_gasoline.csv",
                hash="25d440337a06cbf83423e81d0337a1ce",
                header_time="Week",
                format_time="%m/%d/%Y",
            )
        )


class WineDataset(DatasetLoaderCSV):
    """
    Australian total wine sales by wine makers in bottles <= 1 litre. Monthly between Jan 1980 and Aug 1994.
    Source: [1]_

    References
    ----------
    .. [1] https://www.rdocumentation.org/packages/forecast/versions/8.1/topics/wineind
    """

    def __init__(self):
        super().__init__(
            metadata=DatasetLoaderMetadata(
                "wine.csv",
                uri=_DEFAULT_PATH + "/wineind.csv",
                hash="b68971d7e709ad0b7e6300cab977e3cd",
                header_time="date",
                format_time="%Y-%m-%d",
            )
        )


class WoolyDataset(DatasetLoaderCSV):
    """
    Quarterly production of woollen yarn in Australia: tonnes. Mar 1965 -- Sep 1994.
    Source: [1]_

    References
    ----------
    .. [1] https://www.rdocumentation.org/packages/forecast/versions/8.1/topics/woolyrnq
    """

    def __init__(self):
        super().__init__(
            metadata=DatasetLoaderMetadata(
                "wooly.csv",
                uri=_DEFAULT_PATH + "/woolyrnq.csv",
                hash="4be8b12314db94c8fd76f5c674454bf0",
                header_time="date",
                format_time="%Y-%m-%d",
            )
        )


class ETTh1Dataset(DatasetLoaderCSV):
    """
    The data of 1 Electricity Transformers at 1 stations, including load, oil temperature.
    The dataset ranges from 2016/07 to 2018/07 taken hourly.
    Source: [1][2]_

    Field Descriptions:
    date: The recorded date
    HUFL: High UseFul Load
    HULL: High UseLess Load
    MUFL: Medium UseFul Load
    MULL: Medium UseLess Load
    LUFL: Low UseFul Load
    LULL: Low UseLess Load
    OT: Oil Temperature (Target)

    References
    ----------
    .. [1] https://github.com/zhouhaoyi/ETDataset
    .. [2] https://arxiv.org/abs/2012.07436
    """

    def __init__(self):
        super().__init__(
            metadata=DatasetLoaderMetadata(
                "ETTh1.csv",
                uri=_DEFAULT_PATH + "/ETTh1.csv",
                hash="8381763947c85f4be6ac456c508460d6",
                header_time="date",
                format_time="%Y-%m-%d %H:%M:%S",
            )
        )


class ETTh2Dataset(DatasetLoaderCSV):
    """
    The data of 1 Electricity Transformers at 1 stations, including load, oil temperature.
    The dataset ranges from 2016/07 to 2018/07 taken hourly.
    Source: [1][2]_

    Field Descriptions:
    date: The recorded date
    HUFL: High UseFul Load
    HULL: High UseLess Load
    MUFL: Medium UseFul Load
    MULL: Medium UseLess Load
    LUFL: Low UseFul Load
    LULL: Low UseLess Load
    OT: Oil Temperature (Target)

    References
    ----------
    .. [1] https://github.com/zhouhaoyi/ETDataset
    .. [2] https://arxiv.org/abs/2012.07436
    """

    def __init__(self):
        super().__init__(
            metadata=DatasetLoaderMetadata(
                "ETTh2.csv",
                uri=_DEFAULT_PATH + "/ETTh2.csv",
                hash="51a229a3fc13579dd939364fefe9c7ab",
                header_time="date",
                format_time="%Y-%m-%d %H:%M:%S",
            )
        )


class ETTm1Dataset(DatasetLoaderCSV):
    """
    The data of 1 Electricity Transformers at 1 stations, including load, oil temperature.
    The dataset ranges from 2016/07 to 2018/07 recorded every 15 minutes.
    Source: [1][2]_

    Field Descriptions:
    date: The recorded date
    HUFL: High UseFul Load
    HULL: High UseLess Load
    MUFL: Medium UseFul Load
    MULL: Medium UseLess Load
    LUFL: Low UseFul Load
    LULL: Low UseLess Load
    OT: Oil Temperature (Target)

    References
    ----------
    .. [1] https://github.com/zhouhaoyi/ETDataset
    .. [2] https://arxiv.org/abs/2012.07436
    """

    def __init__(self):
        super().__init__(
            metadata=DatasetLoaderMetadata(
                "ETTm1.csv",
                uri=_DEFAULT_PATH + "/ETTm1.csv",
                hash="82d6bd89109c63d075d99c1077b33f38",
                header_time="date",
                format_time="%Y-%m-%d %H:%M:%S",
            )
        )


class ETTm2Dataset(DatasetLoaderCSV):
    """
    The data of 1 Electricity Transformers at 1 stations, including load, oil temperature.
    The dataset ranges from 2016/07 to 2018/07 recorded every 15 minutes.
    Source: [1][2]_

    Field Descriptions:
    date: The recorded date
    HUFL: High UseFul Load
    HULL: High UseLess Load
    MUFL: Medium UseFul Load
    MULL: Medium UseLess Load
    LUFL: Low UseFul Load
    LULL: Low UseLess Load
    OT: Oil Temperature (Target)

    References
    ----------
    .. [1] https://github.com/zhouhaoyi/ETDataset
    .. [2] https://arxiv.org/abs/2012.07436
    """

    def __init__(self):
        super().__init__(
            metadata=DatasetLoaderMetadata(
                "ETTm2.csv",
                uri=_DEFAULT_PATH + "/ETTm2.csv",
                hash="7687e47825335860bf58bccb31be0c56",
                header_time="date",
                format_time="%Y-%m-%d %H:%M:%S",
            )
        )


class ElectricityDataset(DatasetLoaderCSV):
    """
    Measurements of electric power consumption in one household with 15 minute sampling rate.
    370 client's consumption are recorded in kW.
    Source: [1]_

    Loading this dataset will provide a multivariate timeseries with 370 columns for each household.
    The following code can be used to convert the dataset to a list of univariate timeseries,
    one for each household.


    References
    ----------
    .. [1] https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014

    """

    def __init__(self, multivariate: bool = True):
        """
        Parameters
        ----------
        multivariate: bool
            Whether to return a single multivariate timeseries - if False returns a list of univariate TimeSeries. Default is True.
        """

        def pre_proces_fn(extracted_dir, dataset_path):
            with open(Path(extracted_dir, "LD2011_2014.txt")) as fin:
                with open(dataset_path, "wt", newline="\n") as fout:
                    for line in fin:
                        fout.write(line.replace(",", ".").replace(";", ","))

        super().__init__(
            metadata=DatasetLoaderMetadata(
                "Electricity.csv",
                uri="https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt.zip",
                hash="acfe6783eea43905e510f537add940fd",
                header_time="Unnamed: 0",
                format_time="%Y-%m-%d %H:%M:%S",
                pre_process_zipped_csv_fn=pre_proces_fn,
                multivariate=multivariate,
            )
        )

    def _to_multi_series(self, series: pd.DataFrame) -> List[TimeSeries]:
        """
        Load the electricity dataset as a list of univariate series, one for each household.
        """

        ts_list = []  # list of timeseries
        for label in series:
            srs = series[label]

            # filter column down to the period of recording
            srs = srs.replace(0.0, np.nan)
            start_date = min(srs.fillna(method="ffill").dropna().index)
            end_date = max(srs.fillna(method="bfill").dropna().index)
            active_range = (srs.index >= start_date) & (srs.index <= end_date)
            srs = srs[active_range].fillna(0.0)

            # convert to timeseries
            tmp = pd.DataFrame({"power_usage": srs})
            tmp["date"] = tmp.index
            ts = TimeSeries.from_dataframe(tmp, "date", ["power_usage"])
            ts_list.append(ts)
        return ts_list


class UberTLCDataset(DatasetLoaderCSV):
    """
    14.3 million Uber pickups from January to June 2015. The data is resampled to hourly or daily based sample_freq
    on using the locationID as the target.
    Source: [1]_

    Loading this dataset will provide a multivariate timeseries with 262 columns for each locationID.
    The following code can be used to convert the dataset to a list of univariate timeseries,
    one for each locationID.


    References
    ----------
    .. [1] https://github.com/fivethirtyeight/uber-tlc-foil-response

    """

    def __init__(self, sample_freq: str = "hourly", multivariate: bool = True):
        """
        Parameters
        ----------
        sample_freq: str
            The sampling frequency of the data. Can be "hourly" or "daily". Default is "hourly".
        multivariate: bool
            Whether to return a single multivariate timeseries - if False returns a list of univariate TimeSeries. Default is True.
        """
        valid_sample_freq = ["daily", "hourly"]
        raise_if_not(
            sample_freq in valid_sample_freq,
            f"sample_freq must be one of {valid_sample_freq}",
            logger,
        )

        def pre_proces_fn(extracted_dir, dataset_path):
            df = pd.read_csv(
                Path(extracted_dir, "uber-raw-data-janjune-15.csv"),
                header=0,
                usecols=["Pickup_date", "locationID"],
                index_col=0,
            )

            output_dict = {}
            freq_setting = "1H" if "hourly" in str(dataset_path) else "1D"
            time_series_of_locations = list(df.groupby(by="locationID"))
            for locationID, df in time_series_of_locations:
                df.sort_index()
                df.index = pd.to_datetime(df.index)

                count_series = df.resample(rule=freq_setting).size()

                output_dict[locationID] = count_series
            output_df = pd.DataFrame(output_dict)
            output_df.to_csv(dataset_path, line_terminator="\n")

        super().__init__(
            metadata=DatasetLoaderMetadata(
                f"uber_tlc_{sample_freq}.csv",
                uri="https://github.com/fivethirtyeight/uber-tlc-foil-response/raw/"
                "63bb878b76f47f69b4527d50af57aac26dead983/"
                "uber-trip-data/uber-raw-data-janjune-15.csv.zip",
                hash="9ed84ebe0df4bc664748724b633b3fe6"
                if sample_freq == "hourly"
                else "24f9fd67e4b9e53f0214a90268cd9bee",
                header_time="Pickup_date",
                format_time="%Y-%m-%d %H:%M",
                pre_process_zipped_csv_fn=pre_proces_fn,
                multivariate=multivariate,
            )
        )

    def _to_multi_series(self, series: pd.DataFrame) -> List[TimeSeries]:
        """
        load the Uber TLC dataset as a list of univariate timeseries, one for each locationID.
        """

        ts_list = []  # list of timeseries
        for label in series:
            srs = series[label]

            # filter column down to the period of recording
            start_date = min(srs.fillna(method="ffill").dropna().index)
            end_date = max(srs.fillna(method="bfill").dropna().index)
            active_range = (srs.index >= start_date) & (srs.index <= end_date)
            srs = srs[active_range]

            # convert to timeseries
            tmp = pd.DataFrame({"locationID": srs})
            tmp["date"] = tmp.index
            ts = TimeSeries.from_dataframe(tmp, "date", ["locationID"])
            ts_list.append(ts)
        return ts_list
