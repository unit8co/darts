import numpy as np
import pandas as pd

from darts.dataprocessing.transformers import MissingValuesFiller
from darts.timeseries import TimeSeries


class TestMissingValuesFiller:
    time = pd.date_range("20130101", "20130130")
    static_covariate = pd.DataFrame({"0": [1]})

    const_series = TimeSeries.from_times_and_values(
        time, np.array([2.0] * len(time)), static_covariates=static_covariate
    )
    const_series_with_holes = TimeSeries.from_times_and_values(
        time,
        np.array([2.0] * 10 + [np.nan] * 5 + [2.0] * 10 + [np.nan] * 5),
        static_covariates=static_covariate,
    )

    lin = [float(i) for i in range(len(time))]
    lin_series = TimeSeries.from_times_and_values(time, np.array(lin))
    lin_series_with_holes = TimeSeries.from_times_and_values(
        time, np.array(lin[0:10] + [np.nan] * 5 + lin[15:24] + [np.nan] * 5 + [lin[29]])
    )

    def test_fill_const_series_with_const_value(self):
        const_transformer = MissingValuesFiller(fill=2.0)
        transformed = const_transformer.transform(self.const_series_with_holes)
        assert self.const_series == transformed

    def test_fill_const_series_with_auto_value(self):
        auto_transformer = MissingValuesFiller()
        transformed = auto_transformer.transform(self.const_series_with_holes)
        assert self.const_series == transformed

    def test_fill_lin_series_with_auto_value(self):
        auto_transformer = MissingValuesFiller()
        transformed = auto_transformer.transform(self.lin_series_with_holes)
        assert self.lin_series == transformed

    def test_fill_static_covariates_preserved(self):
        const_transformer = MissingValuesFiller(fill=2.0)
        transformed = const_transformer.transform(self.const_series_with_holes)
        assert (
            self.const_series.static_covariates.values
            == transformed.static_covariates.values
        )
