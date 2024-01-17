import copy

import numpy as np
import pandas as pd
import pytest

from darts.utils.data.tabularization import add_static_covariates_to_lagged_data
from darts.utils.timeseries_generation import linear_timeseries


class TestAddStaticToLaggedData:
    series = linear_timeseries(length=6)
    series = series.stack(series)
    series_stcov_single = series.with_static_covariates(pd.DataFrame({"a": [0.0]}))
    series_stcov_multi = series.with_static_covariates(
        pd.DataFrame({"a": [0.0], "b": [1.0]})
    )
    series_stcov_multivar = series.with_static_covariates(
        pd.DataFrame({"a": [0.0, 1.0], "b": [10.0, 20.0]})
    )
    features = np.empty(shape=(len(series), 2))

    def test_add_static_covs_train(self):
        # training when `last_shape=None`

        # does not use static covariates -> input = output
        features, last_shape = add_static_covariates_to_lagged_data(
            copy.deepcopy(self.features),
            self.series,
            uses_static_covariates=False,
            last_shape=None,
        )
        assert features.shape == self.features.shape
        assert last_shape is None

        with pytest.raises(ValueError):
            # use static covariates enforces that series have matching static covs
            add_static_covariates_to_lagged_data(
                copy.deepcopy(self.features),
                self.series,
                uses_static_covariates=True,
                last_shape=None,
            )

        with pytest.raises(ValueError):
            # some series with static covs, other without should raise an error
            add_static_covariates_to_lagged_data(
                [copy.deepcopy(self.features), copy.deepcopy(self.features)],
                [self.series, self.series_stcov_single],
                uses_static_covariates=True,
                last_shape=None,
            )

        with pytest.raises(ValueError):
            # mismatching static covs should raise an error
            add_static_covariates_to_lagged_data(
                [copy.deepcopy(self.features), copy.deepcopy(self.features)],
                [self.series_stcov_single, self.series_stcov_multi],
                uses_static_covariates=True,
                last_shape=None,
            )

        # single static cov should yield one added column
        expected_shape = (self.features.shape[0], self.features.shape[1] + 1)
        features, last_shape = add_static_covariates_to_lagged_data(
            copy.deepcopy(self.features),
            self.series_stcov_single,
            uses_static_covariates=True,
            last_shape=None,
        )
        assert features.shape == expected_shape
        assert last_shape == self.series_stcov_single.static_covariates.shape

        # multi static cov should yield two added columns
        expected_shape = (self.features.shape[0], self.features.shape[1] + 2)
        features, last_shape = add_static_covariates_to_lagged_data(
            copy.deepcopy(self.features),
            self.series_stcov_multi,
            uses_static_covariates=True,
            last_shape=None,
        )
        assert features.shape == expected_shape
        assert last_shape == self.series_stcov_multi.static_covariates.shape

        # multivariate static cov should yield four added columns
        expected_shape = (self.features.shape[0], self.features.shape[1] + 4)
        features, last_shape = add_static_covariates_to_lagged_data(
            copy.deepcopy(self.features),
            self.series_stcov_multivar,
            uses_static_covariates=True,
            last_shape=None,
        )
        assert features.shape == expected_shape
        assert last_shape == self.series_stcov_multivar.static_covariates.shape

        # multi series with multivariate static cov should yield four added columns per series
        expected_shape = (self.features.shape[0], self.features.shape[1] + 4)
        features, last_shape = add_static_covariates_to_lagged_data(
            [copy.deepcopy(self.features), copy.deepcopy(self.features)],
            [self.series_stcov_multivar, self.series_stcov_multivar],
            uses_static_covariates=True,
            last_shape=None,
        )
        assert [features_.shape == expected_shape for features_ in features]
        assert last_shape == self.series_stcov_multivar.static_covariates.shape
        assert np.all(
            features[0][:, -sum(last_shape) :] == np.array([0.0, 1.0, 10.0, 20.0])
        )

    def test_add_static_covs_predict(self):
        # predicting when `last_shape` other than `None`

        # does not use static covariates -> input = output
        features, last_shape = add_static_covariates_to_lagged_data(
            copy.deepcopy(self.features),
            self.series,
            uses_static_covariates=False,
            last_shape=(10, 10),
        )
        assert features.shape == self.features.shape
        assert last_shape == (10, 10)

        with pytest.raises(ValueError):
            # last static cov shape is given, but no covariates are available
            add_static_covariates_to_lagged_data(
                copy.deepcopy(self.features),
                self.series,
                uses_static_covariates=True,
                last_shape=(10, 10),
            )

        with pytest.raises(ValueError):
            # when last static cov shape is other than current one, raise an error
            add_static_covariates_to_lagged_data(
                copy.deepcopy(self.features),
                self.series_stcov_single,
                uses_static_covariates=True,
                last_shape=(10, 10),
            )

        # single static cov should yield one added column
        expected_shape = (self.features.shape[0], self.features.shape[1] + 1)
        features, last_shape = add_static_covariates_to_lagged_data(
            copy.deepcopy(self.features),
            self.series_stcov_single,
            uses_static_covariates=True,
            last_shape=self.series_stcov_single.static_covariates.shape,
        )
        assert features.shape == expected_shape
        assert last_shape == self.series_stcov_single.static_covariates.shape

        # multi static cov should yield two added columns
        expected_shape = (self.features.shape[0], self.features.shape[1] + 2)
        features, last_shape = add_static_covariates_to_lagged_data(
            copy.deepcopy(self.features),
            self.series_stcov_multi,
            uses_static_covariates=True,
            last_shape=self.series_stcov_multi.static_covariates.shape,
        )
        assert features.shape == expected_shape
        assert last_shape == self.series_stcov_multi.static_covariates.shape

        # multivariate static cov should yield four added columns
        expected_shape = (self.features.shape[0], self.features.shape[1] + 4)
        features, last_shape = add_static_covariates_to_lagged_data(
            copy.deepcopy(self.features),
            self.series_stcov_multivar,
            uses_static_covariates=True,
            last_shape=self.series_stcov_multivar.static_covariates.shape,
        )
        assert features.shape == expected_shape
        assert last_shape == self.series_stcov_multivar.static_covariates.shape

        # multi series with multivariate static cov should yield four added columns per series
        expected_shape = (self.features.shape[0], self.features.shape[1] + 4)
        features, last_shape = add_static_covariates_to_lagged_data(
            [copy.deepcopy(self.features), copy.deepcopy(self.features)],
            [self.series_stcov_multivar, self.series_stcov_multivar],
            uses_static_covariates=True,
            last_shape=self.series_stcov_multivar.static_covariates.shape,
        )
        assert [features_.shape == expected_shape for features_ in features]
        assert last_shape == self.series_stcov_multivar.static_covariates.shape
        assert np.all(
            features[0][:, -sum(last_shape) :] == np.array([0.0, 1.0, 10.0, 20.0])
        )
