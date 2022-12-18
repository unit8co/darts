# from darts import TimeSeries

# from darts.tests.base_test_class import DartsBaseTestClass
# from darts.utils.timeseries_generation import linear_timeseries

# from darts.utils.data.tabularization import create_lagged_training_data

# import pandas as pd
# import numpy as np
# from itertools import product

# import warnings

# from typing import Optional, Sequence

# class CreateLaggedPredictionDataTestCase(DartsBaseTestClass):

#     lag_combos = ({'vals': None, 'max': None},
#     {'vals': [-1, -3], 'max': 3},
#     {'vals': [-3, -1], 'max': 3})
#     max_samples_per_ts_combos = (1, 2, None)

#     @staticmethod
#     def get_feature_times(target: TimeSeries, past: TimeSeries, future: TimeSeries, lags_max: Optional[int],
# lags_past_max: Optional[int], lags_future_max: Optional[int], max_samples_per_ts: Optional[int]) -> pd.Index:
#         times = None
#         for series_i, lags_max_i in zip([target, past, future], [lags_max, lags_past_max, lags_future_max]):
#             if lags_max_i is not None:
#                 series_i = series_i.append_values()
#                 if times is None:
#                     times = series_i.time_index[lags_max_i:]
#                 else:
#                     times = times.intersection(series_i.time_index[lags_max_i:])
#         return times

#     @staticmethod
#     def get_features(series: TimeSeries, feature_times: pd.Index, lags: Optional[Sequence[int]]) -> np.array:
#         if lags is None:
#             features = None
#         else:
#             array_vals = series.all_values(copy=False)
#             features = []
#             for time in feature_times:
#                 feature_i = []
#                 for lag in lags:
#                     lag_time = time + lag*series.freq
#                     idx = np.searchsorted(series.time_index, lag_time)
#                     feature_i.append(array_vals[idx,:,0].reshape(-1))
#                 feature_i = np.concatenate(feature_i, axis=0)
#                 features.append(feature_i)
#             features = np.stack(features, axis=0)
#         return features

#     def test_equal_freq_range_index(self):
#         target = linear_timeseries(start_value=0, end_value=10, start=2, end=20, freq=2)
#         past = linear_timeseries(start_value=10, end_value=20, start=4, end=23, freq=2)
#         future = linear_timeseries(start_value=20, end_value=30, start=6, end=26, freq=2)
#         param_combos = product(self.lag_combos, self.lag_combos, self.lag_combos, self.max_samples_per_ts_combos)
#         for (lags, lags_past, lags_future, output_chunk_length, multi_models,
# max_samples_per_ts) in param_combos:
#             if all(x is None for x in [lags['vals'], lags_past['vals'], lags_future['vals']]):
#                 continue
#             feature_times = self.get_feature_times(target, past, future, lags['max'],
# lags_past['max'], lags_future['max'], max_samples_per_ts)
#             target_features = self.get_features(target, feature_times, lags['vals'])
#             past_features = self.get_features(past, feature_times, lags_past['vals'])
#             future_features = self.get_features(future, feature_times, lags_future['vals'])
#             to_concat = [x for x in (target_features, past_features, future_features) if x is not None]
#             expected_X = np.concatenate(to_concat, axis=1)
#             # `create_lagged_training_data` throws warning when a series is specified, but
#             # the corresponding lag is not - silence this warning:
#             with warnings.catch_warnings():
#                 warnings.simplefilter("ignore")
#                 X, times = create_lagged_training_data(target, output_chunk_length,
#                 past_covariates=past,
#                 future_covariates=future,
#                 lags=lags['vals'],
#                 lags_past_covariates=lags_past['vals'],
#                 lags_future_covariates=lags_future['vals'],
#                 multi_models=multi_models,
#                 max_samples_per_ts=max_samples_per_ts
#                 )
#             self.assertTrue(np.allclose(expected_X, X[:,:,0]))
#             self.assertTrue(feature_times.equals(times))
