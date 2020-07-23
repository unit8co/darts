"""Creating darts TimeSeries from M4 dataset

"""

from darts import TimeSeries
import os
import numpy as np
import pandas as pd
import pickle
from datetime import datetime
from darts.utils import _build_tqdm_iterator

# dataset info
data_categories = ['Yearly', 'Quarterly', 'Monthly', 'Weekly', 'Daily', 'Hourly']
train_datasets = []
test_datasets = []
for cat in data_categories:
    train_datasets.append(pd.read_csv('./dataset/train/{}-train.csv'.format(cat), delimiter=',').set_index('V1').T)
    test_datasets.append(pd.read_csv('./dataset/test/{}-test.csv'.format(cat), delimiter=',').set_index('V1').T)
info_dataset = pd.read_csv('./dataset/M4-info.csv', delimiter=',').set_index('M4id')

# creating time series
for i, dc in _build_tqdm_iterator(enumerate(data_categories), verbose=True):
    if os.path.isfile("dataset/train_"+dc+".pkl") and os.path.isfile("dataset/test_"+dc+".pkl"):
        print(" TimeSeries already created")
        continue
    train_set = train_datasets[i]
    test_set = test_datasets[i]
    ts_train = []
    ts_test = []
    forecast_horizon = test_set.shape[0]
    if dc == 'Yearly':
        index = pd.date_range(pd.Timestamp.min,
                              periods=584,
                              freq=info_dataset.SP.str[0][dc[0] + '1'])
        fallback_index = pd.date_range(pd.Timestamp.min,
                                       periods=train_set.shape[0] + forecast_horizon,
                                       freq='Q')
    else:
        index = pd.date_range(pd.Timestamp.min,
                              periods=train_set.shape[0] + forecast_horizon,
                              freq=info_dataset.SP.str[0][dc[0]+'1'])
    for ts in _build_tqdm_iterator(train_set.columns, verbose=True):
        train_index = index
        if dc == 'Yearly' and train_set.count()[ts] > (len(index) - forecast_horizon):
            print("ts too big, fallback to quarterly frequency")
            train_index = fallback_index
        series_train = TimeSeries.from_series(train_set[ts].dropna().set_axis(train_index[:train_set.count()[ts]]))
        series_test = TimeSeries.from_series(test_set[ts].dropna().
                                             set_axis(train_index[train_set.count()[ts]:
                                                                  train_set.count()[ts]+forecast_horizon]))
        ts_train.append(series_train)
        ts_test.append(series_test)
    pickle.dump(ts_train, open("dataset/train_"+dc+".pkl", "wb"))
    pickle.dump(ts_test, open("dataset/test_"+dc+".pkl", "wb"))
    print(dc+" frequency transformed")
print("All timeseries transformed")
