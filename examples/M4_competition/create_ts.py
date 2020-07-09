"""Creating darts TimeSeries from M4 dataset

"""

from darts import TimeSeries
import os
import numpy as np
import pandas as pd
import pickle
from datetime import datetime
from tqdm import tqdm

# dataset info
data_categories = ['Yearly', 'Quarterly', 'Monthly', 'Weekly', 'Daily', 'Hourly']
train_datasets = []
test_datasets = []
for cat in data_categories:
    train_datasets.append(pd.read_csv('./dataset/train/{}-train.csv'.format(cat), delimiter=',').set_index('V1').T)
    test_datasets.append(pd.read_csv('./dataset/test/{}-test.csv'.format(cat), delimiter=',').set_index('V1').T)
info_dataset = pd.read_csv('./dataset/M4-info.csv', delimiter=',').set_index('M4id')

# creating time series
for i, dc in tqdm(enumerate(data_categories)):
    train_set = train_datasets[i]
    test_set = test_datasets[i]
    ts_train = []
    ts_test = []
    if os.path.isfile("dataset/train_"+dc+".pkl") and os.path.isfile("dataset/test_"+dc+".pkl"):
        print(" TimeSeries already created")
        continue
    for ts in tqdm(train_set.columns):
        if dc == 'Yearly' and train_set.count()[ts] > 490:
            print("ts too big, fallback to quaterly frequency")
            info_dataset.SP[ts] = "Quaterly"
        try:
            train_index = pd.date_range(info_dataset.StartingDate[ts],
                                        periods=train_set.count()[ts],
                                        freq=info_dataset.SP.str[0][ts])
            series_train = TimeSeries.from_series(train_set[ts].dropna().set_axis(train_index))
            test_index = pd.date_range(series_train.time_index()[-1],
                                       periods=info_dataset.Horizon[ts]+1,
                                       freq=info_dataset.SP.str[0][ts])
        except pd.errors.OutOfBoundsDatetime:
            train_index = pd.date_range(info_dataset.StartingDate[ts][:6] + '17' + info_dataset.StartingDate[ts][6:],
                                        periods=train_set.count()[ts],
                                        freq=info_dataset.SP.str[0][ts])
            series_train = TimeSeries.from_series(train_set[ts].dropna().set_axis(train_index))
            test_index = pd.date_range(series_train.time_index()[-1],
                                       periods=info_dataset.Horizon[ts]+1,
                                       freq=info_dataset.SP.str[0][ts])
        series_test = TimeSeries.from_series(test_set[ts].dropna().set_axis(test_index[1:]))
        ts_train.append(series_train)
        ts_test.append(series_test)
    pickle.dump(ts_train, open("dataset/train_"+dc+".pkl", "wb"))
    pickle.dump(ts_test, open("dataset/test_"+dc+".pkl", "wb"))
    print(dc+" frequency transformed")
print("All timeseries transformed")
