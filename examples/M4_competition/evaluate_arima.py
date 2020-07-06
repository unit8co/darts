"""Evaluating AutoARIMA model on M4 timeseries

"""

from darts import TimeSeries
from darts.models import AutoARIMA
from darts.utils.statistics import check_seasonality, remove_seasonality, extract_trend_and_seasonality

import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle as pkl

from M4_metrics import *


if __name__ == "__main__":

    data_categories = ['Yearly', 'Quarterly', 'Monthly', 'Weekly', 'Daily', 'Hourly']

    info_dataset = pd.read_csv('dataset/M4-info.csv', delimiter=',').set_index('M4id')

    for cat in data_categories[::-1]:
        # Load TimeSeries from M4
        ts_train = pkl.load(open("dataset/train_"+cat+".pkl", "rb"))
        ts_test = pkl.load(open("dataset/test_"+cat+".pkl", "rb"))

        # Test models on all time series
        mase_all = []
        smape_all = []
        m = info_dataset.Frequency[cat[0]+"1"]
        for train, test in tqdm(zip(ts_train, ts_test)):
            train_des = train
            seasonOut = 1
            if m > 1:
                if check_seasonality(train, m=int(m), max_lag=2*m):
                    _, season = extract_trend_and_seasonality(train, m, model='multiplicative')
                    train_des = remove_seasonality(train, freq=m, model='multiplicative')
                    seasonOut = season[-m:].shift(m)
                    seasonOut = seasonOut.append_values(seasonOut.values())
                    seasonOut = seasonOut[:len(test)]
            if len(train_des) < 30:
                train_des = train_des.shift(-len(train)).append(train_des)
            autoar = AutoARIMA()
            autoar.fit(train_des)
            forecast_autoar = autoar.predict(len(test)) * seasonOut
            mase_all.append(np.vstack([
                mase_m4(train, test, forecast_autoar, m=m),
                ]))
            smape_all.append(np.vstack([
                smape_m4(test, forecast_autoar),
                ]))
        pkl.dump(mase_all, open("autoar_mase_"+cat+".pkl", "wb"))
        pkl.dump(smape_all, open("autoar_smape_"+cat+".pkl", "wb"))
        print("MASE; ARIMA: {}".format(*tuple(np.nanmean(np.stack(mase_all), axis=(0, 2)))))
        print("sMAPE; ARIMA: {}".format(*tuple(np.nanmean(np.stack(smape_all), axis=(0, 2)))))
        print("OWA: ", OWA_m4(cat, np.nanmean(np.stack(mase_all), axis=(0, 2)),
                              np.nanmean(np.stack(smape_all), axis=(0, 2))))
