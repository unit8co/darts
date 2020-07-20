"""Evaluating AutoARIMA model on M4 timeseries

"""

from darts import ModelMode
from darts.models import AutoARIMA, NaiveSeasonal
from darts.utils.statistics import check_seasonality, remove_from_series, extract_trend_and_seasonality
from darts.utils import _build_tqdm_iterator

import numpy as np
import pandas as pd
import pickle as pkl

from M4_metrics import owa_m4, mase_m4, smape_m4


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
        m = int(info_dataset.Frequency[cat[0]+"1"])
        for train, test in _build_tqdm_iterator(zip(ts_train, ts_test), verbose=True):
            train_des = train
            seasonOut = 1
            if m > 1:
                if check_seasonality(train, m=m, max_lag=2*m):
                    _, season = extract_trend_and_seasonality(train, m, model=ModelMode.MULTIPLICATIVE)
                    train_des = remove_from_series(train, season, model=ModelMode.MULTIPLICATIVE)
                    seasonOut = season[-m:].shift(m)
                    seasonOut = seasonOut.append_values(seasonOut.values())
                    seasonOut = seasonOut[:len(test)]
            # if len(train_des) < 30:
            #     train_des = train_des.shift(-len(train)).append(train_des)
            autoar = AutoARIMA()
            try:
                autoar.fit(train_des)
            except ValueError:
                autoar = NaiveSeasonal(K=1)
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
        print("OWA: ", owa_m4(cat, np.nanmean(np.stack(mase_all), axis=(0, 2)),
                              np.nanmean(np.stack(smape_all), axis=(0, 2))))
