"""Evaluating FFT model on M4 timeseries

"""

from darts import ModelMode
from darts.models import FFT
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
            try:
                try:
                    fft = FFT.gridsearch({'nr_freqs_to_keep': [5, 7, 10, 15, 25, 50],
                                               'trend': ['poly', 'exp'],
                                               'trend_poly_degree': [0, 1, 2, 3],
                                               'required_matches': [None]},
                                              train[:-2*len(test)],
                                         val_series=train[-2 * len(test):],
                                         metric=lambda x, y: np.mean(mase_m4(train[-2*len(test)], x, y, m=m)))
                except ValueError:
                    fft = FFT.gridsearch({'nr_freqs_to_keep': [5, 7, 10, 15, 25, 50],
                                               'trend': ['poly', 'exp'],
                                               'trend_poly_degree': [0, 1, 2, 3],
                                               'required_matches': [None]},
                                              train[:-len(test)],
                                         val_series=train[-len(test):],
                                         metric=lambda x, y: np.mean(mase_m4(train[-len(test)], x, y, m=m)))
                fft.fit(train)
                forecast_fft = fft.predict(len(test))
                mase_all.append(np.vstack([
                    mase_m4(train, test, forecast_fft, m=m),
                    ]))
                smape_all.append(np.vstack([
                    smape_m4(test, forecast_fft),
                    ]))
            except Exception as e:
                print(e)
                pkl.dump(mase_all, open("FFT_mase_"+cat+".pkl", "wb"))
                pkl.dump(smape_all, open("FFT_smape_"+cat+".pkl", "wb"))
                break
        pkl.dump(mase_all, open("FFT_mase_"+cat+"2.pkl", "wb"))
        pkl.dump(smape_all, open("FFT_smape_"+cat+"2.pkl", "wb"))
        print("MASE; fft: {}".format(*tuple(np.nanmean(np.stack(mase_all), axis=(0, 2)))))
        print("sMAPE; fft: {}".format(*tuple(np.nanmean(np.stack(smape_all), axis=(0, 2)))))
        print("OWA: ", owa_m4(cat,
                              np.nanmean(np.stack(smape_all), axis=(0, 2)),
                              np.nanmean(np.stack(mase_all), axis=(0, 2))))
