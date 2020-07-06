"""Evaluating FFT model on M4 timeseries

"""

from darts import TimeSeries
from darts.models import ExponentialSmoothing, Theta, FFT
from darts.metrics import mape, mase
from darts.backtesting import backtest_forecasting, backtest_gridsearch, forecasting_residuals
from darts.backtesting.backtesting import explore_models
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
            try:
                train_des = train
                seasonOut = 1
                if m > 1:
                    if check_seasonality(train, m=int(m), max_lag=2*m):
                        _, season = extract_trend_and_seasonality(train, m, model='multiplicative')
                        train_des = remove_seasonality(train, freq=m, model='multiplicative')
                        seasonOut = season[-m:].shift(m)
                        seasonOut = seasonOut.append_values(seasonOut.values())[:len(test)]

                try:
                    fft = backtest_gridsearch(FFT,
                                            {'nr_freqs_to_keep': [2, 3, 5, 10, 15, 25, 50],
                                             'trend': ['poly', 'exp'],
                                             'trend_poly_degree': [1, 2, 3],
                                             'required_matches': [None]},
                                            train,
                                            fcast_horizon_n=len(test),
                                            num_predictions=3,
                                            metric=lambda x, y: np.mean(mase_m4(train, x, y, m=m)))
                except ValueError:
                    fft = backtest_gridsearch(FFT,
                                            {'nr_freqs_to_keep': [2, 3, 5, 10, 15, 25, 50],
                                             'trend': ['poly', 'exp'],
                                             'trend_poly_degree': [1, 2, 3],
                                             'required_matches': [None]},
                                            train,
                                            fcast_horizon_n=len(test)//2,
                                            num_predictions=3,
                                            metric=lambda x, y: np.mean(mase_m4(train, x, y, m=m)))
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
        print("OWA: ", OWA_m4(cat,
                              np.nanmean(np.stack(mase_all), axis=(0, 2)),
                              np.nanmean(np.stack(smape_all), axis=(0, 2))))
