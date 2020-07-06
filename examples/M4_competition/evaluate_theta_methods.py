"""Evaluating DL models on M4 timeseries

"""

from darts import TimeSeries
from darts.models import Theta
from FourTheta import FourTheta
from darts.backtesting import backtest_forecasting, backtest_gridsearch
from darts.utils.statistics import check_seasonality, remove_seasonality, extract_trend_and_seasonality

from scipy.stats import boxcox, boxcox_normplot, boxcox_normmax
from scipy.special import inv_boxcox

import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle as pkl

from M4_metrics import *
from sklearn.metrics import mean_absolute_error as mae


def train_theta(ts, seasonality, n):
    # should be the same as fitting with mode='multiplicative' and no prior deseasonalization.
    # done to change easily the deseasonalization method
    theta = Theta(theta=2, mode="additive", seasonality_period=1)
    theta.fit(ts)
    forecast = theta.predict(n) * seasonality
    return forecast


def train_theta_boxcox(ts, seasonality, n):
    theta_bc = Theta(theta=2, mode="additive", seasonality_period=1)
    shiftdata = 0
    if (ts < 0).any():
        shiftdata = -ts.min() + 100
        ts = ts + shiftdata
    new_values, lmbd = boxcox(ts.values())
    if lmbd < 0:
        lmbds, value = boxcox_normplot(ts.values(), lmbd - 1, 0, N=100)
        if np.isclose(value[0], 0):
            lmbd = lmbds[np.argmax(value)]
            new_values = boxcox(ts.values(), lmbd)
        if np.isclose(new_values, new_values[0]).all():
            lmbd = 0
            new_values = boxcox(ts.values(), lmbd)
    ts = TimeSeries.from_times_and_values(ts.time_index(), new_values)
    theta_bc.fit(ts)
    forecast = theta_bc.predict(n)

    new_values = inv_boxcox(forecast.values(), lmbd)
    forecast = TimeSeries.from_times_and_values(seasonality.time_index(), new_values)
    if shiftdata > 0:
        forecast = forecast - shiftdata
    forecast = forecast * seasonality
    if (forecast < 0).any():
        indices = seasonality.time_index()[forecast < 0]
        forecast = forecast.update(indices, np.zeros(len(indices)), inplace=True)
    return forecast


def train_4theta(ts, n):
    season_mode = ["additive", "multiplicative"]
    model_mode = ["additive", "multiplicative"]
    drift_mode = ["linear", "exponential"]
    if (ts.values() <= 0).any():
        drift_mode = ["linear"]
        model_mode = ["additive"]
        season_mode = ["additive"]
    fourtheta = backtest_gridsearch(FourTheta,
                                    {"theta": [1, 2, 3],
                                     "mode": model_mode,
                                     "season_mode": season_mode,
                                     "trend": drift_mode,
                                     "seasonality_period": [m]
                                     },
                                    train,
                                    val_series='train',
                                    metric=mae)
    fourtheta.fit(ts)
    forecast = fourtheta.predict(n)
    return forecast


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
                else:
                    m = 1

            forecast_theta = train_theta(train_des, seasonOut, len(test))
            forecast_fourtheta = train_4theta(train, len(test))
            forecast_thetaBC = train_theta_boxcox(train_des, seasonOut, len(test))
            
            m = info_dataset.Frequency[cat[0]+'1']
            mase_all.append(np.vstack([
                mase_m4(train, test, forecast_theta, m=m),
                mase_m4(train, test, forecast_fourtheta, m=m),
                mase_m4(train, test, forecast_thetaBC, m=m),
                ]))
            print(np.mean(mase_all[-1]))
            smape_all.append(np.vstack([
                smape_m4(test, forecast_theta),
                smape_m4(test, forecast_fourtheta),
                smape_m4(test, forecast_thetaBC),
                ]))
        pkl.dump(mase_all, open("theta_mase_"+cat+".pkl", "wb"))
        pkl.dump(smape_all, open("theta_smape_"+cat+".pkl", "wb"))
        print("MASE; Theta: {:.3f}, 4theta: {:.3f},"
              " Theta-BoxCox: {:.3f}".format(*tuple(np.nanmean(np.stack(mase_all), axis=(0, 2)))))
        print("sMAPE; Theta: {:.3f}, 4theta: {:.3f},"
              " Theta-BoxCox: {:.3f}".format(*tuple(np.nanmean(np.stack(smape_all), axis=(0, 2)))))
        print("OWA: ", OWA_m4(cat, np.nanmean(np.stack(mase_all), axis=(0, 2)),
                              np.nanmean(np.stack(smape_all), axis=(0, 2))))
