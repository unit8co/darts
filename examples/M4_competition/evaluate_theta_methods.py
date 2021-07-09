"""Evaluating DL models on M4 timeseries

"""

from darts import TimeSeries, SeasonalityMode
from darts.models import Theta, FourTheta
from darts.utils.statistics import check_seasonality, remove_from_series, extract_trend_and_seasonality
from darts.utils import _build_tqdm_iterator

from scipy.stats import boxcox, boxcox_normplot
from scipy.special import inv_boxcox

import numpy as np
import pandas as pd
import pickle as pkl

from M4_metrics import mase_m4, smape_m4, owa_m4


def train_theta(ts, seasonality, n):
    # should be the same as fitting with mode='multiplicative' and no prior deseasonalization.
    # done to change easily the deseasonalization method
    theta = Theta(theta=0, season_mode=SeasonalityMode.NONE)
    theta.fit(ts)
    forecast = theta.predict(n) * seasonality
    return forecast


def train_theta_boxcox(ts, seasonality, n):
    theta_bc = Theta(theta=0, season_mode=SeasonalityMode.NONE)
    shiftdata = 0
    if (ts.univariate_values() < 0).any():
        shiftdata = -ts.min() + 100
        ts = ts + shiftdata
    new_values, lmbd = boxcox(ts.univariate_values())
    if lmbd < 0:
        lmbds, value = boxcox_normplot(ts.univariate_values(), lmbd - 1, 0, N=100)
        if np.isclose(value[0], 0):
            lmbd = lmbds[np.argmax(value)]
            new_values = boxcox(ts.univariate_values(), lmbd)
        if np.isclose(new_values, new_values[0]).all():
            lmbd = 0
            new_values = boxcox(ts.univariate_values(), lmbd)
    ts = TimeSeries.from_times_and_values(ts.time_index, new_values)
    theta_bc.fit(ts)
    forecast = theta_bc.predict(n)

    new_values = inv_boxcox(forecast.univariate_values(), lmbd)
    forecast = TimeSeries.from_times_and_values(seasonality.time_index, new_values)
    if shiftdata > 0:
        forecast = forecast - shiftdata
    forecast = forecast * seasonality
    if (forecast.univariate_values() < 0).any():
        indices = seasonality.time_index[forecast < 0]
        forecast = forecast.update(indices, np.zeros(len(indices)), inplace=True)
    return forecast


def train_4theta(ts, n):
    fourtheta = FourTheta.select_best_model(ts, [1, 2, 3], m)
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
        m = int(info_dataset.Frequency[cat[0]+"1"])
        for train, test in _build_tqdm_iterator(zip(ts_train, ts_test), verbose=True):
            train_des = train
            seasonOut = 1
            if m > 1:
                if check_seasonality(train, m=m, max_lag=2*m):
                    _, season = extract_trend_and_seasonality(train, m, model=SeasonalityMode.MULTIPLICATIVE)
                    train_des = remove_from_series(train, season, model=SeasonalityMode.MULTIPLICATIVE)
                    seasonOut = season[-m:].shift(m)
                    seasonOut = seasonOut.append_values(seasonOut.values())
                    seasonOut = seasonOut[:len(test)]

            forecast_theta = train_theta(train_des, seasonOut, len(test))
            forecast_fourtheta = train_4theta(train, len(test))
            forecast_thetaBC = train_theta_boxcox(train_des, seasonOut, len(test))

            mase_all.append(np.vstack([
                mase_m4(train, test, forecast_theta, m=m),
                mase_m4(train, test, forecast_fourtheta, m=m),
                mase_m4(train, test, forecast_thetaBC, m=m),
                ]))
            smape_all.append(np.vstack([
                smape_m4(test, forecast_theta),
                smape_m4(test, forecast_fourtheta),
                smape_m4(test, forecast_thetaBC),
                ]))
        pkl.dump(mase_all, open("theta_mase_"+cat+".pkl", "wb"))
        pkl.dump(smape_all, open("theta_smape_"+cat+".pkl", "wb"))
        print("MASE; Theta: {:.3f}, 4theta: {:.3f},"
              " Theta-BoxCox: {:.3f}".format(*tuple(np.nanmean(np.stack(smape_all), axis=(0, 2)))))
        print("sMAPE; Theta: {:.3f}, 4theta: {:.3f},"
              " Theta-BoxCox: {:.3f}".format(*tuple(np.nanmean(np.stack(mase_all), axis=(0, 2)))))
        print("OWA: ", owa_m4(cat, np.nanmean(np.stack(mase_all), axis=(0, 2)),
                              np.nanmean(np.stack(smape_all), axis=(0, 2))))
