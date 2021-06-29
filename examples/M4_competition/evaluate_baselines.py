"""Evaluating baseline models on M4 timeseries

"""

from darts import ModelMode
from darts.models import NaiveSeasonal, NaiveDrift, ExponentialSmoothing
from darts.utils.statistics import check_seasonality, remove_from_series, extract_trend_and_seasonality
from darts.utils import _build_tqdm_iterator

import numpy as np
import pandas as pd
import pickle as pkl

from M4_metrics import mase_m4, smape_m4, owa_m4


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
                if check_seasonality(train, m=m, max_lag=2 * m):
                    _, season = extract_trend_and_seasonality(train, m, model=ModelMode.MULTIPLICATIVE)
                    train_des = remove_from_series(train, season, model=ModelMode.MULTIPLICATIVE)
                    seasonOut = season[-m:].shift(m)
                    seasonOut = seasonOut.append_values(seasonOut.values())
                    seasonOut = seasonOut[:len(test)]
            naive = NaiveDrift()
            naive2 = NaiveSeasonal(K=1)
            naiveSeason = NaiveSeasonal(K=m)
            ses = ExponentialSmoothing(trend=None, seasonal=None, seasonal_periods=m)
            holt = ExponentialSmoothing(seasonal=None, damped=False, trend='additive', seasonal_periods=m)
            damp = ExponentialSmoothing(seasonal=None, damped=True, trend='additive', seasonal_periods=m)
            naive.fit(train)
            naive2.fit(train_des)
            naiveSeason.fit(train)
            try:
                ses.fit(train_des)
            except ValueError:
                # not useful anymore
                train_des = train_des.shift(-len(train_des)).append(train_des)
#                 train_des = TimeSeries.from_times_and_values(train_des.shift(-11).time_index[:11],
#                                                              2*train_des.values()[0]-train_des.values()[10::-1]) \
#                                                             .append(train_des)
                ses.fit(train_des)
            holt.fit(train_des)
            damp.fit(train_des)
            forecast_naiveSeason = naiveSeason.predict(len(test))
            forecast_naiveDrift = naive.predict(len(test))
            forecast_naive = forecast_naiveSeason + forecast_naiveDrift - train.last_value()
            forecast_naive2 = naive2.predict(len(test)) * seasonOut
            forecast_ses = ses.predict(len(test)) * seasonOut
            forecast_holt = holt.predict(len(test)) * seasonOut
            forecast_damp = damp.predict(len(test)) * seasonOut
            forecast_comb = ((forecast_ses + forecast_holt + forecast_damp) / 3)
            
            mase_all.append(np.vstack([
                mase_m4(train, test, forecast_naiveSeason, m=m),
                mase_m4(train, test, forecast_naive, m=m),
                mase_m4(train, test, forecast_naive2, m=m),
                mase_m4(train, test, forecast_ses, m=m),
                mase_m4(train, test, forecast_holt, m=m),
                mase_m4(train, test, forecast_damp, m=m),
                mase_m4(train, test, forecast_comb, m=m),
                ]))
            smape_all.append(np.vstack([
                smape_m4(test, forecast_naiveSeason),
                smape_m4(test, forecast_naive),
                smape_m4(test, forecast_naive2),
                smape_m4(test, forecast_ses),
                smape_m4(test, forecast_holt),
                smape_m4(test, forecast_damp),
                smape_m4(test, forecast_comb),
                ]))
        pkl.dump(mase_all, open("baseline_mase_"+cat+".pkl", "wb"))
        pkl.dump(smape_all, open("baseline_smape_"+cat+".pkl", "wb"))
        print("MASE; sNaive: {:.3f}, sNaive+Drift: {:.3f}, Naive2: {:.3f}, SES: {:.3f}, Holt: {:.3f}, "
              "Damped: {:.3f}, Comb: {:.3f}".format(*tuple(np.nanmean(np.stack(mase_all), axis=(0,2)))))
        print("sMAPE; sNaive: {:.3f}, sNaive+Drift: {:.3f}, Naive2: {:.3f}, SES: {:.3f}, Holt: {:.3f}, "
              "Damped: {:.3f}, Comb: {:.3f}".format(*tuple(np.nanmean(np.stack(smape_all), axis=(0, 2)))))
        print("OWA: ", owa_m4(cat,
                              np.nanmean(np.stack(smape_all), axis=(0, 2)),
                              np.nanmean(np.stack(mase_all), axis=(0, 2))))
