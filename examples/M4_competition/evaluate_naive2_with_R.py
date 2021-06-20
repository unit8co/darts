"""Evaluating DL models on M4 timeseries

"""

from darts import TimeSeries, ModelMode
from darts.models import NaiveSeasonal
from darts.utils.statistics import check_seasonality, remove_from_series, extract_trend_and_seasonality
from darts.utils import _build_tqdm_iterator

import numpy as np
import pandas as pd
import pickle as pkl

from M4_metrics import owa_m4, mase_m4, smape_m4, baseline

import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
ts = robjects.r('ts')
forecast = importr('forecast')
utils = importr('utils')

pandas2ri.activate()


rstring = """
    function(input, fh, m){
        library(forecast)
        input <- ts(input, frequency=m)
        SeasonalityTest <- function(input, ppy){
          #Used to determine whether a time series is seasonal
          tcrit <- 1.645
          if (length(input)<3*ppy){
            test_seasonal <- FALSE
          }else{
            xacf <- acf(input, plot = FALSE)$acf[-1, 1, 1]
            clim <- tcrit/sqrt(length(input)) * sqrt(cumsum(c(1, 2 * xacf^2)))
            test_seasonal <- ( abs(xacf[ppy]) > clim[ppy] )
    
            if (is.na(test_seasonal)==TRUE){ test_seasonal <- FALSE }
          }

          return(test_seasonal)
        }
        ppy <- frequency(input) ; ST <- F
        if (ppy>1){ ST <- SeasonalityTest(input,ppy) }
        if (ST==T){
            Dec <- decompose(input,type="multiplicative")
            des_input <- input/Dec$seasonal
            SIout <- head(rep(Dec$seasonal[(length(Dec$seasonal)-ppy+1):length(Dec$seasonal)], fh), fh)
        }else{
            des_input <- input ; SIout <- rep(1, fh)
        }
            list(des_input, SIout)
    }
"""
# """
#     ts_values: numpy_array
#     fh: forecast horizon int
#     m: frequency int
#     ----
#     des_input: numpy array
#     SIout: seaonality numpy array
# """
test_seasonality = robjects.r(rstring)

rstring = """
    function(input, m){
        library(forecast)
        input <- ts(input, frequency=m)
        lambda<-BoxCox.lambda(input, method="loglik", lower=0, upper=1)
        data.bxcx <- BoxCox(input, lambda)
        list(lambda, data.bxcx)
    }
"""
rboxcox = robjects.r(rstring)

rstring = """
    function(input, lambda, m){
        library(forecast)
        input <- ts(input, frequency=m)
        inv<-InvBoxCox(input, lambda)
        inv
    }
"""
rinvboxcox = robjects.r(rstring)

rstring = """
    function(input, fh, m){
        library(forecast)
        input <- ts(input, frequency=m)
        lambda<-BoxCox.lambda(input, method="loglik", lower=0, upper=1)
        data.bxcx <- BoxCox(input, lambda)
        
        data.forecast <- thetaf(data.bxcx, h=fh)
        
        inv<-InvBoxCox(data.forecast$mean, lambda)
        inv
    }
"""
rboxcoxtheta = robjects.r(rstring)


if __name__ == "__main__":
    data_categories = ['Yearly', 'Quarterly', 'Monthly', 'Weekly', 'Daily', 'Hourly']

    info_dataset = pd.read_csv('dataset/M4-info.csv', delimiter=',').set_index('M4id')

    for cat in data_categories[::-1]:
        # Load TimeSeries from M4
        ts_train = pkl.load(open("dataset/train_"+cat+".pkl", "rb"))
        ts_test = pkl.load(open("dataset/test_"+cat+".pkl", "rb"))

        # Test models on all time series
        season_diff = []
        mase_all = []
        smape_all = []
        m = int(info_dataset.Frequency[cat[0]+"1"])
        for train, test in _build_tqdm_iterator(zip(ts_train, ts_test), verbose=True):
            r_train_des, r_seasonOut = test_seasonality(train.values(), len(test), m)
            train_des = train
            seasonOut = 1
            if m > 1:
                if check_seasonality(train, m=m, max_lag=2*m):
                    _, season = extract_trend_and_seasonality(train, m, model=ModelMode.MULTIPLICATIVE)
                    train_des = remove_from_series(train, season, model=ModelMode.MULTIPLICATIVE)
                    seasonOut = season[-m:].shift(m)
                    seasonOut = seasonOut.append_values(seasonOut.values())
                    seasonOut = seasonOut[:len(test)]
                    
            season_diff.append(np.abs(train_des.values() - r_train_des)/np.abs(r_train_des))
            
            train_des = TimeSeries.from_times_and_values(train.time_index, r_train_des)
            seasonOut = TimeSeries.from_times_and_values(test.time_index, r_seasonOut)
            
            naive2 = NaiveSeasonal(K=1)
            naive2.fit(train_des)
            
            forecast_naive2 = naive2.predict(len(test)) * seasonOut
            
            mase_all.append(np.vstack([
                mase_m4(train, test, forecast_naive2, m=m),
                ]))
            smape_all.append(np.vstack([
                smape_m4(test, forecast_naive2),
                ]))
        rel_error = np.mean(np.hstack(season_diff))
        pkl.dump([mase_all, smape_all, season_diff], open("rnaive2_"+cat+".pkl", "wb"))
        print(np.round(np.nanmean(np.stack(mase_all), axis=(0, 2)), 3) == baseline[cat][1])
        print(np.round(np.nanmean(np.stack(smape_all), axis=(0, 2)), 3) == baseline[cat][0])
        print("MASE; Naive2: {:.3f}".format(*tuple(np.nanmean(np.stack(mase_all), axis=(0, 2)))))
        print("sMAPE; Naive2: {:.3f}".format(*tuple(np.nanmean(np.stack(smape_all), axis=(0, 2)))))
        print("OWA: ", owa_m4(cat, np.nanmean(np.stack(smape_all), axis=(0, 2)),
                              np.nanmean(np.stack(mase_all), axis=(0, 2))))
