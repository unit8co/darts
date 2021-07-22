"""Reproducing 6th place winning model from M4 competition
    Generalised Rolling Origin Evaluation (GROE)
"""

from darts import TimeSeries
from darts.models import NaiveSeasonal
from darts.models.forecasting_model import ForecastingModel

import pandas as pd
import numpy as np
from tqdm import tqdm_notebook as tqdm
import pickle

import rpy2.robjects as robjects
from rpy2.rinterface_lib.embedded import RRuntimeError
from rpy2.robjects import pandas2ri

from M4_metrics import owa_m4, mase_m4, smape_m4


rstring = """
    function(input, fh, fq){
        library(forecast)
        input <- ts(input, frequency=fq)
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

# Test if a seasonality of period m exists, and extract it
test_seasonality = robjects.r(rstring)
# Use example
# des_input, seasonOut = test_seasonality(train.values(), len(test), m)

rstring = """
    function(input, fh, fq){
        library(forecTheta)
        input <- ts(input, frequency=fq)
        out_otm <- otm(input, fh, level=c(95,95))
        out_otm$mean
    }
"""
# OTM model that returns a forecast of horizon fh
rOTM = robjects.r(rstring)
# Use example
# outOTM = rOTM(train.values(), len(test), m)
# forecast_otm = TimeSeries.from_times_and_values(test.time_index, outOTM)

rstring = """
    function(input, fh, fq){
        library(forecTheta)
        input <- ts(input, frequency=fq)
        out_dotm <- dotm(input, fh, level=c(95,95))
        out_dotm$mean
    }
"""
# DOTM model that returns a forecast of horizon fh
rDOTM = robjects.r(rstring)
# Use example
# outDOTM = rDOTM(train.values(), len(test), m)
# forecast_dotm = TimeSeries.from_times_and_values(test.time_index, outDOTM)

rstring = """
    function(input, fq){
        library(forecast)
        input <- ts(input, frequency=fq)
        out_ets <- ets(input)
        paste0(out_ets$components[1:3], collapse='')
    }
"""
# Train an ETS model and return its parameters
getETScomponent = robjects.r(rstring)

rstring = """
    function(input, fh, fq, model){
        library(forecast)
        input <- ts(input, frequency=fq)
        out_ets <- forecast(ets(input, model=model),h=fh, level=0)
        out_ets$mean
    }
"""
# forecast an ETS model given its parameters
rETS = robjects.r(rstring)
# Use example
# modelETS = getETScomponent(train.values(), m)
# out_ets = rETS(train.values(), len(test), m, modelETS)
# forecast_ets = TimeSeries.from_times_and_values(test.time_index, out_ets)

rstring = """
    function(input, fq){
        library(forecast)
        input <- ts(input, frequency=fq)
        out_arima <- auto.arima(input)
        out_arima
    }
"""
# Train an AutoARIMA model
getARIMAcomponent = robjects.r(rstring)

rstring = """
    function(input, fh, fq, model){
        library(forecast)
        input <- ts(input, frequency=fq)
        out_arima <- forecast(Arima(y=input, model=model), h=fh, level=0)
        out_arima$mean
    }
"""
# forecast a given ARIMA model
rARIMA = robjects.r(rstring)
# Use example
# arima_model = getARIMAcomponent(train.values(), m)
# out_arima = rARIMA(train.values(), len(test), m, arima_model)
# forecast_arima = TimeSeries.from_times_and_values(test.time_index, out_arima)

rstring = """
    function(input, fh, fq){
        library(forecast)
        input <- ts(input, frequency=fq)
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
        naive(des_input, h=fh)$mean*SIout
    }
"""
# Apply a naive model on deseasonalized time series, and reseason the forecast (naive2 model)
rNaive2 = robjects.r(rstring)
# Use example
# out_naive2 = rNaive2(train.values(), len(test), m)
# forecast_naive2 = TimeSeries.from_times_and_values(test.time_index, out_naive2)


def groe_owa(ts: TimeSeries, model: ForecastingModel, fq: int, n1: int, m: int, p: int) -> float:
    """
    Implementation of Generalized Rolling Origin Evaluation using OWA score.

    The concept is to cross-validate a model on a time series with rolling origin, using OWA score from M4 competition.

    Parameters
    -----------
    ts
        The time series object to use to cross-validate
    model
        The Darts model to evaluate
    fq
        Period of the seasonality of the time series
    n1
        First origin to use for the cross-validation
    m
        Stride used for rolling the origin
    p
        number of stride to operate
    Returns
    -------
    Float
        sum of OWA score for all different origins
        If there is an error with one of the origin, return 0.
    """
    # todo: Implement generalized version from R
    n = len(ts)
    errors = []
    for i in range(p):
        # if origin is further than end timestamp, end function
        if n1 + i * m >= n:
            break
        ni = n1 + i * m
        npred = n - ni
        train = ts[:ni]
        test = ts[ni:]

        forecast_naive2 = rNaive2(train.values(), npred, fq)
        forecast_naive2 = TimeSeries.from_times_and_values(test.time_index, forecast_naive2)
        try:
            error_ase_n2 = mase_m4(train, test, forecast_naive2)
            error_sape_n2 = smape_m4(test, forecast_naive2)
        except ValueError:
            errors.append(0)
            continue
        try:
            model.fit(train)
            forecast = model.predict(npred)
        except RRuntimeError:
            errors.append(0)
            continue
        try:
            error_ase = mase_m4(train, test, forecast)
            error_sape = smape_m4(test, forecast)
            OWA = 0.5 * (error_sape / error_sape_n2) + 0.5 * (error_ase / error_ase_n2)
            errors.append(np.sum(OWA))
        except ValueError:
            errors.append(0)
    errors = np.sum(errors)
    return errors


class RModel(ForecastingModel):
    """
    Wrapper around R function that takes a time series and return a forecast
    """
    def __init__(self, rmodel, m, **info):
        super().__init__()
        self.rmodel = rmodel
        self.m = m
        self.info = info
        self.values = None

    def fit(self, ts):
        super().fit(ts)
        self.values = ts.values()

    def predict(self, n):
        super().predict(n)
        out = self.rmodel(self.values, n, self.m, **self.info)
        return self._build_forecast_series(out)


def fallback(mase_all, smape_all, train, test, m):
    naive = NaiveSeasonal(K=m)
    naive.fit(train)
    forecast = naive.predict(len(test))
    mase_all.append(np.vstack([
        mase_m4(train, test, forecast, m=m),
    ]))
    smape_all.append(np.vstack([
        smape_m4(test, forecast),
    ]))


if __name__ == "__main__":
    pandas2ri.activate()

    data_categories = ['Macro', 'Micro', 'Demographic', 'Industry', 'Finance', 'Other']
    data_freq = ['Yearly', 'Quarterly', 'Monthly', 'Weekly', 'Daily', 'Hourly']
    info_dataset = pd.read_csv('dataset/m4/M4-info.csv', delimiter=',').set_index('M4id')

    for freq in data_freq:
        ts_train = pickle.load(open("dataset/train_" + freq + ".pkl", "rb"))
        ts_test = pickle.load(open("dataset/test_" + freq + ".pkl", "rb"))
        mase_all = []
        smape_all = []
        m = info_dataset.Frequency[freq[0] + '1']
        for train, test in tqdm(zip(ts_train, ts_test)):
            model_arima = getARIMAcomponent(train.values(), m)
            model_ets = getETScomponent(train.values(), m)
            otm_model = RModel(rOTM, m)
            dotm_model = RModel(rDOTM, m)
            arima_model = RModel(rARIMA, m, model=model_arima)
            ets_model = RModel(rETS, m, model=model_ets)
            models = [otm_model, dotm_model, arima_model, ets_model]
            model_predictions = []
            for model in models:
                try:
                    model.fit(train)
                    model_predictions.append(model.predict(len(test)))
                except RRuntimeError:
                    fallback(mase_all, smape_all, train, test, m)
                    continue
            criterion = [
                groe_owa(train, otm_model, m, max(5, len(train) - len(test)),
                         int(np.floor(len(test) / 6)), 6),
                groe_owa(train, dotm_model, m, max(5, len(train) - len(test)),
                         int(np.floor(len(test) / 6)), 6),
                groe_owa(train, arima_model, m, max(5, len(train) - len(test)),
                         int(np.floor(len(test) / 6)), 6),
                groe_owa(train, ets_model, m, max(5, len(train) - len(test)),
                         int(np.floor(len(test) / 6)), 6)]

            if not np.all(np.array(criterion) > 0):
                fallback(mase_all, smape_all, train, test, m)
                continue

            Score = 1 / np.array(criterion)
            pesos = Score / Score.sum()

            groe_ensemble = 0
            for prediction, weight in zip(model_predictions, pesos):
                groe_ensemble = prediction * weight + groe_ensemble
            if (groe_ensemble.univariate_values() < 0).any():
                indices = test.time_index[groe_ensemble.univariate_values() < 0]
                groe_ensemble = groe_ensemble.update(indices, np.zeros(len(indices)))

            mase_all.append(np.vstack([
                mase_m4(train, test, groe_ensemble, m=m),
            ]))
            smape_all.append(np.vstack([
                smape_m4(test, groe_ensemble),
            ]))
        print("MASE GROE: {:.3f}".format(np.nanmean(np.stack(mase_all), axis=(0, 2))))
        print("sMAPE GROE: {:.3f}".format(np.nanmean(np.stack(smape_all), axis=(0, 2))))
        print("OWA GROE: {:.3f}".format(owa_m4(freq,
                                               np.nanmean(np.stack(smape_all), axis=(0, 2)),
                                               np.nanmean(np.stack(mase_all), axis=(0, 2)))))
        pickle.dump(mase_all, open("groeR_mase_" + freq + ".pkl", "wb"))
        pickle.dump(smape_all, open("groeR_smape_" + freq + ".pkl", "wb"))
