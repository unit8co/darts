"""Testing ensembling models on M4 timeseries

"""

from darts import TimeSeries, ModelMode, SeasonalityMode
from darts.models import NaiveSeasonal, ExponentialSmoothing, Theta, FourTheta, LinearRegressionModel
from darts.models.forecasting_model import ForecastingModel
from darts.utils.statistics import check_seasonality, remove_from_series, extract_trend_and_seasonality
from darts.utils.timeseries_generation import constant_timeseries
from darts.utils import _build_tqdm_iterator

import numpy as np
import pandas as pd
import pickle as pkl
from sklearn.linear_model import LassoCV

from M4_metrics import owa_m4, smape_m4, mase_m4

info_dataset = pd.read_csv('dataset/M4-info.csv', delimiter=',').set_index('M4id')


def naive2_groe(ts: TimeSeries, n: int, m: int):
    """
    Return the prediction of the naive2 baseline
    """
    # It will be better to use R functions
    ts_des = ts
    seasonOut = 1
    if m > 1:
        if check_seasonality(ts, m=int(m), max_lag=2 * m):
            _, season = extract_trend_and_seasonality(ts, m, model=ModelMode.MULTIPLICATIVE)
            ts_des = remove_from_series(ts, season, model=ModelMode.MULTIPLICATIVE)
            seasonOut = season[-m:].shift(m)
            seasonOut = seasonOut.append_values(seasonOut.values())[:n]
    naive2 = NaiveSeasonal(K=1)

    naive2.fit(ts_des)
    return naive2.predict(n) * seasonOut


def groe_owa(ts: TimeSeries, model: ForecastingModel, n1: int, m: int, p: int, fq: int):
    """
    Backtesting.
    Compute the OWA score iteratively on ´p´ timepoints following an expanding window mode.

    Parameters
    ---------
    ts
        TimeSeries on which backtesting will be done.
    model
        model to backtest.
    n1
        minimum number of datapoints to take during training
    m
        step size
    p
        number of steps to iterate over
    fq
        Frequency of the time series.
    Returns
    -------
    Sum of all OWA errors
    """
    n = len(ts)
    errors = []
    for i in range(p):
        if n1 + i * m == n:
            break
        ni = n1 + i * m
        npred = n - ni
        train = ts[:ni]
        if npred >= 3:
            test = ts[ni:]
        else:
            test = TimeSeries(ts.pd_series()[ni:], freq=ts.freq_str)

        forecast_naive2 = naive2_groe(train, npred, fq)
        error_sape_n2 = mase_m4(train, test, forecast_naive2)
        error_ase_n2 = smape_m4(test, forecast_naive2)

        model.fit(train)
        forecast = model.predict(npred)
        try:
            error_sape = mase_m4(train, test, forecast)
            error_ase = smape_m4(test, forecast)
            owa = 0.5 * (error_sape / error_sape_n2) + 0.5 * (error_ase / error_ase_n2)
            errors.append(np.sum(owa))
        except (ZeroDivisionError, ValueError):
            errors.append(0)
    errors = np.sum(errors)
    return errors


class DeseasonForecastingModel(ForecastingModel):
    """
    Wrapper class around ForecastingModel to perform deseasonalization directly,
    and reseason the prediction.
    """
    def __init__(self, model: ForecastingModel, m: int):
        super().__init__()
        self.model = model
        self.m = m

    def fit(self, train: TimeSeries):
        super().fit(train)
        train_des = train
        self.seasonOut = 1
        if self.m > 1:
            if check_seasonality(train, m=self.m, max_lag=2 * self.m):
                _, season = extract_trend_and_seasonality(train, self.m, model=ModelMode.MULTIPLICATIVE)
                train_des = remove_from_series(train, season, model=ModelMode.MULTIPLICATIVE)
                seasonOut = season[-self.m:].shift(self.m)
                self.seasonOut = seasonOut.append_values(seasonOut.values())
        self.model.fit(train_des)

    def predict(self, n: int):
        super().predict(n)
        pred = self.model.predict(n)
        if isinstance(self.seasonOut, int):
            return pred
        else:
            return pred * self.seasonOut[:n]


if __name__ == "__main__":
    data_frequencies = ['Yearly', 'Quarterly', 'Monthly', 'Weekly', 'Daily', 'Hourly']

    for freq in data_frequencies[::-1]:
        # Load TimeSeries from M4
        ts_train = pkl.load(open("dataset/train_" + freq + ".pkl", "rb"))
        ts_test = pkl.load(open("dataset/test_" + freq + ".pkl", "rb"))

        mase_all = []
        smape_all = []
        m = int(info_dataset.Frequency[freq[0] + '1'])
        for train, test in _build_tqdm_iterator(zip(ts_train, ts_test), verbose=True):
            # remove seasonality
            train_des = train
            seasonOut = 1
            season = constant_timeseries(length=len(train), freq=train.freq_str, start_ts=train.start_time())
            if m > 1:
                if check_seasonality(train, m=m, max_lag=2 * m):
                    pass
                    _, season = extract_trend_and_seasonality(train, m, model=ModelMode.MULTIPLICATIVE)
                    train_des = remove_from_series(train, season, model=ModelMode.MULTIPLICATIVE)
                    seasonOut = season[-m:].shift(m)
                    seasonOut = seasonOut.append_values(seasonOut.values())[:len(test)]
            # model selection
            naiveSeason = NaiveSeasonal(K=m)
            naive2 = NaiveSeasonal(K=1)
            ses = ExponentialSmoothing(trend=None, seasonal=None, seasonal_periods=m)
            holt = ExponentialSmoothing(seasonal=None, damped=False, trend='additive', seasonal_periods=m)
            damp = ExponentialSmoothing(seasonal=None, damped=True, trend='additive', seasonal_periods=m)

            fourtheta = FourTheta.select_best_model(train, [1, 2, 3], m)
            theta = Theta(theta=0, season_mode=SeasonalityMode.MULTIPLICATIVE, seasonality_period=m)

            models_simple = [naiveSeason, theta, fourtheta]
            models_des = [naive2, ses, holt, damp]

            # Linear Regression (with constraints)

            def train_pred(id_start=None, id_end=None):
                for m in models_simple:
                    m.fit(train[id_start:id_end])
                for m in models_des:
                    m.fit(train_des[id_start:id_end])
                models_simple_predictions = [m.predict(len(test))
                                             for m in models_simple]
                id_fin = id_end + len(test)
                if id_fin == 0:
                    id_fin = None
                models_des_predictions = [m.predict(len(test)) *
                                          (seasonOut if id_end is None else season[id_end:id_fin])
                                          for m in models_des]

                model_predictions = models_simple_predictions + models_des_predictions

                return model_predictions

            val_predictions = train_pred(id_end=-len(test))
            target_val = train.slice_intersect(val_predictions[0])

            regr_model = LinearRegressionModel(train_n_points=len(test),
                                                 model=LassoCV(positive=True, fit_intercept=False))
            regr_model.fit(val_predictions, target_val)

            for mod in models_simple:
                mod.fit(train)
            for mod in models_des:
                mod.fit(train_des)

            models_simple_predictions = [mod.predict(len(test))
                                         for mod in models_simple]
            models_des_predictions = [mod.predict(len(test)) * seasonOut
                                      for mod in models_des]

            model_predictions = models_simple_predictions + models_des_predictions

            # constraint sum equal to 1
            regr_model.model.coef_ = regr_model.model.coef_ / np.sum(regr_model.model.coef_)

            ensemble_pred = regr_model.predict(model_predictions)

            # Mean ensembling
            mean_pred = 0
            for pred in model_predictions:
                mean_pred = pred + mean_pred
            mean_pred = mean_pred / len(model_predictions)

            # GROE OWA (weight based on score)
            fq = info_dataset.Frequency[freq[0] + '1']
            criterion = [
                groe_owa(train, naiveSeason, max(5, len(train) - len(test)), int(np.floor(len(test) / 6)), 6, fq),
                groe_owa(train, theta, max(5, len(train) - len(test)), int(np.floor(len(test) / 6)), 6, fq),
                groe_owa(train, fourtheta, max(5, len(train) - len(test)), int(np.floor(len(test) / 6)), 6, fq),
                groe_owa(train, DeseasonForecastingModel(NaiveSeasonal(K=1), m), max(5, len(train) - len(test)),
                         int(np.floor(len(test) / 6)), 6, fq),
                groe_owa(train, DeseasonForecastingModel(ses, m), max(5, len(train) - len(test)),
                         int(np.floor(len(test) / 6)), 6, fq),
                groe_owa(train, DeseasonForecastingModel(holt, m), max(5, len(train) - len(test)),
                         int(np.floor(len(test) / 6)), 6, fq),
                groe_owa(train, DeseasonForecastingModel(damp, m), max(5, len(train) - len(test)),
                         int(np.floor(len(test) / 6)), 6, fq)]

            Score = 1 / np.array(criterion)
            pesos = Score / Score.sum()
            groe_ensemble = 0
            for prediction, weight in zip(model_predictions, pesos):
                groe_ensemble = prediction * weight + groe_ensemble

            # BO3 ensembling
            score = np.argsort(Score)[::-1][:3]
            pesos2 = Score[score] / Score[score].sum()

            bo3_ensemble = 0
            bo3_mean = 0
            for i, model in enumerate(score):
                bo3_ensemble = model_predictions[model] * pesos2[i] + bo3_ensemble
                bo3_mean = model_predictions[model] / len(score) + bo3_mean

            # compute score
            m = info_dataset.Frequency[freq[0] + '1']
            mase_all.append(np.vstack([
                mase_m4(train, test, models_des_predictions[0], m=m),
                mase_m4(train, test, ensemble_pred, m=m),
                mase_m4(train, test, mean_pred, m=m),
                mase_m4(train, test, groe_ensemble, m=m),
                mase_m4(train, test, bo3_ensemble, m=m),
                mase_m4(train, test, bo3_mean, m=m),
            ]))
            smape_all.append(np.vstack([
                smape_m4(test, models_des_predictions[0]),
                smape_m4(test, ensemble_pred),
                smape_m4(test, mean_pred),
                smape_m4(test, groe_ensemble),
                smape_m4(test, bo3_ensemble),
                smape_m4(test, bo3_mean),
            ]))
        pkl.dump(mase_all, open("ensembling_mase_"+freq+".pkl", "wb"))
        pkl.dump(smape_all, open("ensembling_smape_"+freq+".pkl", "wb"))
        print("MASE; Naive2: {:.3f}, Linear Regression: {:.3f}, Mean ensembling: {:.3f}, GROE ensembling: {:.3f}, "
              "BO3 ensembling: {:.3f}, BO3 Mean: {:.3f}".format(*tuple(np.nanmean(np.stack(mase_all), axis=(0, 2)))))
        print("sMAPE; Naive2: {:.3f}, Linear Regression: {:.3f}, Mean ensembling: {:.3f}, GROE ensembling: {:.3f}, "
              "BO3 ensembling: {:.3f}, BO3 Mean: {:.3f}".format(*tuple(np.nanmean(np.stack(smape_all), axis=(0, 2)))))
        print("OWA: ", owa_m4(freq,
                              np.nanmean(np.stack(smape_all), axis=(0, 2)),
                              np.nanmean(np.stack(mase_all), axis=(0, 2))))
