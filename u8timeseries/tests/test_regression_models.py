import unittest
import pandas as pd
import numpy as np

from ..timeseries import TimeSeries
from ..utils import timeseries_generation as tg
from ..metrics import mape, overall_percentage_error, mase
from u8timeseries import StandardRegressiveModel

def train_test_split(features, target, split_ts):
    """
    Splits all provided TimeSeries instances into train and test sets according to the provided timestamp.

    :param features: Feature TimeSeries instances to be split.
    :param target: Target TimeSeries instance to be split.
    :return: 4-tuple of the form (train_features, train_target, test_features, test_target)
    """

    # split features
    train_features = []
    test_features = []
    for feature in features:
        train_feature, test_feature = feature.split_after(split_ts)
        train_features.append(train_feature)
        test_features.append(test_feature)

    # split target
    train_target, test_target = target.split_after(split_ts)

    return (train_features, train_target, test_features, test_target)


class RegressionModelsTestCase(unittest.TestCase):

    # number of data points used for training
    regression_window = 5

    # dummy feature and target TimeSeries instances
    ts_periodic = tg.sine_timeseries(length=500)
    ts_gaussian = tg.gaussian_timeseries(length=500)
    ts_sum = ts_periodic + ts_gaussian

    # default regression models
    models = [
        StandardRegressiveModel(regression_window)
    ]
    
    def test_models_runnability(self):
        for model in self.models:
            # training and predicting on same features, since only runnability is tested
            model.fit([self.ts_periodic, self.ts_gaussian], self.ts_sum)
            prediction = model.predict([self.ts_periodic, self.ts_gaussian])
            self.assertTrue(len(prediction) == len(self.ts_periodic))

    def test_models_denoising(self):
        # for every model, test whether it correctly denoises ts_sum using ts_gaussian and ts_periodic as inputs
        train_f, train_t, test_f, test_t = train_test_split([self.ts_gaussian, self.ts_sum], self.ts_periodic,
                                                             pd.Timestamp('20010101'))
        max_mape = 1

        for model in self.models:
            model.fit(train_f, train_t)
            prediction = model.predict(test_f)
            self.assertTrue(mape(prediction, test_t) < max_mape, "{} model was not able to denoise data.".format(str(model)))
