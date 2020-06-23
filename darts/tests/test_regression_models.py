import unittest
import logging

import numpy as np
import pandas as pd

from ..utils import timeseries_generation as tg
from ..metrics import r2_score
from ..models import StandardRegressionModel


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


def test_models_accuracy(test_case, models, features, target, min_r2):
    # for every model, test whether it predicts the target with a minimum r2 score of `min_r2`
    train_f, train_t, test_f, test_t = train_test_split(features, target, pd.Timestamp('20010101'))

    for model in models:
        model.fit(train_f, train_t)
        prediction = model.predict(test_f)
        current_r2 = r2_score(prediction, test_t)
        test_case.assertTrue(current_r2 >= min_r2, "{} model was not able to denoise data."
                             "A r2 score of {} was recorded.".format(str(model), current_r2))


class RegressionModelsTestCase(unittest.TestCase):

    np.random.seed(1)

    # number of data points used for training
    regression_window = 5

    # dummy feature and target TimeSeries instances
    ts_periodic = tg.sine_timeseries(length=500)
    ts_gaussian = tg.gaussian_timeseries(length=500)
    ts_random_walk = tg.random_walk_timeseries(length=500)
    ts_sum = ts_periodic + ts_gaussian
    ts_random_multi = ts_gaussian.stack(ts_random_walk)
    ts_sum_2 = ts_sum + ts_random_walk
    ts_sum_multi = ts_sum.stack(ts_sum_2)

    # default regression models
    models = [
        StandardRegressionModel(regression_window)
    ]

    @classmethod
    def setUpClass(cls):
        logging.disable(logging.CRITICAL)

    def test_models_runnability(self):
        for model in self.models:
            # training and predicting on same features, since only runnability is tested
            model.fit([self.ts_periodic, self.ts_gaussian], self.ts_sum)
            prediction = model.predict([self.ts_periodic, self.ts_gaussian])
            self.assertTrue(len(prediction) == len(self.ts_periodic))

    def test_models_denoising(self):
        # for every model, test whether it correctly denoises ts_sum using ts_gaussian and ts_sum as inputs
        test_models_accuracy(self, self.models, [self.ts_gaussian, self.ts_sum], self.ts_periodic, 1.0)

    def test_models_denoising_multi_input(self):
        # for every model, test whether it correctly denoises ts_sum_2 using ts_random_multi and ts_sum_2 as inputs
        test_models_accuracy(self, self.models, [self.ts_random_multi, self.ts_sum_2], self.ts_periodic, 1.0)

    def test_models_denoising_multi_target(self):
        # for every model, test whether it correctly denoises ts_sum_multi using ts_random_multi and ts_sum_2 as inputs
        test_models_accuracy(self, self.models, [self.ts_random_multi, self.ts_sum_2], self.ts_sum_multi, 1.0)

    def test_wrong_dimensionality(self):
        train_f, train_t, _, _ = train_test_split([self.ts_periodic, self.ts_sum_multi],
                                                  self.ts_sum, pd.Timestamp('20010101'))
        self.models[0].fit(train_f, train_t)
        _, _, test_f, _ = train_test_split([self.ts_sum_multi, self.ts_periodic],
                                           self.ts_sum, pd.Timestamp('20010101'))

        with self.assertRaises(ValueError):
            self.models[0].predict(test_f)
