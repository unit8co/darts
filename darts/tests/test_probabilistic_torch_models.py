import numpy as np

from .base_test_class import DartsBaseTestClass
from ..utils import timeseries_generation as tg
from ..metrics import mae
from ..logging import get_logger

logger = get_logger(__name__)

try:
    from ..models.rnn_model import RNNModel
    from darts.utils.likelihood_models import GaussianLikelihoodModel
    TORCH_AVAILABLE = True
except ImportError:
    logger.warning('Torch not available. TCN tests will be skipped.')
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:

    models_cls_kwargs_errs = [
        (RNNModel, {'input_chunk_length': 2, 'training_length': 10, 'n_epochs': 10, 'random_state': 0,
                    'likelihood': GaussianLikelihoodModel()}, 1.9)
    ]

    class ProbabilisticTorchModelsTestCase(DartsBaseTestClass):

        np.random.seed(0)
        linear_ts = tg.constant_timeseries(length=100)
        linear_noisy_ts = linear_ts + tg.gaussian_timeseries(length=100, std=0.5)
        linear_multivar_ts = linear_ts.stack(linear_ts)
        linear_noisy_multivar_ts = linear_noisy_ts.stack(linear_noisy_ts)
        num_samples = 5

        def test_fit_predict_determinism(self):

            for model_cls, model_kwargs, _ in models_cls_kwargs_errs:

                # whether the first predictions of two models initiated with the same random state are the same
                model = model_cls(**model_kwargs)
                model.fit(self.linear_ts)
                pred1 = model.predict(n=10, num_samples=2).values()

                model = model_cls(**model_kwargs)
                model.fit(self.linear_ts)
                pred2 = model.predict(n=10, num_samples=2).values()

                self.assertTrue((pred1 == pred2).all())

                # test whether the next prediction of the same model is different
                pred3 = model.predict(n=10, num_samples=2).values()
                self.assertTrue((pred2 != pred3).any())

        def test_probabilistic_forecast_accuracy(self):
            for model_cls, model_kwargs, err in models_cls_kwargs_errs:
                self.helper_test_probabilistic_forecast_accuracy(model_cls, model_kwargs, err,
                                                                 self.linear_ts, self.linear_noisy_ts)
                self.helper_test_probabilistic_forecast_accuracy(model_cls, model_kwargs, err,
                                                                 self.linear_multivar_ts,
                                                                 self.linear_noisy_multivar_ts)

        def helper_test_probabilistic_forecast_accuracy(self, model_cls, model_kwargs, err, ts, noisy_ts):
            model = model_cls(**model_kwargs)
            model.fit(noisy_ts[:80], epochs=50)
            pred = model.predict(n=20, num_samples=100)

            # test median accuracy noiseless ts
            mae_err_median = mae(ts[20:], pred)
            self.assertLess(mae_err_median, err)

            # test accuracy for increasing quantiles between 0.7 and 1 (it should decrease)
            tested_quantiles = [0.7, 0.8, 0.9, 0.99]
            mae_err = mae_err_median
            for quantile in tested_quantiles:
                new_mae = mae(ts[20:], pred.quantile_timeseries(quantile=quantile))
                self.assertLess(mae_err, new_mae)
                mae_err = new_mae

            # test accuracy for decreasing quantiles between 0.3 and 0 (it should decrease)
            tested_quantiles = [0.3, 0.2, 0.1, 0.01]
            mae_err = mae_err_median
            for quantile in tested_quantiles:
                new_mae = mae(ts[20:], pred.quantile_timeseries(quantile=quantile))
                self.assertLess(mae_err, new_mae)
                mae_err = new_mae
