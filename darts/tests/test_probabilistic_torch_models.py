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
        (RNNModel, {'input_chunk_length': 2, 'training_length': 12, 'n_epochs': 10, 'random_state': 0,
                    'likelihood': GaussianLikelihoodModel()}, 0.41)
    ]

    class ProbabilisticTorchModelsTestCase(DartsBaseTestClass):

        sine_ts = tg.sine_timeseries(length=100)
        sine_multivar_ts = sine_ts.stack(sine_ts)
        num_samples = 5

        def test_fit_predict_determinism(self):
            for model_cls, model_kwargs, _ in models_cls_kwargs_errs:
                self.helper_test_fit_predict_determinism(model_cls, model_kwargs)

        def helper_test_fit_predict_determinism(self, model_cls, model_kwargs):

            # whether the first predictions of two models initiated with the same random state are the same
            model = model_cls(**model_kwargs)
            model.fit(self.sine_ts)
            pred1 = model.predict(n=10, num_samples=2).values()

            model = model_cls(**model_kwargs)
            model.fit(self.sine_ts)
            pred2 = model.predict(n=10, num_samples=2).values()

            self.assertTrue((pred1 == pred2).all())

            # test whether the next prediction of the same model is different
            pred3 = model.predict(n=10, num_samples=2).values()
            self.assertTrue((pred2 != pred3).any())

        def test_probabilistic_forecast_accuracy(self):
            for model_cls, model_kwargs, err in models_cls_kwargs_errs:
                self.help_test_probabilistic_forecast_accuracy(model_cls, model_kwargs, err, self.sine_ts)
                self.help_test_probabilistic_forecast_accuracy(model_cls, model_kwargs, err, self.sine_multivar_ts)

        def help_test_probabilistic_forecast_accuracy(self, model_cls, model_kwargs, err, ts):
            model = model_cls(**model_kwargs)
            model.fit(ts[:80], epochs=30)
            pred = model.predict(n=20, num_samples=self.num_samples)
            mape_err = mae(ts[20:], pred)
            self.assertTrue(mape_err < err, 'Model {} produces errors too high (one time '
                                            'series). Error = {}'.format(model_cls, mape_err))
