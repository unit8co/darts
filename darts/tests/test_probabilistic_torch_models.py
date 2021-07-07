import numpy as np

from .base_test_class import DartsBaseTestClass
from ..utils import timeseries_generation as tg
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

    models_cls_kwargs = [
        (RNNModel, {'input_chunk_length': 2, 'training_length': 12, 'n_epochs': 10, 'random_state': 0,
                    'likelihood': GaussianLikelihoodModel()})
    ]

    class ProbabilisticTorchModelsTestCase(DartsBaseTestClass):

        def test_fit_predict_determinism(self):
            for model_cls, model_kwargs in models_cls_kwargs:
                self.helper_test_fit_predict_determinism(model_cls, model_kwargs)

        def helper_test_fit_predict_determinism(self, model_cls, model_kwargs):
            ts = tg.constant_timeseries(length=100, value=10)

            # whether the first predictions of two models initiated with the same random state are the same
            model = model_cls(**model_kwargs)
            model.fit(ts)
            pred1 = model.predict(n=10, num_samples=2).values()

            model = model_cls(**model_kwargs)
            model.fit(ts)
            pred2 = model.predict(n=10, num_samples=2).values()

            self.assertTrue((pred1 == pred2).all())

            # test whether the next prediction of the same model is different
            pred3 = model.predict(n=10, num_samples=2).values()
            self.assertTrue((pred2 != pred3).any())
