from numpy.random import RandomState

from darts.logging import get_logger
from darts.tests.base_test_class import DartsBaseTestClass

logger = get_logger(__name__)

try:
    import torch

    from darts.utils.torch import random_method

    TORCH_AVAILABLE = True
except ImportError:
    logger.warning("Torch not available. Torch utils will not be tested.")
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:
    # use a simple torch model mock
    class TorchModelMock:
        @random_method
        def __init__(self, some_params=None, **kwargs):
            self.model = torch.randn(5)
            # super().__init__()

        @random_method
        def fit(self, some_params=None):
            self.fit_value = torch.randn(5)

    class RandomMethodTestCase(DartsBaseTestClass):
        def test_it_raises_error_if_used_on_function(self):
            with self.assertRaises(ValueError):

                @random_method
                def a_random_function():
                    pass

        def test_model_is_random_by_default(self):
            model1 = TorchModelMock()
            model2 = TorchModelMock()
            self.assertFalse(torch.equal(model1.model, model2.model))

        def test_model_is_random_when_None_random_state_specified(self):
            model1 = TorchModelMock(random_state=None)
            model2 = TorchModelMock(random_state=None)
            self.assertFalse(torch.equal(model1.model, model2.model))

        def helper_test_reproducibility(self, model1, model2):
            self.assertTrue(torch.equal(model1.model, model2.model))

            model1.fit()
            model2.fit()
            self.assertTrue(torch.equal(model1.fit_value, model2.fit_value))

        def test_model_is_reproducible_when_seed_specified(self):
            model1 = TorchModelMock(random_state=42)
            model2 = TorchModelMock(random_state=42)
            self.helper_test_reproducibility(model1, model2)

        def test_model_is_reproducible_when_random_instance_specified(self):
            model1 = TorchModelMock(random_state=RandomState(42))
            model2 = TorchModelMock(random_state=RandomState(42))
            self.helper_test_reproducibility(model1, model2)

        def test_model_is_different_for_different_seeds(self):
            model1 = TorchModelMock(random_state=42)
            model2 = TorchModelMock(random_state=43)
            self.assertFalse(torch.equal(model1.model, model2.model))

        def test_model_is_different_for_different_random_instance(self):
            model1 = TorchModelMock(random_state=RandomState(42))
            model2 = TorchModelMock(random_state=RandomState(43))
            self.assertFalse(torch.equal(model1.model, model2.model))

        def helper_test_successive_call_are_different(self, model):
            # different between init and fit
            model.fit()
            self.assertFalse(torch.equal(model.model, model.fit_value))

            # different between 2 fit
            old_fit_value = model.fit_value.clone()
            model.fit()
            self.assertFalse(torch.equal(model.fit_value, old_fit_value))

        def test_successive_call_to_rng_are_different_when_seed_specified(self):
            model = TorchModelMock(random_state=42)
            self.helper_test_successive_call_are_different(model)

        def test_successive_call_to_rng_are_different_when_random_instance_specified(
            self,
        ):
            model = TorchModelMock(random_state=RandomState(42))
            self.helper_test_successive_call_are_different(model)

        def test_no_side_effect_between_rng_with_seeds(self):
            model = TorchModelMock(random_state=42)
            model.fit()
            fit_value = model.fit_value.clone()

            model = TorchModelMock(random_state=42)
            model2 = TorchModelMock(random_state=42)
            model2.fit()
            model.fit()

            self.assertTrue(torch.equal(model.fit_value, fit_value))

        def test_no_side_effect_between_rng_with_random_instance(self):
            model = TorchModelMock(random_state=RandomState(42))
            model.fit()
            fit_value = model.fit_value.clone()

            model = TorchModelMock(random_state=RandomState(42))
            model2 = TorchModelMock(random_state=RandomState(42))
            model2.fit()
            model.fit()

            self.assertTrue(torch.equal(model.fit_value, fit_value))
