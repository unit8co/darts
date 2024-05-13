import pytest
from numpy.random import RandomState

from darts.tests.conftest import TORCH_AVAILABLE

if not TORCH_AVAILABLE:
    pytest.skip(
        f"Torch not available. {__name__} tests will be skipped.",
        allow_module_level=True,
    )
import torch

from darts.utils.torch import random_method


# use a simple torch model mock
class TorchModelMock:
    @random_method
    def __init__(self, some_params=None, **kwargs):
        self.model = torch.randn(5)
        # super().__init__()

    @random_method
    def fit(self, some_params=None):
        self.fit_value = torch.randn(5)


class TestRandomMethod:
    def test_it_raises_error_if_used_on_function(self):
        with pytest.raises(ValueError):

            @random_method
            def a_random_function():
                pass

    def test_model_is_random_by_default(self):
        model1 = TorchModelMock()
        model2 = TorchModelMock()
        assert not torch.equal(model1.model, model2.model)

    def test_model_is_random_when_None_random_state_specified(self):
        model1 = TorchModelMock(random_state=None)
        model2 = TorchModelMock(random_state=None)
        assert not torch.equal(model1.model, model2.model)

    def helper_test_reproducibility(self, model1, model2):
        assert torch.equal(model1.model, model2.model)

        model1.fit()
        model2.fit()
        assert torch.equal(model1.fit_value, model2.fit_value)

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
        assert not torch.equal(model1.model, model2.model)

    def test_model_is_different_for_different_random_instance(self):
        model1 = TorchModelMock(random_state=RandomState(42))
        model2 = TorchModelMock(random_state=RandomState(43))
        assert not torch.equal(model1.model, model2.model)

    def helper_test_successive_call_are_different(self, model):
        # different between init and fit
        model.fit()
        assert not torch.equal(model.model, model.fit_value)

        # different between 2 fit
        old_fit_value = model.fit_value.clone()
        model.fit()
        assert not torch.equal(model.fit_value, old_fit_value)

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

        assert torch.equal(model.fit_value, fit_value)

    def test_no_side_effect_between_rng_with_random_instance(self):
        model = TorchModelMock(random_state=RandomState(42))
        model.fit()
        fit_value = model.fit_value.clone()

        model = TorchModelMock(random_state=RandomState(42))
        model2 = TorchModelMock(random_state=RandomState(42))
        model2.fit()
        model.fit()

        assert torch.equal(model.fit_value, fit_value)
