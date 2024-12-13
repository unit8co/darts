import numpy as np
import pytest

from darts.tests.conftest import TORCH_AVAILABLE

if not TORCH_AVAILABLE:
    pytest.skip(
        f"Torch not available. {__name__} tests will be skipped.",
        allow_module_level=True,
    )
import torch

from darts.models.components.layer_norm_variants import (
    LayerNorm,
    LayerNormNoBias,
    RINorm,
    RMSNorm,
)


class TestLayerNormVariants:
    def test_lnv(self):
        for layer_norm in [RMSNorm, LayerNorm, LayerNormNoBias]:
            ln = layer_norm(4)
            inputs = torch.zeros(1, 4, 4)
            ln(inputs)

    def test_rin(self):
        np.random.seed(42)
        torch.manual_seed(42)

        x = torch.randn(3, 4, 7)
        affine_options = [True, False]

        # test with and without affine and correct input dim
        for affine in affine_options:
            rin = RINorm(input_dim=7, affine=affine)
            x_norm = rin(x)

            # expand dims to simulate probabilistic forecasting
            x_denorm = rin.inverse(x_norm.view(x_norm.shape + (1,))).squeeze(-1)
            assert torch.all(torch.isclose(x, x_denorm)).item()

        # try invalid input_dim
        rin = RINorm(input_dim=3, affine=True)
        with pytest.raises(RuntimeError):
            x_norm = rin(x)
