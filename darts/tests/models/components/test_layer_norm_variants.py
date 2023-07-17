import numpy as np

from darts.logging import get_logger

logger = get_logger(__name__)

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    logger.warning("Torch not available. Loss tests will be skipped.")
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:
    from darts.models.components.layer_norm_variants import (
        LayerNorm,
        LayerNormNoBias,
        RINorm,
        RMSNorm,
    )
    from darts.tests.base_test_class import DartsBaseTestClass

    class LayerNormVariantsTestCase(DartsBaseTestClass):
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
                x_denorm = rin.inverse(x_norm)
                assert torch.all(torch.isclose(x, x_denorm)).item()

            # try invalid input_dim
            rin = RINorm(input_dim=3, affine=True)
            with self.assertRaises(RuntimeError):
                x_norm = rin(x)
