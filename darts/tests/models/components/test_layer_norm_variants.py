import itertools

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
            axis_options = [1, 2, -1, -2]
            input_dim_options = list(x.shape[1:])

            test_runs = itertools.product(
                affine_options, axis_options, input_dim_options
            )

            for run in test_runs:

                affine, axis, input_dim = run

                if (x.shape[axis] is input_dim) and affine:

                    rin = RINorm(axis=axis, input_dim=input_dim, affine=affine)
                    with self.assertRaises(RuntimeError):
                        _ = rin(x, "norm")
                    with self.assertRaises(RuntimeError):
                        _ = rin(x, "denorm")

                    continue

                rin = RINorm(axis=axis, input_dim=input_dim, affine=affine)
                x_norm = rin(x, "norm")
                x_denorm = rin(x_norm, "denorm")
                assert torch.all(torch.isclose(x, x_denorm)).item()
