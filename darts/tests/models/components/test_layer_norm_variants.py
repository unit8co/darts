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
        RMSNorm,
    )
    from darts.tests.base_test_class import DartsBaseTestClass

    class LayerNormVariantsTestCase(DartsBaseTestClass):
        def test_lnv(self):
            for layer_norm in [RMSNorm, LayerNorm, LayerNormNoBias]:
                ln = layer_norm(4)
                inputs = torch.zeros(1, 4, 4)
                ln(inputs)
