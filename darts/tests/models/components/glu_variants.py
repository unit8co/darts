import pytest

from darts.logging import get_logger

logger = get_logger(__name__)

try:
    import torch

    from darts.models.components import glu_variants
    from darts.models.components.glu_variants import GLU_FFN
except ImportError:
    pytest.skip(
        f"Torch not available. {__name__} tests will be skipped.",
        allow_module_level=True,
    )


class TestFFN:
    def test_ffn(self):
        for FeedForward_network in GLU_FFN:
            self.feed_forward_block = getattr(glu_variants, FeedForward_network)(d_model=4, d_ff=16, dropout=0.1)

            inputs = torch.zeros(1, 4, 4)
            self.feed_forward_block(x=inputs)
