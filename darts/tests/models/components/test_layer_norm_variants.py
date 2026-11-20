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

    @pytest.mark.parametrize("bias", [0.0, 0.5])
    @pytest.mark.parametrize(
        "weight",
        # includes weights around `0` and within `eps ** 2` (1e-10 for the default `eps=1e-5`), as well as negatives
        [2.0, 1.0, 1e-3, 1e-10, 5e-11, 0.0, -5e-11, -1e-10, -1e-3, -2.0],
    )
    def test_rin_affine_inverse_is_exact(self, weight, bias):
        """`inverse()` must undo the normalization for any affine weight (including zero / negative values)."""
        torch.manual_seed(42)
        x = torch.randn(3, 4, 2, dtype=torch.float64)

        rin = RINorm(input_dim=2, affine=True).double()
        with torch.no_grad():
            rin.affine_weight.fill_(weight)
            rin.affine_bias.fill_(bias)

        x_norm = rin(x)
        # expand dims to simulate probabilistic forecasting
        x_denorm = rin.inverse(x_norm.view(x_norm.shape + (1,))).squeeze(-1)

        # with a tiny weight (<= `eps ** 2`), `bias` dominates `x_norm` and limits the achievable precision
        torch.testing.assert_close(x_denorm, x, rtol=1e-4, atol=1e-4)

    def test_rin_affine_matches_plain_formula_for_regular_weights(self):
        """Weights with a magnitude >= `eps ** 2` must not be altered by the clipping (same as before)."""
        torch.manual_seed(42)
        x = torch.randn(3, 5, 3)

        rin = RINorm(input_dim=3, affine=True)
        with torch.no_grad():
            rin.affine_weight.copy_(torch.tensor([0.5, 1.0, -2.0]))
            rin.affine_bias.copy_(torch.tensor([0.1, -0.2, 0.3]))

        x_norm = rin(x)
        expected = (x - rin.mean) / rin.stdev * rin.affine_weight + rin.affine_bias
        torch.testing.assert_close(x_norm, expected)

    def test_rin_affine_zero_weight_has_finite_gradients(self):
        """A weight of zero must neither produce NaNs / infs in the output nor in the gradients."""
        torch.manual_seed(42)
        x = torch.randn(3, 4, 2)

        rin = RINorm(input_dim=2, affine=True)
        with torch.no_grad():
            rin.affine_weight.zero_()

        x_norm = rin(x)
        x_denorm = rin.inverse(x_norm.view(x_norm.shape + (1,)))
        assert torch.isfinite(x_denorm).all()

        x_denorm.sum().backward()
        assert torch.isfinite(rin.affine_weight.grad).all()
        assert torch.isfinite(rin.affine_bias.grad).all()

    def test_rin_affine_weight_gradient_flows_through_clip(self):
        """A weight in the clipped dead zone must still receive a non-zero gradient."""
        rin = RINorm(input_dim=2, affine=True)
        with torch.no_grad():
            rin.affine_weight.fill_(5e-11)

        x = torch.randn(4, 10, 2)
        rin.forward(x)  # populate mean/stdev
        out = rin.transform(x)
        out.sum().backward()

        assert rin.affine_weight.grad is not None
        assert torch.all(rin.affine_weight.grad.abs() > 0), (
            f"gradient collapsed to zero: {rin.affine_weight.grad}"
        )
