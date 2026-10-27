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
from darts.utils.likelihood_models.torch import (
    MIN_CAUCHY_GAMMA_SAMPLING,
    CauchyLikelihood,
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

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("affine", [False, True])
    def test_rin_inverse_scale_extremes(self, dtype, affine):
        rin = RINorm(input_dim=1, affine=affine).to(dtype=dtype)
        if affine:
            with torch.no_grad():
                rin.affine_weight.fill_(-2.5)
                rin.affine_bias.fill_(1000.0)
        rin(torch.tensor([[[10.0], [30.0]]], dtype=dtype))
        raw = (
            torch.tensor([-1000.0, -100.0, -2.0, 0.0, 25.0, 1000.0], dtype=dtype)
            .reshape(1, 6, 1, 1)
            .requires_grad_()
        )
        output = rin.inverse_scale(raw)
        factor = rin.stdev.unsqueeze(-1)
        if affine:
            factor = factor / (rin.affine_weight + rin.eps**2).abs().unsqueeze(-1)
        expected = (torch.nn.functional.softplus(raw) * factor).clamp_min(
            torch.finfo(dtype).tiny
        )
        assert torch.isfinite(output).all()
        torch.testing.assert_close(
            torch.nn.functional.softplus(output), expected, atol=0, rtol=1e-5
        )
        # Cauchy's existing gamma floor still applies after the raw-space inverse.
        _, gamma = CauchyLikelihood()._params_from_output(
            torch.cat([torch.zeros_like(output), output], dim=-1)
        )
        torch.testing.assert_close(
            gamma,
            expected.squeeze(-1).clamp_min(MIN_CAUCHY_GAMMA_SAMPLING),
            atol=0,
            rtol=1e-5,
        )
        output.sum().backward()
        assert torch.isfinite(raw.grad).all()
        if affine:
            assert torch.isfinite(rin.affine_weight.grad).all()
