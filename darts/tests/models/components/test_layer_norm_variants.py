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

    def test_rin_requires_at_least_one_group(self):
        with pytest.raises(ValueError):
            RINorm(input_dim=0)

    def test_rin_groups(self):
        np.random.seed(42)
        torch.manual_seed(42)

        series = torch.randn(3, 4, 7)
        past_cov = torch.randn(3, 4, 2)
        future_cov = torch.randn(3, 4, 5)

        for affine in [True, False]:
            rin = RINorm(input_dim=7, past_cov_dim=2, future_cov_dim=5, affine=affine)
            assert rin.has_series and rin.has_past_cov and rin.has_future_cov

            series_out, past_cov_out, future_cov_out = rin(series, past_cov, future_cov)

            series_denorm, past_cov_denorm, future_cov_denorm = rin.inverse(
                x=series_out.view(series_out.shape + (1,)),
                past_cov=past_cov_out.view(past_cov_out.shape + (1,)),
                future_cov=future_cov_out.view(future_cov_out.shape + (1,)),
            )
            series_denorm, past_cov_denorm, future_cov_denorm = (
                series_denorm.squeeze(-1),
                past_cov_denorm.squeeze(-1),
                future_cov_denorm.squeeze(-1),
            )

            assert torch.all(torch.isclose(series, series_denorm)).item()
            assert torch.all(torch.isclose(past_cov, past_cov_denorm)).item()
            assert torch.all(torch.isclose(future_cov, future_cov_denorm)).item()

            # `transform` reuses stats stored by the last `forward()` call for that group,
            # without recomputing them
            future_cov_2 = torch.randn(3, 2, 5)
            _, _, future_cov_2_norm = rin.transform(future_cov=future_cov_2)
            _, _, future_cov_2_denorm = rin.inverse(
                future_cov=future_cov_2_norm.view(future_cov_2_norm.shape + (1,)),
            )
            future_cov_2_denorm = future_cov_2_denorm.squeeze(-1)
            assert torch.all(torch.isclose(future_cov_2, future_cov_2_denorm)).item()

    def test_rin_partial_groups(self):
        # only past covariates active: series/future covariates must be `None`
        rin = RINorm(input_dim=0, past_cov_dim=2)
        assert not rin.has_series
        assert rin.has_past_cov
        assert not rin.has_future_cov

        past_cov = torch.randn(3, 4, 2)
        series_out, past_cov_out, future_cov_out = rin(past_cov=past_cov)
        assert series_out is None
        assert future_cov_out is None
        assert past_cov_out is not None

    def test_rin_parse_config(self):
        # disabled
        assert RINorm.parse_config(False) is None
        assert RINorm.parse_config(None) is None

        # `True` -> series only, default params
        assert RINorm.parse_config(True) == {
            "params": {},
            "series": True,
            "past_covariates": False,
            "future_covariates": False,
        }

        # legacy flat dict -> series only, with given params
        assert RINorm.parse_config({"affine": False}) == {
            "params": {"affine": False},
            "series": True,
            "past_covariates": False,
            "future_covariates": False,
        }
        assert RINorm.parse_config({"eps": 1e-3, "affine": True}) == {
            "params": {"eps": 1e-3, "affine": True},
            "series": True,
            "past_covariates": False,
            "future_covariates": False,
        }

        # new style dict, mixing bool and list[str], with missing keys defaulting to `False`
        parsed = RINorm.parse_config({
            "params": {"affine": False},
            "series": False,
            "past_covariates": ["a", "b"],
        })
        assert parsed == {
            "params": {"affine": False},
            "series": False,
            "past_covariates": ["a", "b"],
            "future_covariates": False,
        }

        # idempotency: re-parsing an already normalized dict yields an equivalent dict
        assert RINorm.parse_config(parsed) == parsed

        # invalid types/keys raise `ValueError`
        with pytest.raises(ValueError):
            RINorm.parse_config(42)
        with pytest.raises(ValueError):
            RINorm.parse_config({"not_a_valid_key": True})
        with pytest.raises(ValueError):
            RINorm.parse_config({"params": {"not_a_valid_param": True}, "series": True})
        with pytest.raises(ValueError):
            RINorm.parse_config({"series": 1})
        with pytest.raises(ValueError):
            # all groups disabled
            RINorm.parse_config({"series": False, "past_covariates": False})
