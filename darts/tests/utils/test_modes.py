import pytest

from darts.utils._modes import ADDITIVE, MULTIPLICATIVE, validate_mode


class TestModes:
    def test_validate_mode_accepts_allowed_value(self):
        assert (
            validate_mode(
                ADDITIVE,
                param="model",
                allowed={ADDITIVE, MULTIPLICATIVE},
            )
            is ADDITIVE
        )

    def test_validate_mode_rejects_wrong_case(self):
        with pytest.raises(ValueError, match="Unknown value for `model`"):
            validate_mode(
                "ADDITIVE",
                param="model",
                allowed={ADDITIVE, MULTIPLICATIVE},
            )

    def test_validate_mode_none_in_allowed(self):
        assert (
            validate_mode(None, param="trend", allowed={ADDITIVE, MULTIPLICATIVE, None})
            is None
        )

    def test_validate_mode_none_not_allowed(self):
        with pytest.raises(ValueError, match="Unknown value for `model`: None"):
            validate_mode(None, param="model", allowed={ADDITIVE, MULTIPLICATIVE})

    def test_validate_mode_invalid(self):
        with pytest.raises(ValueError, match="Unknown value for `trend_mode`"):
            validate_mode(
                "quadratic",
                param="trend_mode",
                allowed={"linear", "exponential"},
            )
