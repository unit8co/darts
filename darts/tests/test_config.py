"""
Tests for the configuration system
"""

import matplotlib as mpl
import pytest

from darts.config import (
    describe_option,
    get_option,
    option_context,
    reset_option,
    set_option,
)
from darts.tests.conftest import PLOTLY_AVAILABLE

if PLOTLY_AVAILABLE:
    import plotly.io as pio


@pytest.fixture(scope="function", autouse=True)
def reset_options():
    """Restores default config after each test."""
    yield
    reset_option("all")


class TestConfig:
    """Test suite for Darts configuration system."""

    def test_get_option_exact_match(self):
        """Test getting option with exact key match."""
        value = get_option("display.max_rows")
        assert isinstance(value, int)
        assert value == 10  # Default value

    def test_get_option_empty_pattern(self):
        """Test getting option with invalid pattern."""
        with pytest.raises(ValueError, match="Pattern must be non-empty"):
            get_option("")

    def test_get_option_invalid_pattern(self):
        """Test getting option with invalid pattern."""
        with pytest.raises(ValueError, match="No option found matching pattern"):
            get_option("invalid.option")

    def test_get_option_ambiguous_pattern(self):
        """Test getting option with ambiguous pattern."""
        # This should work for prefix match returning single category
        # But should fail if the pattern is too broad
        with pytest.raises(ValueError, match="matches multiple options"):
            get_option("display")  # Matches multiple display.* options

    def test_set_option_valid(self):
        """Test setting option with valid value."""
        set_option("display.max_rows", 20)
        assert get_option("display.max_rows") == 20

    def test_set_option_invalid_value(self):
        """Test setting option with invalid value."""
        with pytest.raises(ValueError, match="must be a positive integer"):
            set_option("display.max_rows", -5)

        with pytest.raises(ValueError, match="must be a positive integer"):
            set_option("display.max_rows", 0)

        with pytest.raises(ValueError, match="must be a positive integer"):
            set_option("display.max_rows", "invalid")

    def test_set_option_boolean(self):
        """Test setting boolean option."""
        set_option("plotting.use_darts_style", False)
        assert get_option("plotting.use_darts_style") is False

        set_option("plotting.use_darts_style", True)
        assert get_option("plotting.use_darts_style") is True

    def test_set_option_boolean_invalid(self):
        """Test setting boolean option with invalid value."""
        with pytest.raises(ValueError, match="must be a boolean"):
            set_option("plotting.use_darts_style", "true")

        with pytest.raises(ValueError, match="must be a boolean"):
            set_option("plotting.use_darts_style", 1)

    def test_reset_option_single(self):
        """Test resetting a single option."""
        # Change the value
        set_option("display.max_rows", 50)
        assert get_option("display.max_rows") == 50

        # Reset to default
        reset_option("display.max_rows")
        assert get_option("display.max_rows") == 10  # Default value

    def test_reset_option_multiple(self):
        """Test resetting multiple options with pattern."""
        # Change multiple display options
        set_option("display.max_rows", 50)
        set_option("display.max_cols", 30)

        # Reset all display options
        reset_option("display")

        # Check all are back to defaults
        assert get_option("display.max_rows") == 10
        assert get_option("display.max_cols") == 10

    def test_describe_option_single(self):
        """Test describing a single option."""
        desc = describe_option("display.max_rows")
        assert "display.max_rows" in desc
        assert "Maximum number of rows" in desc
        assert "[default: 10]" in desc

    def test_describe_option_multiple(self):
        """Test describing multiple options."""
        desc = describe_option("display")
        assert "display.max_rows" in desc
        assert "display.max_cols" in desc

    def test_describe_option_all(self):
        """Test describing all options."""
        desc = describe_option("all")
        assert "display.max_rows" in desc
        assert "display.max_cols" in desc
        assert "plotting.use_darts_style" in desc

    def test_option_context_single(self):
        """Test option_context with single option."""
        original = get_option("display.max_rows")

        with option_context("display.max_rows", 100):
            assert get_option("display.max_rows") == 100

        # Value should be restored
        assert get_option("display.max_rows") == original

    def test_option_context_multiple(self):
        """Test option_context with multiple options."""
        original_rows = get_option("display.max_rows")
        original_cols = get_option("display.max_cols")

        with option_context("display.max_rows", 100, "display.max_cols", 50):
            assert get_option("display.max_rows") == 100
            assert get_option("display.max_cols") == 50

        # Values should be restored
        assert get_option("display.max_rows") == original_rows
        assert get_option("display.max_cols") == original_cols

    def test_option_context_nested(self):
        """Test nested option_context."""
        original = get_option("display.max_rows")
        assert original != 100

        with option_context("display.max_rows", 100):
            assert get_option("display.max_rows") == 100

            with option_context("display.max_rows", 200):
                assert get_option("display.max_rows") == 200

            # Back to outer context value
            assert get_option("display.max_rows") == 100

        # Back to original
        assert get_option("display.max_rows") == original

    def test_option_context_exception(self):
        """Test that option_context restores values even if exception occurs."""
        original = get_option("display.max_rows")

        try:
            with option_context("display.max_rows", 100):
                assert get_option("display.max_rows") == 100
                raise ValueError("Test exception")
        except ValueError:
            pass

        # Value should still be restored
        assert get_option("display.max_rows") == original

    def test_option_context_invalid_args(self):
        """Test option_context with invalid number of arguments."""
        with pytest.raises(ValueError, match="even number of arguments"):
            with option_context("display.max_rows"):
                pass

    def test_all_default_options_accessible(self):
        """Test that all default options can be accessed."""
        options = [
            "display.max_rows",
            "display.max_cols",
            "plotting.use_darts_style",
        ]

        for option in options:
            # Should not raise any exception
            value = get_option(option)
            assert value is not None

    def test_option_types(self):
        """Test that options have the expected types."""
        assert isinstance(get_option("display.max_rows"), int)
        assert isinstance(get_option("display.max_cols"), int)
        assert isinstance(get_option("plotting.use_darts_style"), bool)

    def test_plotting_style_callback(self):
        """Test that changing plotting.use_darts_style actually updates matplotlib."""
        # Store original rcParams
        assert mpl.rcParams == mpl.rcParamsDefault

        # Apply Darts style
        set_option("plotting.use_darts_style", True)
        assert mpl.rcParams != mpl.rcParamsDefault

        # Remove Darts style
        set_option("plotting.use_darts_style", False)
        assert mpl.rcParams == mpl.rcParamsDefault

        # Re-apply Darts style
        set_option("plotting.use_darts_style", True)
        assert mpl.rcParams != mpl.rcParamsDefault

        # Reset style
        reset_option("plotting.use_darts_style")
        assert mpl.rcParams == mpl.rcParamsDefault

    def test_plotting_no_user_style_override_on_reset(self):
        """Test that changing plotting.use_darts_style actually updates matplotlib."""
        # Store original rcParams
        mpl.rcParams.update({"font.family": "serif"})
        user_style = mpl.rcParams.copy()
        assert user_style != mpl.rcParamsDefault

        # Resetting style should not override user's style
        reset_option("plotting.use_darts_style")
        assert mpl.rcParams == user_style

        # Setting Darts style and resetting should restore user's style
        set_option("plotting.use_darts_style", True)
        reset_option("plotting.use_darts_style")
        assert mpl.rcParams == user_style

    @pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="requires plotly")
    def test_plotting_style_callback_plotly(self):
        """Test that changing plotting.use_darts_style updates plotly template."""
        # Store original template
        original_template = pio.templates.default

        # Apply Darts style
        set_option("plotting.use_darts_style", True)
        assert pio.templates.default != original_template
        darts_template = pio.templates.default

        # Remove Darts style
        set_option("plotting.use_darts_style", False)
        assert pio.templates.default == original_template

        # Re-apply Darts style
        set_option("plotting.use_darts_style", True)
        assert pio.templates.default == darts_template

        # Reset style
        reset_option("plotting.use_darts_style")
        assert pio.templates.default == original_template

    @pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="requires plotly")
    def test_plotting_no_user_template_override_on_reset_plotly(self):
        """Test that resetting plotting.use_darts_style preserves user's plotly template."""
        # Set a custom template
        pio.templates.default = "plotly_dark"
        user_template = pio.templates.default
        assert user_template == "plotly_dark"

        # Resetting style should not override user's template
        reset_option("plotting.use_darts_style")
        assert pio.templates.default == user_template

        # Setting Darts style and resetting should restore user's template
        set_option("plotting.use_darts_style", True)
        assert pio.templates.default != user_template
        reset_option("plotting.use_darts_style")
        assert pio.templates.default == user_template
