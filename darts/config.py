"""
Configuration
-------------

Darts configuration system for global options and settings.

This module provides functionality to configure global behavior of Darts, similar to pandas' options system.

Available Options
=================

**Display Options**

- ``display.max_rows`` : int (default: 10)
    Maximum number of rows to display in TimeSeries representation. When a TimeSeries has more rows
    than this, the display will be truncated, showing the first and last portion of rows.

- ``display.max_cols`` : int (default: 10)
    Maximum number of columns to display in TimeSeries representation. When a TimeSeries has more
    columns than this, the display will be truncated, showing the first and last portion of columns.

**Plotting Options**

- ``plotting.use_darts_style`` : bool (default: False)
    Whether to apply Darts' custom plotting style to both matplotlib and plotly. When True, Darts will
    configure both backends with a custom style optimized for time series visualization. When False,
    the default or user-configured styles will be used. Changes to this option take effect immediately.

Examples
========
>>> from darts import get_option, set_option, option_context
>>> # Get current display settings
>>> get_option('display.max_rows')
10
>>> # Change display settings globally
>>> set_option('display.max_rows', 20)
>>> # Temporarily change settings within a context
>>> with option_context('display.max_rows', 5):
...     print(my_timeseries)  # Shows only 5 rows
>>> # Settings automatically restored after context
>>> get_option('display.max_rows')
20
"""

from collections.abc import Generator
from contextlib import contextmanager
from typing import Any, Callable, Optional

from darts.logging import get_logger, raise_log

logger = get_logger(__name__)

# Darts color palette used for both matplotlib and plotly plotting
_DARTS_COLORS = [
    "#000000",
    "#003dfd",
    "#b512b8",
    "#11a9ba",
    "#0d780f",
    "#f77f07",
    "#ba0f0f",
]

# Register Darts Plotly template (if plotly is available)
try:
    import plotly.graph_objects as go
    import plotly.io as pio

    PLOTLY_AVAILABLE = True

    pio.templates["darts"] = go.layout.Template(
        layout=go.Layout(
            font=dict(family="Arial, sans-serif", size=14, color="black"),
            paper_bgcolor="white",
            plot_bgcolor="white",
            colorway=_DARTS_COLORS,
            showlegend=True,
            legend=dict(
                bgcolor="rgba(255, 255, 255, 0.8)",
                x=1,
                y=1,
                yanchor="top",
                xanchor="right",
                font=dict(size=14),
                borderwidth=0,
            ),
            xaxis=dict(
                showline=True,
                linecolor="#dedede",
                showgrid=False,
                title=dict(font=dict(size=16, color="black")),
            ),
            yaxis=dict(
                showline=False,
                showgrid=True,
                gridcolor="#dedede",
                gridwidth=1,
                zeroline=True,
                zerolinecolor="#dedede",
            ),
            margin=dict(l=50, r=50, t=50, b=50),
        ),
        data=dict(scatter=[go.Scatter(line=dict(width=3))]),
    )
except ImportError:
    PLOTLY_AVAILABLE = False
    pass


class _Option:
    """Internal class representing a single configuration option."""

    def __init__(
        self,
        key: str,
        default_value: Any,
        description: str,
        validator: Optional[Callable] = None,
        callback: Optional[Callable] = None,
    ):
        self.key = key
        self.default_value = default_value
        self.description = description
        self.validator = validator
        self.callback = callback
        self.value = default_value

    def set(self, value: Any) -> None:
        """Set the option value with validation."""
        if self.validator is not None:
            self.validator(value)
        self.value = value
        # Call the callback if one is registered
        if self.callback is not None:
            self.callback(value)

    def reset(self) -> None:
        """Reset the option to its default value."""
        old_value = self.value
        self.value = self.default_value
        # Call the callback if value changed
        if self.callback is not None and old_value != self.default_value:
            self.callback(self.default_value)

    def get(self) -> Any:
        """Get the current option value."""
        return self.value


class _OptionsManager:
    def __init__(self):
        """Manager for all Darts configuration options."""
        # Display options
        display_max_rows = _Option(
            key="display.max_rows",
            default_value=10,
            description="Maximum number of rows to display in TimeSeries representation. "
            "When a TimeSeries has more rows than this, the display will be truncated, "
            "showing the first and last portion of rows.",
            validator=self._validate_positive_int,
        )

        display_max_cols = _Option(
            key="display.max_cols",
            default_value=10,
            description="Maximum number of columns to display in TimeSeries representation. "
            "When a TimeSeries has more columns than this, the display will be truncated, "
            "showing the first and last portion of columns.",
            validator=self._validate_positive_int,
        )

        # Plotting options
        plotting_use_darts_style = _Option(
            key="plotting.use_darts_style",
            default_value=False,
            description="Whether to apply Darts' custom plotting style to both matplotlib and plotly. "
            "When True, Darts will configure both backends with a custom style optimized for "
            "time series visualization. When False, the default or user-configured styles will be used.",
            validator=self._validate_bool,
            callback=self._on_plotting_style_change,
        )

        self._options = {
            opt.key: opt
            for opt in [
                display_max_rows,
                display_max_cols,
                plotting_use_darts_style,
            ]
        }
        # remember if user applied Darts style
        self._darts_plotting_style_applied = False
        # store original plotly template to restore later
        self._original_plotly_template = None

    @staticmethod
    def _validate_positive_int(value: Any):
        """Validator for positive integers."""
        if not isinstance(value, int) or value <= 0:
            raise_log(ValueError("Value must be a positive integer"), logger)

    @staticmethod
    def _validate_bool(value: Any):
        """Validator for boolean values."""
        if not isinstance(value, bool):
            raise_log(ValueError("Value must be a boolean"), logger)

    def _on_plotting_style_change(self, value: bool) -> None:
        """Callback for when plotting.use_darts_style changes."""
        # matplotlib
        import matplotlib as mpl
        from matplotlib import cycler

        if value:
            # apply Darts plotting style to matplotlib
            colors = cycler(color=_DARTS_COLORS)
            u8plots_mplstyle = {
                "font.family": "sans serif",
                "axes.edgecolor": "black",
                "axes.grid": True,
                "axes.labelcolor": "#333333",
                "axes.labelweight": 600,
                "axes.linewidth": 1,
                "axes.prop_cycle": colors,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "axes.spines.bottom": False,
                "axes.spines.left": False,
                "grid.color": "#dedede",
                "legend.frameon": False,
                "lines.linewidth": 1.3,
                "xtick.color": "#333333",
                "xtick.labelsize": "small",
                "ytick.color": "#333333",
                "ytick.labelsize": "small",
                "xtick.bottom": False,
            }
            mpl.rcParams.update(u8plots_mplstyle)
        elif self._darts_plotting_style_applied:
            # restore default matplotlib options
            mpl.rcParams.update(mpl.rcParamsDefault)

        # plotly
        if PLOTLY_AVAILABLE:
            import plotly.io as pio

            if value:
                # store existing default to restore later
                if self._original_plotly_template is None:
                    self._original_plotly_template = pio.templates.default

                # apply the registered 'darts' plotly template
                pio.templates.default = "darts"
            else:
                # restore the previous default
                if self._original_plotly_template is not None:
                    pio.templates.default = self._original_plotly_template
                    self._original_plotly_template = None

        # update the state tracker
        self._darts_plotting_style_applied = value

    def _find_option(self, pattern: str, check_unique: bool = False) -> list[_Option]:
        """Find options matching a pattern (supports both exact match and prefix match)."""
        if not pattern:
            raise_log(
                ValueError("Pattern must be non-empty"),
                logger,
            )

        if pattern == "all":
            return list(self._options.values())

        # Exact match
        if pattern in self._options:
            return [self._options[pattern]]

        # Prefix match (e.g., 'display' matches all 'display.*' options)
        matches = [
            opt
            for key, opt in self._options.items()
            if key.split(".")[0].startswith(pattern)
        ]
        if matches:
            if check_unique and len(matches) > 1:
                raise_log(
                    ValueError(
                        f"Pattern '{pattern}' matches multiple options: {[opt.key for opt in matches]}"
                        ". "
                        "Give a specific option."
                    ),
                    logger,
                )
            return matches

        raise_log(ValueError(f"No option found matching pattern: '{pattern}'"), logger)

    def get_option(self, pattern: str) -> Any:
        """Get the value of an option."""
        matches = self._find_option(pattern, check_unique=True)
        return matches[0].get()

    def set_option(self, pattern: str, value: Any) -> None:
        """Set the value of an option."""
        matches = self._find_option(pattern, check_unique=True)
        matches[0].set(value)

    def reset_option(self, pattern: str) -> None:
        """Reset option(s) to default value(s)."""
        matches = self._find_option(pattern, check_unique=False)
        for opt in matches:
            opt.reset()

    def describe_option(self, pattern: str) -> str:
        """Describe option(s) matching the pattern."""
        matches = self._find_option(pattern, check_unique=False)

        description_parts = []
        for opt in sorted(matches, key=lambda x: x.key):
            desc = f"{opt.key} : {type(opt.default_value).__name__}\n"
            desc += f"    {opt.description}\n"
            desc += f"    [default: {opt.default_value}] [currently: {opt.get()}]"
            description_parts.append(desc)

        return "\n\n".join(description_parts)

    @contextmanager
    def option_context(self, *args) -> Generator[None, None, None]:
        """Context manager to temporarily set options."""
        if len(args) % 2 != 0:
            raise_log(
                ValueError(
                    "option_context requires an even number of arguments (option-value pairs)"
                ),
                logger,
            )

        # Store original values
        original_values = {}
        try:
            # Set new values
            for pattern, value in zip(args[::2], args[1::2]):
                original_values[pattern] = self.get_option(pattern)
                self.set_option(pattern, value)

            yield
        finally:
            # Restore original values
            for key, value in original_values.items():
                self.set_option(key, value)


# Global options manager instance
_global_options = _OptionsManager()


# Public API functions
def get_option(pat: str) -> Any:
    """
    Retrieves the value of the specified option.

    Available Options:

    - display.[max_rows, max_cols]
    - plotting.use_darts_style

    Parameters
    ----------
    pat
        The option key to retrieve. Must uniquely identify a single option.

    Returns
    -------
    Any
        The current value of the option.

    Raises
    ------
    ValueError
        If no option matches the pattern, or if the pattern is ambiguous.

    Examples
    --------
    >>> from darts import get_option
    >>> get_option('display.max_rows')
    10
    """
    return _global_options.get_option(pat)


def set_option(pat: str, value: Any) -> None:
    """
    Sets the value of the specified option.

    Available Options:

    - display.[max_rows, max_cols]
    - plotting.use_darts_style

    Parameters
    ----------
    pat
        The option key to set. Must uniquely identify a single option.
    value
        The new value for the option. Must be valid according to the option's validator.

    Raises
    ------
    ValueError
        If no option matches the pattern, if the pattern is ambiguous, or if the value is invalid for the option.

    Examples
    --------
    >>> from darts import set_option
    >>> set_option('display.max_rows', 20)
    """
    _global_options.set_option(pat, value)


def reset_option(pat: str) -> None:
    """
    Reset one or more options to their default value.

    Available Options:

    - display.[max_rows, max_cols]
    - plotting.use_darts_style

    Parameters
    ----------
    pat
        The option key or pattern to reset. Can match multiple options. Use `"all"` to reset all options.

    Raises
    ------
    ValueError
        If no option matches the pattern.

    Examples
    --------
    >>> from darts import reset_option
    >>> reset_option('display.max_rows')  # Reset single option
    >>> reset_option('display')  # Reset all display options
    >>> reset_option('all')  # Reset all options
    """
    _global_options.reset_option(pat)


def describe_option(pat: str) -> str:
    """
    Describe one or more options.

    Available Options:

    - display.[max_rows, max_cols]
    - plotting.use_darts_style

    Parameters
    ----------
    pat
        The option key or pattern to describe. Can match multiple options. Use `"all"` to describe all options.

    Returns
    -------
    str
        The description for the specified options.

    Raises
    ------
    ValueError
        If no option matches the pattern.

    Examples
    --------
    >>> from darts import describe_option
    >>> describe_option('display.max_rows')
    display.max_rows : int
        Maximum number of rows to display in TimeSeries representation...
        [default: 10] [currently: 10]

    >>> describe_option('display')  # Describe all display options
    >>> describe_option('all')  # Describe all options
    """
    return _global_options.describe_option(pat)


@contextmanager
def option_context(*args) -> Generator[None, None, None]:
    """
    Context manager to temporarily set options in the `with` statement context.

    Available Options:

    - display.[max_rows, max_cols]
    - plotting.use_darts_style

    Parameters
    ----------
    *args
        Pairs of (key, value) for options to set temporarily.

    Yields
    ------
    None

    Examples
    --------
    >>> from darts import option_context, get_option
    >>> get_option('display.max_rows')
    10
    >>> with option_context('display.max_rows', 20, 'display.max_cols', 15):
    ...     print(get_option('display.max_rows'))
    ...     print(get_option('display.max_cols'))
    20
    15
    >>> get_option('display.max_rows')  # Back to original
    10
    """
    with _global_options.option_context(*args):
        yield
