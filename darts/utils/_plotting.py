"""
Plotting utilities for TimeSeries visualization using matplotlib and plotly.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from copy import deepcopy
from typing import TYPE_CHECKING, Any

import matplotlib.axes
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from darts.logging import get_logger, raise_log

if TYPE_CHECKING:
    import plotly.graph_objects as go

    from darts import TimeSeries

logger = get_logger(__name__)


def _prepare_plot_params(
    series: TimeSeries,
    central_quantile: float | str,
    low_quantile: float | None,
    high_quantile: float | None,
    label: str | Sequence[str],
    max_nr_components: int,
    color: Any,
    c: Any,
    alpha: float | None,
) -> dict:
    """Shared input validation and parameter preparation for plot() and plotly()."""

    # quantile validation
    if central_quantile != "mean":
        if not (isinstance(central_quantile, float) and 0.0 <= central_quantile <= 1.0):
            raise_log(
                ValueError(
                    'central_quantile must be either "mean", or a float between 0 and 1.'
                ),
                logger,
            )

    if high_quantile is not None and low_quantile is not None:
        if not (0.0 <= low_quantile <= 1.0 and 0.0 <= high_quantile <= 1.0):
            raise_log(
                ValueError(
                    "confidence interval low and high quantiles must be between 0 and 1."
                ),
                logger,
            )
        if low_quantile >= high_quantile:
            raise_log(
                ValueError(
                    f"low_quantile ({low_quantile}) must be less than high_quantile ({high_quantile})."
                ),
                logger,
            )

    # component slicing
    n_components_to_plot = (
        series.n_components
        if max_nr_components == -1
        else min(series.n_components, max_nr_components)
    )
    if series.n_components > n_components_to_plot:
        logger.warning(
            f"Number of series components ({series.n_components}) is larger than the maximum number of "
            f"components to plot ({max_nr_components}). Plotting only the first `{n_components_to_plot}` "
            f"components. You can adjust the number of components to plot using `max_nr_components`."
        )

    # label resolution
    custom_labels = not isinstance(label, str) and isinstance(label, Sequence)
    if custom_labels:
        if len(label) != series.n_components and len(label) != n_components_to_plot:
            raise_log(
                ValueError(
                    f"The `label` sequence must have the same length as the number of series components "
                    f"({series.n_components}) or as the number of plotted components ({n_components_to_plot}). "
                    f"Received length `{len(label)}`."
                ),
                logger,
            )

    resolved_labels = []
    for i, comp_name in enumerate(series.components[:n_components_to_plot]):
        if custom_labels:
            lbl = label[i]
        elif label == "":
            lbl = comp_name
        elif series.n_components == 1:
            lbl = label
        else:
            lbl = f"{label}_{comp_name}"
        resolved_labels.append(lbl)

    # color resolution
    if color and c:
        raise_log(
            ValueError(
                "`color` and `c` must not be used simultaneously, use one or the other."
            ),
            logger,
        )
    color = color if color is not None else c

    # color sequence length validation
    if (
        isinstance(color, Sequence)
        and not isinstance(color, str)
        and not isinstance(color, tuple)
    ):
        if len(color) not in {series.n_components, n_components_to_plot}:
            raise_log(
                ValueError(
                    f"The `color` sequence must have the same length as the number of series components "
                    f"({series.n_components}) or as the number of plotted components ({n_components_to_plot}). "
                    f"Received length `{len(color)}`."
                ),
                logger,
            )

    # alpha preprocessing
    alpha_ci = alpha if alpha is not None else 0.25
    alpha_line = 1 if series.is_stochastic else alpha

    return {
        "n_components_to_plot": n_components_to_plot,
        "resolved_labels": resolved_labels,
        "color": color,
        "plot_ci": series.is_stochastic
        and low_quantile is not None
        and high_quantile is not None,
        "alpha_ci": alpha_ci,
        "alpha_line": alpha_line,
    }


def _compute_central_series(
    comp_ts: TimeSeries,
    central_quantile: float | str,
) -> TimeSeries:
    """Compute the central TimeSeries for a component."""
    if comp_ts.is_stochastic:
        if central_quantile == "mean":
            return comp_ts.mean()
        else:
            return comp_ts.quantile(q=central_quantile)
    else:
        return comp_ts


def _compute_quantile_bounds(
    comp_ts: TimeSeries,
    low_quantile: float,
    high_quantile: float,
) -> tuple[TimeSeries, TimeSeries]:
    """Compute the low and high quantile TimeSeries for confidence intervals.

    Returns
    -------
    tuple[TimeSeries, TimeSeries]
        A tuple of (low_ts, high_ts) representing the confidence interval bounds.
    """
    low_ts = comp_ts.quantile(q=low_quantile)
    high_ts = comp_ts.quantile(q=high_quantile)
    return low_ts, high_ts


def plot(
    series: TimeSeries,
    new_plot: bool = False,
    central_quantile: float | str = 0.5,
    low_quantile: float | None = 0.05,
    high_quantile: float | None = 0.95,
    default_formatting: bool = True,
    title: str | None = None,
    label: str | Sequence[str] | None = "",
    max_nr_components: int = 10,
    ax: matplotlib.axes.Axes | None = None,
    alpha: float | None = None,
    color: str | tuple | Sequence[str, tuple] | None = None,
    c: str | tuple | Sequence[str, tuple] | None = None,
    *args,
    **kwargs,
) -> matplotlib.axes.Axes:
    """Plot a TimeSeries using Matplotlib.

    Parameters
    ----------
    series
        The TimeSeries to plot.
    new_plot
        Whether to spawn a new axis to plot on. See also parameter `ax`.
    central_quantile
        The quantile (between 0 and 1) to plot as a "central" value, if the series is stochastic (i.e., if
        it has multiple samples). This will be applied on each component separately (i.e., to display quantiles
        of the components' marginal distributions). For instance, setting `central_quantile=0.5` will plot the
        median of each component. `central_quantile` can also be set to 'mean'.
    low_quantile
        The quantile to use for the lower bound of the plotted confidence interval. Similar to `central_quantile`,
        this is applied to each component separately (i.e., displaying marginal distributions). No confidence
        interval is shown if `low_quantile` is None (default 0.05).
    high_quantile
        The quantile to use for the upper bound of the plotted confidence interval. Similar to `central_quantile`,
        this is applied to each component separately (i.e., displaying marginal distributions). No confidence
        interval is shown if `high_quantile` is None (default 0.95).
    default_formatting
        Whether to use the darts default scheme.
    title
        Optionally, a plot title.
    label
        Can either be a string or list of strings. If a string and the series only has a single component, it is
        used as the label for that component. If a string and the series has multiple components, it is used as
        a prefix for each component name. If a list of strings with length equal to the number of components in
        the series, the labels will be mapped to the components in order.
    max_nr_components
        The maximum number of components of a series to plot. -1 means all components will be plotted.
    ax
        Optionally, an axis to plot on. If `None`, and `new_plot=False`, will use the current axis. If
        `new_plot=True`, will create a new axis.
    alpha
        Optionally, set the line alpha for deterministic series, or the confidence interval alpha for
        probabilistic series.
    color
        Can either be a single color or list of colors. Any matplotlib color is accepted (string, hex string,
        RGB/RGBA tuple). If a single color and the series only has a single component, it is used as the color
        for that component. If a single color and the series has multiple components, it is used as the color
        for each component. If a list of colors with length equal to the number of components in the series, the
        colors will be mapped to the components in order.
    c
        An alias for `color`.
    args
        some positional arguments for the `plot()` method
    kwargs
        some keyword arguments for the `plot()` method

    Returns
    -------
    matplotlib.axes.Axes
        Either the passed `ax` axis, a newly created one if `new_plot=True`, or the existing one.
    """
    # parameter preparation
    prepared_params = _prepare_plot_params(
        series,
        central_quantile,
        low_quantile,
        high_quantile,
        label,
        max_nr_components,
        color,
        c,
        alpha,
    )
    n_components_to_plot = prepared_params["n_components_to_plot"]
    resolved_labels = prepared_params["resolved_labels"]
    color = prepared_params["color"]
    plot_ci = prepared_params["plot_ci"]
    alpha_ci = prepared_params["alpha_ci"]
    alpha_line = prepared_params["alpha_line"]

    # determine if custom colors (sequence of colors) are provided
    custom_colors = isinstance(color, Sequence) and not isinstance(color, str | tuple)

    kwargs["alpha"] = alpha_line
    if not any(lw in kwargs for lw in ["lw", "linewidth"]):
        kwargs["lw"] = 2

    if new_plot:
        fig, ax = plt.subplots()
    else:
        if ax is None:
            ax = plt.gca()

    for i, comp_name in enumerate(series.components[:n_components_to_plot]):
        comp_ts = series[comp_name]

        central_ts = _compute_central_series(comp_ts, central_quantile)
        central_series = central_ts.to_series()  # shape: (time,)

        kwargs["label"] = resolved_labels[i]
        kwargs["c"] = color[i] if custom_colors else color

        kwargs_central = deepcopy(kwargs)
        # line plot
        if len(central_series) > 1:
            p = central_series.plot(
                *args,
                ax=ax,
                **kwargs_central,
            )
            color_used = p.get_lines()[-1].get_color() if default_formatting else None
        # point plot
        elif len(central_series) == 1:
            p = ax.plot(
                [series.start_time()],
                central_series.values[0],
                "o",
                *args,
                **kwargs_central,
            )
            color_used = p[0].get_color() if default_formatting else None
        # empty plot
        else:
            p = ax.plot(
                [],
                [],
                *args,
                **kwargs_central,
            )
            color_used = p[0].get_color() if default_formatting else None
        ax.set_xlabel(series.time_dim)

        # Optionally show confidence intervals
        if plot_ci:
            low_ts, high_ts = _compute_quantile_bounds(
                comp_ts, low_quantile, high_quantile
            )
            low_series = low_ts.to_series()
            high_series = high_ts.to_series()
            # filled area
            if len(low_series) > 1:
                ax.fill_between(
                    series.time_index,
                    low_series,
                    high_series,
                    color=color_used,
                    alpha=alpha_ci,
                )
            # filled line
            elif len(low_series) == 1:
                ax.plot(
                    [series.start_time(), series.start_time()],
                    [low_series.values[0], high_series.values[0]],
                    "-+",
                    color=color_used,
                    lw=2,
                )

    ax.legend()
    ax.set_title(title if title is not None else "")
    return ax


def plotly(
    series: TimeSeries,
    fig: go.Figure | None = None,
    central_quantile: float | str = 0.5,
    low_quantile: float | None = 0.05,
    high_quantile: float | None = 0.95,
    title: str | None = None,
    label: str | Sequence[str] | None = "",
    max_nr_components: int = 10,
    alpha: float | None = None,
    color: str | Sequence[str] | None = None,
    c: str | Sequence[str] | None = None,
    downsample_threshold: int = 100_000,
    **kwargs,
) -> go.Figure:
    """Plot a TimeSeries using Plotly.

    Parameters
    ----------
    series
        The TimeSeries to plot.
    fig
        Optionally, a Plotly `go.Figure` object to plot on. If provided, the series will be added to this
        figure. If None, a new figure will be created.
    central_quantile
        The quantile (between 0 and 1) to plot as a "central" value, if the series is stochastic (i.e., if
        it has multiple samples). This will be applied on each component separately (i.e., to display quantiles
        of the components' marginal distributions). For instance, setting `central_quantile=0.5` will plot the
        median of each component. `central_quantile` can also be set to 'mean'.
    low_quantile
        The quantile to use for the lower bound of the plotted confidence interval. Similar to `central_quantile`,
        this is applied to each component separately (i.e., displaying marginal distributions). No confidence
        interval is shown if `low_quantile` is None (default 0.05).
    high_quantile
        The quantile to use for the upper bound of the plotted confidence interval. Similar to `central_quantile`,
        this is applied to each component separately (i.e., displaying marginal distributions). No confidence
        interval is shown if `high_quantile` is None (default 0.95).
    title
        Optionally, a plot title.
    label
        Can either be a string or list of strings. If a string and the series only has a single component, it is
        used as the label for that component. If a string and the series has multiple components, it is used as
        a prefix for each component name. If a list of strings with length equal to the number of components in
        the series, the labels will be mapped to the components in order.
    max_nr_components
        The maximum number of components of a series to plot. -1 means all components will be plotted.
    alpha
        Optionally, set the line alpha for deterministic series, or the confidence interval alpha for
        probabilistic series.
    color
        Set the line color(s). Can be a single color string (name or hex), or a sequence of
        strings (one per component). If a sequence, it must match the number of components.
        By default, colors are pulled from the active Plotly template.
    c
        An alias for `color`.
    downsample_threshold
        The maximum number of total data points (time steps * components * traces) to plot.
        If exceeded, the series will be automatically downsampled using a constant step
        size to avoid rendering crashes. Set to -1 to disable downsampling. Defaults to 100,000.
    **kwargs
        Additional keyword arguments to pass to `plotly.graph_objects.Scatter()` for trace customization
        (e.g., `line_dash`, `line_width`, `marker_symbol`, `opacity`, or `hovertemplate`).

    Returns
    -------
    plotly.graph_objects.Figure
        The Plotly figure object containing the plot. Call `.show()` on the returned figure to display it.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise_log(
            ImportError(
                "Plotly is not installed. Please install it with: `pip install plotly`"
            ),
            logger,
        )

    def _get_active_colorway(fig):
        plotly_default_colors = [
            "#636EFA",
            "#EF553B",
            "#00CC96",
            "#AB63FA",
            "#FFA15A",
            "#19D3F3",
            "#FF6692",
            "#B6E880",
            "#FF97FF",
            "#FECB52",
        ]
        template = fig.layout.template
        colorway = (
            getattr(template.layout, "colorway", None)
            if hasattr(template, "layout")
            else None
        )
        return list(colorway) if colorway else plotly_default_colors

    def _modify_color_opacity(color, alpha):
        # if color is already in rgba/hsla format, just replace alpha
        if "(" in color:
            # find prefix (rgb, rgba, hsl, hsla)
            prefix = re.search(r"^[a-z]+", color.lower()).group().rstrip("a")
            # find numeric values
            values = re.findall(r"[-+]?\d*\.?\d+%?", color)[:3]
            # reconstruct with new alpha
            return f"{prefix}a({', '.join(values)}, {alpha})"
        # else, convert to rgba and set alpha
        r, g, b, _ = mcolors.to_rgba(color)
        return f"rgba({int(r * 255)}, {int(g * 255)}, {int(b * 255)}, {alpha})"

    # parameter preparation
    prepared_params = _prepare_plot_params(
        series,
        central_quantile,
        low_quantile,
        high_quantile,
        label,
        max_nr_components,
        color,
        c,
        alpha,
    )
    n_components_to_plot = prepared_params["n_components_to_plot"]
    resolved_labels = prepared_params["resolved_labels"]
    color = prepared_params["color"]
    plot_ci = prepared_params["plot_ci"]
    alpha_ci = prepared_params["alpha_ci"]
    alpha_line = prepared_params["alpha_line"]

    # initialize figure
    fig = fig or go.Figure()

    color_cycle_start_idx = 0
    if color is None:
        # use colors from figure template
        resolved_colors = _get_active_colorway(fig)
        # count existing main traces for correct color cycling if plotting multiple series
        color_cycle_start_idx = len([t for t in fig.data if t.hoverinfo != "skip"])
    elif isinstance(color, str):
        # single color string provided: wrap in list to allow infinite cycling
        resolved_colors = [color]
    elif isinstance(color, Sequence):
        # sequence of strings provided (length already validated in _prepare_plot_params)
        resolved_colors = list(color)
    else:
        raise_log(
            ValueError(
                "`color` and `c` must be a string, a sequence of strings, or None."
            ),
            logger,
        )

    # downsampling
    step = 1
    points_to_plot = len(series) * n_components_to_plot * (3 if plot_ci else 1)
    if downsample_threshold != -1 and points_to_plot > downsample_threshold:
        step = np.pow(
            2, np.ceil(np.log2(points_to_plot / downsample_threshold))
        ).astype(int)

        logger.warning(
            f"Plotting {points_to_plot:,} data points exceeds `downsample_threshold={downsample_threshold:,}`.\n"
            f"Automatically downsampling by a factor of {step} to avoid crashing the rendering engine.\n"
            f"To adjust this, increase `downsample_threshold` or set it to `-1` to disable downsampling."
        )

    time_idx = series.time_index[::step]
    is_single_point = len(time_idx) == 1

    for i, comp_name in enumerate(series.components[:n_components_to_plot]):
        comp_ts = series[comp_name]

        # determine central series
        central_ts = _compute_central_series(comp_ts, central_quantile)
        central_values = central_ts.values(copy=False).flatten()[::step]

        # determine label & color
        curr_label = resolved_labels[i]
        curr_color = resolved_colors[(color_cycle_start_idx + i) % len(resolved_colors)]

        # plot confidence intervals (if stochastic)
        ci_data = None
        error_y = None
        if plot_ci:
            low_ts, high_ts = _compute_quantile_bounds(
                comp_ts, low_quantile, high_quantile
            )
            low_values = low_ts.values(copy=False).flatten()[::step]
            high_values = high_ts.values(copy=False).flatten()[::step]
            fill_color = _modify_color_opacity(curr_color, alpha_ci)

            if not is_single_point:
                # ci upper bound
                fig.add_trace(
                    go.Scatter(
                        x=time_idx,
                        y=high_values,
                        mode="lines",
                        line=dict(width=0),
                        showlegend=False,
                        legendgroup=curr_label,
                        hoverinfo="skip",
                    )
                )
                # ci lower bound (filled area)
                fig.add_trace(
                    go.Scatter(
                        x=time_idx,
                        y=low_values,
                        mode="lines",
                        line=dict(width=0),
                        fill="tonexty",
                        fillcolor=fill_color,
                        showlegend=False,
                        legendgroup=curr_label,
                        hoverinfo="skip",
                    )
                )
            else:
                # add error bars for single stochastic point
                error_y = dict(
                    type="data",
                    symmetric=False,
                    array=high_values - central_values,
                    arrayminus=central_values - low_values,
                    color=curr_color,
                )

            # stylized probabilistic hovertemplate
            ci_data = np.stack([low_values, high_values], axis=-1)
            q_label = {0.5: "Median", "mean": "Mean"}.get(
                central_quantile, f"{central_quantile}-Q"
            )
            hovertemplate = (
                f"<b>{curr_label}</b><br>"
                + f"{q_label}: %{{y:.3g}}<br>"
                + f"{int(round((high_quantile - low_quantile) * 100))}% CI: "
                + "[%{customdata[0]:.3g}, %{customdata[1]:.3g}]"
                + "<extra></extra>"
            )
        else:
            # simple deterministic hovertemplate
            hovertemplate = f"<b>{curr_label}</b>: %{{y:.3g}}<extra></extra>"

        # central line
        trace_params = {
            "x": time_idx,
            "y": central_values,
            "name": curr_label,
            "legendgroup": curr_label,
            "mode": kwargs.get("mode", "lines+markers" if is_single_point else "lines"),
            "line": {**{"color": curr_color}, **kwargs.get("line", {})},
            "customdata": ci_data,
            "hovertemplate": hovertemplate,
            "error_y": error_y,
            **({"opacity": alpha_line} if alpha_line is not None else {}),
        }

        fig.add_trace(
            go.Scatter(**{
                **trace_params,
                **{k: v for k, v in kwargs.items() if k not in ["mode", "line"]},
            })
        )

    # layout
    fig.update_layout(
        title=title,
        xaxis_title=series.time_dim,
        hovermode="x unified",
    )

    return fig
