"""
Plotting utilities for TimeSeries visualization using matplotlib and plotly.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Optional, Union

import matplotlib.axes
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from darts.logging import get_logger, raise_log

if TYPE_CHECKING:
    import plotly.graph_objects as go

    from darts import TimeSeries

logger = get_logger(__name__)


def prepare_plot_params(
    series: TimeSeries,
    central_quantile: Union[float, str],
    low_quantile: Optional[float],
    high_quantile: Optional[float],
    label: Union[str, Sequence[str]],
    max_nr_components: int,
    color: Any,
    c: Any,
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

    return {
        "n_components_to_plot": n_components_to_plot,
        "resolved_labels": resolved_labels,
        "color": color,
        "plot_ci": series.is_stochastic
        and low_quantile is not None
        and high_quantile is not None,
    }


def plot(
    series: TimeSeries,
    new_plot: bool = False,
    central_quantile: Union[float, str] = 0.5,
    low_quantile: Optional[float] = 0.05,
    high_quantile: Optional[float] = 0.95,
    default_formatting: bool = True,
    title: Optional[str] = None,
    label: Optional[Union[str, Sequence[str]]] = "",
    max_nr_components: int = 10,
    ax: Optional[matplotlib.axes.Axes] = None,
    alpha: Optional[float] = None,
    color: Optional[Union[str, tuple, Sequence[str, tuple]]] = None,
    c: Optional[Union[str, tuple, Sequence[str, tuple]]] = None,
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
        interval is shown if `confidence_low_quantile` is None (default 0.05).
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
    alpha_confidence_intvls = 0.25

    # parameter preparation
    prepared_params = prepare_plot_params(
        series,
        central_quantile,
        low_quantile,
        high_quantile,
        label,
        max_nr_components,
        color,
        c,
    )
    n_components_to_plot = prepared_params["n_components_to_plot"]
    resolved_labels = prepared_params["resolved_labels"]
    color = prepared_params["color"]
    plot_ci = prepared_params["plot_ci"]

    # separate color validation
    if not isinstance(color, (str, tuple)) and isinstance(color, Sequence):
        if len(color) != series.n_components and len(color) != n_components_to_plot:
            raise_log(
                ValueError(
                    f"The `color` sequence must have the same length as the number of series components "
                    f"({series.n_components}) or as the number of plotted components ({n_components_to_plot}). "
                    f"Received length `{len(label)}`."
                ),
                logger,
            )
        custom_colors = True
    else:
        custom_colors = False

    kwargs["alpha"] = alpha
    if not any(lw in kwargs for lw in ["lw", "linewidth"]):
        kwargs["lw"] = 2

    if new_plot:
        fig, ax = plt.subplots()
    else:
        if ax is None:
            ax = plt.gca()

    for i, comp_name in enumerate(series.components[:n_components_to_plot]):
        comp_ts = series[comp_name]

        if series.is_stochastic:
            if central_quantile == "mean":
                central_ts = comp_ts.mean()
            else:
                central_ts = comp_ts.quantile(q=central_quantile)
        else:
            central_ts = comp_ts

        central_series = central_ts.to_series()  # shape: (time,)

        kwargs["label"] = resolved_labels[i]
        kwargs["c"] = color[i] if custom_colors else color

        kwargs_central = deepcopy(kwargs)
        if series.is_stochastic:
            kwargs_central["alpha"] = 1
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
            low_series = comp_ts.quantile(q=low_quantile).to_series()
            high_series = comp_ts.quantile(q=high_quantile).to_series()
            # filled area
            if len(low_series) > 1:
                ax.fill_between(
                    series.time_index,
                    low_series,
                    high_series,
                    color=color_used,
                    alpha=(alpha if alpha is not None else alpha_confidence_intvls),
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
    fig: Optional[go.Figure] = None,
    central_quantile: Union[float, str] = 0.5,
    low_quantile: Optional[float] = 0.05,
    high_quantile: Optional[float] = 0.95,
    title: Optional[str] = None,
    label: Optional[Union[str, Sequence[str]]] = "",
    max_nr_components: int = 10,
    alpha: Optional[float] = None,
    color: Optional[Union[str, Sequence[str]]] = None,
    c: Optional[Union[str, Sequence[str]]] = None,
    downsample_threshold: int = 100_000,
    width: Optional[int] = None,
    height: Optional[int] = None,
    template: Optional[str] = None,
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
        The quantile (between 0 and 1) to plot as a "central" value if the series is probabilistic
        (stochastic). Default is 0.5 (median). Can also be set to 'mean' to display the average instead.
    low_quantile
        The quantile to use for the lower bound of the plotted confidence interval.
        Defaults to 0.05. If None, no confidence interval is displayed.
    high_quantile
        The quantile to use for the upper bound of the plotted confidence interval.
        Defaults to 0.95. If None, no confidence interval is displayed.
    title
        The title of the figure.
    label
        The label(s) for the traces. If a single string and the series is univariate, it is used as the
        trace name. If a single string and the series is multivariate, it is used as a prefix for
        component names. If a sequence, must match the number of components being plotted.
        If empty string (default), component names are used.
    max_nr_components
        The maximum number of components to plot. If the series has more, only the first
        `max_nr_components` are plotted. Defaults to 10. Use -1 to plot all components.
    alpha
        Opacity for the confidence interval fill. Defaults to None. For deterministic series lines,
        opacity can be passed via `kwargs`.
    color
        Set the line color(s). Can be a single color string (name or hex), or a sequence of
        strings (one per component). If a sequence, it must match the number of components.
        By default, colors are pulled from the active Plotly template.
    c
        Alias for `color` parameter. Cannot be used simultaneously with `color`.
    downsample_threshold
        The maximum number of total data points (time steps * components * traces) to plot.
        If exceeded, the series will be automatically downsampled using a constant step
        size to avoid rendering crashes. Set to -1 to disable downsampling. Defaults to 100,000.
    width
        Optionally, the width of the figure in pixels.
    height
        Optionally, the height of the figure in pixels.
    template
        Optionally, the name of a Plotly template to use for the figure (e.g., 'plotly', 'plotly_white', 'none').
        Setting `template='darts'` will use the Darts-specific colorway and formatting.
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
        import plotly.io as pio
    except ImportError:
        raise_log(
            ImportError(
                "Plotly is not installed. Please install it with: `pip install plotly`"
            ),
            logger,
        )

    def _get_active_colorway(template_name=None):
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
        target = template_name or pio.templates.default
        template = pio.templates[target] if target in pio.templates else None
        colorway = (
            getattr(template.layout, "colorway", None)
            if hasattr(template, "layout")
            else None
        )
        return list(colorway) if colorway else plotly_default_colors

    def _modify_color_opacity(color, alpha):
        if "(" in color:
            prefix = re.search(r"^[a-z]+", color.lower()).group().rstrip("a")
            values = re.findall(r"[-+]?\d*\.?\d+%?", color)[:3]
            return f"{prefix}a({', '.join(values)}, {alpha})"
        r, g, b, _ = mcolors.to_rgba(color)
        return f"rgba({int(r * 255)}, {int(g * 255)}, {int(b * 255)}, {alpha})"

    # parameter preparation
    prepared_params = prepare_plot_params(
        series,
        central_quantile,
        low_quantile,
        high_quantile,
        label,
        max_nr_components,
        color,
        c,
    )
    n_components_to_plot = prepared_params["n_components_to_plot"]
    resolved_labels = prepared_params["resolved_labels"]
    color = prepared_params["color"]
    plot_ci = prepared_params["plot_ci"]

    # initialize figure
    fig = fig or go.Figure()

    color_cycle_start_idx = 0
    if color is None:
        # use colors from provided template or default template
        resolved_colors = _get_active_colorway(template_name=template)
        # count existing main traces for correct color cycling if plotting multiple series
        color_cycle_start_idx = len([t for t in fig.data if t.hoverinfo != "skip"])
    elif isinstance(color, str):
        # single color string provided: wrap in list to allow infinite cycling
        resolved_colors = [color]
    elif isinstance(color, Sequence):
        # sequence of strings provided: validate length
        if len(color) not in {series.n_components, n_components_to_plot}:
            expected_str = " or ".join(
                map(str, sorted({series.n_components, n_components_to_plot}))
            )
            raise_log(
                ValueError(
                    f"The `color` sequence length ({len(color)}) is invalid. "
                    f"It must match the number of components ({expected_str})."
                ),
                logger,
            )
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

    alpha_ci = alpha or 0.25
    time_idx = series.time_index[::step]
    is_single_point = len(time_idx) == 1

    for i, comp_name in enumerate(series.components[:n_components_to_plot]):
        comp_ts = series[comp_name]

        # determine central series
        if not series.is_stochastic:
            central_values = comp_ts.values(copy=False).flatten()
        else:
            if central_quantile == "mean":
                central_values = comp_ts.mean().values(copy=False).flatten()
            else:
                central_values = (
                    comp_ts.quantile(central_quantile).values(copy=False).flatten()
                )
        central_values = central_values[::step]

        # determine label & color
        curr_label = resolved_labels[i]
        curr_color = resolved_colors[(color_cycle_start_idx + i) % len(resolved_colors)]

        # plot confidence intervals (if stochastic)
        ci_data = None
        error_y = None
        if plot_ci:
            low_values = (
                comp_ts.quantile(low_quantile).values(copy=False).flatten()[::step]
            )
            high_values = (
                comp_ts.quantile(high_quantile).values(copy=False).flatten()[::step]
            )
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
        template=template or pio.templates.default,
        hovermode="x unified",
        width=width,
        height=height,
    )

    return fig
