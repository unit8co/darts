"""Generate the Timestral quickstart figure from a seven-day Zurich holdout.

Run from the repository root after installing the optional Timestral dependency:
    uv run python examples/timestral_quickstart.py

The split and settings are fixed before forecasting. This is an illustrative
holdout in the Darts workflow, not a benchmark or a guarantee that these dates
were excluded from upstream pretraining.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import StrMethodFormatter

from darts.datasets import ElectricityConsumptionZurichDataset
from darts.models import TimestralModel
from darts.utils.likelihood_models import QuantileRegression

HORIZON = 168
CONTEXT = 2048
REVISION = "bdd20cfc15a3fea0e4cb5ccef37f8d3ca7aa335f"


def main():
    torch.set_num_threads(1)
    series = ElectricityConsumptionZurichDataset().load().astype("float32")
    series = series[["Value_NE5"]].resample("1h", method="mean")
    train, actual = series[:-HORIZON], series[-HORIZON:]
    model = TimestralModel(
        input_chunk_length=CONTEXT,
        output_chunk_length=HORIZON,
        hub_model_name="mistralai/Timestral-5M",
        hub_model_revision=REVISION,
        likelihood=QuantileRegression([0.1, 0.5, 0.9]),
        pl_trainer_kwargs={
            "accelerator": "cpu",
            "logger": False,
            "enable_progress_bar": False,
        },
    )
    model.fit(train)
    forecast = model.predict(HORIZON, predict_likelihood_parameters=True)
    assert forecast.time_index.equals(actual.time_index)
    lower, median, upper = forecast.values().T
    truth = actual.univariate_values()
    baseline = train[-HORIZON:].univariate_values()
    assert np.isfinite(np.stack((truth, baseline, lower, median, upper))).all()
    assert ((lower <= median) & (median <= upper)).all()
    mae = np.mean(np.abs(truth - median))
    baseline_mae = np.mean(np.abs(truth - baseline))
    mape = 100 * np.mean(np.abs((truth - median) / truth))
    baseline_mape = 100 * np.mean(np.abs((truth - baseline) / truth))
    coverage = 100 * np.mean((lower <= truth) & (truth <= upper))
    print(f"Holdout: {actual.start_time()} to {actual.end_time()} (UTC)")
    print(f"Timestral-5M: MAE={mae:.2f}, MAPE={mape:.2f}%")
    print(f"Repeat last week: MAE={baseline_mae:.2f}, MAPE={baseline_mape:.2f}%")
    print(f"10–90% interval coverage: {coverage:.1f}%")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, ax = plt.subplots(figsize=(13.5, 5.8), facecolor="white")
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.25, top=0.79)
    fig.text(
        0.08, 0.94, "Timestral-5M · Zurich electricity", fontsize=19, weight="bold"
    )
    fig.text(
        0.08,
        0.885,
        "7-day holdout · 2,048 hours of context · zero-shot · no covariates · CPU",
        fontsize=11,
        color="#526071",
    )
    history = train[-HORIZON:]
    ax.plot(
        history.time_index,
        history.univariate_values(),
        color="#64748b",
        lw=1.6,
        label="Observed history",
    )
    ax.fill_between(
        actual.time_index,
        lower,
        upper,
        color="#3b82f6",
        alpha=0.18,
        label="10–90% forecast interval",
    )
    ax.plot(actual.time_index, truth, color="#172033", lw=1.8, label="Actual holdout")
    ax.plot(
        actual.time_index, median, color="#2563eb", lw=1.8, label="Timestral median"
    )
    ax.plot(
        actual.time_index,
        baseline,
        color="#b7791f",
        lw=1.2,
        ls="--",
        alpha=0.8,
        label="Repeat last week",
    )
    ax.axvline(actual.start_time(), color="#94a3b8", lw=1, ls=":")
    ax.text(
        actual.start_time(),
        1.025,
        " Forecast starts",
        transform=ax.get_xaxis_transform(),
        color="#526071",
        fontsize=9,
    )
    ax.set_ylabel("Consumption (kWh, hourly mean)")
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.set_xlim(history.start_time(), actual.end_time())
    ax.grid(axis="y", color="#e2e8f0", lw=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["bottom", "left"]].set_color("#cbd5e1")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=5,
        frameon=False,
        fontsize=9,
    )
    fig.text(
        0.08,
        0.075,
        f"Holdout MAE: Timestral {mae:,.0f} vs. repeat last week {baseline_mae:,.0f} kWh"
        f"    |    MAPE: {mape:.2f}% vs. {baseline_mape:.2f}%"
        f"    |    Interval coverage: {coverage:.1f}%",
        fontsize=10,
    )
    fig.text(
        0.08,
        0.025,
        "Source: Darts ElectricityConsumptionZurichDataset, Value_NE5 · 24–31 Aug 2022 (UTC). "
        "One illustrative split, not a benchmark.",
        fontsize=9,
        color="#64748b",
    )
    destination = (
        Path(__file__).parent / "static" / "images" / "timestral_quickstart.png"
    )
    fig.savefig(destination, dpi=160)
    plt.close(fig)
    print(f"Saved {destination}")


if __name__ == "__main__":
    main()
