"""Plot fairness and distribution-balance metrics for test-set conditions."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


CONDITION_ORDER = ["none", "activity", "eccentricity", "both"]
CONDITION_LABELS = {
    "none": "No balancing",
    "activity": "Balance\nactivity",
    "eccentricity": "Balance\neccentricity",
    "both": "Balance\nboth",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot EO and KS statistics from MF test-balancing results."
    )
    parser.add_argument(
        "--results-file", type=Path, default=Path("mf_balanced_test.csv")
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("figures/mf_balanced_test_metrics.png"),
    )
    parser.add_argument(
        "--title",
        default="Test-set balancing: fairness and distribution distances",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    results = pd.read_csv(args.results_file)
    results["eo_signed"] = results["rmse_A"] - results["rmse_B"]
    summary = results.groupby("condition")[[
        "eo_signed",
        "activity_ks",
        "eccentricity_ks",
    ]].agg(["mean", "std"]).reindex(CONDITION_ORDER)

    metrics = [
        ("eo_signed", "Signed EO (RMSE A - RMSE B)"),
        ("activity_ks", "Activity KS statistic"),
        ("eccentricity_ks", "Eccentricity KS statistic"),
    ]
    colors = ["#5B6472", "#D97745", "#3B7A78", "#B04A5A"]
    labels = [CONDITION_LABELS[name] for name in CONDITION_ORDER]
    positions = range(len(CONDITION_ORDER))

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.4), sharex=True)
    for axis, (column, title) in zip(axes, metrics):
        means = summary[(column, "mean")].to_numpy()
        stds = summary[(column, "std")].to_numpy()
        axis.bar(
            positions,
            means,
            yerr=stds,
            capsize=4,
            color=colors,
            edgecolor="#20252B",
            linewidth=0.6,
            error_kw={"elinewidth": 1.0, "ecolor": "#20252B"},
        )
        axis.set_title(title)
        axis.set_xticks(list(positions), labels)
        if column == "eo_signed":
            extent = (means + stds).max()
            extent = max(extent, -(means - stds).min(), 0.001) * 1.2
            axis.set_ylim(-extent, extent)
            axis.axhline(0, color="#20252B", linewidth=0.8)
        else:
            axis.set_ylim(bottom=0)
        axis.grid(axis="y", color="#D9DDE2", linewidth=0.8)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.yaxis.set_major_formatter(lambda value, _: f"{value:.3f}")

    axes[0].set_ylabel("Mean across folds (error bars: +/- 1 SD)")
    fig.suptitle(args.title, y=1.02)
    fig.tight_layout()

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {args.output_file}")


if __name__ == "__main__":
    main()