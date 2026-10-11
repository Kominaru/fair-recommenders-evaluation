"""Plot additive signed-EO components for all predictors and constructions."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CONSTRUCTION_ORDER = [
    "Gender",
    "Age",
    "Artificial activity",
    "Artificial eccentricity",
]
PREDICTOR_ORDER = ["mf", "random_uniform", "user_item_average"]
PREDICTOR_LABELS = {
    "mf": "MF",
    "random_uniform": "Uniform random",
    "user_item_average": "User/item average",
}
COMPONENTS = [
    ("activity_component", "Activity"),
    ("eccentricity_component", "Eccentricity"),
    ("other_component", "Other"),
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-file", type=Path, default=Path("mf_test_decomposition.csv"))
    parser.add_argument(
        "--output-file", type=Path, default=Path("figures/mf_test_decomposition.png")
    )
    return parser.parse_args()


def main():
    args = parse_args()
    results = pd.read_csv(args.results_file)
    summary = results.groupby(["construction_label", "predictor"])[
        [component for component, _ in COMPONENTS]
    ].mean()
    positions = np.arange(len(COMPONENTS))
    width = 0.24
    colors = ["#D97745", "#8C8C8C", "#8C8C8C"]
    hatches = ["", "//", ".."]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2), sharey=False)
    axes = axes.ravel()
    for axis, construction in zip(axes, CONSTRUCTION_ORDER):
        panel_values = []
        for index, predictor in enumerate(PREDICTOR_ORDER):
            values = [
                summary.loc[(construction, predictor), component]
                for component, _ in COMPONENTS
            ]
            panel_values.extend(values)
            axis.bar(
                positions + (index - 1) * width,
                values,
                width=width,
                label=PREDICTOR_LABELS[predictor],
                color=colors[index],
                hatch=hatches[index],
                alpha=0.8,
                edgecolor="#20252B",
                linewidth=0.5,
            )
        limit = max(abs(value) for value in panel_values) * 1.15
        axis.set_ylim(-limit, limit)
        axis.set_title(construction)
        axis.axhline(0, color="#20252B", linewidth=0.8)
        axis.set_xticks(positions, [title for _, title in COMPONENTS])
        axis.grid(axis="y", color="#D9DDE2", linewidth=0.8)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.yaxis.set_major_formatter(lambda value, _: f"{value:.3f}")
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle("Additive decomposition of observed signed EO", y=0.99)
    row_label = (
        "Group A                         Group B\n"
        "favored       \u27f5 EO \u27f6       favored"
    )
    for axis in (axes[0], axes[2]):
        axis.set_ylabel(row_label, rotation=90, labelpad=18, va="center")
    fig.subplots_adjust(left=0.18, right=0.98, top=0.91, bottom=0.10, hspace=0.32)
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {args.output_file}")


if __name__ == "__main__":
    main()