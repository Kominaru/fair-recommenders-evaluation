"""Create a contribution graphic for hidden-factor-aware EO analysis."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle


COLORS = {
    "ink": "#263238",
    "muted": "#607D8B",
    "paper": "#F7F4EE",
    "activity": "#D97745",
    "eccentricity": "#3B7A78",
    "other": "#7A7185",
    "lens": "#DCEBE7",
}


def rounded_panel(axis, xy, width, height, facecolor):
    axis.add_patch(
        FancyBboxPatch(
            xy,
            width,
            height,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            linewidth=1.0,
            edgecolor=COLORS["ink"],
            facecolor=facecolor,
        )
    )


def draw_activity_icon(axis, center):
    x, y = center
    for offset, height in zip((-0.045, 0.0, 0.045), (0.06, 0.12, 0.18)):
        axis.add_patch(Rectangle((x + offset - 0.014, y - 0.08), 0.028, height, color=COLORS["activity"]))
    axis.plot([x - 0.075, x + 0.075], [y - 0.08, y - 0.08], color=COLORS["ink"], linewidth=1)


def draw_eccentricity_icon(axis, center):
    x, y = center
    axis.add_patch(Circle((x, y), 0.075, fill=False, linewidth=1.4, edgecolor=COLORS["eccentricity"]))
    axis.add_patch(Circle((x + 0.035, y + 0.025), 0.018, color=COLORS["eccentricity"]))
    axis.plot([x, x + 0.035], [y, y + 0.025], color=COLORS["eccentricity"], linewidth=1)


def draw_other_icon(axis, center):
    x, y = center
    axis.add_patch(Circle((x, y), 0.07, fill=False, linewidth=1.4, edgecolor=COLORS["other"]))
    axis.text(x, y - 0.005, "?", ha="center", va="center", fontsize=18, color=COLORS["other"])


def draw_decomposition(axis):
    x, y, width, height = 0.62, 0.28, 0.29, 0.20
    values = (("Activity", 0.32, COLORS["activity"]), ("Eccentricity", 0.39, COLORS["eccentricity"]), ("Other", 0.29, COLORS["other"]))
    left = x
    for label, share, color in values:
        segment_width = width * share
        axis.add_patch(Rectangle((left, y), segment_width, height, color=color, linewidth=0))
        axis.text(left + segment_width / 2, y + height / 2, label, ha="center", va="center", fontsize=9, color="white")
        left += segment_width
    axis.text(x + width / 2, y + height + 0.075, "Observed signed EO", ha="center", va="center", fontsize=13, fontweight="bold")
    axis.text(x + width / 2, y - 0.055, "not one uniquely identified cause", ha="center", va="center", fontsize=9, color=COLORS["muted"])


def main():
    output = Path("figures/contributions_pipeline.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(10, 4.8), facecolor=COLORS["paper"])
    axis.set_facecolor(COLORS["paper"])
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    axis.text(0.50, 0.94, "A hidden-factor-aware view of group error", ha="center", va="center", fontsize=16, fontweight="bold", color=COLORS["ink"])
    axis.text(0.50, 0.875, "The same checkpoint is evaluated after the test distribution is made comparable.", ha="center", va="center", fontsize=10, color=COLORS["muted"])

    rounded_panel(axis, (0.045, 0.18), 0.25, 0.59, "#E8EEF2")
    axis.text(0.17, 0.71, "Hidden factors", ha="center", va="center", fontsize=13, fontweight="bold", color=COLORS["ink"])
    for center, title, subtitle, drawer in (
        ((0.105, 0.56), "Activity", "how much evidence", draw_activity_icon),
        ((0.235, 0.56), "Eccentricity", "how atypical", draw_eccentricity_icon),
        ((0.17, 0.29), "Other", "representation + group-specific", draw_other_icon),
    ):
        drawer(axis, center)
        axis.text(center[0], center[1] - 0.13, title, ha="center", va="center", fontsize=10, fontweight="bold", color=COLORS["ink"])
        axis.text(center[0], center[1] - 0.17, subtitle, ha="center", va="center", fontsize=8, color=COLORS["muted"])

    axis.add_patch(Circle((0.47, 0.48), 0.14, facecolor=COLORS["lens"], edgecolor=COLORS["ink"], linewidth=1.2))
    axis.add_patch(Circle((0.47, 0.48), 0.08, fill=False, edgecolor=COLORS["ink"], linewidth=1.2))
    axis.plot([0.525, 0.59], [0.425, 0.36], color=COLORS["ink"], linewidth=3, solid_capstyle="round")
    axis.text(0.47, 0.48, "fixed\ncheckpoint", ha="center", va="center", fontsize=10, fontweight="bold", color=COLORS["ink"])
    axis.text(0.47, 0.22, "common target\ndistributions", ha="center", va="center", fontsize=9, color=COLORS["muted"])
    draw_decomposition(axis)
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved figure to {output}")


if __name__ == "__main__":
    main()
