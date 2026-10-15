"""Shapley split of train-side EO into representation, activity and eccentricity.

Each training condition aligns a subset S of three factors between groups A
and B, all at the same rating count:

    r  representation: the A:B user ratio is set to 1:1;
    a  activity: within-group user distributions over activity strata match;
    e  eccentricity: within-group user distributions over eccentricity strata
       match.

With v(S) the EO of the model trained under S, the contribution of factor i is

    Phi_i = sum_{S not containing i} |S|!(n-|S|-1)!/n! [v(S) - v(S + {i})],

and Phi_o = v({r, a, e}) is the residual, so EO(T_0) = sum_i Phi_i + Phi_o.
With n = 2 and no representation factor this is the test-side split in
decompose_test_results.py.
"""

import argparse
from itertools import combinations
from math import factorial
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from decompose_test_results import CONSTRUCTION_LABELS as TEST_CONSTRUCTION_LABELS


CONSTRUCTION_LABELS = {**TEST_CONSTRUCTION_LABELS, "placebo": "Placebo", "placebo_age": "Placebo (age sizes)"}
FACTORS = ("representation", "activity", "eccentricity")
FACTOR_LABELS = {
    "representation": "Representation",
    "activity": "Activity",
    "eccentricity": "Eccentricity",
}
# Aligned factors -> balance condition in balanced_subset.CONDITIONS.
LATTICE = {
    frozenset(): "none",
    frozenset({"representation"}): "group",
    frozenset({"activity"}): "activity_aligned",
    frozenset({"eccentricity"}): "eccentricity_aligned",
    frozenset({"activity", "eccentricity"}): "activity_eccentricity_aligned",
    frozenset({"representation", "activity"}): "activity_group",
    frozenset({"representation", "eccentricity"}): "eccentricity_group",
    frozenset(FACTORS): "all",
}
CONDITION_ORDER = list(LATTICE.values())
CONDITION_SYMBOLS = {
    condition: "$\\mathcal T_{"
    + ("".join(factor[0] for factor in FACTORS if factor in aligned) or "0")
    + "}$"
    for aligned, condition in LATTICE.items()
}
# The resized control aligns nothing; T_r would clash with representation.
CONDITION_SYMBOLS["none"] = r"$\mathcal T_\varnothing$"
COMPONENTS = ("global", *(f"{factor}_component" for factor in FACTORS), "other_component")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "results",
        type=Path,
        nargs="+",
        help="Per-fold CSVs written by run_mf_balanced_conditions.py.",
    )
    parser.add_argument(
        "--output-csv", type=Path, default=Path("mf_train_decomposition.csv")
    )
    parser.add_argument(
        "--output-tex", type=Path, default=Path("mf_train_decomposition.tex")
    )
    parser.add_argument(
        "--allow-fallback",
        action="store_true",
        help="Keep folds where a condition silently fell back to T_b.",
    )
    return parser.parse_args()


def load_results(paths, allow_fallback):
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        if "construction" not in frame:
            # Runs before --group-by existed always used gender.
            frame["construction"] = "gender"
        frames.append(frame)
    frame = pd.concat(frames, ignore_index=True)
    frame = frame[frame["condition"].isin(CONDITION_ORDER)].copy()
    complete = frame.groupby(["construction", "fold"])["condition"].nunique()
    incomplete = complete[complete < len(CONDITION_ORDER)]
    if len(incomplete):
        raise ValueError(
            "Every fold needs all eight conditions "
            f"({', '.join(CONDITION_ORDER)}); incomplete:\n{incomplete}"
        )
    fallback = frame["fallback"].astype(bool)
    if fallback.any() and not allow_fallback:
        rows = frame.loc[fallback, ["construction", "condition", "fold"]]
        raise ValueError(
            "These runs fell back to the fully balanced subset, so they do "
            f"not measure their own condition:\n{rows.to_string(index=False)}"
        )
    if "evaluation" in frame and frame["evaluation"].nunique() > 1:
        raise ValueError("Results mix standardized and per-condition evaluation.")
    unknown = set(frame["construction"]) - set(CONSTRUCTION_LABELS)
    if unknown:
        raise ValueError(f"Unknown constructions: {sorted(unknown)}")
    return frame


def shapley(values):
    """Return each factor's Shapley contribution from v(S) for all subsets S."""

    n = len(FACTORS)
    contributions = {}
    for factor in FACTORS:
        others = [other for other in FACTORS if other != factor]
        total = 0.0
        for size in range(n):
            weight = factorial(size) * factorial(n - size - 1) / factorial(n)
            for subset in combinations(others, size):
                aligned = frozenset(subset)
                total += weight * (
                    values[LATTICE[aligned]] - values[LATTICE[aligned | {factor}]]
                )
        contributions[f"{factor}_component"] = total
    contributions["other_component"] = values[LATTICE[frozenset(FACTORS)]]
    return contributions


def decompose(frame):
    signed = frame.pivot_table(
        index=["construction", "fold"],
        columns="condition",
        values="eo_rmse",
        aggfunc="first",
    ).reset_index()
    components = pd.DataFrame(
        [shapley(row) for _, row in signed.iterrows()], index=signed.index
    )
    signed = pd.concat([signed, components], axis=1)
    signed["global"] = signed["none"]
    signed["reconstructed_global"] = signed[list(COMPONENTS[1:])].sum(axis=1)
    signed["construction"] = pd.Categorical(
        signed["construction"], categories=list(CONSTRUCTION_LABELS), ordered=True
    )
    return signed.sort_values(["construction", "fold"])


def summarize_conditions(frame):
    summary = (
        frame.groupby(["construction", "condition"])[["eo_rmse", "rmse_A", "rmse_B"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary["construction"] = pd.Categorical(
        summary["construction"], categories=list(CONSTRUCTION_LABELS), ordered=True
    )
    summary["condition"] = pd.Categorical(
        summary["condition"], categories=CONDITION_ORDER, ordered=True
    )
    return summary.sort_values(["construction", "condition"])


def signed(value, digits=3):
    """Format a value without printing a negative zero."""

    return f"{round(value, digits) + 0.0:.{digits}f}"


def format_cell(row, column):
    return f"{signed(row[(column, 'mean')])} $\\pm$ {signed(row[(column, 'std')])}"


def aligned_marks(condition):
    aligned = next(key for key, value in LATTICE.items() if value == condition)
    return [r"\checkmark" if factor in aligned else "" for factor in FACTORS]


def corrected_cell(values):
    """Mean and corrected resampled-t 95% half-width over K folds."""

    n = len(values)
    se = values.std() * np.sqrt(1 / n + 1 / (n - 1))
    mean = values.mean()
    p = 2 * stats.t.sf(abs(mean / se), n - 1) if se > 0 else np.nan
    mark = r"$^{*}$" if p < 0.05 else ""
    return f"{signed(mean)}{mark} $\\pm$ {signed(stats.t.ppf(0.975, n - 1) * se)}"


def write_latex(conditions, decomposed, output_path, conditions_path=None):
    """Write the condition table and the decomposition table.

    With ``conditions_path`` the condition table goes to its own file.
    """

    lines = [
        r"\begin{table}[t]",
        r"    \centering",
        r"    \caption{Training conditions at matched rating count. A check "
        r"marks each factor aligned between groups. All models are evaluated "
        r"on the same standardized test population; values are means $\pm$ "
        r"standard deviations across folds (not confidence intervals; those "
        r"of the derived components are in Table~\ref{tab:training-decomposition}).}",
        r"    \label{tab:training-results}",
        r"    \resizebox{\linewidth}{!}{%",
        r"    \begin{tabular}{@{}llcccrrr@{}}",
        r"        \toprule",
        r"        Construction & Condition & Repr. & Act. & Ecc. & EO & "
        r"RMSE A & RMSE B \\",
        r"        \midrule",
    ]
    for index, (construction, rows) in enumerate(
        conditions.groupby("construction", observed=True)
    ):
        if index:
            lines.append(r"        \midrule")
        label = CONSTRUCTION_LABELS[construction].replace(" ", "~")
        label = f"\\multirow{{{len(rows)}}}{{*}}{{{label}}}"
        for position, (_, row) in enumerate(rows.iterrows()):
            condition = row[("condition", "")]
            cells = [
                label if position == 0 else "",
                CONDITION_SYMBOLS[condition],
                *aligned_marks(condition),
                format_cell(row, "eo_rmse"),
                format_cell(row, "rmse_A"),
                format_cell(row, "rmse_B"),
            ]
            lines.append("        " + " & ".join(cells) + r" \\")
    lines.extend(
        [
            r"        \bottomrule",
            r"    \end{tabular}%",
            r"    }",
            r"\end{table}",
            "",
        ]
    )

    if conditions_path is not None:
        conditions_path.write_text("\n".join(lines), encoding="utf-8")
        lines = []
    factor_headers = " & ".join(FACTOR_LABELS[factor] for factor in FACTORS)
    lines.extend(
        [
            r"\begin{table}[t]",
            r"    \centering",
            r"    \caption{Train-side signed EO decomposition of the resized "
            r"control, with every condition scored on the jointly balanced "
            r"test population $\mathcal C_{ae}$. Positive EO favors group A. "
            r"Means over ten folds $\pm$ 95\% confidence half-widths from the "
            r"corrected resampled $t$-test; $^{*}$ marks $p<0.05$.}",
            r"    \label{tab:training-decomposition}",
            r"    \resizebox{\linewidth}{!}{%",
            r"    \begin{tabular}{@{}lrrrrr@{}}",
            r"        \toprule",
            r"        \multirow{2}{*}{Construction} & "
            r"\multirow{2}{*}{$\mathrm{EO}(\mathcal T_\varnothing)$} & "
            r"\multicolumn{4}{c}{Decomposition of $\mathrm{EO}(\mathcal T_\varnothing)$} \\",
            f"        & & {factor_headers} & Other \\\\",
            r"        \midrule",
        ]
    )
    for construction, rows in decomposed.groupby("construction", observed=True):
        cells = [CONSTRUCTION_LABELS[construction]]
        cells += [corrected_cell(rows[component]) for component in COMPONENTS]
        lines.append("        " + " & ".join(cells) + r" \\")
    lines.extend(
        [
            r"        \bottomrule",
            r"    \end{tabular}%",
            r"    }",
            r"\end{table}",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    frame = load_results(args.results, args.allow_fallback)
    decomposed = decompose(frame)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_tex.parent.mkdir(parents=True, exist_ok=True)
    decomposed.to_csv(args.output_csv, index=False)
    write_latex(summarize_conditions(frame), decomposed, args.output_tex)
    max_error = (
        decomposed["global"] - decomposed["reconstructed_global"]
    ).abs().max()
    print(f"Saved {len(decomposed):,} fold-level decompositions to {args.output_csv}")
    print(f"Saved LaTeX tables to {args.output_tex}")
    print(f"Maximum reconstruction error: {max_error:.3e}")


if __name__ == "__main__":
    main()
