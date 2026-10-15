"""Combine weighted test experiments and compute additive signed-EO components."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


CONDITIONS = ("none", "activity", "eccentricity", "both")
CONSTRUCTION_LABELS = {
    "gender": "Gender",
    "age": "Age",
    "artificial_activity": "Induced activity",
    "artificial_eccentricity": "Induced eccentricity",
    "placebo": "Placebo",
}
PREDICTOR_ORDER = ("mf", "random_uniform", "user_item_average")
PREDICTOR_LABELS = {
    "mf": "MF",
    "random_uniform": "Uniform random",
    "user_item_average": "User--item average",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("."))
    parser.add_argument(
        "--output-csv", type=Path, default=Path("mf_test_decomposition.csv")
    )
    parser.add_argument(
        "--output-tex", type=Path, default=Path("mf_test_decomposition.tex")
    )
    parser.add_argument(
        "--legacy-sign",
        action="store_true",
        help="Negate EO for CSVs written with the old RMSE_A - RMSE_B convention.",
    )
    return parser.parse_args()


def load_results(input_dir, legacy_sign=False):
    frames = []
    for construction in CONSTRUCTION_LABELS:
        if construction == "placebo":
            # Only produced by run_joint_decomposition.py.
            continue
        if construction == "artificial_activity":
            candidates = (
                input_dir / "mf_balanced_test_artificial_activity_300_400_sd200.csv",
                input_dir / "mf_balanced_test_artificial_activity_decomposition.csv",
            )
        else:
            candidates = (input_dir / f"mf_balanced_test_{construction}_decomposition.csv",)
        path = next((candidate for candidate in candidates if candidate.exists()), None)
        if path is None:
            raise FileNotFoundError(
                f"No result file found for {construction}: {candidates}"
            )
        frame = pd.read_csv(path)
        if legacy_sign:
            # CSVs written before the sign change used EO = RMSE_A - RMSE_B.
            frame["eo_rmse"] = -frame["eo_rmse"]
        frame["construction"] = construction
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def decompose(frame):
    signed = frame.pivot_table(
        index=["construction", "predictor", "fold"],
        columns="condition",
        values="eo_rmse",
        aggfunc="first",
    ).reset_index()
    missing = set(CONDITIONS) - set(signed.columns)
    if missing:
        raise ValueError(f"Missing test conditions: {sorted(missing)}")
    signed["global"] = signed["none"]
    signed["activity_component"] = (
        (signed["none"] - signed["activity"])
        + (signed["eccentricity"] - signed["both"])
    ) / 2
    signed["eccentricity_component"] = (
        (signed["none"] - signed["eccentricity"])
        + (signed["activity"] - signed["both"])
    ) / 2
    signed["other_component"] = signed["both"]
    signed["reconstructed_global"] = signed[
        ["activity_component", "eccentricity_component", "other_component"]
    ].sum(axis=1)
    signed["construction_label"] = signed["construction"].map(CONSTRUCTION_LABELS)
    signed["construction"] = pd.Categorical(
        signed["construction"],
        categories=list(CONSTRUCTION_LABELS),
        ordered=True,
    )
    signed["predictor"] = pd.Categorical(
        signed["predictor"], categories=PREDICTOR_ORDER, ordered=True
    )
    return signed


def corrected_table_values(decomposed, columns):
    """Means, corrected 95% half-widths and p-values per construction and predictor."""

    grouped = decomposed.groupby(
        ["construction", "construction_label", "predictor"], observed=True
    )[list(columns)]
    means = grouped.mean()
    folds = grouped.size()
    # Corrected resampled t-test (Nadeau and Bengio): K-fold CV has a
    # test-to-training ratio of 1/(K-1).
    se = grouped.std().mul(np.sqrt(1 / folds + 1 / (folds - 1)), axis=0)
    half_widths = se.mul(stats.t.ppf(0.975, folds - 1), axis=0)
    p_values = pd.DataFrame(
        2 * stats.t.sf((means / se).abs(), np.asarray(folds - 1)[:, None]),
        index=means.index,
        columns=means.columns,
    )
    return means, half_widths, p_values


def write_latex_summary(decomposed, output_path, label_free=None):
    """Write the test-side table.

    ``label_free`` is the decomposition with label-free eccentricity strata; its
    eccentricity and residual terms are added as two further columns.
    """

    columns = (
        "global",
        "activity_component",
        "eccentricity_component",
        "other_component",
    )
    means, half_widths, p_values = corrected_table_values(decomposed, columns)
    if label_free is not None:
        free_columns = ("eccentricity_component", "other_component")
        free_means, free_half, free_p = corrected_table_values(label_free, free_columns)

    def format_value(value):
        return f"{value:.3f}".replace("-0.000", "0.000")

    def cell(mean, half, p):
        mark = r"$^{*}$" if p < 0.05 else ""
        return f"{format_value(mean)}{mark} $\\pm$ {format_value(half)}"

    extra = 2 if label_free is not None else 0
    lines = [
        r"\begin{table}[t]",
        r"    \centering",
        r"    \caption{Test-side decomposition of the signed EO of each model trained on the full fold. Positive EO favors group A. The last two columns repeat the eccentricity and residual terms with label-free eccentricity strata (the user's training eccentricity). Means over ten folds $\pm$ 95\% confidence half-widths from the corrected resampled $t$-test; $^{*}$ marks $p<0.05$.}"
        if label_free is not None else
        r"    \caption{Test-side decomposition of the signed EO of each model trained on the full fold. Positive EO favors group A. Means over ten folds $\pm$ 95\% confidence half-widths from the corrected resampled $t$-test; $^{*}$ marks $p<0.05$.}",
        r"    \label{tab:test-decomposition}",
        r"    \resizebox{\linewidth}{!}{%",
        r"    \begin{tabular}{@{}ll" + "r" * (4 + extra) + r"@{}}",
        r"        \toprule",
        r"        \multirow{2}{*}{Construction} & \multirow{2}{*}{Predictor} & \multirow{2}{*}{Global EO} & \multicolumn{3}{c}{Decomposition of global EO}"
        + (r" & \multicolumn{2}{c}{Label-free strata} \\" if label_free is not None else r" \\"),
        r"        & & & Activity & Eccentricity & Other"
        + (r" & Eccentricity & Other \\" if label_free is not None else r" \\"),
        r"        \midrule",
    ]
    previous_label = None
    labels = means.index.get_level_values("construction_label")
    for key in means.index:
        _, raw_label, predictor = key
        values = [
            cell(means.loc[key, c], half_widths.loc[key, c], p_values.loc[key, c])
            for c in columns
        ]
        if label_free is not None:
            values += [
                cell(free_means.loc[key, c], free_half.loc[key, c], free_p.loc[key, c])
                for c in free_columns
            ]
        if raw_label == previous_label:
            label = ""
        else:
            if previous_label is not None:
                lines.append(r"        \cline{1-2}")
            rows = int((labels == raw_label).sum())
            label = f"\\multirow{{{rows}}}{{*}}{{{raw_label.replace(' ', '~')}}}"
            previous_label = raw_label
        name = PREDICTOR_LABELS.get(predictor, predictor.replace("_", " "))
        lines.append("        " + " & ".join([label, name, *values]) + r" \\")
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
    decomposed = decompose(load_results(args.input_dir, args.legacy_sign))
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_tex.parent.mkdir(parents=True, exist_ok=True)
    decomposed.to_csv(args.output_csv, index=False)
    write_latex_summary(decomposed, args.output_tex)
    max_error = (
        decomposed["global"] - decomposed["reconstructed_global"]
    ).abs().max()
    print(f"Saved {len(decomposed):,} fold-level decompositions to {args.output_csv}")
    print(f"Saved LaTeX summary to {args.output_tex}")
    print(f"Maximum reconstruction error: {max_error:.3e}")


if __name__ == "__main__":
    main()