"""Combine weighted test experiments and compute additive signed-EO components."""

import argparse
from pathlib import Path

import pandas as pd


CONDITIONS = ("none", "activity", "eccentricity", "both")
CONSTRUCTION_LABELS = {
    "gender": "Gender",
    "age": "Age",
    "artificial_activity": "Artificial activity",
    "artificial_eccentricity": "Artificial eccentricity",
}
PREDICTOR_ORDER = ("mf", "random_uniform", "user_item_average")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("."))
    parser.add_argument(
        "--output-csv", type=Path, default=Path("mf_test_decomposition.csv")
    )
    parser.add_argument(
        "--output-tex", type=Path, default=Path("mf_test_decomposition.tex")
    )
    return parser.parse_args()


def load_results(input_dir):
    frames = []
    for construction in CONSTRUCTION_LABELS:
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
        # Existing CSVs were generated with EO = RMSE_A - RMSE_B. Convert
        # them to the report convention: positive EO favors group A.
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


def write_latex_summary(decomposed, output_path):
    summary = (
        decomposed.groupby(["construction", "construction_label", "predictor"])[
            [
                "global",
                "activity_component",
                "eccentricity_component",
                "other_component",
            ]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )

    def format_value(value):
        return f"{value:.3f}"

    lines = [
        r"\begin{table}[H]",
        r"    \centering",
        r"    \caption{Signed EO decomposition across ten folds. Positive EO favors group A. Values are means $\pm$ standard deviations.}",
        r"    \label{tab:test-decomposition}",
        r"    \resizebox{\linewidth}{!}{%",
        r"    \begin{tabular}{@{}llrrrr@{}}",
        r"        \toprule",
        r"        \multirow{2}{*}{Construction} & \multirow{2}{*}{Predictor} & \multirow{2}{*}{Global EO} & \multicolumn{3}{c}{Decomposition of global EO} \\",
        r"        & & & Activity & Eccentricity & Other \\",
        r"        \midrule",
    ]
    previous_label = None
    first_group = True
    for _, row in summary.iterrows():
        values = []
        for component in (
            "global",
            "activity_component",
            "eccentricity_component",
            "other_component",
        ):
            values.append(
                f"{format_value(row[(component, 'mean')])} "
                f"$\\pm$ {format_value(row[(component, 'std')])}"
            )
        raw_label = row[("construction_label", "")]
        label = raw_label.replace(" ", "~")
        if raw_label == previous_label:
            label = ""
        else:
            if not first_group:
                lines.append(r"        \cline{1-2}")
            construction_rows = int(
                (summary[("construction_label", "")] == raw_label).sum()
            )
            label = f"\\multirow{{{construction_rows}}}{{*}}{{{label}}}"
            previous_label = raw_label
            first_group = False
        predictor = row[("predictor", "")].replace("_", " ")
        if predictor == "mf":
            predictor = "MF"
        lines.append(
            "        "
            + " & ".join([label, predictor, *values])
            + r" \\"
        )
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
    decomposed = decompose(load_results(args.input_dir))
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