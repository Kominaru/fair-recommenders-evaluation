"""Decompose group-specific RMSE across test-side weighting conditions."""

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
GROUPS = ("A", "B")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute group-wise Shapley RMSE decompositions."
    )
    parser.add_argument("--input-dir", type=Path, default=Path("."))
    parser.add_argument(
        "--output-csv", type=Path, default=Path("group_rmse_decomposition.csv")
    )
    parser.add_argument(
        "--output-tex", type=Path, default=Path("group_rmse_decomposition.tex")
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
            candidates = (
                input_dir / f"mf_balanced_test_{construction}_decomposition.csv",
            )
        path = next((candidate for candidate in candidates if candidate.exists()), None)
        if path is None:
            raise FileNotFoundError(
                f"No result file found for {construction}: {candidates}"
            )
        frame = pd.read_csv(path)
        required = {"condition", "fold", "predictor", "rmse_A", "rmse_B"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        frame["construction"] = construction
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def decompose(frame):
    index = ["construction", "predictor", "fold"]
    decomposed = []
    for group in GROUPS:
        values = frame.pivot_table(
            index=index,
            columns="condition",
            values=f"rmse_{group}",
            aggfunc="first",
        ).reset_index()
        missing = set(CONDITIONS) - set(values.columns)
        if missing:
            raise ValueError(
                f"Missing conditions for group {group}: {sorted(missing)}"
            )
        values["group"] = group
        values["starting_rmse"] = values["none"]
        values["activity_component"] = (
            (values["none"] - values["activity"])
            + (values["eccentricity"] - values["both"])
        ) / 2
        values["eccentricity_component"] = (
            (values["none"] - values["eccentricity"])
            + (values["activity"] - values["both"])
        ) / 2
        values["remaining_rmse_component"] = values["both"]
        values["reconstructed_starting_rmse"] = values[
            [
                "activity_component",
                "eccentricity_component",
                "remaining_rmse_component",
            ]
        ].sum(axis=1)
        decomposed.append(values)

    result = pd.concat(decomposed, ignore_index=True)
    result["construction_label"] = result["construction"].map(CONSTRUCTION_LABELS)
    result["construction"] = pd.Categorical(
        result["construction"],
        categories=list(CONSTRUCTION_LABELS),
        ordered=True,
    )
    result["predictor"] = pd.Categorical(
        result["predictor"], categories=PREDICTOR_ORDER, ordered=True
    )
    result["group"] = pd.Categorical(result["group"], categories=GROUPS, ordered=True)
    return result


def write_latex_summary(decomposed, output_path):
    summary = (
        decomposed.groupby(
            ["construction", "construction_label", "predictor", "group"]
        )[
            [
                "starting_rmse",
                "activity_component",
                "eccentricity_component",
                "remaining_rmse_component",
            ]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )

    def format_value(value, signed=False):
        return f"{value:{'+' if signed else ''}.3f}"

    lines = [
        r"\begin{table}[H]",
        r"    \centering",
        r"    \caption{Group-wise RMSE decomposition across ten folds. The starting RMSE is decomposed into activity, eccentricity, and remaining RMSE contributions using the same Shapley construction as the EO decomposition. Positive contributions indicate increases in RMSE; negative contributions indicate decreases. Values are means $\pm$ standard deviations.}",
        r"    \label{tab:group-rmse-decomposition}",
        r"    \resizebox{\linewidth}{!}{%",
        r"    \begin{tabular}{@{}lllrrrr@{}}",
        r"        \toprule",
        r"        Construction & Predictor & Group & Starting RMSE & Activity & Eccentricity & Remaining RMSE \\",
        r"        \midrule",
    ]
    previous_label = None
    previous_predictor = None
    first_group = True
    for _, row in summary.iterrows():
        raw_label = row[("construction_label", "")]
        raw_predictor = row[("predictor", "")]
        raw_group = row[("group", "")]
        label = raw_label.replace(" ", "~")
        predictor = {
            "mf": "MF",
            "random_uniform": "random uniform",
            "user_item_average": "user/item average",
        }[raw_predictor]
        if raw_label == previous_label:
            label = ""
        else:
            if not first_group:
                lines.append(r"        \cline{1-3}")
            construction_rows = int(
                (summary[("construction_label", "")] == raw_label).sum()
            )
            label = f"\\multirow{{{construction_rows}}}{{*}}{{{label}}}"
            previous_label = raw_label
            previous_predictor = None
            first_group = False
        if raw_predictor == previous_predictor:
            predictor = ""
        previous_predictor = raw_predictor
        values = []
        for component in (
            "starting_rmse",
            "activity_component",
            "eccentricity_component",
            "remaining_rmse_component",
        ):
            signed = component in {"activity_component", "eccentricity_component"}
            values.append(
                f"{format_value(row[(component, 'mean')], signed=signed)} "
                f"$\\pm$ {format_value(row[(component, 'std')])}"
            )
        lines.append(
            "        "
            + " & ".join([label, predictor, raw_group, *values])
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
        decomposed["starting_rmse"] - decomposed["reconstructed_starting_rmse"]
    ).abs().max()
    print(f"Saved {len(decomposed):,} fold-level group decompositions to {args.output_csv}")
    print(f"Saved LaTeX summary to {args.output_tex}")
    print(f"Maximum reconstruction error: {max_error:.3e}")


if __name__ == "__main__":
    main()
