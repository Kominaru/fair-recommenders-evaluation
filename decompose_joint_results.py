"""Five-player Shapley split of the measured EO into train and test disparities.

Players are the train-side disparities r^tr (representation), a^tr (activity),
e^tr (eccentricity) and the test-side disparities a^te and e^te. With
v(S) = EO of the model trained on T_{S cap train} and scored on test population
C_{S cap test},

    EO_obs = Phi_size + sum_i Phi_i + Phi_o,

where Phi_i is player i's Shapley value, Phi_o = v(all players) is the gap
left when every disparity is aligned, and Phi_size = EO_obs - v(empty set) is
the effect of training on the common reduced size instead of the full fold.
Seeds are averaged within each cell before decomposing.

Uncertainty uses the corrected resampled t-test of Nadeau and Bengio (2003):
cross-validation folds share training data, so the fold variance is inflated
by n_test / n_train before forming confidence intervals.
"""

import argparse
from itertools import combinations
from math import factorial
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import decompose_group_rmse
import decompose_test_results
import decompose_train_results
from decompose_train_results import CONSTRUCTION_LABELS, LATTICE, signed


TRAIN_PLAYERS = {"r_T": "representation", "a_T": "activity", "e_T": "eccentricity"}
TEST_PLAYERS = {"a_V": "activity", "e_V": "eccentricity"}
PLAYERS = (*TRAIN_PLAYERS, *TEST_PLAYERS)
POPULATIONS = {
    frozenset(): "none",
    frozenset({"a_V"}): "activity",
    frozenset({"e_V"}): "eccentricity",
    frozenset({"a_V", "e_V"}): "both",
}
PLAYER_LABELS = {
    "r_T": r"$\Phi_r^{\mathrm{tr}}$",
    "a_T": r"$\Phi_a^{\mathrm{tr}}$",
    "e_T": r"$\Phi_e^{\mathrm{tr}}$",
    "a_V": r"$\Phi_a^{\mathrm{te}}$",
    "e_V": r"$\Phi_e^{\mathrm{te}}$",
}
INTERACTIONS = (("a_T", "a_V"), ("e_T", "e_V"))
COMPONENT_ROWS = (
    ("eo_observed", r"$\mathrm{EO}_{\mathrm{obs}}$"),
    ("size_component", r"$\Phi_{\mathrm{size}}$"),
    *((f"{player}_component", label) for player, label in PLAYER_LABELS.items()),
    ("other_component", r"$\Phi_o$"),
    ("interaction_a_T_a_V", r"$I(a^{\mathrm{tr}},a^{\mathrm{te}})$"),
    ("interaction_e_T_e_V", r"$I(e^{\mathrm{tr}},e^{\mathrm{te}})$"),
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path, nargs="+")
    parser.add_argument(
        "--output-csv", type=Path, default=Path("mf_joint_decomposition.csv")
    )
    parser.add_argument("--summary-csv", type=Path, default=None)
    parser.add_argument(
        "--output-tex", type=Path, default=Path("mf_joint_decomposition.tex")
    )
    parser.add_argument(
        "--baseline-tex",
        type=Path,
        default=None,
        help="Joint table for the user-item average baseline.",
    )
    parser.add_argument(
        "--train-tex",
        type=Path,
        default=None,
        help="Train-side tables of decompose_train_results.py, using MF on the "
        "jointly balanced test population.",
    )
    parser.add_argument(
        "--test-tex",
        type=Path,
        default=None,
        help="Test-side table (full training fold) in the format of "
        "decompose_test_results.py; its CSV goes next to it.",
    )
    parser.add_argument("--group-rmse-tex", type=Path, default=None)
    parser.add_argument(
        "--test-variant",
        default="rating10",
        help="Test populations to use: rating10 (main), user10 (label-free "
        "eccentricity) or rating50 (full training fold only).",
    )
    parser.add_argument(
        "--panel-correction",
        action="store_true",
        help="Subtract from every condition the EO change that the full model "
        "shows on that condition's evaluated test ratings.",
    )
    parser.add_argument(
        "--predictors",
        default="mf,user_item_average",
        help="Predictors to decompose; random_uniform has no training to vary.",
    )
    return parser.parse_args()


def cell(players):
    """Return the (training, population) pair that aligns ``players``."""

    players = frozenset(players)
    train = frozenset(TRAIN_PLAYERS[p] for p in players if p in TRAIN_PLAYERS)
    test = frozenset(p for p in players if p in TEST_PLAYERS)
    return LATTICE[train], POPULATIONS[test]


def weight(size, n):
    return factorial(size) * factorial(n - size - 1) / factorial(n)


def shapley(value, players=PLAYERS):
    """Shapley values of ``players`` with contributions v(S) - v(S + {i})."""

    n = len(players)
    result = {}
    for player in players:
        others = [other for other in players if other != player]
        result[player] = sum(
            weight(size, n)
            * (value(frozenset(subset)) - value(frozenset(subset) | {player}))
            for size in range(n)
            for subset in combinations(others, size)
        )
    return result


def interaction(value, first, second, players=PLAYERS):
    """Shapley interaction index of two players under the same convention."""

    n = len(players)
    others = [other for other in players if other not in (first, second)]
    total = 0.0
    for size in range(n - 1):
        scale = factorial(size) * factorial(n - size - 2) / factorial(n - 1)
        for subset in combinations(others, size):
            base = frozenset(subset)
            # Change in first's contribution once second is aligned:
            # negative = the two substitute for each other.
            total += scale * (
                value(base | {first}) + value(base | {second})
                - value(base) - value(base | {first, second})
            )
    return total


def load(paths):
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    if "test_variant" not in frame:
        # Runs before test variants existed used rating-level eccentricity.
        frame["test_variant"] = "rating10"
    if "panel" not in frame:
        frame["panel"] = frame["training"]
    unknown = set(frame["construction"]) - set(CONSTRUCTION_LABELS)
    if unknown:
        raise ValueError(f"Unknown constructions: {sorted(unknown)}")
    return frame


def cell_values(frame, variant, panel_correction=False):
    """Seed-averaged EO for every (training, population) cell.

    With ``panel_correction``, each condition's EO is reduced by the change
    the full model shows when scored on that condition's evaluable ratings,
    so that only the effect of training remains.
    """

    keys = ["construction", "predictor", "fold", "training", "population"]
    rows = frame[
        (frame["test_variant"] == variant)
        & frame["population"].isin(POPULATIONS.values())
    ]
    own = rows[rows["panel"] == rows["training"]].groupby(keys)["eo_rmse"].mean()
    if not panel_correction:
        return own
    full = rows[(rows["training"] == "full") & (rows["panel"] != "full")]
    on_panel = full.groupby(
        ["construction", "predictor", "fold", "panel", "population"]
    )["eo_rmse"].mean()
    on_panel.index = on_panel.index.set_names(keys)
    baseline = own.xs("full", level="training")
    shift = on_panel - baseline.reindex(on_panel.index.droplevel("training")).to_numpy()
    corrected = own.copy()
    common = corrected.index.intersection(shift.index)
    corrected.loc[common] = corrected.loc[common] - shift.loc[common]
    return corrected


def decompose(frame, variant="rating10", panel_correction=False, size_calibration="population"):
    """``size_calibration`` removes the effect of training on the common reduced
    size: ``population`` shifts every condition scored on a test population by
    the gap between the full model and the size-matched null model T_empty on
    that population; ``global`` shifts every condition by the one shift measured
    on the unaligned population; ``none`` keeps Phi_size as a separate term.
    """

    cells = cell_values(frame, variant, panel_correction)
    main = frame[(frame["test_variant"] == variant) & (frame["panel"] == frame["training"])]
    noise = (
        main[main["training"].isin(["full", "none"]) & (main["population"] == "none")]
        .groupby(["construction", "predictor", "fold", "training"])["eo_rmse"]
        .std()
        .groupby(["construction", "predictor", "fold"])
        .mean()
    )
    needed = {cell(subset) for size in range(len(PLAYERS) + 1)
              for subset in combinations(PLAYERS, size)}
    needed |= {("full", population) for population in POPULATIONS.values()}
    rows = []
    for (construction, predictor, fold), group in cells.groupby(
        ["construction", "predictor", "fold"]
    ):
        values = group.droplevel([0, 1, 2]).to_dict()
        missing = needed - set(values)
        if missing:
            raise ValueError(
                f"{construction} {predictor} fold {fold} lacks {sorted(missing)}"
            )

        observed = values[("full", "none")]
        size_shift = {
            population: values[("full", population)] - values[("none", population)]
            for population in POPULATIONS.values()
        }
        if size_calibration != "none":
            values = {
                (training, population): level
                + (
                    0.0
                    if training == "full"
                    else size_shift[population if size_calibration == "population" else "none"]
                )
                for (training, population), level in values.items()
            }

        def value(players):
            return values[cell(players)]

        contributions = shapley(value)
        row = {
            "construction": construction,
            "predictor": predictor,
            "fold": fold,
            "eo_observed": observed,
            "size_component": observed - value(frozenset()),
            **{f"{player}_component": contributions[player] for player in PLAYERS},
            "other_component": value(frozenset(PLAYERS)),
            **{
                f"interaction_{first}_{second}": interaction(value, first, second)
                for first, second in INTERACTIONS
            },
            # The size effect measured on each test population.
            **{f"size_{population}": shift for population, shift in size_shift.items()},
            "noise_sd": noise.get((construction, predictor, fold), float("nan")),
        }
        row["train_total"] = sum(row[f"{p}_component"] for p in TRAIN_PLAYERS)
        row["test_total"] = sum(row[f"{p}_component"] for p in TEST_PLAYERS)
        row["reconstructed"] = (
            row["size_component"] + row["train_total"] + row["test_total"]
            + row["other_component"]
        )
        test_only = shapley(
            lambda players: values[("full", POPULATIONS[frozenset(players)])],
            players=tuple(TEST_PLAYERS),
        )
        row.update({f"test_only_{p}": test_only[p] for p in TEST_PLAYERS})
        rows.append(row)
    return pd.DataFrame(rows)


def corrected_summary(decomposed, folds=None):
    """Mean, Nadeau-Bengio standard error, 95% interval and p-value."""

    numeric = [
        column for column in decomposed.columns
        if column not in {"construction", "predictor", "fold"}
    ]
    rows = []
    for (construction, predictor), group in decomposed.groupby(["construction", "predictor"]):
        n = len(group)
        k = folds or n
        factor = np.sqrt(1 / n + 1 / (k - 1))
        critical = stats.t.ppf(0.975, n - 1)
        for column in numeric:
            values = group[column].dropna()
            if values.empty:
                continue
            mean, sd = values.mean(), values.std()
            se = sd * factor
            t = mean / se if se > 0 else np.nan
            rows.append(
                {
                    "construction": construction,
                    "predictor": predictor,
                    "component": column,
                    "mean": mean,
                    "fold_sd": sd,
                    "corrected_se": se,
                    "ci95": critical * se,
                    "t": t,
                    "p": 2 * stats.t.sf(abs(t), n - 1) if np.isfinite(t) else np.nan,
                    "positive_folds": int((values > 0).sum()),
                    "folds": int(len(values)),
                }
            )
    return pd.DataFrame(rows)


def format_estimate(row):
    mark = r"$^{*}$" if row["p"] < 0.05 else ""
    return f"{signed(row['mean'])}{mark} $\\pm$ {signed(row['ci95'])}"


def write_latex(summary, predictor, output_path, label, caption):
    """Components as rows, constructions as columns."""

    summary = summary[summary["predictor"] == predictor]
    order = [c for c in CONSTRUCTION_LABELS if c in set(summary["construction"])]
    indexed = summary.set_index(["component", "construction"])
    lines = [
        r"\begin{table}[t]",
        r"    \centering",
        f"    \\caption{{{caption}}}",
        f"    \\label{{{label}}}",
        r"    \resizebox{\linewidth}{!}{%",
        r"    \begin{tabular}{@{}l" + "r" * len(order) + r"@{}}",
        r"        \toprule",
        "        Component & "
        + " & ".join(CONSTRUCTION_LABELS[c] for c in order) + r" \\",
        r"        \midrule",
    ]
    for column, row_label in COMPONENT_ROWS:
        if column == "size_component" or column.startswith("interaction_a"):
            lines.append(r"        \midrule")
        cells = [format_estimate(indexed.loc[(column, c)]) for c in order]
        lines.append(f"        {row_label} & " + " & ".join(cells) + r" \\")
    lines.extend([r"        \bottomrule", r"    \end{tabular}%", r"    }", r"\end{table}", ""])
    output_path.write_text("\n".join(lines), encoding="utf-8")


def full_model_frame(frame, variant="rating10"):
    """Full-fold rows in the format of balanced_test.py, seeds averaged."""

    rows = frame[
        (frame["training"] == "full")
        & (frame["panel"] == "full")
        & (frame["test_variant"] == variant)
        & frame["population"].isin(POPULATIONS.values())
    ]
    return (
        rows.groupby(["construction", "predictor", "fold", "population"])[
            ["eo_rmse", "rmse_A", "rmse_B"]
        ]
        .mean()
        .reset_index()
        .rename(columns={"population": "condition"})
    )


def train_side_frame(frame, variant="rating10"):
    """Rows in the format of run_mf_balanced_conditions.py for the train split.

    Every training condition is scored on the fixed, jointly balanced test
    population, i.e. the train-side split at v(S | {a_te, e_te}).
    """

    rows = frame[
        (frame["predictor"] == "mf")
        & (frame["population"] == "both")
        & (frame["training"] != "full")
        & (frame["test_variant"] == variant)
        & (frame["panel"] == frame["training"])
    ]
    rows = (
        rows.groupby(["construction", "fold", "training"])[
            ["eo_rmse", "rmse_A", "rmse_B"]
        ]
        .mean()
        .reset_index()
        .rename(columns={"training": "condition"})
    )
    rows["fallback"] = False
    return rows


def write_train_tables(frame, output_path, variant="rating10"):
    train = train_side_frame(frame, variant)
    if train.empty:
        raise ValueError("No MF rows on the jointly balanced test population.")
    decomposed = decompose_train_results.decompose(train)
    decompose_train_results.write_latex(
        decompose_train_results.summarize_conditions(train), decomposed, output_path,
        conditions_path=output_path.with_name("mf_train_conditions.tex"),
    )
    return decomposed


def main():
    args = parse_args()
    frame = load(args.results)
    predictors = tuple(args.predictors.split(","))
    decomposed = decompose(
        frame[frame["predictor"].isin(predictors)],
        args.test_variant,
        args.panel_correction,
    )
    summary = corrected_summary(decomposed)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    decomposed.to_csv(args.output_csv, index=False)
    if args.summary_csv is not None:
        summary.to_csv(args.summary_csv, index=False)
    caption = (
        "Joint train/test decomposition of the measured EO for {name}. "
        "Positive values favor group A. Means over ten folds $\\pm$ 95\\% "
        "confidence half-widths from the corrected resampled $t$-test; "
        "$^{{*}}$ marks $p<0.05$. The interaction indices are diagnostics and "
        "are not part of the additive split."
    )
    if "mf" in predictors:
        write_latex(
            summary, "mf", args.output_tex, "tab:joint-decomposition",
            caption.format(name="matrix factorization"),
        )
    if args.baseline_tex is not None:
        write_latex(
            summary, "user_item_average", args.baseline_tex,
            "tab:joint-decomposition-baseline",
            caption.format(name="the user--item average baseline"),
        )
    if args.train_tex is not None:
        write_train_tables(frame, args.train_tex, args.test_variant)
        print(f"Saved train-side tables to {args.train_tex}")
    if args.test_tex is not None or args.group_rmse_tex is not None:
        full = full_model_frame(frame, args.test_variant)
        if args.test_tex is not None:
            test = decompose_test_results.decompose(full)
            test.to_csv(args.test_tex.with_suffix(".csv"), index=False)
            label_free = None
            if args.test_variant == "rating10" and "user10" in set(frame["test_variant"]):
                label_free = decompose_test_results.decompose(full_model_frame(frame, "user10"))
            decompose_test_results.write_latex_summary(test, args.test_tex, label_free)
        if args.group_rmse_tex is not None:
            decompose_group_rmse.write_latex_summary(
                decompose_group_rmse.decompose(full), args.group_rmse_tex
            )
    error = (decomposed["eo_observed"] - decomposed["reconstructed"]).abs().max()
    table = summary.pivot_table(
        index="component", columns=["construction", "predictor"], values="mean"
    )
    print(table.to_string(float_format=lambda value: f"{value:+.4f}"))
    print(f"Saved {len(decomposed):,} fold-level decompositions to {args.output_csv}")
    print(f"Maximum reconstruction error: {error:.3e}")


if __name__ == "__main__":
    main()
