"""Compare MF bias after fixed-size user-removal balancing conditions."""

import argparse
import logging
from pathlib import Path
import time

logging.getLogger("torch.utils.flop_counter").setLevel(logging.ERROR)

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from scipy.stats import ks_2samp

import warnings

warnings.filterwarnings(
    "ignore",
    message=r"`isinstance\(treespec, LeafSpec\)` is deprecated.*",
)

warnings.filterwarnings(
    "ignore",
    message="Please migrate to LitLogger.*",
    category=UserWarning,
)
logging.getLogger("lightning_utilities.core.rank_zero").setLevel(
    logging.WARNING
)
logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(
    logging.WARNING
)

from balanced_subset import (
    CONDITIONS,
    balance_users,
    condition_cells,
    hybrid_balance_users,
    thin_users_to_size,
    user_strata,
)
from balanced_test import (
    TestBalanceCondition,
    balance_test_ratings,
    print_test_statistics,
    standardize_test_ratings,
    standardized_test_reference,
)
from dataset_loader import load_movielens_ratings
from user_balancing import add_user_statistics

def parse_condition_names(value):
    names = tuple(name.strip() for name in value.split(",") if name.strip())
    available = {condition.name for condition in CONDITIONS}
    unknown = set(names) - available
    if unknown:
        raise argparse.ArgumentTypeError(
            f"Unknown conditions: {sorted(unknown)}; choose from {sorted(available)}"
        )
    return names


def print_data_summary(frame, label, reference_ratings=None):
    """Print frame statistics and optional reference-based user statistics."""

    if reference_ratings is None:
        reference_ratings = frame
    reference_users = add_user_statistics(frame, reference_ratings)
    user_means = frame.groupby("user_id")["rating"].mean()
    item_means = frame.groupby("item_id")["rating"].mean()
    statistics = frame.assign(
        activity=frame["user_id"].map(
            reference_users.set_index("user_id")["activity"]
        ),
        eccentricity=(
            frame["rating"]
            - (frame["user_id"].map(user_means) + frame["item_id"].map(item_means))
            / 2
        ).abs(),
    )
    users = (
        statistics.groupby("user_id", as_index=False)
        .agg(
            group=("group", "first"),
            activity=("activity", "first"),
        )
    )
    users = users.merge(
        reference_users[["user_id", "eccentricity"]],
        on="user_id",
        how="left",
        suffixes=("_frame", "_reference"),
    )
    users = users.rename(columns={"eccentricity_reference": "eccentricity"})
    summary = statistics.groupby("group").agg(
        ratings=("rating", "size"),
        users=("user_id", "nunique"),
        mean_rating=("rating", "mean"),
        rating_mean_activity=("activity", "mean"),
        rating_eccentricity=("eccentricity", "mean"),
    )
    user_summary = users.groupby("group").agg(
        mean_activity=("activity", "mean"),
        user_eccentricity=("eccentricity", "mean"),
    )
    summary = summary.join(user_summary).reindex(["A", "B"])
    group_a = users[users["group"] == "A"]
    group_b = users[users["group"] == "B"]
    if group_a.empty or group_b.empty:
        activity_ks = eccentricity_ks = np.nan
        activity_pvalue = eccentricity_pvalue = np.nan
    else:
        activity_test = ks_2samp(group_a["activity"], group_b["activity"])
        eccentricity_test = ks_2samp(
            group_a["eccentricity"], group_b["eccentricity"]
        )
        activity_ks = activity_test.statistic
        activity_pvalue = activity_test.pvalue
        eccentricity_ks = eccentricity_test.statistic
        eccentricity_pvalue = eccentricity_test.pvalue
    print(f"\n{label}")
    print(
        f"  total: ratings={len(frame):,}, users={frame['user_id'].nunique():,}, "
        f"items={frame['item_id'].nunique():,}, mean_rating={frame['rating'].mean():.4f}"
    )
    print(
        summary.to_string(
            float_format=lambda value: f"{value:.4f}",
            na_rep="nan",
        )
    )
    print(
        f"  A/B KS statistic: activity={activity_ks:.3f}, "
        f"eccentricity={eccentricity_ks:.3f}"
    )
    print(
        f"  A/B KS p-value: activity={activity_pvalue:.3f}, "
        f"eccentricity={eccentricity_pvalue:.3f}"
    )
    return {
        "activity_ks": activity_ks,
        "activity_ks_pvalue": activity_pvalue,
        "eccentricity_ks": eccentricity_ks,
        "eccentricity_ks_pvalue": eccentricity_pvalue,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare MF EO at each fold's minimal balanced size."
    )
    parser.add_argument("--original-data-dir", type=Path, default=Path("movielens_data/ml-1m"))
    parser.add_argument(
        "--group-by",
        choices=(
            "gender",
            "age",
            "artificial_activity",
            "artificial_eccentricity",
            "placebo",
        ),
        default="gender",
        help="User-level A/B group construction.",
    )
    parser.add_argument("--age-threshold", type=int, default=45)
    for feature, means, stds in (
        ("activity", (300.0, 400.0), (200.0, 200.0)),
        ("eccentricity", (0.65, 0.85), (0.15, 0.15)),
    ):
        parser.add_argument(
            f"--artificial-{feature}-means",
            type=float,
            nargs=2,
            default=means,
            metavar=("A", "B"),
        )
        parser.add_argument(
            f"--artificial-{feature}-stds",
            type=float,
            nargs=2,
            default=stds,
            metavar=("A", "B"),
        )
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2**14)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=727)
    parser.add_argument("--activity-quantiles", type=int, default=10)
    parser.add_argument("--eccentricity-quantiles", type=int, default=10)
    parser.add_argument("--ratio-tolerance", type=float, default=1.001)
    parser.add_argument(
        "--target-size-ratio-tolerance",
        type=float,
        default=0.001,
        help="Allowed relative deviation from the target rating count.",
    )
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument(
        "--eccentricity-updates",
        choices=("frozen", "dynamic"),
        default="frozen",
        help=(
            "frozen: strata use each user's eccentricity on the full training "
            "fold (Algorithm 1); dynamic: re-bin users as removals change it. "
            "Thinning to the common size always uses the frozen strata."
        ),
    )
    parser.add_argument(
        "--evaluation",
        choices=("standardized", "per-condition"),
        default="standardized",
        help=(
            "standardized: weight every condition's test ratings to one joint "
            "activity-eccentricity target fixed from the full training fold; "
            "per-condition: rebuild the joint weighting from each subset."
        ),
    )
    parser.add_argument(
        "--balancer",
        choices=("dynamic", "hybrid"),
        default="dynamic",
        help="Train-side balancing algorithm.",
    )
    parser.add_argument("--hybrid-max-passes", type=int, default=8)
    parser.add_argument("--hybrid-max-swaps-per-pass", type=int, default=40)
    parser.add_argument(
        "--conditions",
        type=parse_condition_names,
        default=tuple(condition.name for condition in CONDITIONS),
        help="Comma-separated conditions to compare.",
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("mf_balanced_conditions"))
    parser.add_argument("--results-file", type=Path, default=Path("mf_balanced_conditions.csv"))
    parser.add_argument(
        "--summary-file",
        type=Path,
        default=Path("mf_balanced_conditions_summary.csv"),
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--balance-only",
        action="store_true",
        help="Build and check every training subset on CPU without training.",
    )
    parser.add_argument(
        "--fold-limit",
        type=int,
        default=None,
        help="Stop after this many folds.",
    )
    return parser.parse_args()


def balanced_bases(args, fold, train_frame):
    """Return each condition's largest balanced subset for one fold."""

    condition_map = {condition.name: condition for condition in CONDITIONS}
    bases = {}
    for name in args.conditions:
        if name == "none":
            bases[name] = train_frame
            continue
        bases[name], _ = balance_users(
            train_frame,
            condition_map[name],
            activity_quantiles=args.activity_quantiles,
            eccentricity_quantiles=args.eccentricity_quantiles,
            ratio_tolerance=args.ratio_tolerance,
            random_state=args.seed + fold,
            max_iterations=args.max_iterations,
            dynamic_eccentricity=args.eccentricity_updates == "dynamic",
        )
    return bases


def build_training_subset(args, condition, fold, base, size, strata):
    """Thin a condition's balanced subset to the common rating count.

    Users are removed at random within each cell of group and balanced
    strata, so the subset keeps its user ratio and balanced shape.
    """

    return thin_users_to_size(
        base,
        size,
        condition_cells(strata, condition),
        random_state=args.seed + fold,
        target_size_ratio_tolerance=args.target_size_ratio_tolerance,
    )


def _group_distance(users, column):
    """Total variation distance between the A and B user distributions."""

    shares = pd.crosstab(users[column], users["group"], normalize="columns")
    shares = shares.reindex(columns=["A", "B"], fill_value=0)
    return 0.5 * float((shares["A"] - shares["B"]).abs().sum())


def describe_condition(
    args,
    condition,
    fold,
    base,
    size,
    reference_condition,
    test_reference,
    features,
):
    """Build one training subset and report how well it meets its condition."""

    start = time.perf_counter()
    balanced_train = build_training_subset(
        args, condition, fold, base, size, features
    )
    used_fallback = False
    seconds = time.perf_counter() - start
    users = features.loc[balanced_train["user_id"].unique()]
    user_counts = users["group"].value_counts().reindex(["A", "B"], fill_value=0)
    rating_counts = balanced_train["group"].value_counts().reindex(
        ["A", "B"], fill_value=0
    )
    row = {
        "construction": args.group_by,
        "fold": fold,
        "condition": condition.name,
        "reference_size": size,
        "balanced_ratings": len(base),
        "reference_condition": reference_condition,
        "train_ratings": len(balanced_train),
        "size_matched": abs(len(balanced_train) - size)
        <= size * args.target_size_ratio_tolerance,
        "fallback": used_fallback,
        "seconds": seconds,
        "group_A_users": int(user_counts["A"]),
        "group_B_users": int(user_counts["B"]),
        "user_ratio": user_counts["A"] / max(user_counts["B"], 1),
        "rating_ratio": rating_counts["A"] / max(rating_counts["B"], 1),
        "activity_tvd": _group_distance(users, "activity_bin"),
        "eccentricity_tvd": _group_distance(users, "eccentricity_bin"),
        "joint_tvd": _group_distance(users, "joint_bin"),
    }
    if test_reference is not None:
        evaluated, test_info = standardize_test_ratings(balanced_train, test_reference)
        row.update(
            {
                "test_ratings": len(evaluated),
                "test_users": evaluated["user_id"].nunique(),
                "test_target_coverage_A": test_info["target_coverage_A"],
                "test_target_coverage_B": test_info["target_coverage_B"],
            }
        )
    print(
        f"Fold {fold:2d} {condition.name:30s} N={len(balanced_train):,} "
        f"users A:B={row['user_ratio']:.2f} ratings A:B={row['rating_ratio']:.2f} "
        f"TVD act={row['activity_tvd']:.3f} ecc={row['eccentricity_tvd']:.3f} "
        f"joint={row['joint_tvd']:.3f} (balanced N={len(base):,}, thinning {seconds:.1f}s)"
    )
    return row


def summarize_balancing(results, results_file):
    """Save per-fold balancing checks and print their means by condition."""

    columns = [
        "train_ratings",
        "user_ratio",
        "rating_ratio",
        "activity_tvd",
        "eccentricity_tvd",
        "joint_tvd",
        "seconds",
    ]
    columns += [
        column
        for column in ("test_users", "test_target_coverage_A", "test_target_coverage_B")
        if column in results
    ]
    order = list(dict.fromkeys(results["condition"]))
    summary = results.groupby("condition")[columns].mean().reindex(order)
    summary["fallbacks"] = results.groupby("condition")["fallback"].sum()
    summary["size_misses"] = (~results["size_matched"]).groupby(
        results["condition"]
    ).sum()
    print("\nBalancing checks by condition (means across folds)")
    print(summary.to_string(float_format=lambda value: f"{value:.3f}"))
    results.to_csv(results_file, index=False)
    print(f"Saved balancing checks to {results_file}")


def train_condition(
    args,
    condition,
    fold,
    train_frame,
    test_frame,
    size,
    base,
    strata,
    test_reference=None,
):
    if args.balancer == "hybrid":
        balanced_train, balance_info = hybrid_balance_users(
            train_frame,
            condition,
            target_size=size,
            target_size_ratio_tolerance=args.target_size_ratio_tolerance,
            activity_quantiles=args.activity_quantiles,
            eccentricity_quantiles=args.eccentricity_quantiles,
            ratio_tolerance=args.ratio_tolerance,
            random_state=args.seed + fold,
            max_passes=args.hybrid_max_passes,
            max_swaps_per_pass=args.hybrid_max_swaps_per_pass,
        )
        used_fallback = False
    else:
        balanced_train = build_training_subset(
            args, condition, fold, base, size, strata
        )
        used_fallback = False
    rating_counts = balanced_train["group"].value_counts().reindex(
        ["A", "B"], fill_value=0
    )
    user_counts = (
        balanced_train[["user_id", "group"]]
        .drop_duplicates("user_id")
        ["group"]
        .value_counts()
        .reindex(["A", "B"], fill_value=0)
    )
    if test_reference is not None:
        balanced_test, test_info = standardize_test_ratings(
            balanced_train, test_reference
        )
    else:
        balanced_test, test_info = balance_test_ratings(
            balanced_train,
            test_frame,
            TestBalanceCondition("both", ("activity", "eccentricity")),
            activity_quantiles=args.activity_quantiles,
            eccentricity_quantiles=args.eccentricity_quantiles,
            random_state=args.seed + fold,
        )
    train_statistics = print_data_summary(
        balanced_train, f"Fold {fold} condition {condition.name}: train subset"
    )
    print_data_summary(
        balanced_test,
        f"Fold {fold} condition {condition.name}: evaluation subset",
        reference_ratings=balanced_train,
    )
    print_test_statistics(
        balanced_train,
        balanced_test,
        f"Fold {fold} condition {condition.name}: weighted evaluation set",
    )
    from mf_train_eval import train_and_evaluate

    metrics, checkpoint_path = train_and_evaluate(
        train_frame=balanced_train,
        test_frame=balanced_test,
        num_users=args.num_users,
        num_items=args.num_items,
        embedding_dim=args.embedding_dim,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        checkpoint_dir=args.checkpoint_dir / condition.name / f"fold_{fold}",
        early_stopping_patience=5,
        quiet=args.quiet,
    )
    metrics.update(
        {
            "condition": condition.name,
            "construction": args.group_by,
            "fold": fold,
            "target_size": size,
            "size_matched": abs(len(balanced_train) - size)
            <= size * args.target_size_ratio_tolerance,
            "train_ratings": len(balanced_train),
            "train_users": balanced_train["user_id"].nunique(),
            "test_ratings": len(balanced_test),
            "group_A_users": int(user_counts["A"]),
            "group_B_users": int(user_counts["B"]),
            "group_A_ratings": int(rating_counts["A"]),
            "group_B_ratings": int(rating_counts["B"]),
            "achieved_user_ratio": user_counts["A"] / max(user_counts["B"], 1),
            "achieved_rating_ratio": rating_counts["A"] / max(rating_counts["B"], 1),
            "fallback": used_fallback,
            "train_balancer": args.balancer,
            "eccentricity_updates": args.eccentricity_updates,
            "train_balance_iterations": balance_info.get("iterations", 0)
            if args.balancer == "hybrid"
            else np.nan,
            "train_balance_swaps": balance_info.get("swaps", 0)
            if args.balancer == "hybrid"
            else np.nan,
            "train_balance_deviation": balance_info.get("maximum_deviation", np.nan)
            if args.balancer == "hybrid"
            else np.nan,
            "checkpoint": checkpoint_path,
            "condition_test_ratings": len(balanced_test),
            "test_users": balanced_test["user_id"].nunique(),
            "evaluation": args.evaluation,
            "test_balance_excluded_ratings": test_info.get("excluded_ratings", np.nan),
            "test_target_coverage_A": test_info.get("target_coverage_A", np.nan),
            "test_target_coverage_B": test_info.get("target_coverage_B", np.nan),
            "test_balance_zero_weight_ratings": test_info["zero_weight_ratings"],
            "train_activity_ks": train_statistics["activity_ks"],
            "train_eccentricity_ks": train_statistics["eccentricity_ks"],
        }
    )
    return metrics, set(balanced_train["user_id"])


def run(args):
    if not args.balance_only:
        import lightning as L
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is required but is not available to PyTorch.")
        L.seed_everything(args.seed, workers=True)
    ratings = load_movielens_ratings(
        args.original_data_dir,
        group_by=args.group_by,
        age_threshold=args.age_threshold,
        artificial_activity_means=args.artificial_activity_means,
        artificial_activity_standard_deviations=args.artificial_activity_stds,
        artificial_eccentricity_means=args.artificial_eccentricity_means,
        artificial_eccentricity_standard_deviations=args.artificial_eccentricity_stds,
    )
    ratings["user_index"], user_ids = pd.factorize(ratings["user_id"], sort=True)
    ratings["item_index"], item_ids = pd.factorize(ratings["item_id"], sort=True)
    args.num_users = len(user_ids)
    args.num_items = len(item_ids)

    splitter = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    results = []
    condition_map = {condition.name: condition for condition in CONDITIONS}
    for fold, (train_indices, test_indices) in enumerate(splitter.split(ratings), start=1):
        if args.fold_limit is not None and fold > args.fold_limit:
            break
        train_frame = ratings.iloc[train_indices].copy()
        test_frame = ratings.iloc[test_indices].copy()
        print_data_summary(train_frame, f"Fold {fold}: original train data")
        print_data_summary(
            test_frame,
            f"Fold {fold}: original test data",
            reference_ratings=train_frame,
        )
        bases = balanced_bases(args, fold, train_frame)
        reference_condition = min(bases, key=lambda name: len(bases[name]))
        size = len(bases[reference_condition])
        strata = user_strata(
            train_frame,
            activity_quantiles=args.activity_quantiles,
            eccentricity_quantiles=args.eccentricity_quantiles,
        )
        print(
            f"Fold {fold:2d}: common reference size N={size:,} "
            f"set by {reference_condition}"
        )
        test_reference = (
            standardized_test_reference(
                train_frame,
                test_frame,
                activity_quantiles=args.activity_quantiles,
                eccentricity_quantiles=args.eccentricity_quantiles,
            )
            if args.evaluation == "standardized"
            else None
        )
        if args.balance_only:
            for condition_name in args.conditions:
                results.append(
                    describe_condition(
                        args,
                        condition_map[condition_name],
                        fold,
                        bases[condition_name],
                        size,
                        reference_condition,
                        test_reference,
                        strata,
                    )
                )
            continue
        fold_results = []
        for condition_name in args.conditions:
            metrics, retained_users = train_condition(
                args,
                condition_map[condition_name],
                fold,
                train_frame,
                test_frame,
                size,
                bases[condition_name],
                strata,
                test_reference,
            )
            fold_results.append(metrics)

        for metrics in fold_results:
            condition_metrics = {
                f"condition_{name}": value
                for name, value in metrics.items()
                if name in {"rmse", "mae", "rmse_A", "rmse_B", "eo_rmse"}
            }
            metrics.update(condition_metrics)
            metrics.update(
                {
                    "test_ratings": metrics["test_ratings"],
                    "weighted_test_rmse": metrics["rmse"],
                    "weighted_test_mae": metrics["mae"],
                    "weighted_test_rmse_A": metrics["rmse_A"],
                    "weighted_test_rmse_B": metrics["rmse_B"],
                    "weighted_test_eo_rmse": metrics["eo_rmse"],
                }
            )
            results.append(metrics)
            print(
                f"Fold {fold:2d} {metrics['condition']:24s}: "
                f"N={metrics['train_ratings']:,}, "
                f"EO(weighted test)={metrics['eo_rmse']:.4f}, "
                f"EO(condition)={metrics['condition_eo_rmse']:.4f}, "
                f"fallback={metrics['fallback']}"
            )

    results = pd.DataFrame(results)
    if args.balance_only:
        summarize_balancing(results, args.results_file)
        return
    condition_labels = {
        "all": "Fully balanced",
        "none": "Resized (no balancing)",
    }
    global_summary = (
        results.assign(
            condition=results["condition"].map(condition_labels).fillna(
                results["condition"]
            )
        )
        .groupby("condition")[
            ["eo_rmse", "condition_eo_rmse", "rmse_A", "rmse_B"]
        ]
        .agg(["mean", "std"])
    )
    global_summary.columns = [
        f"{metric}_{stat}"
        for metric, stat in global_summary.columns
    ]
    print("\nGlobal results by condition (mean +/- std across folds)")
    print(global_summary.to_string(float_format=lambda value: f"{value:.6f}"))
    results.to_csv(args.results_file, index=False)
    global_summary.to_csv(args.summary_file)
    print(f"Saved results to {args.results_file}")
    print(f"Saved global summary to {args.summary_file}")


if __name__ == "__main__":
    run(parse_args())
