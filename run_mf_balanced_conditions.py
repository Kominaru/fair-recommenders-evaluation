"""Compare MF bias after fixed-size user-removal balancing conditions."""

import argparse
import logging
from pathlib import Path

logging.getLogger("torch.utils.flop_counter").setLevel(logging.ERROR)

import lightning as L
import numpy as np
import pandas as pd
import torch
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
    find_reference_size,
    trim_users_to_size,
)
from mf_train_eval import train_and_evaluate
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


def select_protected_users(
    reference_train,
    test_frame,
    fraction,
    activity_quantiles,
    eccentricity_quantiles,
    random_state,
):
    """Select a fixed, approximately stratified test-user cohort."""

    if not 0 < fraction <= 1:
        raise ValueError("protected fraction must be in (0, 1].")
    test_user_ids = set(test_frame["user_id"])
    users = add_user_statistics(reference_train)
    users = users[users["user_id"].isin(test_user_ids)].copy()
    if users.empty:
        raise RuntimeError("No fully balanced users are present in the test fold.")
    activity_edges = np.unique(
        np.quantile(
            users["activity"],
            np.linspace(0, 1, activity_quantiles + 1),
        )
    )
    eccentricity_edges = np.unique(
        np.quantile(
            users["eccentricity"],
            np.linspace(0, 1, eccentricity_quantiles + 1),
        )
    )
    users["activity_bin"] = np.searchsorted(
        activity_edges[1:-1], users["activity"], side="right"
    )
    users["eccentricity_bin"] = np.searchsorted(
        eccentricity_edges[1:-1], users["eccentricity"], side="right"
    )
    users["stratum"] = list(
        zip(users["activity_bin"], users["eccentricity_bin"])
    )
    selected = []
    for (_, group), group_users in users.groupby(["stratum", "group"]):
        sample_size = int(np.floor(fraction * len(group_users)))
        if sample_size == 0:
            continue
        selected.extend(
            group_users.sort_values("user_id")["user_id"].head(sample_size).tolist()
        )
    protected = set(selected)
    if not protected:
        raise RuntimeError("Protected-user sampling produced an empty cohort.")
    return protected


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare MF EO at each fold's minimal balanced size."
    )
    parser.add_argument("--original-data-dir", type=Path, default=Path("movielens_data/ml-1m"))
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2**14)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--protected-fraction",
        type=float,
            default=0.2,
        help="Fraction of eligible fully balanced users protected for testing.",
    )
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
    return parser.parse_args()


def train_condition(
    args,
    condition,
    fold,
    train_frame,
    common_test,
    size,
    fallback_train,
    protected_users,
):
    if condition.name == "none":
        balanced_train = trim_users_to_size(
            train_frame,
            size,
            random_state=args.seed + fold,
            protected_user_ids=protected_users,
            target_size_ratio_tolerance=args.target_size_ratio_tolerance,
        )
        used_fallback = False
    else:
        used_fallback = False
        try:
            balanced_train, balance_info = balance_users(
                train_frame,
                condition,
                target_size=size,
                target_size_ratio_tolerance=args.target_size_ratio_tolerance,
                activity_quantiles=args.activity_quantiles,
                eccentricity_quantiles=args.eccentricity_quantiles,
                ratio_tolerance=args.ratio_tolerance,
                random_state=args.seed + fold,
                max_iterations=args.max_iterations,
                protected_user_ids=protected_users,
            )
        except RuntimeError:
            if condition.name == "activity_eccentricity":
                raise
            balanced_train = fallback_train
            used_fallback = True
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
    retained_users = set(balanced_train["user_id"])
    missing_protected = protected_users - retained_users
    if missing_protected:
        raise RuntimeError(
            f"Condition {condition.name} removed protected test users: "
            f"{sorted(missing_protected)[:5]}"
        )
    train_statistics = print_data_summary(
        balanced_train, f"Fold {fold} condition {condition.name}: train subset"
    )
    print_data_summary(
        common_test,
        f"Fold {fold} condition {condition.name}: evaluation subset",
        reference_ratings=balanced_train,
    )
    metrics, checkpoint_path = train_and_evaluate(
        train_frame=balanced_train,
        test_frame=common_test,
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
            "fold": fold,
            "target_size": size,
            "size_matched": abs(len(balanced_train) - size)
            <= size * args.target_size_ratio_tolerance,
            "train_ratings": len(balanced_train),
            "train_users": balanced_train["user_id"].nunique(),
            "test_ratings": len(common_test),
            "group_A_users": int(user_counts["A"]),
            "group_B_users": int(user_counts["B"]),
            "group_A_ratings": int(rating_counts["A"]),
            "group_B_ratings": int(rating_counts["B"]),
            "achieved_user_ratio": user_counts["A"] / max(user_counts["B"], 1),
            "achieved_rating_ratio": rating_counts["A"] / max(rating_counts["B"], 1),
            "fallback": used_fallback,
            "checkpoint": checkpoint_path,
            "condition_test_ratings": len(common_test),
            "common_test_users": common_test["user_id"].nunique(),
            "train_activity_ks": train_statistics["activity_ks"],
            "train_eccentricity_ks": train_statistics["eccentricity_ks"],
        }
    )
    return metrics, retained_users


def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required but is not available to PyTorch.")
    L.seed_everything(args.seed, workers=True)
    ratings = load_movielens_ratings(args.original_data_dir)
    ratings["user_index"], user_ids = pd.factorize(ratings["user_id"], sort=True)
    ratings["item_index"], item_ids = pd.factorize(ratings["item_id"], sort=True)
    args.num_users = len(user_ids)
    args.num_items = len(item_ids)

    splitter = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    results = []
    condition_map = {condition.name: condition for condition in CONDITIONS}
    for fold, (train_indices, test_indices) in enumerate(splitter.split(ratings), start=1):
        train_frame = ratings.iloc[train_indices].copy()
        test_frame = ratings.iloc[test_indices].copy()
        print_data_summary(train_frame, f"Fold {fold}: original train data")
        print_data_summary(
            test_frame,
            f"Fold {fold}: original test data",
            reference_ratings=train_frame,
        )
        reference_train, reference_info = find_reference_size(
            train_frame,
            activity_quantiles=args.activity_quantiles,
            eccentricity_quantiles=args.eccentricity_quantiles,
            ratio_tolerance=args.ratio_tolerance,
            random_state=args.seed + fold,
            max_iterations=args.max_iterations,
        )
        print(
            f"Fold {fold:2d}: strict reference size N={reference_info['size']:,}; "
            f"A:B={reference_info['ratio']:.4f}"
        )
        size = reference_info["size"]
        protected_users = select_protected_users(
            reference_train,
            test_frame,
            args.protected_fraction,
            args.activity_quantiles,
            args.eccentricity_quantiles,
            args.seed + fold,
        )
        print(
            f"Fold {fold:2d}: protected test users={len(protected_users):,} "
            f"(fraction={args.protected_fraction:.3f})"
        )
        common_test = test_frame[
            test_frame["user_id"].isin(protected_users)
        ].copy()
        if common_test.empty:
            raise RuntimeError(f"Fold {fold} has no protected test ratings.")
        print(
            f"Fold {fold:2d}: common evaluation users="
            f"{common_test['user_id'].nunique():,}, "
            f"ratings={len(common_test):,}"
        )
        fold_results = []
        for condition_name in args.conditions:
            metrics, retained_users = train_condition(
                args,
                condition_map[condition_name],
                fold,
                train_frame,
                common_test,
                size,
                reference_train,
                protected_users,
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
                    "test_ratings": len(common_test),
                    "common_rmse": metrics["rmse"],
                    "common_mae": metrics["mae"],
                    "common_rmse_A": metrics["rmse_A"],
                    "common_rmse_B": metrics["rmse_B"],
                    "common_eo_rmse": metrics["eo_rmse"],
                }
            )
            results.append(metrics)
            print(
                f"Fold {fold:2d} {metrics['condition']:24s}: "
                f"N={metrics['train_ratings']:,}, "
                f"EO(common)={metrics['eo_rmse']:.4f}, "
                f"EO(condition)={metrics['condition_eo_rmse']:.4f}, "
                f"fallback={metrics['fallback']}"
            )

    results = pd.DataFrame(results)
    condition_labels = {
        "all": "Fully balanced",
        "none": "Resized (no balancing)",
    }
    global_summary = (
        results.assign(condition=results["condition"].map(condition_labels))
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
