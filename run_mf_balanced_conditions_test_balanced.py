"""Run iterative training balancing with a common test set balanced both ways."""

import argparse
import logging
from pathlib import Path

logging.getLogger("torch.utils.flop_counter").setLevel(logging.ERROR)

import pandas as pd
import torch
import lightning as L
from sklearn.model_selection import KFold

from balanced_subset import (
    CONDITIONS,
    balance_users,
    find_reference_size,
    trim_users_to_size,
)
from balanced_test import TestBalanceCondition, balance_test_ratings, print_test_statistics
from dataset_loader import load_movielens_ratings
from run_mf_balanced_conditions import (
    parse_condition_names,
    print_data_summary,
)
from mf_train_eval import train_and_evaluate


TEST_BALANCE_BOTH = TestBalanceCondition("both", ("activity", "eccentricity"))


def train_condition_with_balanced_test(
    args,
    condition,
    fold,
    train_frame,
    test_frame,
    size,
    fallback_train,
):
    if condition.name == "none":
        balanced_train = trim_users_to_size(
            train_frame,
            size,
            random_state=args.seed + fold,
            target_size_ratio_tolerance=args.target_size_ratio_tolerance,
        )
        used_fallback = False
    else:
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
            )
            used_fallback = False
        except RuntimeError:
            if condition.name == "activity_eccentricity":
                raise
            balanced_train = fallback_train
            used_fallback = True

    balanced_test, test_info = balance_test_ratings(
        balanced_train,
        test_frame,
        TEST_BALANCE_BOTH,
        activity_quantiles=args.activity_quantiles,
        eccentricity_quantiles=args.eccentricity_quantiles,
        random_state=args.seed + fold,
    )
    train_statistics = print_data_summary(
        balanced_train,
        f"Fold {fold} condition {condition.name}: train subset",
    )
    print_test_statistics(
        balanced_train,
        balanced_test,
        f"Fold {fold} condition {condition.name}: balanced evaluation set",
    )
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
    rating_counts = balanced_train["group"].value_counts().reindex(
        ["A", "B"], fill_value=0
    )
    user_counts = (
        balanced_train[["user_id", "group"]]
        .drop_duplicates("user_id")["group"]
        .value_counts()
        .reindex(["A", "B"], fill_value=0)
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
            "test_ratings": len(balanced_test),
            "group_A_users": int(user_counts["A"]),
            "group_B_users": int(user_counts["B"]),
            "group_A_ratings": int(rating_counts["A"]),
            "group_B_ratings": int(rating_counts["B"]),
            "achieved_user_ratio": user_counts["A"] / max(user_counts["B"], 1),
            "achieved_rating_ratio": rating_counts["A"] / max(rating_counts["B"], 1),
            "fallback": used_fallback,
            "checkpoint": checkpoint_path,
            "test_balance": "activity_eccentricity",
            "test_balance_removed_ratings": test_info["removed_ratings"],
            "common_test_users": balanced_test["user_id"].nunique(),
            "train_activity_ks": train_statistics["activity_ks"],
            "train_eccentricity_ks": train_statistics["eccentricity_ks"],
        }
    )
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare MF EO with a common activity/eccentricity-balanced test set."
    )
    parser.add_argument("--original-data-dir", type=Path, default=Path("movielens_data/ml-1m"))
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2**14)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--activity-quantiles", type=int, default=10)
    parser.add_argument("--eccentricity-quantiles", type=int, default=10)
    parser.add_argument("--ratio-tolerance", type=float, default=1.001)
    parser.add_argument("--target-size-ratio-tolerance", type=float, default=0.001)
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument(
        "--conditions",
        type=parse_condition_names,
        default=tuple(condition.name for condition in CONDITIONS),
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("mf_balanced_conditions_test_balanced"))
    parser.add_argument("--results-file", type=Path, default=Path("mf_balanced_conditions_test_balanced.csv"))
    parser.add_argument("--summary-file", type=Path, default=Path("mf_balanced_conditions_test_balanced_summary.csv"))
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


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
        fold_results = []
        for condition_name in args.conditions:
            metrics = train_condition_with_balanced_test(
                args,
                condition_map[condition_name],
                fold,
                train_frame,
                test_frame,
                reference_info["size"],
                reference_train,
            )
            fold_results.append(metrics)
        for metrics in fold_results:
            results.append(metrics)

    results_frame = pd.DataFrame(results)
    args.results_file.parent.mkdir(parents=True, exist_ok=True)
    args.summary_file.parent.mkdir(parents=True, exist_ok=True)
    results_frame.to_csv(args.results_file, index=False)
    summary = results_frame.groupby("condition").agg(
        folds=("fold", "count"),
        mean_eo_rmse=("eo_rmse", "mean"),
        std_eo_rmse=("eo_rmse", "std"),
        mean_rmse=("rmse", "mean"),
        mean_mae=("mae", "mean"),
        mean_train_activity_ks=("train_activity_ks", "mean"),
        mean_train_eccentricity_ks=("train_eccentricity_ks", "mean"),
    )
    summary.to_csv(args.summary_file)
    print(summary.to_string(float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    run(parse_args())
