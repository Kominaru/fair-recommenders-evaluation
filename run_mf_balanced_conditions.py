"""Compare MF bias after fixed-size user-removal balancing conditions."""

import argparse
import logging
from pathlib import Path

logging.getLogger("torch.utils.flop_counter").setLevel(logging.ERROR)
import lightning as L
import pandas as pd
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import KFold



import warnings

warnings.filterwarnings(
    "ignore",
    message=r"`isinstance\(treespec, LeafSpec\)` is deprecated.*",
)

from balanced_subset import (
    CONDITIONS,
    balance_users,
    find_reference_size,
    trim_users_to_size,
)
from run_mf_experiments import (
    MatrixFactorizationModel,
    evaluate,
    load_ratings,
    make_dataloader,
)

def parse_condition_names(value):
    names = tuple(name.strip() for name in value.split(",") if name.strip())
    available = {condition.name for condition in CONDITIONS}
    unknown = set(names) - available
    if unknown:
        raise argparse.ArgumentTypeError(
            f"Unknown conditions: {sorted(unknown)}; choose from {sorted(available)}"
        )
    return names


def print_data_summary(frame, label):
    """Print rating, group, activity, and eccentricity statistics for a frame."""

    user_activity = frame.groupby("user_id")["rating"].size()
    user_means = frame.groupby("user_id")["rating"].mean()
    item_means = frame.groupby("item_id")["rating"].mean()
    statistics = frame.assign(
        activity=frame["user_id"].map(user_activity),
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
            eccentricity=("eccentricity", "mean"),
        )
    )
    summary = statistics.groupby("group").agg(
        ratings=("rating", "size"),
        users=("user_id", "nunique"),
        mean_rating=("rating", "mean"),
        rating_eccentricity=("eccentricity", "mean"),
    )
    user_summary = users.groupby("group").agg(
        mean_activity=("activity", "mean"),
        user_eccentricity=("eccentricity", "mean"),
    )
    summary = summary.join(user_summary).reindex(["A", "B"])
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
    parser.add_argument("--activity-quantiles", type=int, default=10)
    parser.add_argument("--eccentricity-quantiles", type=int, default=10)
    parser.add_argument("--ratio-tolerance", type=float, default=1.001)
    parser.add_argument(
        "--target-size-tolerance",
        type=int,
        default=0,
        help="Allowed absolute deviation from the requested training size N.",
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
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def train_condition(
    args,
    condition,
    fold,
    train_frame,
    test_frame,
    size,
    fallback_train,
):
    if condition.name == "none":
        balanced_train = trim_users_to_size(train_frame, size)
        counts = balanced_train["group"].value_counts().reindex(
            ["A", "B"], fill_value=0
        )
        balance_info = {
            "group_counts": counts,
            "ratio": counts["A"] / max(counts["B"], 1),
        }
        used_fallback = False
    else:
        used_fallback = False
        try:
            balanced_train, balance_info = balance_users(
                train_frame,
                condition,
                target_size=size,
                target_size_tolerance=args.target_size_tolerance,
                activity_quantiles=args.activity_quantiles,
                eccentricity_quantiles=args.eccentricity_quantiles,
                ratio_tolerance=args.ratio_tolerance,
                random_state=args.seed + fold,
                max_iterations=args.max_iterations,
            )
        except RuntimeError:
            balanced_train = fallback_train
            counts = balanced_train["group"].value_counts().reindex(
                ["A", "B"], fill_value=0
            )
            balance_info = {"group_counts": counts, "ratio": counts["A"] / max(counts["B"], 1)}
            used_fallback = True
    retained_users = set(balanced_train["user_id"])
    evaluation_test = test_frame[test_frame["user_id"].isin(retained_users)].copy()
    if evaluation_test.empty:
        raise RuntimeError(
            f"Condition {condition.name} removed every user from the test fold."
        )
    print_data_summary(balanced_train, f"Fold {fold} condition {condition.name}: train subset")
    print_data_summary(
        evaluation_test,
        f"Fold {fold} condition {condition.name}: evaluation subset",
    )
    model = MatrixFactorizationModel(
        num_users=args.num_users,
        num_items=args.num_items,
        embedding_dim=args.embedding_dim,
        learning_rate=args.learning_rate,
    )
    checkpoint = ModelCheckpoint(
        dirpath=args.checkpoint_dir / condition.name / f"fold_{fold}",
        filename="best",
        monitor="test_rmse",
        mode="min",
        save_top_k=1,
    )
    trainer = L.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=args.max_epochs,
        callbacks=[
            checkpoint,
            EarlyStopping(monitor="test_rmse", mode="min", patience=5),
        ],
        logger=False,
        enable_model_summary=False,
        enable_progress_bar=not args.quiet,
    )
    trainer.fit(
        model,
        train_dataloaders=make_dataloader(balanced_train, args.batch_size, True),
        val_dataloaders=make_dataloader(evaluation_test, args.batch_size, False),
    )
    best_model = MatrixFactorizationModel.load_from_checkpoint(
        checkpoint.best_model_path
    ).to(model.device)
    metrics = evaluate(
        best_model,
        make_dataloader(evaluation_test, args.batch_size, False),
        evaluation_test["group"].to_numpy(),
    )
    metrics.update(
        {
            "condition": condition.name,
            "fold": fold,
            "target_size": size,
            "train_ratings": len(balanced_train),
            "train_users": balanced_train["user_id"].nunique(),
            "test_ratings": len(evaluation_test),
            "group_A_ratings": int(balance_info["group_counts"]["A"]),
            "group_B_ratings": int(balance_info["group_counts"]["B"]),
            "achieved_ratio": balance_info["ratio"],
            "fallback": used_fallback,
        }
    )
    return metrics


def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required but is not available to PyTorch.")
    L.seed_everything(args.seed, workers=True)
    ratings = load_ratings(args)
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
        print_data_summary(test_frame, f"Fold {fold}: original test data")
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
        for condition_name in args.conditions:
            metrics = train_condition(
                args,
                condition_map[condition_name],
                fold,
                train_frame,
                test_frame,
                size,
                reference_train,
            )
            results.append(metrics)
            print(
                f"Fold {fold:2d} {condition_name:24s}: "
                f"N={metrics['train_ratings']:,}, EO={metrics['eo_rmse']:.4f}, "
                f"fallback={metrics['fallback']}"
            )

    results = pd.DataFrame(results)
    print("\nMean EO by condition")
    print(results.groupby("condition")["eo_rmse"].agg(["mean", "std"]).to_string())
    results.to_csv(args.results_file, index=False)
    print(f"Saved results to {args.results_file}")


if __name__ == "__main__":
    run(parse_args())
