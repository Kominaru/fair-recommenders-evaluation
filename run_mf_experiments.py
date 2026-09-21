"""Run 10-fold cross-validation for a matrix-factorization rating model."""

import argparse
import logging
from pathlib import Path
import warnings

logging.getLogger("torch.utils.flop_counter").setLevel(logging.ERROR)

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

import lightning as L
import numpy as np
import pandas as pd
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import KFold
from torch import nn
from torch.utils.data import DataLoader, Dataset

from equalize_dataset import balance_groups
from equalize_users import balance_users

class MatrixFactorizationModel(L.LightningModule):
    """Predict ratings using the dot product of user and item embeddings."""

    def __init__(self, num_users, num_items, embedding_dim, learning_rate):
        super().__init__()
        self.save_hyperparameters()
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)
        nn.init.xavier_uniform_(self.user_embeddings.weight)
        nn.init.xavier_uniform_(self.item_embeddings.weight)
        self.learning_rate = learning_rate
        self.register_buffer("train_squared_error", torch.tensor(0.0), persistent=False)
        self.register_buffer("train_rating_count", torch.tensor(0), persistent=False)
        self.register_buffer("test_squared_error", torch.tensor(0.0), persistent=False)
        self.register_buffer("test_rating_count", torch.tensor(0), persistent=False)

    def forward(self, user_ids, item_ids):
        user_vectors = self.user_embeddings(user_ids)
        item_vectors = self.item_embeddings(item_ids)
        logits = (user_vectors * item_vectors).sum(dim=1)
        return 1 + 4 * torch.sigmoid(logits)

    def training_step(self, batch, batch_index):
        user_ids, item_ids, ratings = batch
        predictions = self(user_ids, item_ids)
        squared_errors = (predictions - ratings).square()
        loss = squared_errors.mean()
        self.train_squared_error += squared_errors.detach().sum()
        self.train_rating_count += ratings.numel()
        self.log("train_mse", loss, on_epoch=True)
        return loss

    def on_train_epoch_end(self):
        train_rmse = torch.sqrt(
            self.train_squared_error / self.train_rating_count
        )
        self.log("train_rmse", train_rmse, prog_bar=True)
        self.train_squared_error.zero_()
        self.train_rating_count.zero_()

    def validation_step(self, batch, batch_index):
        user_ids, item_ids, ratings = batch
        predictions = self(user_ids, item_ids)
        squared_errors = (predictions - ratings).square()
        self.test_squared_error += squared_errors.detach().sum()
        self.test_rating_count += ratings.numel()

    def on_validation_epoch_end(self):
        test_rmse = torch.sqrt(
            self.test_squared_error / self.test_rating_count
        )
        self.log("test_rmse", test_rmse, prog_bar=True)
        self.test_squared_error.zero_()
        self.test_rating_count.zero_()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)


class RatingsDataset(Dataset):
    """Expose indexed ratings directly from contiguous NumPy arrays."""

    def __init__(self, frame):
        self.user_ids = frame["user_index"].to_numpy(
            dtype=np.int64,
            copy=False,
        )
        self.item_ids = frame["item_index"].to_numpy(
            dtype=np.int64,
            copy=False,
        )
        self.ratings = frame["rating"].to_numpy(
            dtype=np.float32,
            copy=False,
        )

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, index):
        return (
            self.user_ids[index],
            self.item_ids[index],
            self.ratings[index],
        )


def make_dataloader(frame, batch_size, shuffle):
    """Create a DataLoader from an indexed ratings frame."""

    return DataLoader(
        RatingsDataset(frame),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )


def evaluate(model, data_loader, groups):
    """Return overall and per-group test metrics."""

    model.eval()
    squared_error_sum = 0.0
    absolute_error_sum = 0.0
    rating_count = 0
    group_squared_error = {group: 0.0 for group in ("A", "B")}
    group_rating_count = {group: 0 for group in ("A", "B")}
    batch_start = 0

    with torch.no_grad():
        for user_ids, item_ids, batch_ratings in data_loader:
            user_ids = user_ids.to(model.device, non_blocking=True)
            item_ids = item_ids.to(model.device, non_blocking=True)
            batch_ratings = batch_ratings.to(
                model.device,
                non_blocking=True,
            )
            batch_predictions = model(user_ids, item_ids)
            errors = batch_predictions - batch_ratings
            squared_error_sum += errors.square().sum().item()
            absolute_error_sum += errors.abs().sum().item()
            rating_count += len(batch_ratings)
            batch_groups = groups[batch_start:batch_start + len(batch_ratings)]
            for group in ("A", "B"):
                group_mask = torch.as_tensor(
                    batch_groups == group,
                    device=errors.device,
                )
                group_squared_error[group] += (
                    errors[group_mask].square().sum().item()
                )
                group_rating_count[group] += int(group_mask.sum().item())
            batch_start += len(batch_ratings)

    group_rmse = {
        group: (
            group_squared_error[group] / group_rating_count[group]
        ) ** 0.5
        for group in ("A", "B")
    }

    return {
        "rmse": (squared_error_sum / rating_count) ** 0.5,
        "mae": absolute_error_sum / rating_count,
        "rmse_A": group_rmse["A"],
        "rmse_B": group_rmse["B"],
        "eo_rmse": abs(group_rmse["A"] - group_rmse["B"]),
    }


def load_ratings(args):
    """Load the original MovieLens ratings and user groups."""

    ratings_path = args.original_data_dir / "ratings.dat"
    users_path = args.original_data_dir / "users.dat"
    ratings = pd.read_csv(
        ratings_path,
        sep="::",
        engine="python",
        names=["user_id", "item_id", "rating", "timestamp"],
        encoding="latin-1",
    )
    users = pd.read_csv(
        users_path,
        sep="::",
        engine="python",
        names=["user_id", "gender", "age", "occupation", "zip_code"],
        encoding="latin-1",
    )
    ratings = ratings.merge(
        users[["user_id", "gender"]],
        on="user_id",
        how="inner",
    )
    ratings["group"] = ratings["gender"].map({"M": "A", "F": "B"})

    required_columns = {"user_id", "item_id", "rating", "group"}
    missing_columns = required_columns - set(ratings.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    ratings = ratings[ratings["group"].isin(["A", "B"])].copy()
    return ratings[["user_id", "item_id", "rating", "group"]]


def print_dataset_summary(ratings):
    """Print basic dataset and per-group activity statistics."""

    user_activity = ratings.groupby("user_id")["rating"].transform("size")
    user_means = ratings.groupby("user_id")["rating"].transform("mean")
    item_means = ratings.groupby("item_id")["rating"].transform("mean")
    ratings = ratings.assign(
        user_activity=user_activity,
        eccentricity=(
            ratings["rating"]
            - (user_means + item_means) / 2
        ).abs(),
    )
    user_summary = (
        ratings[["user_id", "group", "user_activity"]]
        .drop_duplicates("user_id")
        .groupby("group", sort=True)["user_activity"]
        .mean()
    )

    print("\nDataset summary")
    print("---------------")
    print(f"Examples (ratings): {len(ratings):,}")
    print(f"Users: {ratings['user_id'].nunique():,}")
    print(f"Items: {ratings['item_id'].nunique():,}")
    print(f"Mean rating: {ratings['rating'].mean():.4f}")
    print("\nPer-group statistics:")
    group_summary = (
        ratings.groupby("group", sort=True)
        .agg(
            ratings=("rating", "size"),
            users=("user_id", "nunique"),
            mean_rating=("rating", "mean"),
            mean_eccentricity=("eccentricity", "mean"),
        )
    )
    group_summary["mean_activity"] = user_summary
    group_summary = group_summary[
        [
            "ratings",
            "users",
            "mean_rating",
            "mean_activity",
            "mean_eccentricity",
        ]
    ]
    print(group_summary.to_string(float_format=lambda value: f"{value:.4f}"))


def print_test_fold_statistics(test_frame, fold):
    """Print the rating-level statistics used to balance a test fold."""

    user_activity = test_frame.groupby("user_id")["rating"].transform("size")
    user_means = test_frame.groupby("user_id")["rating"].transform("mean")
    item_means = test_frame.groupby("item_id")["rating"].transform("mean")
    test_statistics = test_frame.assign(
        activity=user_activity,
        eccentricity=(
            test_frame["rating"]
            - (user_means + item_means) / 2
        ).abs(),
    )
    overall = test_statistics[["activity", "eccentricity"]].mean()
    group_means = test_statistics.groupby("group")[
        ["activity", "eccentricity"]
    ].mean()
    print(
        f"Fold {fold:2d} test-rating statistics: "
        f"mean activity={overall['activity']:.4f}, "
        f"mean eccentricity={overall['eccentricity']:.4f}; "
        f"A activity={group_means.loc['A', 'activity']:.4f}, "
        f"A eccentricity={group_means.loc['A', 'eccentricity']:.4f}; "
        f"B activity={group_means.loc['B', 'activity']:.4f}, "
        f"B eccentricity={group_means.loc['B', 'eccentricity']:.4f}"
    )


def run_cross_validation(args):
    """Train and evaluate one independent MF model per cross-validation fold."""

    L.seed_everything(args.seed, workers=True)
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is required but is not available to PyTorch."
        )

    ratings = load_ratings(args)
    print_dataset_summary(ratings)
    ratings["user_index"], user_ids = pd.factorize(ratings["user_id"], sort=True)
    ratings["item_index"], item_ids = pd.factorize(ratings["item_id"], sort=True)
    splitter = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    fold_results = []

    for fold, (train_indices, test_indices) in enumerate(
        splitter.split(ratings), start=1
    ):
        train_frame = ratings.iloc[train_indices].copy()
        test_frame = ratings.iloc[test_indices].copy()

        if args.rebalance_split in {"train", "both"}:
            original_train_users = set(train_frame["user_id"])
            train_frame, balance_info = balance_users(
                train_frame,
                activity_quantiles=args.rebalance_activity_quantiles,
                eccentricity_quantiles=args.rebalance_eccentricity_quantiles,
                balance_over=args.rebalance_over,
                random_state=args.seed + fold,
                ratio_tolerance=args.rebalance_ratio_tolerance,
            )
            print(
                f"Fold {fold:2d}: rebalanced train users "
                f"{balance_info['original_users']:,} -> "
                f"{balance_info['balanced_users']:,}"
            )

            retained_train_users = set(train_frame["user_id"])
            removed_train_users = original_train_users - retained_train_users
            test_frame = test_frame[
                ~test_frame["user_id"].isin(removed_train_users)
            ].copy()
            print(
                f"Fold {fold:2d}: removed train users from test ratings="
                f"{len(removed_train_users):,}"
            )

        if args.rebalance_split in {"test", "both"}:
            test_frame, balance_info = balance_groups(
                test_frame,
                activity_quantiles=args.rebalance_activity_quantiles,
                eccentricity_quantiles=args.rebalance_eccentricity_quantiles,
                ratio_mode="equal",
                balance_over=args.rebalance_over,
                random_state=args.seed + fold,
                ratio_tolerance=args.rebalance_ratio_tolerance,
            )
            print(
                f"Fold {fold:2d}: rebalanced test ratings "
                f"{balance_info['original_size']:,} -> "
                f"{balance_info['balanced_size']:,}"
            )
        print_test_fold_statistics(test_frame, fold)
        model = MatrixFactorizationModel(
            num_users=len(user_ids),
            num_items=len(item_ids),
            embedding_dim=args.embedding_dim,
            learning_rate=args.learning_rate,
        )
        checkpoint_callback = ModelCheckpoint(
            dirpath=args.checkpoint_dir / f"fold_{fold}",
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
                checkpoint_callback,
                EarlyStopping(
                    monitor="test_rmse",
                    mode="min",
                    patience=5,
                )
            ],
            logger=False,
            enable_model_summary=False,
            enable_progress_bar=not args.quiet,
        )
        trainer.fit(
            model,
            train_dataloaders=make_dataloader(
                train_frame, args.batch_size, shuffle=True
            ),
            val_dataloaders=make_dataloader(
                test_frame, args.batch_size, shuffle=False
            ),
        )
        best_model = MatrixFactorizationModel.load_from_checkpoint(
            checkpoint_callback.best_model_path,
        ).to(model.device)
        metrics = evaluate(
            best_model,
            make_dataloader(test_frame, args.batch_size, shuffle=False),
            test_frame["group"].to_numpy(),
        )
        metrics["fold"] = fold
        metrics["train_ratings"] = len(train_frame)
        metrics["test_ratings"] = len(test_frame)
        fold_results.append(metrics)
        print(
            f"Fold {fold:2d}: RMSE={metrics['rmse']:.4f}, "
            f"MAE={metrics['mae']:.4f}, "
            f"RMSE_A={metrics['rmse_A']:.4f}, "
            f"RMSE_B={metrics['rmse_B']:.4f}, "
            f"EO={metrics['eo_rmse']:.4f}, "
            f"checkpoint={checkpoint_callback.best_model_path}"
        )

    results = pd.DataFrame(fold_results)
    print(f"\n{args.folds}-fold cross-validation results")
    print(results.to_string(index=False))
    print(
        "\nMean metrics: "
        f"RMSE={results['rmse'].mean():.4f} (+/- {results['rmse'].std():.4f}), "
        f"MAE={results['mae'].mean():.4f} (+/- {results['mae'].std():.4f})"
    )
    print(f"Mean EO (RMSE gap): {results['eo_rmse'].mean():.4f}")
    if args.results_file:
        results.to_csv(args.results_file, index=False)
        print(f"\nSaved fold metrics to: {args.results_file}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="K-fold MF experiments on MovieLens ratings.",
    )
    parser.add_argument(
        "--original-data-dir",
        type=Path,
        default=Path("movielens_data/ml-1m"),
        help="Directory containing the original ratings.dat and users.dat files.",
    )
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("mf_checkpoints"),
        help="Directory where the best checkpoint for each fold is saved.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=5,
        help="Epochs without test RMSE improvement before stopping.",
    )
    parser.add_argument("--batch-size", type=int, default=2**14)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--results-file",
        type=Path,
        default=Path("mf_cross_validation_results.csv"),
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--rebalance-split",
        choices=("none", "train", "test", "both"),
        default="none",
        help=(
            "Rebalance no fold, train, test, or both folds. In both mode, "
            "train is balanced first and test uses the balanced train."
        ),
    )
    parser.add_argument("--rebalance-activity-quantiles", type=int, default=10)
    parser.add_argument(
        "--rebalance-over",
        choices=("activity", "eccentricity", "both"),
        default="both",
        help="User distribution dimensions used for fold rebalancing.",
    )
    parser.add_argument(
        "--rebalance-eccentricity-quantiles",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--rebalance-ratio-tolerance",
        type=float,
        default=1.001,
        help="Maximum per-stratum multiplicative A:B user-ratio deviation.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_cross_validation(parse_args())