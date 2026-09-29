"""Train MF models after balancing test ratings using fixed training statistics."""

import argparse
from dataclasses import dataclass
import logging
from pathlib import Path
import warnings
logging.getLogger("torch.utils.flop_counter").setLevel(logging.ERROR)
import torch
import lightning as L
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


import lightning as L
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold

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
from sklearn.model_selection import KFold

from dataset_loader import load_movielens_ratings
from mf_train_eval import evaluate_checkpoint, train_model
from user_balancing import add_user_statistics




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


@dataclass(frozen=True)
class TestBalanceCondition:
    name: str
    dimensions: tuple[str, ...]


CONDITIONS = (
    TestBalanceCondition("none", ()),
    TestBalanceCondition("activity", ("activity",)),
    TestBalanceCondition("eccentricity", ("eccentricity",)),
    TestBalanceCondition("both", ("activity", "eccentricity")),
)


def _quantile_edges(values, quantiles):
    return np.unique(np.quantile(values, np.linspace(0, 1, quantiles + 1)))


def _fixed_bins(values, edges):
    bins = np.searchsorted(edges[1:-1], values, side="right")
    return np.clip(bins, 0, max(0, len(edges) - 2))


def _group_ks_test(values, groups):
    """Return the two-sample KS test result for groups A and B."""

    values = np.asarray(values)
    group_labels = np.asarray(groups)
    group_a = values[group_labels == "A"]
    group_b = values[group_labels == "B"]
    return ks_2samp(group_a, group_b)


def _sample_matching_histogram(frame, strata, random_state):
    """Sample A and B with the same normalized histogram over strata."""

    working = frame.copy()
    working["_matching_stratum"] = strata
    counts = pd.crosstab(working["_matching_stratum"], working["group"])
    for group in ("A", "B"):
        if group not in counts:
            counts[group] = 0
    counts = counts[["A", "B"]].sort_index()
    shared = (counts["A"] > 0) & (counts["B"] > 0)
    if not shared.any():
        raise RuntimeError(
            "No stratum contains ratings from both groups; matching the "
            "group distributions is impossible."
        )
    counts = counts.loc[shared]
    totals = counts.sum(axis=0)
    normalized = counts.divide(totals, axis="columns")
    common_shape = normalized.min(axis=1)
    common_shape = common_shape / common_shape.sum()
    target_totals = {
        group: int(np.floor((counts[group] / common_shape).min()))
        for group in ("A", "B")
    }
    target_counts = {}
    for group in ("A", "B"):
        raw_counts = target_totals[group] * common_shape
        integer_counts = np.floor(raw_counts).astype(int)
        remainder = target_totals[group] - int(integer_counts.sum())
        if remainder:
            fractional = (raw_counts - integer_counts).sort_values(
                ascending=False
            )
            for stratum in fractional.index:
                if remainder == 0:
                    break
                if integer_counts.loc[stratum] < counts.loc[stratum, group]:
                    integer_counts.loc[stratum] += 1
                    remainder -= 1
        target_counts[group] = integer_counts
    rng = np.random.default_rng(random_state)
    retained_parts = []
    for stratum in counts.index:
        stratum_rows = working[working["_matching_stratum"] == stratum]
        for group in ("A", "B"):
            retained_parts.append(
                stratum_rows[stratum_rows["group"] == group].sample(
                    n=int(target_counts[group].loc[stratum]),
                    random_state=rng,
                )
            )
    balanced = pd.concat(retained_parts, axis=0).sort_index()
    return balanced.drop(
        columns=[column for column in balanced if column.startswith("_")]
    ).reset_index(drop=True), {
        "counts_before": counts.to_dict("index"),
        "target_totals": target_totals,
        "common_shape": common_shape.to_dict(),
    }


def _training_user_statistics(train_frame, test_frame):

    """Map fixed training activity statistics onto test users."""

    train_users = add_user_statistics(train_frame).set_index("user_id")
    test_user_ids = test_frame["user_id"].drop_duplicates()
    missing = test_user_ids[~test_user_ids.isin(train_users.index)]
    if not missing.empty:
        raise RuntimeError(
            "Test contains users absent from training, so fixed training "
            f"statistics are unavailable: {missing.tolist()[:5]}"
        )
    return train_users.loc[test_user_ids]


def _test_ratings_with_training_entities(train_frame, test_frame):
    """Keep test ratings whose user and item occur in the training fold."""

    train_user_ids = set(train_frame["user_id"])
    train_item_ids = set(train_frame["item_id"])
    has_training_user = test_frame["user_id"].isin(train_user_ids)
    has_training_item = test_frame["item_id"].isin(train_item_ids)
    eligible = test_frame[has_training_user & has_training_item].copy()
    return eligible, {
        "unavailable_user_ratings": int((~has_training_user).sum()),
        "unavailable_item_ratings": int(
            (has_training_user & ~has_training_item).sum()
        ),
    }


def _training_rating_statistics(train_frame, test_frame):
    """Return one fixed-statistics record for every test-rating row.

    Activity is the training rating count of the rating's user and is repeated
    for every test rating by that user. Eccentricity is computed independently
    for each test rating.
    """

    test_users = _training_user_statistics(train_frame, test_frame)
    user_means = train_frame.groupby("user_id")["rating"].mean()
    item_means = train_frame.groupby("item_id")["rating"].mean()
    user_mean = test_frame["user_id"].map(user_means)
    item_mean = test_frame["item_id"].map(item_means)
    if user_mean.isna().any() or item_mean.isna().any():
        raise RuntimeError(
            "Training statistics are missing for a test user or item."
        )
    return pd.DataFrame(
        {
            "activity": test_frame["user_id"].map(test_users["activity"]),
            "eccentricity": (
                test_frame["rating"] - (user_mean + item_mean) / 2
            ).abs(),
        },
        index=test_frame.index,
    )


def balance_test_ratings(
    train_frame,
    test_frame,
    condition,
    activity_quantiles=10,
    eccentricity_quantiles=10,
    random_state=42,
):
    """Balance test ratings without changing training-derived user statistics.

    For a selected condition, every shared fixed activity/eccentricity stratum
    is sampled toward the eligible test set's overall A:B ratio. Integer
    rounding is used for the sampled count, so the resulting ratios can differ
    slightly. Strata containing only one group cannot satisfy the constraint
    and are excluded.
    """

    eligible_test, unavailable = _test_ratings_with_training_entities(
        train_frame, test_frame
    )
    unavailable_user_ratings = unavailable["unavailable_user_ratings"]
    unavailable_item_ratings = unavailable["unavailable_item_ratings"]
    if eligible_test.empty:
        raise RuntimeError("No test ratings belong to users present in training.")

    if condition.name == "none":
        return eligible_test.reset_index(drop=True), {
            "removed_ratings": unavailable_user_ratings + unavailable_item_ratings,
            "unavailable_user_ratings": unavailable_user_ratings,
            "unavailable_item_ratings": unavailable_item_ratings,
            "strata": 1,
            "stratum_counts_before": {0: len(eligible_test)},
            "stratum_counts_after": {0: len(eligible_test)},
            "stratum_ratios_before": {},
            "stratum_ratios_after": {},
        }

    test_statistics = _training_rating_statistics(train_frame, eligible_test)
    rating_frame = eligible_test.copy()
    rating_frame["_activity"] = test_statistics["activity"].to_numpy()
    rating_frame["_eccentricity"] = test_statistics["eccentricity"].to_numpy()
    activity_edges = _quantile_edges(
        rating_frame["_activity"].to_numpy(), activity_quantiles
    )
    activity_bins = _fixed_bins(
        rating_frame["_activity"].to_numpy(), activity_edges
    )
    eccentricity_edges = _quantile_edges(
        rating_frame["_eccentricity"].to_numpy(), eccentricity_quantiles
    )
    eccentricity_bins = _fixed_bins(
        rating_frame["_eccentricity"].to_numpy(), eccentricity_edges
    )

    if condition.dimensions == ("activity",):
        strata = activity_bins
    elif condition.dimensions == ("eccentricity",):
        strata = eccentricity_bins
    else:
        eccentricity_count = max(1, len(eccentricity_edges) - 1)
        strata = activity_bins * eccentricity_count + eccentricity_bins
    balanced, matching_info = _sample_matching_histogram(
        rating_frame, strata, random_state
    )
    return balanced, {
        "removed_ratings": len(test_frame) - len(balanced),
        "unavailable_user_ratings": unavailable_user_ratings,
        "unavailable_item_ratings": unavailable_item_ratings,
        "strata": len(matching_info["common_shape"]),
        "stratum_counts_before": matching_info["counts_before"],
        "stratum_counts_after": {},
        "target_totals": matching_info["target_totals"],
        "common_shape": matching_info["common_shape"],
    }

def print_test_statistics(
    train_frame,
    test_frame,
    label,
):
    """Print test counts and fixed training-derived activity statistics."""

    rating_statistics = _training_rating_statistics(train_frame, test_frame)
    statistics = test_frame.assign(
        activity=rating_statistics["activity"],
        eccentricity=rating_statistics["eccentricity"],
    )
    activity_test = _group_ks_test(
        statistics["activity"].to_numpy(),
        statistics["group"],
    )
    eccentricity_test = _group_ks_test(
        statistics["eccentricity"].to_numpy(),
        statistics["group"],
    )
    summary = statistics.groupby("group").agg(
        ratings=("rating", "size"),
        users=("user_id", "nunique"),
        mean_rating=("rating", "mean"),
        rating_mean_activity=("activity", "mean"),
        rating_std_activity=("activity", "std"),
        rating_eccentricity=("eccentricity", "mean"),
        rating_std_eccentricity=("eccentricity", "std"),
    ).reindex(["A", "B"])
    print(f"\n{label}")
    print(
        f"  total: ratings={len(test_frame):,}, "
        f"users={test_frame['user_id'].nunique():,}, "
        f"mean_rating={test_frame['rating'].mean():.3f}"
    )
    print(
        f"  overall: rating_mean_activity={statistics['activity'].mean():.3f}, "
        f"rating_std_activity={statistics['activity'].std():.3f}, "
        f"rating_eccentricity={statistics['eccentricity'].mean():.3f}, "
        f"rating_std_eccentricity={statistics['eccentricity'].std():.3f}"
    )
    print(
        f"  A/B KS statistic: activity={activity_test.statistic:.3f}, "
        f"eccentricity={eccentricity_test.statistic:.3f}"
    )
    print(
        f"  A/B KS p-value: activity={activity_test.pvalue:.3f}, "
        f"eccentricity={eccentricity_test.pvalue:.3f}"
    )
    print(summary.to_string(float_format=lambda value: f"{value:.3f}"))
    return {
        "activity_ks": activity_test.statistic,
        "eccentricity_ks": eccentricity_test.statistic,
    }


def parse_condition_names(value):
    names = tuple(name.strip() for name in value.split(",") if name.strip())
    available = {condition.name for condition in CONDITIONS}
    unknown = set(names) - available
    if unknown:
        raise argparse.ArgumentTypeError(
            f"Unknown conditions: {sorted(unknown)}; choose from {sorted(available)}"
        )
    return names


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare MF models after balancing test ratings."
    )
    parser.add_argument(
        "--original-data-dir", type=Path, default=Path("movielens_data/ml-1m")
    )
    parser.add_argument(
        "--group-by",
        choices=("gender", "age", "artificial_activity", "artificial_eccentricity"),
        default="gender",
        help="User attribute defining groups A and B.",
    )
    parser.add_argument(
        "--age-threshold",
        type=int,
        default=45,
        help="Age split threshold: A is age > threshold; B is age <= threshold.",
    )
    parser.add_argument(
        "--artificial-activity-means",
        type=float,
        nargs=2,
        default=(200.0, 400.0),
        metavar=("A", "B"),
        help="Target Normal means for artificial activity populations A and B.",
    )
    parser.add_argument(
        "--artificial-activity-stds",
        type=float,
        nargs=2,
        default=(50.0, 50.0),
        metavar=("A", "B"),
        help="Normal standard deviations for artificial activity populations.",
    )
    parser.add_argument(
        "--artificial-eccentricity-means",
        type=float,
        nargs=2,
        default=(0.65, 0.85),
        metavar=("A", "B"),
        help="Target Normal means for artificial eccentricity populations A and B.",
    )
    parser.add_argument(
        "--artificial-eccentricity-stds",
        type=float,
        nargs=2,
        default=(0.15, 0.15),
        metavar=("A", "B"),
        help="Normal standard deviations for artificial eccentricity populations.",
    )
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2**14)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--activity-quantiles", type=int, default=10)
    parser.add_argument("--eccentricity-quantiles", type=int, default=10)
    parser.add_argument(
        "--conditions",
        type=parse_condition_names,
        default=tuple(condition.name for condition in CONDITIONS),
        help="Comma-separated conditions: none, activity, eccentricity, both.",
    )
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=Path("mf_balanced_test")
    )
    parser.add_argument(
        "--results-file", type=Path, default=Path("mf_balanced_test.csv")
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def run(args):
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
        random_state=args.seed,
    )
    ratings["user_index"], user_ids = pd.factorize(ratings["user_id"], sort=True)
    ratings["item_index"], item_ids = pd.factorize(ratings["item_id"], sort=True)
    args.num_users = len(user_ids)
    args.num_items = len(item_ids)

    condition_map = {condition.name: condition for condition in CONDITIONS}
    splitter = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    results = []
    for fold, (train_indices, test_indices) in enumerate(
        splitter.split(ratings), start=1
    ):
        train_frame = ratings.iloc[train_indices].copy()
        test_frame = ratings.iloc[test_indices].copy()
        checkpoint_path = train_model(
            train_frame=train_frame,
            validation_frame=test_frame,
            num_users=args.num_users,
            num_items=args.num_items,
            embedding_dim=args.embedding_dim,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            checkpoint_dir=args.checkpoint_dir / f"fold_{fold}",
            early_stopping_patience=5,
            quiet=True,
        )
        for condition_name in args.conditions:
            condition = condition_map[condition_name]
            balanced_test, balance_info = balance_test_ratings(
                train_frame,
                test_frame,
                condition,
                activity_quantiles=args.activity_quantiles,
                eccentricity_quantiles=args.eccentricity_quantiles,
                random_state=args.seed + fold,
            )
            if balanced_test.empty:
                raise RuntimeError(
                    f"Fold {fold} condition {condition.name} produced an empty test set."
                )
            distribution_metrics = print_test_statistics(
                train_frame,
                balanced_test,
                f"Fold {fold} condition {condition.name}: test statistics",
            )
            metrics = evaluate_checkpoint(
                checkpoint_path=checkpoint_path,
                test_frame=balanced_test,
                batch_size=args.batch_size,
            )
            metrics.update(
                {
                    "condition": condition.name,
                    "fold": fold,
                    "train_ratings": len(train_frame),
                    "train_users": train_frame["user_id"].nunique(),
                    "test_ratings": len(balanced_test),
                    "test_users": balanced_test["user_id"].nunique(),
                    "removed_test_ratings": balance_info["removed_ratings"],
                    "unavailable_user_ratings": balance_info[
                        "unavailable_user_ratings"
                    ],
                    "unavailable_item_ratings": balance_info[
                        "unavailable_item_ratings"
                    ],
                    "test_strata": balance_info["strata"],
                    "checkpoint": checkpoint_path,
                    **distribution_metrics,
                }
            )
            results.append(metrics)
            print(
                f"Fold {fold:2d} {condition.name:14s}: "
                f"train={len(train_frame):,}, test={len(balanced_test):,}, "
                f"EO={metrics['eo_rmse']:.3f}"
            )

    results = pd.DataFrame(results)
    results.to_csv(args.results_file, index=False)
    print(f"Saved results to {args.results_file}")
    print(
        results.groupby("condition")[
            [
                "rmse",
                "mae",
                "eo_rmse",
                "activity_ks",
                "eccentricity_ks",
                "test_ratings",
            ]
        ]
        .agg(["mean", "std"])
        .to_string(float_format=lambda value: f"{value:.3f}")
    )


if __name__ == "__main__":
    run(parse_args())
