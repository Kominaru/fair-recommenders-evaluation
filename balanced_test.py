"""Evaluate MF and baseline predictors under weighted test constructions."""

import argparse
from dataclasses import dataclass
import logging
from pathlib import Path
import warnings
logging.getLogger("torch.utils.flop_counter").setLevel(logging.ERROR)
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


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


def _stratum_importance_weights(frame, strata):
    """Assign per-rating weights for a common A/B stratum distribution."""

    working = frame.assign(_stratum=strata)
    counts = pd.crosstab(working["_stratum"], working["group"])
    counts = counts.reindex(columns=["A", "B"], fill_value=0).sort_index()
    totals = counts.sum(axis=0)
    normalized = counts.divide(totals.replace(0, np.nan), axis="columns")
    shared = (counts["A"] > 0) & (counts["B"] > 0)
    target = normalized.loc[shared, ["A", "B"]].min(axis=1)
    if not target.sum():
        return np.ones(len(working), dtype=float), {
            "counts_before": counts.to_dict("index"),
            "common_shape": {},
            "zero_weight_ratings": 0,
        }
    target = target / target.sum()

    weights = np.zeros(len(working), dtype=float)
    for group in ("A", "B"):
        group_mask = working["group"].to_numpy() == group
        group_share = normalized[group]
        valid = group_mask & working["_stratum"].isin(target.index).to_numpy()
        weights[valid] = (
            target.reindex(working.loc[valid, "_stratum"]).to_numpy()
            / group_share.reindex(working.loc[valid, "_stratum"]).to_numpy()
        )

    return weights, {
        "counts_before": counts.to_dict("index"),
        "common_shape": target.to_dict(),
        "zero_weight_ratings": int((weights == 0).sum()),
    }


def balance_test_ratings(
    train_frame,
    test_frame,
    condition,
    activity_quantiles=10,
    eccentricity_quantiles=10,
    random_state=42,
):
    """Reweight eligible test ratings without removing any test examples."""

    eligible_test, unavailable = _test_ratings_with_training_entities(
        train_frame, test_frame
    )
    unavailable_user_ratings = unavailable["unavailable_user_ratings"]
    unavailable_item_ratings = unavailable["unavailable_item_ratings"]
    if eligible_test.empty:
        raise RuntimeError("No test ratings belong to users present in training.")

    if condition.name == "none":
        result = eligible_test.reset_index(drop=True)
        result["test_weight"] = 1.0
        return result, {
            "excluded_ratings": unavailable_user_ratings + unavailable_item_ratings,
            "unavailable_user_ratings": unavailable_user_ratings,
            "unavailable_item_ratings": unavailable_item_ratings,
            "zero_weight_ratings": 0,
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
    weights, weighting_info = _stratum_importance_weights(rating_frame, strata)
    result = eligible_test.reset_index(drop=True)
    result["test_weight"] = weights
    return result, {
        "excluded_ratings": unavailable_user_ratings + unavailable_item_ratings,
        "unavailable_user_ratings": unavailable_user_ratings,
        "unavailable_item_ratings": unavailable_item_ratings,
        "zero_weight_ratings": weighting_info["zero_weight_ratings"],
        "strata": len(weighting_info["common_shape"]),
        "stratum_counts_before": weighting_info["counts_before"],
        "stratum_counts_after": weighting_info["common_shape"],
        "common_shape": weighting_info["common_shape"],
    }

TEST_POPULATIONS = ("none", "activity", "eccentricity", "both")


def _user_training_eccentricity(train_frame, test_frame):
    """Each test rating's user-level eccentricity e_u from training only.

    Unlike the rating-level eccentricity, this does not use the test label.
    """

    user_means = train_frame.groupby("user_id")["rating"].mean()
    item_means = train_frame.groupby("item_id")["rating"].mean()
    eccentricity = (
        train_frame["rating"]
        - (
            train_frame["user_id"].map(user_means)
            + train_frame["item_id"].map(item_means)
        )
        / 2
    ).abs()
    user_eccentricity = eccentricity.groupby(train_frame["user_id"]).mean()
    return test_frame["user_id"].map(user_eccentricity)


def joint_test_reference(
    train_frame,
    test_frame,
    activity_quantiles=10,
    eccentricity_quantiles=10,
    eccentricity="rating",
):
    """Fix joint test strata and the per-group target of every test population.

    Statistics and pooled bin edges come from the full training fold and its
    eligible test ratings, so every model is scored against the same
    populations. For group g over joint strata s = (a, e) the targets are
    ``none`` p_g(s), ``activity`` p_g(e | a) q_a, ``eccentricity``
    p_g(a | e) q_e and ``both`` q_s, where q is the normalized minimum of the
    two groups' shares over strata both groups occupy. On the full model these
    reproduce ``balance_test_ratings`` for the same bins.

    ``eccentricity="rating"`` stratifies on each test rating's eccentricity,
    which uses its label; ``"user"`` stratifies on the user's mean training
    eccentricity instead, a label-free covariate on the same scale as the
    train-side strata.
    """

    eligible_test, _ = _test_ratings_with_training_entities(train_frame, test_frame)
    statistics = _training_rating_statistics(train_frame, eligible_test)
    if eccentricity == "user":
        statistics["eccentricity"] = _user_training_eccentricity(
            train_frame, eligible_test
        )
    elif eccentricity != "rating":
        raise ValueError("eccentricity must be 'rating' or 'user'.")
    activity_edges = _quantile_edges(
        statistics["activity"].to_numpy(), activity_quantiles
    )
    eccentricity_edges = _quantile_edges(
        statistics["eccentricity"].to_numpy(), eccentricity_quantiles
    )
    eccentricity_count = max(1, len(eccentricity_edges) - 1)
    activity_bins = _fixed_bins(statistics["activity"].to_numpy(), activity_edges)
    eccentricity_bins = _fixed_bins(
        statistics["eccentricity"].to_numpy(), eccentricity_edges
    )
    strata = activity_bins * eccentricity_count + eccentricity_bins
    eligible_test = eligible_test.assign(_stratum=strata)

    shares = pd.crosstab(eligible_test["_stratum"], eligible_test["group"])
    shares = shares.reindex(columns=["A", "B"], fill_value=0)
    shares = shares / shares.sum(axis=0)
    activity_of = pd.Series(shares.index // eccentricity_count, index=shares.index)
    eccentricity_of = pd.Series(shares.index % eccentricity_count, index=shares.index)

    def common(frame):
        shared = (frame["A"] > 0) & (frame["B"] > 0)
        target = frame.min(axis=1).where(shared, 0.0)
        return target / target.sum()

    targets = {"none": shares}
    for name, factor in (("activity", activity_of), ("eccentricity", eccentricity_of)):
        marginal = shares.groupby(factor).sum()
        conditional = shares / marginal.reindex(factor).to_numpy()
        common_marginal = common(marginal).reindex(factor).to_numpy()
        targets[name] = (
            conditional.multiply(common_marginal, axis=0)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )
    joint = common(shares)
    targets["both"] = pd.DataFrame({"A": joint, "B": joint})
    return {"test": eligible_test, "targets": targets}


def weight_test_population(model_train_frame, reference, population):
    """Weight one model's evaluable test ratings to a fixed test population.

    Only ratings whose user and item occur in ``model_train_frame`` can be
    evaluated. Within each group they are reweighted so that their joint
    stratum distribution equals the population's target; target strata that a
    group no longer covers are dropped and the remaining target renormalized.
    """

    test = reference["test"]
    evaluable = test[
        test["user_id"].isin(set(model_train_frame["user_id"]))
        & test["item_id"].isin(set(model_train_frame["item_id"]))
    ]
    if population == "both_own":
        # Joint target rebuilt from this model's evaluable ratings, with the
        # fixed bins: the per-condition alternative to the fixed target.
        shares = pd.crosstab(evaluable["_stratum"], evaluable["group"])
        shares = shares.reindex(columns=["A", "B"], fill_value=0)
        shares = shares / shares.sum(axis=0)
        shared = (shares["A"] > 0) & (shares["B"] > 0)
        joint = shares.min(axis=1).where(shared, 0.0)
        joint = joint / joint.sum()
        target = pd.DataFrame({"A": joint, "B": joint})
    else:
        target = reference["targets"][population]
    weights = np.zeros(len(evaluable), dtype=float)
    coverage = {}
    for group in ("A", "B"):
        mask = (evaluable["group"] == group).to_numpy()
        strata = evaluable.loc[mask, "_stratum"]
        counts = strata.value_counts()
        group_target = target[group][target[group] > 0]
        covered = group_target[group_target.index.isin(counts.index)]
        coverage[group] = float(covered.sum())
        if covered.empty:
            raise RuntimeError(f"Group {group} covers no target stratum.")
        covered = covered / covered.sum()
        shares = counts / counts.sum()
        weights[mask] = (
            covered.reindex(strata).fillna(0).to_numpy()
            / shares.reindex(strata).to_numpy()
        )
    result = evaluable.drop(columns="_stratum").reset_index(drop=True)
    result["test_weight"] = weights
    return result, {
        "zero_weight_ratings": int((weights == 0).sum()),
        "target_coverage_A": coverage["A"],
        "target_coverage_B": coverage["B"],
    }


def standardized_test_reference(
    train_frame,
    test_frame,
    activity_quantiles=10,
    eccentricity_quantiles=10,
):
    """Fixed test populations for scoring training conditions alike."""

    return joint_test_reference(
        train_frame, test_frame, activity_quantiles, eccentricity_quantiles
    )


def standardize_test_ratings(model_train_frame, reference):
    """Weight one condition's evaluable test ratings to the joint target."""

    return weight_test_population(model_train_frame, reference, "both")


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


def evaluate_predictions(test_frame, predictions):
    """Evaluate fixed predictions with the test frame's importance weights."""

    actual = test_frame["rating"].to_numpy(dtype=float)
    weights = test_frame["test_weight"].to_numpy(dtype=float)
    groups = test_frame["group"].to_numpy()
    errors = predictions - actual
    total_weight = weights.sum()
    group_rmse = {}
    for group in ("A", "B"):
        group_weights = weights[groups == group]
        group_errors = errors[groups == group]
        group_rmse[group] = np.sqrt(
            np.sum(group_weights * group_errors**2) / group_weights.sum()
        )
    return {
        "rmse": np.sqrt(np.sum(weights * errors**2) / total_weight),
        "mae": np.sum(weights * np.abs(errors)) / total_weight,
        "rmse_A": group_rmse["A"],
        "rmse_B": group_rmse["B"],
        # Positive EO means lower error for group A.
        "eo_rmse": group_rmse["B"] - group_rmse["A"],
    }


def baseline_predictions(train_frame, test_frame, random_state):
    """Return uniform-random and user/item-average test predictions."""

    rng = np.random.default_rng(random_state)
    user_means = train_frame.groupby("user_id")["rating"].mean()
    item_means = train_frame.groupby("item_id")["rating"].mean()
    user_part = test_frame["user_id"].map(user_means).to_numpy(dtype=float)
    item_part = test_frame["item_id"].map(item_means).to_numpy(dtype=float)
    # A rating of a user or item that the fit lacks (its only training rating
    # was held out for validation) falls back to the other mean, then to the
    # global mean; without this the group RMSE becomes NaN.
    global_mean = train_frame["rating"].mean()
    user_part = np.where(np.isnan(user_part), np.where(np.isnan(item_part), global_mean, item_part), user_part)
    item_part = np.where(np.isnan(item_part), user_part, item_part)
    average_prediction = (user_part + item_part) / 2
    return {
        "random_uniform": rng.uniform(1.0, 5.0, len(test_frame)),
        "user_item_average": average_prediction,
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
        default=(300.0, 400.0),
        metavar=("A", "B"),
        help="Target Normal means for artificial activity populations A and B.",
    )
    parser.add_argument(
        "--artificial-activity-stds",
        type=float,
        nargs=2,
        default=(200.0, 200.0),
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
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2**14)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=727)
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
        "--reuse-checkpoints",
        action="store_true",
        help="Reuse existing fold checkpoints instead of retraining identical MF models.",
    )
    parser.add_argument(
        "--results-file", type=Path, default=Path("mf_balanced_test.csv")
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def run(args):
    import lightning as L
    import torch

    from mf_train_eval import evaluate_checkpoint, train_model

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
        existing_checkpoint = (
            args.checkpoint_dir / f"fold_{fold}" / "best.ckpt"
        )
        if args.reuse_checkpoints and existing_checkpoint.exists():
            checkpoint_path = str(existing_checkpoint)
        else:
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
            common_metadata = {
                "condition": condition.name,
                "fold": fold,
                "train_ratings": len(train_frame),
                "train_users": train_frame["user_id"].nunique(),
                "test_ratings": len(balanced_test),
                "test_users": balanced_test["user_id"].nunique(),
                "excluded_test_ratings": balance_info["excluded_ratings"],
                "unavailable_user_ratings": balance_info[
                    "unavailable_user_ratings"
                ],
                "unavailable_item_ratings": balance_info[
                    "unavailable_item_ratings"
                ],
                "zero_weight_ratings": balance_info["zero_weight_ratings"],
                "test_strata": balance_info["strata"],
                "checkpoint": checkpoint_path,
                **distribution_metrics,
            }
            results.append({**common_metadata, **metrics, "predictor": "mf"})
            print(
                f"Fold {fold:2d} {condition.name:14s} MF: "
                f"train={len(train_frame):,}, test={len(balanced_test):,}, "
                f"EO={metrics['eo_rmse']:.3f}"
            )
            for predictor, predictions in baseline_predictions(
                train_frame, balanced_test, args.seed + fold
            ).items():
                baseline_metrics = evaluate_predictions(balanced_test, predictions)
                results.append(
                    {**common_metadata, **baseline_metrics, "predictor": predictor}
                )
                print(
                    f"Fold {fold:2d} {condition.name:14s} {predictor}: "
                    f"EO={baseline_metrics['eo_rmse']:.3f}"
                )

    results = pd.DataFrame(results)
    results.to_csv(args.results_file, index=False)
    print(f"Saved results to {args.results_file}")
    print(
        results.groupby(["predictor", "condition"])[
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
