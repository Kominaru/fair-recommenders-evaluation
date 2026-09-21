"""Run MF experiments after balancing user activity and eccentricity.

The script performs shuffled K-fold cross-validation on MovieLens ratings.
For each training fold it computes fixed activity/eccentricity bins once per
user, removes complete users from overrepresented group strata, and removes
the same users from the test fold. It then trains and evaluates a matrix
factorization model on the retained users.

Diagnostics printed for every fold compare average ratings per user and two
forms of eccentricity before and after balancing. A corresponding histogram
figure is saved in the configured diagnostics directory.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from run_mf_experiments import (
    MatrixFactorizationModel,
    evaluate,
    load_ratings,
    make_dataloader,
)
from equalize_users import add_user_statistics

logging.getLogger("torch.utils.flop_counter").setLevel(logging.ERROR)

def _rating_statistics(ratings):
    """Return activity and absolute eccentricity for every rating row."""

    user_means = ratings.groupby("user_id")["rating"].transform("mean")
    item_means = ratings.groupby("item_id")["rating"].transform("mean")
    return pd.DataFrame(
        {
            "activity": ratings.groupby("user_id")["rating"].transform("size"),
            "eccentricity": (
                ratings["rating"] - (user_means + item_means) / 2
            ).abs(),
        },
        index=ratings.index,
    )


def _diagnostic_statistics(ratings):
    """Return rating-level statistics and one activity record per user."""

    statistics = _rating_statistics(ratings)
    user_activity = ratings.groupby("user_id")["rating"].size()
    user_eccentricity = statistics["eccentricity"].groupby(
        ratings["user_id"]
    ).mean()
    per_user = ratings[["user_id", "group"]].drop_duplicates("user_id").copy()
    per_user["activity"] = per_user["user_id"].map(user_activity)
    per_user["eccentricity"] = per_user["user_id"].map(user_eccentricity)
    return statistics, per_user


def _print_and_plot_diagnostics(before, after, fold, output_dir):
    """Print fold diagnostics and save before/after comparison histograms."""

    import matplotlib.pyplot as plt

    before_statistics, before_users = _diagnostic_statistics(before)
    after_statistics, after_users = _diagnostic_statistics(after)
    frames = {"before": (before_statistics, before_users), "after": (after_statistics, after_users)}

    print(f"\nFold {fold} balancing diagnostics")
    for stage, (statistics, users) in frames.items():
        print(f"{stage.capitalize()}:")
        activity = users.groupby("group")["activity"].mean()
        user_eccentricity = users.groupby("group")["eccentricity"].mean()
        rating_eccentricity = statistics.assign(group=before["group"] if stage == "before" else after["group"]).groupby("group")["eccentricity"].mean()
        print(
            "  average ratings per user: "
            f"A={activity.get('A', float('nan')):.4f}, "
            f"B={activity.get('B', float('nan')):.4f}, "
            f"global={users['activity'].mean():.4f}"
        )
        print(
            "  average eccentricity, user-wise: "
            f"A={user_eccentricity.get('A', float('nan')):.4f}, "
            f"B={user_eccentricity.get('B', float('nan')):.4f}, "
            f"global={users['eccentricity'].mean():.4f}"
        )
        print(
            "  average eccentricity, all ratings: "
            f"A={rating_eccentricity.get('A', float('nan')):.4f}, "
            f"B={rating_eccentricity.get('B', float('nan')):.4f}, "
            f"global={statistics['eccentricity'].mean():.4f}"
        )

    figure, axes = plt.subplots(3, 2, figsize=(14, 13), constrained_layout=True)
    colors = {"A": "tab:blue", "B": "tab:orange"}
    specifications = (
        ("activity", "Ratings per user", "activity"),
        ("eccentricity", "Mean eccentricity per user", "user_eccentricity"),
        ("eccentricity", "Eccentricity per rating", "rating_eccentricity"),
    )
    for column, (stage, (statistics, users)) in enumerate(frames.items()):
        for row, (_, title, kind) in enumerate(specifications):
            if kind == "activity":
                values = users["activity"]
                groups = users["group"]
            elif kind == "user_eccentricity":
                values = users["eccentricity"]
                groups = users["group"]
            else:
                values = statistics["eccentricity"]
                groups = before["group"] if stage == "before" else after["group"]
            for group in ("A", "B"):
                axes[row, column].hist(
                    values[groups == group],
                    bins=30,
                    alpha=0.6,
                    color=colors[group],
                    label=f"Group {group}",
                )
            axes[row, column].set_title(f"{stage.capitalize()}: {title}")
            axes[row, column].set_xlabel(title)
            axes[row, column].set_ylabel("Users" if kind != "rating_eccentricity" else "Ratings")
            axes[row, column].legend()

    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_dir / f"fold_{fold}_diagnostics.png", dpi=150)
    plt.close(figure)


def _fixed_bins(values, quantiles):
    """Create fixed quantile edges and integer bins."""

    edges = np.unique(
        np.quantile(values, np.linspace(0, 1, quantiles + 1))
    )
    bins = np.searchsorted(edges[1:-1], values, side="right")
    return edges, bins


def _balance_user_strata(
    ratings,
    balance_over,
    activity_quantiles,
    eccentricity_quantiles,
    ratio_tolerance,
    random_state,
    max_iterations,
):
    """Remove users until every shared user stratum has a balanced ratio."""

    original = ratings.reset_index(drop=True).copy()
    users = add_user_statistics(original).reset_index(drop=True)
    activity_edges, _ = _fixed_bins(users["activity"].to_numpy(), activity_quantiles)
    eccentricity_edges, _ = _fixed_bins(
        users["eccentricity"].to_numpy(), eccentricity_quantiles
    )
    activity_bins = np.searchsorted(
        activity_edges[1:-1], users["activity"], side="right"
    )
    eccentricity_bins = np.searchsorted(
        eccentricity_edges[1:-1], users["eccentricity"], side="right"
    )
    if balance_over == "activity":
        strata = activity_bins
        number_of_strata = len(activity_edges) - 1
    elif balance_over == "eccentricity":
        strata = eccentricity_bins
        number_of_strata = len(eccentricity_edges) - 1
    else:
        eccentricity_bin_count = len(eccentricity_edges) - 1
        strata = activity_bins * eccentricity_bin_count + eccentricity_bins
        number_of_strata = (len(activity_edges) - 1) * eccentricity_bin_count

    group_codes = users["group"].eq("B").to_numpy(dtype=np.int64)
    counts = np.zeros((number_of_strata, 2), dtype=np.int64)
    np.add.at(counts.ravel(), strata * 2 + group_codes, 1)
    active = np.ones(len(users), dtype=bool)
    rng = np.random.default_rng(random_state)
    iteration_limit = len(users) if max_iterations is None else max_iterations

    for iteration in range(iteration_limit + 1):
        shared = (counts[:, 0] > 0) & (counts[:, 1] > 0)
        ratios = np.full(number_of_strata, np.nan)
        ratios[shared] = counts[shared, 0] / counts[shared, 1]
        deviations = np.full(number_of_strata, np.nan)
        deviations[shared] = np.maximum(ratios[shared], 1 / ratios[shared])
        largest_deviation = float(np.nanmax(deviations)) if shared.any() else np.inf
        if not shared.any():
            raise RuntimeError("No user stratum contains both groups.")
        if largest_deviation <= ratio_tolerance:
            break
        if iteration == iteration_limit:
            raise RuntimeError("User balancing did not converge.")

        stratum = int(np.nanargmax(deviations))
        group_code = 0 if ratios[stratum] > 1 else 1
        candidates = np.flatnonzero(
            active & (strata == stratum) & (group_codes == group_code)
        )
        rng.shuffle(candidates)
        removed = None
        for candidate in candidates:
            removed = int(candidate)
            break
        if removed is None:
            raise RuntimeError("No removable user found in the selected stratum.")
        active[removed] = False
        counts[strata[removed], group_codes[removed]] -= 1
    else:
        raise RuntimeError("User balancing did not converge.")

    balanced = original[original["user_id"].isin(users.loc[active, "user_id"])].copy()
    balanced.reset_index(drop=True, inplace=True)
    final_counts = balanced["group"].value_counts().reindex(["A", "B"], fill_value=0)
    original_user_counts = users["group"].value_counts().reindex(["A", "B"], fill_value=0)
    balanced_user_counts = users.loc[active, "group"].value_counts().reindex(
        ["A", "B"], fill_value=0
    )
    original_rating_counts = original["group"].value_counts().reindex(["A", "B"], fill_value=0)
    return balanced, {
        "original_ratings": len(original),
        "balanced_ratings": len(balanced),
        "removed_users": int((~active).sum()),
        "original_ratio": original_rating_counts["A"] / original_rating_counts["B"],
        "balanced_ratio": final_counts["A"] / final_counts["B"],
        "original_user_ratio": original_user_counts["A"] / original_user_counts["B"],
        "balanced_user_ratio": balanced_user_counts["A"] / balanced_user_counts["B"],
        "original_user_counts": original_user_counts,
        "balanced_user_counts": balanced_user_counts,
        "balanced_group_counts": final_counts,
        "iterations": iteration,
        "activity_edges": activity_edges,
        "eccentricity_edges": eccentricity_edges,
        "balance_over": balance_over,
    }


def balance_users_preserving_ratio(
    ratings,
    balance_over="both",
    activity_quantiles=10,
    eccentricity_quantiles=10,
    ratio_tolerance=1.001,
    random_state=42,
    max_iterations=None,
):
    """Balance user strata while removing complete users.

    The returned ``original_ratio`` and ``balanced_ratio`` describe rating
    counts. The corresponding ``*_user_ratio`` fields describe user counts;
    the balancing criterion applies to the latter within each fixed stratum.
    """

    if balance_over not in {"activity", "eccentricity", "both"}:
        raise ValueError("balance_over must be activity, eccentricity, or both.")
    if ratio_tolerance < 1:
        raise ValueError("ratio_tolerance must be at least 1.")
    if set(ratings["group"].dropna().unique()) != {"A", "B"}:
        raise ValueError("ratings must contain both groups A and B.")

    return _balance_user_strata(
        ratings,
        balance_over,
        activity_quantiles,
        eccentricity_quantiles,
        ratio_tolerance,
        random_state,
        max_iterations,
    )

def parse_args():
    """Parse dataset, balancing, model, and output configuration."""

    parser = argparse.ArgumentParser(
        description="MF experiments with user-level activity/eccentricity balancing."
    )
    parser.add_argument("--original-data-dir", type=Path, default=Path("movielens_data/ml-1m"))
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2**14)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("mf_user_ratio_checkpoints"))
    parser.add_argument("--results-file", type=Path, default=Path("mf_user_ratio_results.csv"))
    parser.add_argument("--activity-quantiles", type=int, default=10)
    parser.add_argument("--eccentricity-quantiles", type=int, default=10)
    parser.add_argument("--balance-over", choices=("activity", "eccentricity", "both"), default="both")
    parser.add_argument("--ratio-tolerance", type=float, default=1.001)
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--diagnostics-dir", type=Path, default=Path("mf_user_ratio_diagnostics"))
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def run(args):
    """Balance each training fold, train MF, evaluate, and save results."""

    import torch
    import lightning as L
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

    L.seed_everything(args.seed, workers=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required but is not available to PyTorch.")
    ratings = load_ratings(args)
    ratings["user_index"], user_ids = pd.factorize(ratings["user_id"], sort=True)
    ratings["item_index"], item_ids = pd.factorize(ratings["item_id"], sort=True)
    splitter = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    fold_results = []

    for fold, (train_indices, test_indices) in enumerate(splitter.split(ratings), start=1):
        train_frame = ratings.iloc[train_indices].copy()
        test_frame = ratings.iloc[test_indices].copy()
        train_frame, balance_info = balance_users_preserving_ratio(
            train_frame,
            balance_over=args.balance_over,
            activity_quantiles=args.activity_quantiles,
            eccentricity_quantiles=args.eccentricity_quantiles,
            ratio_tolerance=args.ratio_tolerance,
            random_state=args.seed + fold,
            max_iterations=args.max_iterations,
        )
        _print_and_plot_diagnostics(
            ratings.iloc[train_indices],
            train_frame,
            fold,
            args.diagnostics_dir,
        )
        removed_users = set(ratings.iloc[train_indices]["user_id"]) - set(train_frame["user_id"])
        test_frame = test_frame[~test_frame["user_id"].isin(removed_users)].copy()
        print(
            f"Fold {fold:2d}: train ratings {balance_info['original_ratings']:,} -> "
            f"{balance_info['balanced_ratings']:,}; "
            f"A:B {balance_info['original_ratio']:.4f} -> "
            f"{balance_info['balanced_ratio']:.4f}; "
            f"removed users={balance_info['removed_users']:,}"
        )
        model = MatrixFactorizationModel(len(user_ids), len(item_ids), args.embedding_dim, args.learning_rate)
        checkpoint = ModelCheckpoint(
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
            callbacks=[checkpoint, EarlyStopping(monitor="test_rmse", mode="min", patience=5)],
            logger=False,
            enable_model_summary=False,
            enable_progress_bar=not args.quiet,
        )
        trainer.fit(
            model,
            train_dataloaders=make_dataloader(train_frame, args.batch_size, True),
            val_dataloaders=make_dataloader(test_frame, args.batch_size, False),
        )
        best_model = MatrixFactorizationModel.load_from_checkpoint(checkpoint.best_model_path).to(model.device)
        metrics = evaluate(best_model, make_dataloader(test_frame, args.batch_size, False), test_frame["group"].to_numpy())
        metrics.update({"fold": fold, "train_ratings": len(train_frame), "test_ratings": len(test_frame)})
        fold_results.append(metrics)

    results = pd.DataFrame(fold_results)
    print(results.to_string(index=False))
    results.to_csv(args.results_file, index=False)


if __name__ == "__main__":
    run(parse_args())
