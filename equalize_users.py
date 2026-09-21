import argparse
import io
import ssl
import urllib.request
import zipfile

import certifi
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm


MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
GROUPS = ("A", "B")


def download_movielens_1m():
    """Download and load MovieLens 1M ratings with A/B group labels."""

    print("Downloading MovieLens 1M...")
    ssl_context = ssl.create_default_context(cafile=certifi.where())

    with urllib.request.urlopen(MOVIELENS_URL, context=ssl_context) as response:
        zip_data = response.read()

    with zipfile.ZipFile(io.BytesIO(zip_data)) as archive:
        ratings = pd.read_csv(
            archive.open("ml-1m/ratings.dat"),
            sep="::",
            engine="python",
            names=["user_id", "item_id", "rating", "timestamp"],
        )
        users = pd.read_csv(
            archive.open("ml-1m/users.dat"),
            sep="::",
            engine="python",
            names=["user_id", "gender", "age", "occupation", "zip_code"],
        )

    ratings = ratings.merge(
        users[["user_id", "gender"]],
        on="user_id",
        how="inner",
    )
    ratings["group"] = ratings["gender"].map({"M": "A", "F": "B"})
    return ratings[["user_id", "item_id", "rating", "group"]].dropna()


def add_user_statistics(ratings, reference_ratings=None):
    """Return one record per candidate user using optional reference data."""

    if reference_ratings is None:
        reference_ratings = ratings

    user_mean = reference_ratings.groupby("user_id")["rating"].mean()
    item_mean = reference_ratings.groupby("item_id")["rating"].mean()
    reference = reference_ratings.assign(
        user_mean=reference_ratings["user_id"].map(user_mean),
        item_mean=reference_ratings["item_id"].map(item_mean),
    )
    reference["rating_eccentricity"] = (
        reference["rating"]
        - (reference["user_mean"] + reference["item_mean"]) / 2
    ).abs()

    reference_users = (
        reference.groupby("user_id", as_index=False)
        .agg(
            activity=("rating", "size"),
            eccentricity=("rating_eccentricity", "mean"),
        )
    )
    candidate_users = ratings[["user_id", "group"]].drop_duplicates("user_id")
    users = candidate_users.merge(reference_users, on="user_id", how="left")
    return users


def print_user_group_statistics(users, title):
    """Print mean activity and eccentricity for each user group."""

    summary = (
        users.groupby("group", sort=True)
        .agg(
            users=("user_id", "size"),
            mean_activity=("activity", "mean"),
            mean_eccentricity=("eccentricity", "mean"),
        )
    )
    print(f"\n{title}")
    print(summary.to_string(float_format=lambda value: f"{value:.4f}"))


def print_group_statistics(ratings, title):
    """Print user and item exposure statistics for each group."""

    users = add_user_statistics(ratings)
    print_user_group_statistics(users, title)

    item_activity = ratings.groupby("item_id")["rating"].size()
    item_summary = ratings.assign(
        item_activity=ratings["item_id"].map(item_activity),
    )
    rating_group_summary = (
        item_summary.groupby("group", sort=True)
        .agg(
            ratings=("rating", "size"),
            unique_items=("item_id", "nunique"),
            mean_item_activity_per_rating=("item_activity", "mean"),
            mean_item_rating=("rating", "mean"),
        )
    )
    item_group_summary = (
        item_summary[["group", "item_id", "item_activity"]]
        .drop_duplicates(["group", "item_id"])
        .groupby("group", sort=True)
        .agg(mean_item_activity_per_item=("item_activity", "mean"))
    )
    item_group_summary = rating_group_summary.join(item_group_summary)
    print("\nItem exposure statistics:")
    print(
        item_group_summary.to_string(
            float_format=lambda value: f"{value:.4f}"
        )
    )


def plot_user_distributions(original_users, balanced_users):
    """Plot activity and eccentricity distributions before and after balancing."""

    combined_users = pd.concat(
        [original_users, balanced_users],
        ignore_index=True,
    )
    activity_edges = np.histogram_bin_edges(
        combined_users["activity"],
        bins=30,
    )
    eccentricity_edges = np.histogram_bin_edges(
        combined_users["eccentricity"],
        bins=30,
    )

    figure, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(12, 8),
        constrained_layout=True,
        sharex="col",
        sharey="col",
    )
    colors = {"A": "tab:blue", "B": "tab:orange"}

    for row, (title, users) in enumerate(
        [("Before balancing", original_users), ("After balancing", balanced_users)]
    ):
        for group in GROUPS:
            group_users = users[users["group"] == group]
            axes[row, 0].hist(
                group_users["activity"],
                bins=activity_edges,
                density=True,
                alpha=0.6,
                color=colors[group],
                label=f"Group {group}",
            )
            axes[row, 1].hist(
                group_users["eccentricity"],
                bins=eccentricity_edges,
                density=True,
                alpha=0.6,
                color=colors[group],
                label=f"Group {group}",
            )
        axes[row, 0].set_title(f"{title}: activity")
        axes[row, 1].set_title(f"{title}: mean eccentricity")
        axes[row, 0].legend()
        axes[row, 1].legend()

    axes[1, 0].set_xlabel("Ratings per user")
    axes[1, 1].set_xlabel("Mean absolute eccentricity per user")
    axes[0, 0].set_ylabel("Density")
    axes[1, 0].set_ylabel("Density")
    plt.show()


def add_fixed_bins(
    users,
    activity_quantiles,
    eccentricity_quantiles,
    balance_over="both",
):
    """Assign fixed quantile bins calculated from the original users."""

    users = users.copy()
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
        activity_edges[1:-1],
        users["activity"],
        side="right",
    )
    users["eccentricity_bin"] = np.searchsorted(
        eccentricity_edges[1:-1],
        users["eccentricity"],
        side="right",
    )
    if balance_over == "activity":
        users["stratum"] = users["activity_bin"]
    elif balance_over == "eccentricity":
        users["stratum"] = users["eccentricity_bin"]
    else:
        users["stratum"] = (
            users["activity_bin"] * (len(eccentricity_edges) - 1)
            + users["eccentricity_bin"]
        )
    return users, activity_edges, eccentricity_edges


def balance_users(
    ratings,
    reference_ratings=None,
    activity_quantiles=10,
    eccentricity_quantiles=10,
    random_state=42,
    ratio_tolerance=1.001,
    max_iterations=None,
    balance_over="both",
):
    """Remove users until A/B user distributions are approximately equal."""

    if ratio_tolerance < 1:
        raise ValueError("ratio_tolerance must be at least 1.")
    if balance_over not in {"activity", "eccentricity", "both"}:
        raise ValueError(
            "balance_over must be 'activity', 'eccentricity', or 'both'."
        )

    groups = set(ratings["group"].dropna().unique())
    if groups != set(GROUPS):
        raise ValueError(f"Expected groups A and B, found {sorted(groups)}.")

    users = add_user_statistics(ratings, reference_ratings)
    missing_reference = users["activity"].isna()
    candidate_user_count = len(users)
    if missing_reference.any():
        if reference_ratings is None:
            raise ValueError("User statistics could not be computed.")
        users = users.loc[~missing_reference].copy()
    original_users = users.copy()
    users, activity_edges, eccentricity_edges = add_fixed_bins(
        users,
        activity_quantiles,
        eccentricity_quantiles,
        balance_over,
    )
    users["active"] = True

    n_activity_bins = len(activity_edges) - 1
    n_eccentricity_bins = len(eccentricity_edges) - 1
    n_strata = {
        "activity": n_activity_bins,
        "eccentricity": n_eccentricity_bins,
        "both": n_activity_bins * n_eccentricity_bins,
    }[balance_over]
    group_codes = users["group"].eq("B").to_numpy(dtype=np.int64)
    strata = users["stratum"].to_numpy(dtype=np.int64)
    counts = np.zeros((n_strata, 2), dtype=np.int64)
    np.add.at(counts.ravel(), strata * 2 + group_codes, 1)

    rng = np.random.default_rng(random_state)
    iteration_limit = len(users) if max_iterations is None else max_iterations
    progress = tqdm(total=iteration_limit, desc="Removing users", unit="user")

    for iteration in range(iteration_limit + 1):
        shared = (counts[:, 0] > 0) & (counts[:, 1] > 0)
        ratios = np.full(n_strata, np.nan)
        ratios[shared] = counts[shared, 0] / counts[shared, 1]
        deviations = np.full(n_strata, 1.0)
        deviations[shared] = np.maximum(ratios[shared], 1 / ratios[shared])
        largest_deviation = float(np.nanmax(deviations[shared])) if shared.any() else np.inf
        progress.set_postfix(largest_ratio=f"{largest_deviation:.4f}")

        if not shared.any():
            raise RuntimeError("No stratum contains both groups.")
        if largest_deviation <= ratio_tolerance:
            break
        if iteration == iteration_limit:
            raise RuntimeError("User balancing did not converge within the iteration limit.")

        stratum = int(np.nanargmax(np.where(shared, deviations, np.nan)))
        group_code = 0 if ratios[stratum] > 1 else 1
        candidates = np.flatnonzero(
            users["active"].to_numpy()
            & (strata == stratum)
            & (group_codes == group_code)
        )
        if len(candidates) == 0:
            raise RuntimeError("No removable user found in the selected stratum.")

        removed_user = int(rng.choice(candidates))
        users.iloc[removed_user, users.columns.get_loc("active")] = False
        counts[stratum, group_code] -= 1
        progress.update(1)
    else:
        raise RuntimeError("User balancing did not converge.")

    progress.close()
    retained_users = set(users.loc[users["active"], "user_id"])
    balanced = ratings[ratings["user_id"].isin(retained_users)].copy()
    balanced = balanced.reset_index(drop=True)

    info = {
        "original_users": candidate_user_count,
        "excluded_missing_reference_users": int(missing_reference.sum()),
        "balanced_users": int(users["active"].sum()),
        "original_ratings": len(ratings),
        "balanced_ratings": len(balanced),
        "original_user_counts": (
            ratings[["user_id", "group"]]
            .drop_duplicates("user_id")["group"]
            .value_counts()
            .reindex(GROUPS, fill_value=0)
        ),
        "balanced_user_counts": users.loc[users["active"], "group"].value_counts().reindex(GROUPS, fill_value=0),
        "activity_edges": activity_edges,
        "eccentricity_edges": eccentricity_edges,
        "largest_ratio": largest_deviation,
        "original_users_frame": original_users,
        "balanced_users_frame": users.loc[users["active"]].copy(),
    }
    return balanced, info


def parse_args():
    parser = argparse.ArgumentParser(description="Balance MovieLens by removing users.")
    parser.add_argument("--activity-quantiles", type=int, default=10)
    parser.add_argument("--eccentricity-quantiles", type=int, default=10)
    parser.add_argument("--ratio-tolerance", type=float, default=1.001)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument(
        "--output-file",
        type=str,
        default="movielens-1m-user-balanced.csv",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ratings = download_movielens_1m()
    print_group_statistics(ratings, "Before balancing")
    balanced, info = balance_users(
        ratings,
        activity_quantiles=args.activity_quantiles,
        eccentricity_quantiles=args.eccentricity_quantiles,
        random_state=args.random_state,
        ratio_tolerance=args.ratio_tolerance,
        max_iterations=args.max_iterations,
    )
    print_group_statistics(
        balanced,
        "After balancing",
    )

    plot_user_distributions(
        info["original_users_frame"],
        info["balanced_users_frame"],
    )

    print("\nUser-level balancing results")
    print(f"Users: {info['original_users']:,} -> {info['balanced_users']:,}")
    print(f"Ratings: {info['original_ratings']:,} -> {info['balanced_ratings']:,}")
    print("Original users by group:")
    print(info["original_user_counts"])
    print("Balanced users by group:")
    print(info["balanced_user_counts"])
    print(f"Largest final A:B ratio deviation: {info['largest_ratio']:.4f}")
    balanced.to_csv(args.output_file, index=False)
    print(f"Saved: {args.output_file}")


if __name__ == "__main__":
    main()
