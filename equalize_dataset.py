import argparse
import cProfile
import io
import pstats
import ssl
import zipfile
import urllib.request

import certifi
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm


MOVIELENS_URL = (
    "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
)


def plot_group_distributions(
    df,
    title_prefix,
    activity_bins=None,
    eccentricity_bins=None,
):
    """Display per-group activity and eccentricity distributions."""

    if activity_bins is None:
        activity_bins = np.histogram_bin_edges(
            df["user_activity"],
            bins=30,
        )
    if eccentricity_bins is None:
        eccentricity_bins = np.histogram_bin_edges(
            df["eccentricity"],
            bins=30,
        )

    fig, axes = plt.subplots(
        ncols=2,
        figsize=(12, 4),
        constrained_layout=True,
    )

    for group, color in [("A", "tab:blue"), ("B", "tab:orange")]:
        group_data = df[df["group"] == group]

        axes[0].hist(
            group_data["user_activity"],
            bins=activity_bins,
            density=True,
            alpha=0.6,
            label=f"Group {group}",
            color=color,
        )
        axes[1].hist(
            group_data["eccentricity"],
            bins=eccentricity_bins,
            density=True,
            alpha=0.6,
            label=f"Group {group}",
            color=color,
        )

    axes[0].set_xscale("log")
    axes[0].set_title(f"{title_prefix} User Activity")
    axes[0].set_xlabel("Ratings per user")
    axes[0].set_ylabel("Density")
    axes[1].set_title(f"{title_prefix} Absolute Eccentricity")
    axes[1].set_xlabel("Absolute eccentricity")
    axes[1].set_ylabel("Density")

    for axis in axes:
        axis.legend()

    plt.show()

    return activity_bins, eccentricity_bins


def download_movielens_1m():
    """Download and load MovieLens 1M ratings and user information."""

    print("Downloading MovieLens 1M...")

    ssl_context = ssl.create_default_context(
        cafile=certifi.where(),
    )

    with urllib.request.urlopen(
        MOVIELENS_URL,
        context=ssl_context,
    ) as response:
        zip_data = response.read()

    with zipfile.ZipFile(io.BytesIO(zip_data)) as z:
        ratings = pd.read_csv(
            z.open("ml-1m/ratings.dat"),
            sep="::",
            engine="python",
            names=[
                "user_id",
                "item_id",
                "rating",
                "timestamp",
            ],
        )

        users = pd.read_csv(
            z.open("ml-1m/users.dat"),
            sep="::",
            engine="python",
            names=[
                "user_id",
                "gender",
                "age",
                "occupation",
                "zip_code",
            ],
        )

    # We only need gender from the user information.
    users = users[["user_id", "gender"]]

    # Join gender onto the ratings.
    df = ratings.merge(
        users,
        on="user_id",
        how="inner",
    )

    # Rename gender to group so it can be passed directly to
    # the balancing function.
    df = df.rename(columns={"gender": "group"})

    return df


def add_strata(df, activity_quantiles, eccentricity_quantiles):
    """Recompute statistics and quantile bins for the current dataframe."""

    df = df.copy()
    df["user_activity"] = (
        df.groupby("user_id")["rating"].transform("size")
    )
    df["user_mean"] = (
        df.groupby("user_id")["rating"].transform("mean")
    )
    df["item_mean"] = (
        df.groupby("item_id")["rating"].transform("mean")
    )
    df["eccentricity"] = (
        df["rating"] - (df["user_mean"] + df["item_mean"]) / 2
    ).abs()
    df["activity_bin"], activity_edges = pd.qcut(
        df["user_activity"],
        q=activity_quantiles,
        duplicates="drop",
        retbins=True,
    )
    df["eccentricity_bin"], eccentricity_edges = pd.qcut(
        df["eccentricity"],
        q=eccentricity_quantiles,
        duplicates="drop",
        retbins=True,
    )
    return df, activity_edges, eccentricity_edges


def make_numpy_state(df):
    """Create numeric arrays used by the iterative balancing loop."""

    user_codes, _ = pd.factorize(df["user_id"], sort=True)
    item_codes, _ = pd.factorize(df["item_id"], sort=True)
    ratings = df["rating"].to_numpy(dtype=np.float64)
    groups = df["group"].to_numpy()
    user_counts = np.bincount(user_codes)
    user_sums = np.bincount(user_codes, weights=ratings)
    item_counts = np.bincount(item_codes)
    item_sums = np.bincount(item_codes, weights=ratings)
    user_means = user_sums / user_counts
    item_means = item_sums / item_counts
    user_rows = [np.flatnonzero(user_codes == code) for code in range(len(user_counts))]
    item_rows = [np.flatnonzero(item_codes == code) for code in range(len(item_counts))]

    return {
        "user_codes": user_codes,
        "item_codes": item_codes,
        "ratings": ratings,
        "groups": groups,
        "active": np.ones(len(df), dtype=bool),
        "user_counts": user_counts,
        "user_sums": user_sums,
        "user_means": user_means,
        "item_counts": item_counts,
        "item_sums": item_sums,
        "item_means": item_means,
        "user_rows": user_rows,
        "item_rows": item_rows,
        "row_activity": user_counts[user_codes].astype(np.float64),
        "row_eccentricity": np.abs(
            ratings - (user_means[user_codes] + item_means[item_codes]) / 2
        ),
    }


def set_fixed_bin_edges(state, activity_quantiles, eccentricity_quantiles):
    """Freeze quantile boundaries from the original ratings."""

    state["activity_edges"] = np.unique(
        np.quantile(
            state["row_activity"],
            np.linspace(0, 1, activity_quantiles + 1),
        )
    )
    state["eccentricity_edges"] = np.unique(
        np.quantile(
            state["row_eccentricity"],
            np.linspace(0, 1, eccentricity_quantiles + 1),
        )
    )
    state["activity_bins"] = np.searchsorted(
        state["activity_edges"][1:-1],
        state["row_activity"],
        side="right",
    )
    state["eccentricity_bins"] = np.searchsorted(
        state["eccentricity_edges"][1:-1],
        state["row_eccentricity"],
        side="right",
    )


def update_numpy_strata(
    state,
    activity_quantiles,
    eccentricity_quantiles,
    balance_over,
):
    """Update numeric statistics and return the current stratum per row."""

    return calculate_numpy_strata(
        state["activity_bins"],
        state["eccentricity_bins"],
        state,
        balance_over,
    )


def calculate_numpy_strata(
    activity_bins,
    eccentricity_bins,
    state,
    balance_over,
):
    """Calculate strata for all rows or a selected subset of rows."""

    activity_bin_count = len(state["activity_edges"]) - 1
    eccentricity_bin_count = len(state["eccentricity_edges"]) - 1

    if balance_over == "activity":
        strata = activity_bins
        number_of_strata = activity_bin_count
    elif balance_over == "eccentricity":
        strata = eccentricity_bins
        number_of_strata = eccentricity_bin_count
    else:
        strata = (
            activity_bins * eccentricity_bin_count + eccentricity_bins
        )
        number_of_strata = activity_bin_count * eccentricity_bin_count

    return strata, number_of_strata


def update_numpy_cache(state, row_index):
    """Update cached activity and eccentricity for affected active rows."""

    user_code = state["user_codes"][row_index]
    item_code = state["item_codes"][row_index]
    state["user_counts"][user_code] -= 1
    state["user_sums"][user_code] -= state["ratings"][row_index]
    state["item_counts"][item_code] -= 1
    state["item_sums"][item_code] -= state["ratings"][row_index]

    if state["user_counts"][user_code] > 0:
        state["user_means"][user_code] = (
            state["user_sums"][user_code] / state["user_counts"][user_code]
        )
    if state["item_counts"][item_code] > 0:
        state["item_means"][item_code] = (
            state["item_sums"][item_code] / state["item_counts"][item_code]
        )

    affected_rows = np.union1d(
        state["user_rows"][user_code],
        state["item_rows"][item_code],
    )
    affected_rows = affected_rows[state["active"][affected_rows]]
    state["row_activity"][state["user_rows"][user_code]] = (
        state["user_counts"][user_code]
    )
    user_rows = state["user_rows"][user_code]
    state["activity_bins"][user_rows] = np.searchsorted(
        state["activity_edges"][1:-1],
        state["row_activity"][user_rows],
        side="right",
    )
    state["row_eccentricity"][affected_rows] = np.abs(
        state["ratings"][affected_rows]
        - (
            state["user_means"][state["user_codes"][affected_rows]]
            + state["item_means"][state["item_codes"][affected_rows]]
        )
        / 2
    )
    state["eccentricity_bins"][affected_rows] = np.searchsorted(
        state["eccentricity_edges"][1:-1],
        state["row_eccentricity"][affected_rows],
        side="right",
    )
    return affected_rows


def balance_groups(
    df,
    activity_quantiles=10,
    eccentricity_quantiles=10,
    random_state=42,
    plot_distributions=False,
    balance_over="both",
    ratio_mode="best",
    max_iterations=None,
    ratio_tolerance=1.001,
):
    """
    Balance groups A and B within selected quantile strata based on:

        1. Quantile bins of ratings per user
        2. Quantile bins of absolute rating eccentricity

    Absolute eccentricity:

        |r_ui - (r_u + r_i) / 2|

    User/item means and user activity are calculated from the
    original dataset before any observations are removed.

    Balancing is iterative: strata are recomputed after every removal,
    and one observation is removed from the group with the largest current
    difference from the selected target ratio.

    ``balance_over`` selects activity bins, eccentricity bins, or both.
    ``ratio_mode="best"`` uses the largest feasible initial global A:B
    ratio, which must be greater than 1. ``ratio_mode="equal"`` targets
    A:B = 1.

    Set plot_distributions to True to display the original
    per-group activity and eccentricity distributions.
    """

    required_columns = {
        "user_id",
        "item_id",
        "group",
        "rating",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    groups = set(df["group"].dropna().unique())

    if groups != {"A", "B"}:
        raise ValueError(
            "The balancing function expects groups A and B. "
            f"Found: {sorted(groups)}"
        )

    if activity_quantiles < 1:
        raise ValueError(
            "activity_quantiles must be at least 1."
        )

    if eccentricity_quantiles < 1:
        raise ValueError(
            "eccentricity_quantiles must be at least 1."
        )

    if balance_over not in {"activity", "eccentricity", "both"}:
        raise ValueError(
            "balance_over must be 'activity', 'eccentricity', or 'both'."
        )

    if ratio_mode not in {"best", "equal"}:
        raise ValueError(
            "ratio_mode must be 'best' or 'equal'."
        )

    if ratio_tolerance < 1:
        raise ValueError(
            "ratio_tolerance must be at least 1."
        )

    df, activity_edges, eccentricity_edges = add_strata(
        df,
        activity_quantiles,
        eccentricity_quantiles,
    )
    original_df = df.copy()

    strata_columns = {
        "activity": ["activity_bin"],
        "eccentricity": ["eccentricity_bin"],
        "both": ["activity_bin", "eccentricity_bin"],
    }[balance_over]

    state = make_numpy_state(original_df)
    set_fixed_bin_edges(
        state,
        activity_quantiles,
        eccentricity_quantiles,
    )
    group_codes = (state["groups"] == "B").astype(np.int64)
    strata, number_of_strata = update_numpy_strata(
        state,
        activity_quantiles,
        eccentricity_quantiles,
        balance_over,
    )
    state["strata"] = strata.copy()
    counts = np.bincount(
        strata * 2 + group_codes,
        minlength=number_of_strata * 2,
    ).reshape(number_of_strata, 2)
    common_strata = counts[(counts[:, 0] > 0) & (counts[:, 1] > 0)]

    if len(common_strata) == 0:
        raise ValueError(
            "No joint stratum contains both groups."
        )

    if ratio_mode == "equal":
        target_ratio = 1.0
    else:
        if (common_strata[:, 0] <= common_strata[:, 1]).any():
            raise ValueError(
                "No ratio greater than 1 is feasible: group A must have "
                "more ratings than group B in every shared stratum."
            )
        target_ratio = (
            common_strata[:, 0] / common_strata[:, 1]
        ).min()

    rng = np.random.RandomState(random_state)
    iteration_limit = (
        len(df)
        if max_iterations is None
        else max_iterations
    )
    iteration = 0
    progress = tqdm(
        total=iteration_limit,
        desc="Balancing",
        unit="removed",
    )

    while True:
        strata = state["strata"]
        shared_mask = (counts[:, 0] > 0) & (counts[:, 1] > 0)

        if not shared_mask.any():
            raise ValueError(
                "Balancing removed all shared strata before converging."
            )

        differences = np.full(number_of_strata, -np.inf)
        current_ratios = np.zeros(number_of_strata)
        current_ratios[shared_mask] = (
            counts[shared_mask, 0] / counts[shared_mask, 1]
        )
        differences[shared_mask] = np.maximum(
            current_ratios[shared_mask] / target_ratio,
            target_ratio / current_ratios[shared_mask],
        )
        largest_difference = float(differences.max())
        progress.set_postfix(
            largest_ratio=f"{largest_difference:.4f}",
            removed=iteration,
        )

        if largest_difference <= ratio_tolerance:
            break

        stratum_number = int(np.argmax(differences))
        group_code = (
            0
            if current_ratios[stratum_number] > target_ratio
            else 1
        )
        removable_rows = np.flatnonzero(
            state["active"]
            & (strata == stratum_number)
            & (group_codes == group_code)
        )
        row_to_remove = int(rng.choice(removable_rows))
        removed_key = state["strata"][row_to_remove] * 2 + group_code
        state["active"][row_to_remove] = False
        affected_rows = update_numpy_cache(state, row_to_remove)
        counts.flat[removed_key] -= 1

        old_strata = state["strata"][affected_rows]
        new_strata, _ = calculate_numpy_strata(
            state["activity_bins"][affected_rows],
            state["eccentricity_bins"][affected_rows],
            state,
            balance_over,
        )
        old_keys = old_strata * 2 + group_codes[affected_rows]
        new_keys = new_strata * 2 + group_codes[affected_rows]
        moved = old_keys != new_keys
        np.add.at(counts.ravel(), old_keys[moved], -1)
        np.add.at(counts.ravel(), new_keys[moved], 1)
        state["strata"][affected_rows] = new_strata
        iteration += 1
        progress.update(1)

        if (
            max_iterations is not None
            and iteration >= max_iterations
        ):
            progress.close()
            raise RuntimeError(
                f"Stopped after {max_iterations} iterations for profiling."
            )

        if iteration >= iteration_limit:
            raise RuntimeError(
                "Iterative balancing did not converge within the iteration limit."
            )

    progress.close()

    df = original_df.iloc[np.flatnonzero(state["active"])].copy()
    balanced, _, _ = add_strata(
        df,
        activity_quantiles,
        eccentricity_quantiles,
    )
    active_rows = np.flatnonzero(state["active"])
    balanced["activity_bin"] = state["activity_bins"][active_rows]
    balanced["eccentricity_bin"] = state["eccentricity_bins"][active_rows]
    activity_edges = state["activity_edges"]
    eccentricity_edges = state["eccentricity_edges"]
    global_ratio = target_ratio

    # Use the final recomputed dataframe for all post-balancing statistics.
    df = balanced

    counts = (
        df.groupby(
            strata_columns + ["group"],
            observed=True,
        )
        .size()
        .unstack(fill_value=0)
    )

    counts["B_retained"] = counts["B"]
    counts["A_retained"] = counts["A"]

    # ------------------------------------------------------------
    # 11. Post-balancing stratum ratios
    # ------------------------------------------------------------

    # Recompute the statistics and bins after sampling. The retained ratings
    # can change user activity, user means, and item means.

    balanced["user_activity"] = (
        balanced.groupby("user_id")["rating"]
        .transform("size")
    )
    balanced["user_mean"] = (
        balanced.groupby("user_id")["rating"]
        .transform("mean")
    )
    balanced["item_mean"] = (
        balanced.groupby("item_id")["rating"]
        .transform("mean")
    )
    balanced["eccentricity"] = (
        balanced["rating"]
        - (balanced["user_mean"] + balanced["item_mean"]) / 2
    ).abs()

    if plot_distributions:
        plot_data = pd.concat(
            [original_df, balanced],
            ignore_index=True,
        )
        activity_plot_bins = np.histogram_bin_edges(
            plot_data["user_activity"],
            bins=30,
        )
        eccentricity_plot_bins = np.histogram_bin_edges(
            plot_data["eccentricity"],
            bins=30,
        )
        plot_group_distributions(
            original_df,
            "Original",
            activity_bins=activity_plot_bins,
            eccentricity_bins=eccentricity_plot_bins,
        )
        plot_group_distributions(
            balanced,
            "Balanced",
            activity_bins=activity_plot_bins,
            eccentricity_bins=eccentricity_plot_bins,
        )

    balanced_stratum_counts = (
        balanced.groupby(
            strata_columns + ["group"],
            observed=True,
        )
        .size()
        .unstack(fill_value=0)
    )

    for group in ["A", "B"]:
        if group not in balanced_stratum_counts.columns:
            balanced_stratum_counts[group] = 0

    shared_balanced_strata = balanced_stratum_counts[
        (balanced_stratum_counts["A"] > 0)
        & (balanced_stratum_counts["B"] > 0)
    ]

    balanced_stratum_ratios = (
        shared_balanced_strata["A"]
        / shared_balanced_strata["B"]
    )

    # ------------------------------------------------------------
    # 12. Statistics
    # ------------------------------------------------------------

    original_counts = (
        original_df["group"]
        .value_counts()
        .sort_index()
    )

    balanced_counts = (
        balanced["group"]
        .value_counts()
        .sort_index()
    )

    discarded_counts = (
        original_counts
        .subtract(
            balanced_counts,
            fill_value=0,
        )
        .astype(int)
    )

    info = {
        "original_size": len(original_df),
        "balanced_size": len(balanced),
        "original_group_counts": original_counts,
        "balanced_group_counts": balanced_counts,
        "discarded_group_counts": discarded_counts,
        "activity_quantiles": activity_quantiles,
        "activity_edges": activity_edges,
        "eccentricity_quantiles": eccentricity_quantiles,
        "eccentricity_edges": eccentricity_edges,
        "balance_over": balance_over,
        "ratio_mode": ratio_mode,
        "global_a_to_b_ratio": global_ratio,
        "balanced_stratum_ratios": balanced_stratum_ratios,
        "balanced_min_a_to_b_ratio": (
            balanced_stratum_ratios.min()
        ),
        "balanced_max_a_to_b_ratio": (
            balanced_stratum_ratios.max()
        ),
        "number_of_strata": len(counts),
        "number_of_used_strata": int(
            (counts["B_retained"] > 0).sum()
        ),
    }

    return balanced, info


def main():

    parser = argparse.ArgumentParser(
        description="Equalize MovieLens ratings across groups."
    )
    parser.add_argument(
        "--balance-over",
        choices=("activity", "eccentricity", "both"),
        default="both",
        help=(
            "Dimensions used for balancing strata."
        ),
    )
    parser.add_argument(
        "--ratio-mode",
        choices=("best", "equal"),
        default="best",
        help=(
            "Find the largest feasible A:B ratio greater than 1, "
            "or force equal group sizes."
        ),
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Profile balancing and stop after 2,000 iterations.",
    )
    parser.add_argument(
        "--profile-output",
        default="equalize_dataset.prof",
        help="Path for the cProfile output file.",
    )
    parser.add_argument(
        "--ratio-tolerance",
        type=float,
        default=1.01,
        help="Maximum accepted multiplicative A:B ratio deviation.",
    )
    args = parser.parse_args()

    # =========================================================
    # Download MovieLens
    # =========================================================

    df = download_movielens_1m()

    print()
    print("MovieLens 1M:")
    print(f"Ratings: {len(df):,}")
    print(f"Users:   {df['user_id'].nunique():,}")
    print(f"Items:   {df['item_id'].nunique():,}")

    print()
    print("Original gender distribution:")
    print(df["group"].value_counts())

    # =========================================================
    # Map gender to A/B
    #
    # MovieLens:
    #   M = male
    #   F = female
    # =========================================================

    df["group"] = df["group"].map({
        "M": "A",
        "F": "B",
    })

    # =========================================================
    # Balance
    #
    # 10 quantile-based user-activity and eccentricity bins
    # =========================================================

    profiler = cProfile.Profile() if args.profile else None
    if profiler is not None:
        profiler.enable()

    try:
        balanced, info = balance_groups(
            df,
            activity_quantiles=10,
            eccentricity_quantiles=10,
            random_state=42,
            plot_distributions=True,
            balance_over=args.balance_over,
            ratio_mode=args.ratio_mode,
            max_iterations=2000 if args.profile else None,
            ratio_tolerance=args.ratio_tolerance,
        )
    except RuntimeError as error:
        if not args.profile:
            raise
        print(error)
        return
    finally:
        if profiler is not None:
            profiler.disable()
            profiler.dump_stats(args.profile_output)
            print()
            print(f"Profile saved to: {args.profile_output}")
            pstats.Stats(profiler).sort_stats("cumulative").print_stats(30)

    # =========================================================
    # Results
    # =========================================================

    print()
    print("=" * 60)
    print("BALANCING RESULTS")
    print("=" * 60)

    print()
    print(
        f"Original ratings: {info['original_size']:,}"
    )

    print(
        f"Balanced ratings: {info['balanced_size']:,}"
    )

    print(
        f"Balance over: {info['balance_over']}"
    )

    print(
        f"Ratio mode: {info['ratio_mode']}"
    )

    print()
    print("Original groups:")
    print(info["original_group_counts"])

    print()
    print("Balanced groups:")
    print(info["balanced_group_counts"])

    print()
    print("Discarded ratings:")
    print(info["discarded_group_counts"])

    print()
    print(
        "Target A:B ratio: "
        f"{info['global_a_to_b_ratio']:.4f}"
    )
    print(
        "Rebalanced stratum A:B ratio range: "
        f"{info['balanced_min_a_to_b_ratio']:.4f} "
        "to "
        f"{info['balanced_max_a_to_b_ratio']:.4f}"
    )

    print()
    print(
        f"Joint strata: {info['number_of_strata']}"
    )

    print(
        f"Used strata: {info['number_of_used_strata']}"
    )

    # =========================================================
    # Verify joint distribution
    # =========================================================

    joint_distribution = pd.crosstab(
        [balanced[column] for column in {
            "activity": ["activity_bin"],
            "eccentricity": ["eccentricity_bin"],
            "both": ["activity_bin", "eccentricity_bin"],
        }[args.balance_over]],
        balanced["group"],
    )

    print()
    print("=" * 60)
    print("JOINT DISTRIBUTION")
    print("=" * 60)

    print(joint_distribution)

    # =========================================================
    # Verify exact equality
    # =========================================================

    if "A" not in joint_distribution.columns:
        joint_distribution["A"] = 0

    if "B" not in joint_distribution.columns:
        joint_distribution["B"] = 0

    is_balanced = (
        joint_distribution["A"]
        == joint_distribution["B"]
    ).all()

    print()
    print(
        f"Joint distribution balanced: {is_balanced}"
    )

    # =========================================================
    # Save
    # =========================================================

    output_file = (
        "movielens-1m-gender-balanced-"
        f"over-{args.balance_over}-ratio-{args.ratio_mode}.csv"
    )

    balanced.to_csv(
        output_file,
        index=False,
    )

    print()
    print(f"Saved balanced dataset to: {output_file}")


if __name__ == "__main__":
    main()