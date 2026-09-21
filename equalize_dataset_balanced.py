import os
import math
import random
import shutil
import ssl
import zipfile
import urllib.request

import certifi
import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"

OUTPUT_FILE = "movielens-1m-gender-balanced.csv"

RANDOM_STATE = 42

# Number of quantile bins for user activity.
N_ACTIVITY_QUANTILES = 10

# Width of eccentricity bins.
ECCENTRICITY_BIN_WIDTH = 0.5

# Maximum numerator/denominator considered when searching
# for the optimal A:B ratio.
#
# For example, max_ratio_part=500 allows ratios such as:
#   1:1
#   2:1
#   1:2
#   499:500
#   500:499
# etc.
#
# The actual ratios are reduced by gcd before being evaluated.
MAX_RATIO_PART = 500


# ============================================================
# Download MovieLens
# ============================================================

def download_movielens(output_dir="movielens_data"):
    """
    Download and extract MovieLens 1M if it is not already present.

    Returns
    -------
    str
        Path to the extracted ml-1m directory.
    """

    os.makedirs(output_dir, exist_ok=True)

    zip_path = os.path.join(output_dir, "ml-1m.zip")
    extracted_dir = os.path.join(output_dir, "ml-1m")

    ratings_path = os.path.join(extracted_dir, "ratings.dat")

    if os.path.exists(ratings_path):
        print("MovieLens 1M already downloaded.")
        return extracted_dir

    print("Downloading MovieLens 1M...")

    ssl_context = ssl.create_default_context(
        cafile=certifi.where(),
    )

    with urllib.request.urlopen(
        MOVIELENS_URL,
        context=ssl_context,
    ) as response, open(zip_path, "wb") as zip_file:
        shutil.copyfileobj(response, zip_file)

    print("Extracting MovieLens 1M...")

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(output_dir)

    print("MovieLens 1M ready.")

    return extracted_dir


# ============================================================
# Load MovieLens
# ============================================================

def load_movielens(data_dir):
    """
    Load MovieLens 1M ratings and user gender.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
            user_id
            item_id
            group
            rating
    """

    ratings_path = os.path.join(data_dir, "ratings.dat")
    users_path = os.path.join(data_dir, "users.dat")

    ratings = pd.read_csv(
        ratings_path,
        sep="::",
        engine="python",
        names=[
            "user_id",
            "item_id",
            "rating",
            "timestamp"
        ],
        encoding="latin-1"
    )

    users = pd.read_csv(
        users_path,
        sep="::",
        engine="python",
        names=[
            "user_id",
            "gender",
            "age",
            "occupation",
            "zip_code"
        ],
        encoding="latin-1"
    )

    df = ratings.merge(
        users[["user_id", "gender"]],
        on="user_id",
        how="inner"
    )

    df = df.rename(columns={"gender": "group"})

    # Keep only the two gender groups.
    df = df[df["group"].isin(["M", "F"])].copy()

    # Map to the generic A/B notation used by the balancing algorithm.
    df["group"] = df["group"].map({
        "M": "A",
        "F": "B"
    })

    return df[
        ["user_id", "item_id", "group", "rating"]
    ].reset_index(drop=True)


# ============================================================
# Calculate activity and eccentricity
# ============================================================

def add_balancing_variables(df, n_activity_quantiles=10,
                            eccentricity_bin_width=0.5):
    """
    Calculate:

        user activity
        activity quantile
        absolute rating eccentricity
        eccentricity bin

    Eccentricity is defined as:

        |r_ui - (r_u + r_i) / 2|

    where r_u is the user's mean rating and r_i is the item's
    mean rating.

    The activity quantile boundaries are calculated once on the
    original dataset.
    """

    df = df.copy()

    # --------------------------------------------------------
    # User activity
    # --------------------------------------------------------

    df["user_activity"] = (
        df.groupby("user_id")["rating"]
        .transform("size")
    )

    # --------------------------------------------------------
    # Activity quantiles
    # --------------------------------------------------------
    #
    # qcut can have problems when many users have exactly the
    # same activity. duplicates="drop" handles that case.
    #
    # We calculate this before any subsampling.
    # --------------------------------------------------------

    activity_quantile, activity_edges = pd.qcut(
        df["user_activity"],
        q=n_activity_quantiles,
        labels=False,
        retbins=True,
        duplicates="drop"
    )

    df["activity_quantile"] = activity_quantile.astype(int)

    # --------------------------------------------------------
    # User and item mean ratings
    # --------------------------------------------------------

    user_mean = (
        df.groupby("user_id")["rating"]
        .mean()
        .rename("user_mean")
    )

    item_mean = (
        df.groupby("item_id")["rating"]
        .mean()
        .rename("item_mean")
    )

    df = df.join(user_mean, on="user_id")
    df = df.join(item_mean, on="item_id")

    # --------------------------------------------------------
    # Absolute eccentricity
    # --------------------------------------------------------

    df["eccentricity"] = (
        df["rating"]
        - (df["user_mean"] + df["item_mean"]) / 2.0
    ).abs()

    # --------------------------------------------------------
    # Eccentricity bins
    # --------------------------------------------------------

    max_eccentricity = df["eccentricity"].max()

    # Bins start at zero and have fixed width.
    n_eccentricity_bins = (
        int(math.ceil(
            max_eccentricity / eccentricity_bin_width
        ))
        + 1
    )

    eccentricity_edges = (
        np.arange(
            0,
            n_eccentricity_bins * eccentricity_bin_width
            + eccentricity_bin_width,
            eccentricity_bin_width
        )
    )

    df["eccentricity_bin"] = pd.cut(
        df["eccentricity"],
        bins=eccentricity_edges,
        right=False,
        include_lowest=True,
        labels=False
    )

    # Safety for numerical edge cases.
    df["eccentricity_bin"] = (
        df["eccentricity_bin"]
        .fillna(len(eccentricity_edges) - 2)
        .astype(int)
    )

    return df, {
        "activity_edges": activity_edges,
        "eccentricity_edges": eccentricity_edges
    }


# ============================================================
# Find optimal integer A:B ratio
# ============================================================

def find_optimal_ratio(
    stratum_counts_a,
    stratum_counts_b,
    max_ratio_part=500
):
    """
    Find the integer ratio q:p that maximizes the number of
    retained ratings.

    For a ratio

        N_A / N_B = q / p

    each joint stratum s must contain:

        x_A,s = q * k_s
        x_B,s = p * k_s

    ratings.

    The largest possible k_s is:

        min(
            floor(A_s / q),
            floor(B_s / p)
        )

    Therefore:

        K = sum_s k_s

        N_A = q * K
        N_B = p * K

    and total retention is:

        N_A + N_B = (q + p) * K

    The ratio q:p is reduced to lowest terms before evaluation.
    """

    a = np.asarray(stratum_counts_a, dtype=np.int64)
    b = np.asarray(stratum_counts_b, dtype=np.int64)

    # Original group size ratio, useful for tie-breaking.
    original_a = a.sum()
    original_b = b.sum()

    original_ratio = (
        original_a / original_b
        if original_b > 0
        else 1.0
    )

    # --------------------------------------------------------
    # Generate unique reduced ratios.
    # --------------------------------------------------------

    ratios = set()

    for q in range(1, max_ratio_part + 1):
        for p in range(1, max_ratio_part + 1):

            g = math.gcd(q, p)

            q_reduced = q // g
            p_reduced = p // g

            ratios.add((q_reduced, p_reduced))

    # --------------------------------------------------------
    # Evaluate every candidate ratio.
    # --------------------------------------------------------

    best = None

    for q, p in ratios:

        # Number of common "units" that can be retained in
        # every stratum.
        k = np.minimum(
            a // q,
            b // p
        )

        K = int(k.sum())

        if K == 0:
            continue

        n_a = q * K
        n_b = p * K
        total = n_a + n_b

        ratio = n_a / n_b

        # Primary objective:
        #   maximize total retained ratings.
        #
        # Tie-break 1:
        #   maximize the smaller group.
        #
        # Tie-break 2:
        #   stay closer to the original group-size ratio.
        score = (
            total,
            min(n_a, n_b),
            -abs(math.log(ratio / original_ratio))
        )

        if best is None or score > best["score"]:
            best = {
                "q": q,
                "p": p,
                "k": k,
                "K": K,
                "n_a": n_a,
                "n_b": n_b,
                "total": total,
                "ratio": ratio,
                "score": score
            }

    if best is None:
        raise RuntimeError(
            "No feasible common distribution was found."
        )

    return best


# ============================================================
# Construct balanced dataset
# ============================================================

def balance_groups_max_retention(
    df,
    n_activity_quantiles=10,
    eccentricity_bin_width=0.5,
    max_ratio_part=500,
    random_state=42
):
    """
    Maximum-retention subsampling based on the joint distribution
    of activity quantile and absolute eccentricity.

    The final A and B sample sizes do NOT need to be equal or
    similar.

    The algorithm finds the A:B ratio that maximizes:

        N_A + N_B

    while requiring the two groups to have the same joint
    distribution over:

        (activity_quantile, eccentricity_bin)

    up to the unavoidable integer structure.

    Returns
    -------
    balanced_df : pd.DataFrame
    info : dict
    """

    rng = np.random.default_rng(random_state)

    # --------------------------------------------------------
    # Calculate activity/eccentricity variables.
    # --------------------------------------------------------

    work_df, binning_info = add_balancing_variables(
        df,
        n_activity_quantiles=n_activity_quantiles,
        eccentricity_bin_width=eccentricity_bin_width
    )

    # --------------------------------------------------------
    # Create a joint stratum ID.
    # --------------------------------------------------------

    work_df["stratum"] = list(zip(
        work_df["activity_quantile"],
        work_df["eccentricity_bin"]
    ))

    # --------------------------------------------------------
    # Get all possible strata.
    # --------------------------------------------------------

    strata = sorted(
        work_df["stratum"].unique()
    )

    stratum_to_index = {
        s: i
        for i, s in enumerate(strata)
    }

    work_df["stratum_id"] = (
        work_df["stratum"]
        .map(stratum_to_index)
        .astype(int)
    )

    # --------------------------------------------------------
    # Count A and B observations in every stratum.
    # --------------------------------------------------------

    counts = pd.crosstab(
        work_df["stratum_id"],
        work_df["group"]
    )

    counts = counts.reindex(
        index=range(len(strata)),
        columns=["A", "B"],
        fill_value=0
    )

    counts_a = counts["A"].to_numpy(dtype=np.int64)
    counts_b = counts["B"].to_numpy(dtype=np.int64)

    # --------------------------------------------------------
    # Find optimal ratio.
    # --------------------------------------------------------

    optimum = find_optimal_ratio(
        counts_a,
        counts_b,
        max_ratio_part=max_ratio_part
    )

    q = optimum["q"]
    p = optimum["p"]
    k = optimum["k"]

    # Target counts for each stratum.
    target_a = q * k
    target_b = p * k

    # --------------------------------------------------------
    # Subsample each stratum.
    # --------------------------------------------------------

    selected_indices = []

    for stratum_id in range(len(strata)):

        target_a_s = int(target_a[stratum_id])
        target_b_s = int(target_b[stratum_id])

        if target_a_s > 0:

            idx_a = work_df.index[
                (work_df["stratum_id"] == stratum_id)
                & (work_df["group"] == "A")
            ].to_numpy()

            selected_a = rng.choice(
                idx_a,
                size=target_a_s,
                replace=False
            )

            selected_indices.extend(
                selected_a.tolist()
            )

        if target_b_s > 0:

            idx_b = work_df.index[
                (work_df["stratum_id"] == stratum_id)
                & (work_df["group"] == "B")
            ].to_numpy()

            selected_b = rng.choice(
                idx_b,
                size=target_b_s,
                replace=False
            )

            selected_indices.extend(
                selected_b.tolist()
            )

    # --------------------------------------------------------
    # Build final dataframe.
    # --------------------------------------------------------

    balanced_df = (
        work_df.loc[selected_indices]
        .sample(
            frac=1.0,
            random_state=random_state
        )
        .reset_index(drop=True)
    )

    # Keep the original requested columns.
    balanced_df = balanced_df[
        ["user_id", "item_id", "group", "rating"]
    ]

    # --------------------------------------------------------
    # Diagnostics.
    # --------------------------------------------------------

    original_counts = (
        df["group"]
        .value_counts()
        .reindex(["A", "B"])
        .fillna(0)
        .astype(int)
    )

    final_counts = (
        balanced_df["group"]
        .value_counts()
        .reindex(["A", "B"])
        .fillna(0)
        .astype(int)
    )

    info = {
        **binning_info,

        "strata": strata,

        "original_group_counts": {
            "A": int(original_counts["A"]),
            "B": int(original_counts["B"])
        },

        "final_group_counts": {
            "A": int(final_counts["A"]),
            "B": int(final_counts["B"])
        },

        "retained_total": int(len(balanced_df)),

        "retention_rate": (
            len(balanced_df) / len(df)
        ),

        "ratio_q": q,
        "ratio_p": p,

        "final_ratio_A_over_B": (
            final_counts["A"] / final_counts["B"]
        ),

        "target_a_by_stratum": target_a,
        "target_b_by_stratum": target_b,

        "stratum_counts_a": counts_a,
        "stratum_counts_b": counts_b,

        "stratum_count_table": counts.copy()
    }

    return balanced_df, info


# ============================================================
# Verification
# ============================================================

def verify_balanced_distribution(
    original_df,
    balanced_df,
    n_activity_quantiles=10,
    eccentricity_bin_width=0.5
):
    """
    Independently verify that the resulting A/B distributions
    match over the joint activity/eccentricity strata.

    The variables are recalculated using the ORIGINAL dataset,
    not the subsampled dataset.
    """

    # Calculate variables on original data.
    original_with_vars, _ = add_balancing_variables(
        original_df,
        n_activity_quantiles=n_activity_quantiles,
        eccentricity_bin_width=eccentricity_bin_width
    )

    # Attach the precomputed variables to the selected ratings
    # using their original row identity.

    original_with_vars["_row_id"] = np.arange(
        len(original_with_vars)
    )

    # Build an identifier for each rating.
    #
    # MovieLens user/item/rating combinations are sufficient
    # for this dataset because each user-item pair occurs once.
    original_with_vars["rating_key"] = list(zip(
        original_with_vars["user_id"],
        original_with_vars["item_id"]
    ))

    balanced_keys = set(zip(
        balanced_df["user_id"],
        balanced_df["item_id"]
    ))

    selected = original_with_vars[
        original_with_vars["rating_key"].isin(balanced_keys)
    ].copy()

    selected["stratum"] = list(zip(
        selected["activity_quantile"],
        pd.cut(
            selected["eccentricity"],
            bins=(
                np.arange(
                    0,
                    math.ceil(
                        original_with_vars["eccentricity"].max()
                        / eccentricity_bin_width
                    ) * eccentricity_bin_width
                    + eccentricity_bin_width,
                    eccentricity_bin_width
                )
            ),
            right=False,
            include_lowest=True,
            labels=False
        )
    ))

    selected["stratum"] = selected["stratum"].apply(
        lambda x: (
            x[0],
            int(x[1])
        )
    )

    table = pd.crosstab(
        selected["stratum"],
        selected["group"]
    ).reindex(
        columns=["A", "B"],
        fill_value=0
    )

    # Normalize each group independently.
    proportions = table.div(
        table.sum(axis=0),
        axis=1
    )

    max_difference = (
        proportions["A"] - proportions["B"]
    ).abs().max()

    return table, proportions, max_difference


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("MovieLens 1M maximum-retention distribution balancing")
    print("=" * 70)

    # --------------------------------------------------------
    # Download/load data
    # --------------------------------------------------------

    data_dir = download_movielens()

    df = load_movielens(data_dir)

    print()
    print("Original dataset:")
    print(f"  Ratings: {len(df):,}")
    print(
        "  Group A (M): "
        f"{(df['group'] == 'A').sum():,}"
    )
    print(
        "  Group B (F): "
        f"{(df['group'] == 'B').sum():,}"
    )

    # --------------------------------------------------------
    # Balance
    # --------------------------------------------------------

    balanced_df, info = balance_groups_max_retention(
        df,
        n_activity_quantiles=N_ACTIVITY_QUANTILES,
        eccentricity_bin_width=ECCENTRICITY_BIN_WIDTH,
        max_ratio_part=MAX_RATIO_PART,
        random_state=RANDOM_STATE
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("Optimization result")
    print("=" * 70)

    print(
        f"A:B ratio = "
        f"{info['ratio_q']}:{info['ratio_p']}"
    )

    print(
        f"Final A/B ratio = "
        f"{info['final_ratio_A_over_B']:.6f}"
    )

    print(
        f"Final A ratings: "
        f"{info['final_group_counts']['A']:,}"
    )

    print(
        f"Final B ratings: "
        f"{info['final_group_counts']['B']:,}"
    )

    print(
        f"Total retained: "
        f"{info['retained_total']:,}"
    )

    print(
        f"Overall retention: "
        f"{100 * info['retention_rate']:.2f}%"
    )

    # --------------------------------------------------------
    # Compare with simple min() balancing
    # --------------------------------------------------------

    simple_min_total = int(
        np.minimum(
            info["stratum_counts_a"],
            info["stratum_counts_b"]
        ).sum()
    )

    simple_min_total_both_groups = (
        2 * simple_min_total
    )

    print()
    print("=" * 70)
    print("Comparison with simple min()")
    print("=" * 70)

    print(
        f"min() retained per group: "
        f"{simple_min_total:,}"
    )

    print(
        f"min() total retained: "
        f"{simple_min_total_both_groups:,}"
    )

    print(
        f"Optimized total retained: "
        f"{info['retained_total']:,}"
    )

    print(
        f"Additional ratings retained: "
        f"{info['retained_total'] - simple_min_total_both_groups:,}"
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    balanced_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print()
    print(f"Saved balanced dataset to: {OUTPUT_FILE}")

    # --------------------------------------------------------
    # Show distribution table
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("Joint-stratum counts")
    print("=" * 70)

    table = info["stratum_count_table"].copy()

    table["A"] = table["A"].astype(int)
    table["B"] = table["B"].astype(int)

    print(table.to_string())

    # --------------------------------------------------------
    # Verify normalized distributions
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("Distribution verification")
    print("=" * 70)

    verification_table = info["stratum_count_table"].copy()

    final_a = info["target_a_by_stratum"]
    final_b = info["target_b_by_stratum"]

    verification = pd.DataFrame({
        "A": final_a,
        "B": final_b
    })

    verification["A_proportion"] = (
        verification["A"]
        / verification["A"].sum()
    )

    verification["B_proportion"] = (
        verification["B"]
        / verification["B"].sum()
    )

    verification["absolute_difference"] = (
        verification["A_proportion"]
        - verification["B_proportion"]
    ).abs()

    print(
        verification[
            [
                "A",
                "B",
                "A_proportion",
                "B_proportion",
                "absolute_difference"
            ]
        ].to_string()
    )

    print()
    print(
        "Maximum proportional difference: "
        f"{verification['absolute_difference'].max():.12f}"
    )

    print()
    print("Done.")


if __name__ == "__main__":
    main()