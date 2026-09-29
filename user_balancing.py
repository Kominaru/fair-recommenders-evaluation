"""Reusable user-level balancing utilities."""

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


GROUPS = ("A", "B")


def add_user_statistics(ratings, reference_ratings=None):
    """Return one record per candidate user with activity and eccentricity."""

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
    reference_users = reference.groupby("user_id", as_index=False).agg(
        activity=("rating", "size"),
        eccentricity=("rating_eccentricity", "mean"),
    )
    candidate_users = ratings[["user_id", "group"]].drop_duplicates("user_id")
    return candidate_users.merge(reference_users, on="user_id", how="left")


def _add_fixed_bins(users, activity_quantiles, eccentricity_quantiles, balance_over):
    users = users.copy()
    activity_edges = np.unique(
        np.quantile(users["activity"], np.linspace(0, 1, activity_quantiles + 1))
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
    if balance_over == "activity":
        users["stratum"] = users["activity_bin"]
    elif balance_over == "eccentricity":
        users["stratum"] = users["eccentricity_bin"]
    else:
        users["stratum"] = users["activity_bin"] * (len(eccentricity_edges) - 1) + users["eccentricity_bin"]
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
    """Remove complete users until user strata have similar group ratios."""

    if ratio_tolerance < 1:
        raise ValueError("ratio_tolerance must be at least 1.")
    if balance_over not in {"activity", "eccentricity", "both"}:
        raise ValueError("balance_over must be activity, eccentricity, or both.")
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
    users, activity_edges, eccentricity_edges = _add_fixed_bins(
        users, activity_quantiles, eccentricity_quantiles, balance_over
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
    try:
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
                users["active"].to_numpy() & (strata == stratum) & (group_codes == group_code)
            )
            if len(candidates) == 0:
                raise RuntimeError("No removable user found in the selected stratum.")
            removed_user = int(rng.choice(candidates))
            users.iloc[removed_user, users.columns.get_loc("active")] = False
            counts[stratum, group_code] -= 1
            progress.update(1)
    finally:
        progress.close()

    retained_users = set(users.loc[users["active"], "user_id"])
    balanced = ratings[ratings["user_id"].isin(retained_users)].reset_index(drop=True)
    return balanced, {
        "original_users": candidate_user_count,
        "excluded_missing_reference_users": int(missing_reference.sum()),
        "balanced_users": int(users["active"].sum()),
        "original_ratings": len(ratings),
        "balanced_ratings": len(balanced),
        "original_user_counts": ratings[["user_id", "group"]].drop_duplicates("user_id")["group"].value_counts().reindex(GROUPS, fill_value=0),
        "balanced_user_counts": users.loc[users["active"], "group"].value_counts().reindex(GROUPS, fill_value=0),
        "activity_edges": activity_edges,
        "eccentricity_edges": eccentricity_edges,
        "largest_ratio": largest_deviation,
        "original_users_frame": original_users,
        "balanced_users_frame": users.loc[users["active"]].copy(),
    }
