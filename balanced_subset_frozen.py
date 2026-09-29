"""Fixed-strata user-removal samplers for MF experiments.

User activity/eccentricity strata are computed once from the input frame and
remain fixed while users are removed.
"""

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix
from tqdm.auto import tqdm

from balanced_subset import BalanceCondition, CONDITIONS
from user_balancing import add_user_statistics


def _deviations(counts, target_ratio, ratio_mode, eligible=None):
    shared = (counts[:, 0] > 0) & (counts[:, 1] > 0)
    used = (counts[:, 0] > 0) | (counts[:, 1] > 0)
    ratios = counts[:, 0] / np.maximum(counts[:, 1], 1)
    deviations = np.zeros(len(counts), dtype=float)
    if eligible is None:
        eligible = np.ones(len(counts), dtype=bool)
    deviations[used & ~shared] = 0
    deviations[~eligible] = 0
    if ratio_mode == "absolute":
        deviations[shared] = np.abs(ratios[shared] - target_ratio)
    else:
        deviations[shared] = np.maximum(
            ratios[shared] / target_ratio,
            target_ratio / ratios[shared],
        )
    return shared, ratios, deviations


def _size_window(target_size, tolerance):
    if target_size is None:
        return None, None
    if tolerance < 0:
        raise ValueError("target_size_ratio_tolerance must be non-negative.")
    return (
        int(np.ceil(target_size * (1 - tolerance))),
        int(np.floor(target_size * (1 + tolerance))),
    )


def balance_users(
    ratings,
    condition,
    target_size=None,
    target_size_ratio_tolerance=0.001,
    activity_quantiles=10,
    eccentricity_quantiles=10,
    ratio_tolerance=1.001,
    random_state=42,
    max_iterations=None,
    protected_user_ids=None,
):
    """Select users with a mixed-integer model and fixed initial strata."""

    if set(ratings["group"].dropna().unique()) != {"A", "B"}:
        raise ValueError("ratings must contain both groups A and B.")
    if ratio_tolerance < 0 or (
        condition.ratio_mode == "multiplicative" and ratio_tolerance < 1
    ):
        raise ValueError("Invalid ratio_tolerance for the selected condition.")

    users = add_user_statistics(ratings).reset_index(drop=True)
    user_ids = users["user_id"].to_numpy()
    protected_user_ids = set() if protected_user_ids is None else set(protected_user_ids)
    unknown = protected_user_ids - set(user_ids)
    if unknown:
        raise ValueError("Protected users must occur in the ratings frame.")
    protected = np.isin(user_ids, list(protected_user_ids))
    user_index = pd.Index(user_ids)
    rating_user_codes = user_index.get_indexer(ratings["user_id"])
    user_rating_counts = np.bincount(
        rating_user_codes, minlength=len(users)
    ).astype(np.int64)
    groups = (users["group"].to_numpy() == "B").astype(np.int64)

    activity_edges = np.unique(
        np.quantile(users["activity"], np.linspace(0, 1, activity_quantiles + 1))
    )
    eccentricity_edges = np.unique(
        np.quantile(
            users["eccentricity"],
            np.linspace(0, 1, eccentricity_quantiles + 1),
        )
    )
    activity_bins = np.searchsorted(
        activity_edges[1:-1], users["activity"], side="right"
    )
    eccentricity_bins = np.searchsorted(
        eccentricity_edges[1:-1], users["eccentricity"], side="right"
    )
    activity_count = max(1, len(activity_edges) - 1)
    eccentricity_count = max(1, len(eccentricity_edges) - 1)
    if condition.dimensions == ():
        strata = np.zeros(len(users), dtype=np.int64)
        number_of_strata = 1
    elif condition.dimensions == ("activity",):
        strata = activity_bins
        number_of_strata = activity_count
    elif condition.dimensions == ("eccentricity",):
        strata = eccentricity_bins
        number_of_strata = eccentricity_count
    else:
        strata = activity_bins * eccentricity_count + eccentricity_bins
        number_of_strata = activity_count * eccentricity_count

    original_counts = ratings["group"].value_counts()
    target_ratio = (
        1.0
        if condition.equal_groups
        else original_counts["A"] / original_counts["B"]
    )

    minimum_size, maximum_size = _size_window(
        target_size, target_size_ratio_tolerance
    )
    protected_rating_count = int(user_rating_counts[protected].sum())
    if maximum_size is not None and protected_rating_count > maximum_size:
        raise RuntimeError("Protected users exceed the requested target size.")

    n_users = len(users)
    objective = -user_rating_counts.astype(float)
    objective += np.random.default_rng(random_state).uniform(0, 1e-6, n_users)
    lower_bounds = np.zeros(n_users)
    upper_bounds = np.ones(n_users)
    lower_bounds[protected] = 1
    upper_bounds[protected] = 1
    row_indices = []
    column_indices = []
    coefficients = []
    constraint_lower = []
    constraint_upper = []

    def add_constraint(indices, values, lower, upper):
        row = len(constraint_lower)
        row_indices.extend([row] * len(indices))
        column_indices.extend(indices)
        coefficients.extend(values)
        constraint_lower.append(lower)
        constraint_upper.append(upper)

    if condition.equal_groups:
        for stratum in range(number_of_strata):
            members = np.flatnonzero(strata == stratum)
            if len(members):
                members_a = members[groups[members] == 0]
                members_b = members[groups[members] == 1]
                add_constraint(
                    np.r_[members_a, members_b],
                    np.r_[np.ones(len(members_a)), -ratio_tolerance * np.ones(len(members_b))],
                    -np.inf,
                    0,
                )
                add_constraint(
                    np.r_[members_a, members_b],
                    np.r_[-ratio_tolerance * np.ones(len(members_a)), np.ones(len(members_b))],
                    -np.inf,
                    0,
                )
    elif condition.global_ratio_constraint:
        members_a = np.flatnonzero(groups == 0)
        members_b = np.flatnonzero(groups == 1)
        lower_ratio = target_ratio - ratio_tolerance
        upper_ratio = target_ratio + ratio_tolerance
        add_constraint(
            np.r_[members_a, members_b],
            np.r_[user_rating_counts[members_a], -upper_ratio * user_rating_counts[members_b]],
            -np.inf,
            0,
        )
        add_constraint(
            np.r_[members_a, members_b],
            np.r_[-user_rating_counts[members_a], lower_ratio * user_rating_counts[members_b]],
            -np.inf,
            0,
        )

    if target_size is not None:
        add_constraint(
            np.arange(n_users),
            user_rating_counts,
            minimum_size,
            maximum_size,
        )

    matrix = coo_matrix(
        (coefficients, (row_indices, column_indices)),
        shape=(len(constraint_lower), n_users),
    ).tocsr()
    result = milp(
        c=objective,
        integrality=np.ones(n_users),
        bounds=Bounds(lower_bounds, upper_bounds),
        constraints=LinearConstraint(
            matrix, np.asarray(constraint_lower), np.asarray(constraint_upper)
        ),
        options={"time_limit": 300},
    )
    if not result.success:
        raise RuntimeError(
            f"Frozen optimization failed: {result.message}"
        )
    active = result.x > 0.5
    active_rating_count = int(user_rating_counts[active].sum())
    selected_counts = np.zeros((number_of_strata, 2), dtype=np.int64)
    np.add.at(
        selected_counts.ravel(),
        strata[active] * 2 + groups[active],
        user_rating_counts[active] if not condition.equal_groups else 1,
    )
    global_counts = selected_counts.sum(axis=0)
    global_ratio = global_counts[0] / max(global_counts[1], 1)
    maximum_deviation = (
        abs(global_ratio - target_ratio)
        if condition.ratio_mode == "absolute"
        else max(global_ratio / target_ratio, target_ratio / global_ratio)
    )
    iteration = 0
    retained_users = set(users.loc[active, "user_id"])
    balanced = ratings[ratings["user_id"].isin(retained_users)].reset_index(drop=True)
    final_counts = users.loc[active, "group"].value_counts().reindex(
        ["A", "B"], fill_value=0
    )
    return balanced, {
        "size": len(balanced),
        "users": int(active.sum()),
        "removed_users": int((~active).sum()),
        "group_counts": final_counts,
        "ratio": final_counts["A"] / max(final_counts["B"], 1),
        "target_ratio": target_ratio,
        "maximum_deviation": maximum_deviation,
        "iterations": iteration,
        "condition": condition.name,
        "frozen_strata": True,
        "solver": "scipy.milp",
        "solver_message": result.message,
    }


def trim_users_to_size(
    ratings,
    target_size,
    random_state=42,
    protected_user_ids=None,
    target_size_ratio_tolerance=0.001,
):
    """Randomly remove complete users inside a relative size window."""

    minimum_size, maximum_size = _size_window(
        target_size, target_size_ratio_tolerance
    )
    if target_size > len(ratings) or target_size < 0:
        raise ValueError("target_size must be within the rating count.")
    protected_user_ids = set() if protected_user_ids is None else set(protected_user_ids)
    if protected_user_ids - set(ratings["user_id"]):
        raise ValueError("Protected users must occur in the ratings frame.")
    protected_count = int(
        ratings.loc[ratings["user_id"].isin(protected_user_ids)].shape[0]
    )
    if protected_count > maximum_size:
        raise RuntimeError("Protected users exceed the requested size window.")
    user_sizes = ratings.loc[
        ~ratings["user_id"].isin(protected_user_ids)
    ].groupby("user_id").size()
    users = user_sizes.index.to_numpy()
    sizes = user_sizes.to_numpy(dtype=np.int64)
    rng = np.random.default_rng(random_state)
    order = rng.permutation(len(users))
    users, sizes = users[order], sizes[order]
    minimum_removal = len(ratings) - maximum_size
    maximum_removal = len(ratings) - minimum_size
    mask = (1 << (maximum_removal + 1)) - 1
    reachable = 1
    states = [reachable]
    for size_block in sizes:
        reachable |= (reachable << int(size_block)) & mask
        states.append(reachable)
    targets = [
        removal
        for removal in range(minimum_removal, maximum_removal + 1)
        if (reachable >> removal) & 1
    ]
    if not targets:
        raise RuntimeError("No complete-user subset fits the requested size window.")
    removal_target = min(targets, key=lambda value: (abs(len(ratings) - value - target_size), value))
    removed = set()
    total = removal_target
    for index in range(len(users) - 1, -1, -1):
        if (states[index] >> total) & 1:
            continue
        total -= int(sizes[index])
        removed.add(users[index])
    if total != 0:
        raise RuntimeError("Could not reconstruct the complete-user subset.")
    return ratings[~ratings["user_id"].isin(removed)].copy().reset_index(drop=True)


def find_reference_size(ratings, **kwargs):
    strict = next(condition for condition in CONDITIONS if condition.name == "all")
    return balance_users(ratings, strict, **kwargs)
