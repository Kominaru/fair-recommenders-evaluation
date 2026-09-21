"""User-removal samplers for fixed-size, rating-level balanced subsets."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from equalize_users import add_user_statistics


CRITERIA = ("activity", "eccentricity", "group")


@dataclass(frozen=True)
class BalanceCondition:
    """Constraints applied to one subset experiment."""

    name: str
    dimensions: tuple[str, ...]
    equal_groups: bool


CONDITIONS = (
    BalanceCondition("none", (), False),
    BalanceCondition("all", ("activity", "eccentricity"), True),
    BalanceCondition("activity_eccentricity", ("activity", "eccentricity"), False),
    BalanceCondition("activity_group", ("activity",), True),
    BalanceCondition("eccentricity_group", ("eccentricity",), True),
    BalanceCondition("activity", ("activity",), False),
    BalanceCondition("eccentricity", ("eccentricity",), False),
    BalanceCondition("group", (), True),
)


def _rating_features(ratings):
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


def _quantile_edges(values, quantiles):
    return np.unique(np.quantile(values, np.linspace(0, 1, quantiles + 1)))


def _make_fixed_bins(values, edges):
    return np.searchsorted(edges[1:-1], values, side="right")


def _target_ratio(ratings, equal_groups):
    counts = ratings["group"].value_counts()
    return 1.0 if equal_groups else counts["A"] / counts["B"]


def _state(ratings, condition, activity_edges, eccentricity_edges):
    features = _rating_features(ratings)
    activity_bins = _make_fixed_bins(features["activity"], activity_edges)
    eccentricity_bins = _make_fixed_bins(
        features["eccentricity"], eccentricity_edges
    )
    if condition.dimensions == ():
        strata = np.zeros(len(ratings), dtype=np.int64)
        number_of_strata = 1
    elif condition.dimensions == ("activity",):
        strata = activity_bins
        number_of_strata = len(activity_edges) - 1
    elif condition.dimensions == ("eccentricity",):
        strata = eccentricity_bins
        number_of_strata = len(eccentricity_edges) - 1
    else:
        eccentricity_count = len(eccentricity_edges) - 1
        strata = activity_bins * eccentricity_count + eccentricity_bins
        number_of_strata = (len(activity_edges) - 1) * eccentricity_count
    groups = (ratings["group"].to_numpy() == "B").astype(np.int64)
    counts = np.zeros((number_of_strata, 2), dtype=np.int64)
    np.add.at(counts.ravel(), strata * 2 + groups, 1)
    return features, strata, counts


def _make_cached_state(ratings, activity_edges, eccentricity_edges):
    """Build NumPy caches used by the user-removal loop."""

    user_codes, _ = pd.factorize(ratings["user_id"], sort=True)
    item_codes, _ = pd.factorize(ratings["item_id"], sort=True)
    values = ratings["rating"].to_numpy(dtype=np.float64)
    groups = (ratings["group"].to_numpy() == "B").astype(np.int64)
    user_count = np.bincount(user_codes).astype(np.int64)
    user_sum = np.bincount(user_codes, weights=values)
    item_count = np.bincount(item_codes).astype(np.int64)
    item_sum = np.bincount(item_codes, weights=values)
    user_rows = [np.flatnonzero(user_codes == code) for code in range(len(user_count))]
    item_rows = [np.flatnonzero(item_codes == code) for code in range(len(item_count))]
    activity = user_count[user_codes].astype(np.float64)
    user_means = user_sum / user_count
    item_means = item_sum / item_count
    eccentricity = np.abs(
        values - (user_means[user_codes] + item_means[item_codes]) / 2
    )
    return {
        "user_codes": user_codes,
        "item_codes": item_codes,
        "values": values,
        "groups": groups,
        "active": np.ones(len(ratings), dtype=bool),
        "user_count": user_count,
        "user_sum": user_sum,
        "item_count": item_count,
        "item_sum": item_sum,
        "user_rows": user_rows,
        "item_rows": item_rows,
        "activity": activity,
        "eccentricity": eccentricity,
        "activity_bins": _make_fixed_bins(activity, activity_edges),
        "eccentricity_bins": _make_fixed_bins(eccentricity, eccentricity_edges),
        "activity_edges": activity_edges,
        "eccentricity_edges": eccentricity_edges,
    }


def _cached_strata(state, condition, activity_edges, eccentricity_edges):
    if condition.dimensions == ():
        return np.zeros(len(state["active"]), dtype=np.int64), 1
    if condition.dimensions == ("activity",):
        return state["activity_bins"], len(activity_edges) - 1
    if condition.dimensions == ("eccentricity",):
        return state["eccentricity_bins"], len(eccentricity_edges) - 1
    eccentricity_count = len(eccentricity_edges) - 1
    return (
        state["activity_bins"] * eccentricity_count
        + state["eccentricity_bins"],
        (len(activity_edges) - 1) * eccentricity_count,
    )


def _remove_cached_user(state, user_code, strata, counts):
    """Remove a user and update affected rows and stratum counts."""

    removed_rows = state["user_rows"][user_code]
    old_keys = strata[removed_rows] * 2 + state["groups"][removed_rows]
    np.add.at(counts.ravel(), old_keys, -1)
    state["active"][removed_rows] = False
    state["user_count"][user_code] = 0
    state["user_sum"][user_code] = 0

    affected_items = np.unique(state["item_codes"][removed_rows])
    affected_rows = np.unique(
        np.concatenate([state["item_rows"][item] for item in affected_items])
    )
    affected_rows = affected_rows[state["active"][affected_rows]]
    state["item_count"][affected_items] -= np.bincount(
        state["item_codes"][removed_rows],
        minlength=len(state["item_count"]),
    )[affected_items]
    state["item_sum"][affected_items] -= np.bincount(
        state["item_codes"][removed_rows],
        weights=state["values"][removed_rows],
        minlength=len(state["item_sum"]),
    )[affected_items]

    old_keys = strata[affected_rows] * 2 + state["groups"][affected_rows]
    user_codes = state["user_codes"][affected_rows]
    item_codes = state["item_codes"][affected_rows]
    user_means = np.divide(
        state["user_sum"][user_codes],
        state["user_count"][user_codes],
        out=np.zeros(len(affected_rows)),
        where=state["user_count"][user_codes] > 0,
    )
    item_means = np.divide(
        state["item_sum"][item_codes],
        state["item_count"][item_codes],
        out=np.zeros(len(affected_rows)),
        where=state["item_count"][state["item_codes"][affected_rows]] > 0,
    )
    state["eccentricity"][affected_rows] = np.abs(
        state["values"][affected_rows] - (user_means + item_means) / 2
    )
    state["eccentricity_bins"][affected_rows] = np.searchsorted(
        state["eccentricity_edges"][1:-1],
        state["eccentricity"][affected_rows],
        side="right",
    )
    new_strata = strata[affected_rows].copy()
    if state.get("condition_dimensions") == ("eccentricity",):
        new_strata = state["eccentricity_bins"][affected_rows]
    elif state.get("condition_dimensions") == ("activity", "eccentricity"):
        eccentricity_count = len(state["eccentricity_edges"]) - 1
        new_strata = (
            state["activity_bins"][affected_rows] * eccentricity_count
            + state["eccentricity_bins"][affected_rows]
        )
    new_keys = new_strata * 2 + state["groups"][affected_rows]
    moved = old_keys != new_keys
    np.add.at(counts.ravel(), old_keys[moved], -1)
    np.add.at(counts.ravel(), new_keys[moved], 1)
    strata[affected_rows] = new_strata


def _deviations(counts, target_ratio):
    shared = (counts[:, 0] > 0) & (counts[:, 1] > 0)
    used = (counts[:, 0] > 0) | (counts[:, 1] > 0)
    ratios = counts[:, 0] / np.maximum(counts[:, 1], 1)
    deviations = np.zeros(len(counts), dtype=float)
    deviations[used & ~shared] = np.inf
    deviations[shared] = np.maximum(
        ratios[shared] / target_ratio,
        target_ratio / ratios[shared],
    )
    return shared, ratios, deviations


def _choose_user(
    active,
    strata,
    counts,
    target_ratio,
    target_size,
    rng,
):
    shared, ratios, deviations = _deviations(counts, target_ratio)
    used = (counts[:, 0] > 0) | (counts[:, 1] > 0)
    if not used.any():
        raise RuntimeError("No ratings remain feasible.")
    global_ratio = counts[:, 0].sum() / max(counts[:, 1].sum(), 1)
    global_deviation = max(
        global_ratio / target_ratio,
        target_ratio / global_ratio,
    )
    stratum_deviation = float(np.max(deviations[shared]))

    worst_stratum = int(np.argmax(deviations))
    if np.isinf(deviations[worst_stratum]):
        stratum = worst_stratum
        if counts[stratum, 0] > 0 and counts[stratum, 1] == 0:
            group = "A"
        else:
            group = "B"
        candidate_rows = np.flatnonzero(
            (strata == stratum) & (active["group"].to_numpy() == group)
        )
    elif global_deviation >= stratum_deviation:
        group = "A" if global_ratio > target_ratio else "B"
        candidate_rows = np.flatnonzero(active["group"].to_numpy() == group)
    else:
        stratum = worst_stratum
        group = "A" if ratios[stratum] > target_ratio else "B"
        candidate_rows = np.flatnonzero(
            (strata == stratum) & (active["group"].to_numpy() == group)
        )

    candidate_users = active.iloc[candidate_rows]["user_id"].unique()
    rng.shuffle(candidate_users)
    for user_id in candidate_users:
        user_mask = active["user_id"].to_numpy() == user_id
        user_size = int(user_mask.sum())
        if len(active) - user_size < target_size:
            continue
        return user_id
    raise RuntimeError("No complete-user removal can reach the requested size.")


def _balance_user_records(
    ratings,
    condition,
    target_size,
    target_size_tolerance,
    activity_quantiles,
    eccentricity_quantiles,
    ratio_tolerance,
    random_state,
    max_iterations,
):
    """Balance one cached record per user, matching equalize_users.py."""

    users = add_user_statistics(ratings).reset_index(drop=True)
    activity_edges = _quantile_edges(users["activity"], activity_quantiles)
    eccentricity_edges = _quantile_edges(
        users["eccentricity"], eccentricity_quantiles
    )
    users["activity_bin"] = _make_fixed_bins(users["activity"], activity_edges)
    users["eccentricity_bin"] = _make_fixed_bins(
        users["eccentricity"], eccentricity_edges
    )
    activity_count = len(activity_edges) - 1
    eccentricity_count = len(eccentricity_edges) - 1
    if condition.dimensions == ():
        strata = np.zeros(len(users), dtype=np.int64)
        number_of_strata = 1
    elif condition.dimensions == ("activity",):
        strata = users["activity_bin"].to_numpy(dtype=np.int64)
        number_of_strata = activity_count
    elif condition.dimensions == ("eccentricity",):
        strata = users["eccentricity_bin"].to_numpy(dtype=np.int64)
        number_of_strata = eccentricity_count
    else:
        strata = (
            users["activity_bin"].to_numpy(dtype=np.int64) * eccentricity_count
            + users["eccentricity_bin"].to_numpy(dtype=np.int64)
        )
        number_of_strata = activity_count * eccentricity_count

    groups = (users["group"].to_numpy() == "B").astype(np.int64)
    counts = np.zeros((number_of_strata, 2), dtype=np.int64)
    np.add.at(counts.ravel(), strata * 2 + groups, 1)
    target_ratio = _target_ratio(users, condition.equal_groups)
    active = np.ones(len(users), dtype=bool)
    rating_counts = (
        ratings.groupby("user_id").size()
        .reindex(users["user_id"])
        .to_numpy(dtype=np.int64)
    )
    active_rating_count = int(rating_counts.sum())
    minimum_size = None if target_size is None else max(
        0, target_size - target_size_tolerance
    )
    maximum_size = None if target_size is None else target_size + target_size_tolerance
    rng = np.random.default_rng(random_state)
    limit = len(users) if max_iterations is None else max_iterations
    progress = tqdm(total=limit, desc=f"Balancing {condition.name}", unit="user")

    try:
        for iteration in range(limit + 1):
            shared, ratios, deviations = _deviations(counts, target_ratio)
            if not shared.any():
                raise RuntimeError("No stratum contains both groups.")
            maximum_deviation = float(np.max(deviations[shared]))
            progress.set_postfix(
                ratings=f"{active_rating_count:,}",
                deviation=f"{maximum_deviation:.4f}",
            )
            size_ok = target_size is None or minimum_size <= active_rating_count <= maximum_size
            if size_ok and maximum_deviation <= ratio_tolerance:
                break
            if minimum_size is not None and active_rating_count < minimum_size:
                raise RuntimeError("User removal crossed the requested target size.")
            if iteration == limit:
                raise RuntimeError("User balancing did not converge within the iteration limit.")

            stratum = int(np.nanargmax(np.where(shared, deviations, np.nan)))
            group_code = 0 if ratios[stratum] > target_ratio else 1
            candidates = np.flatnonzero(
                active & (strata == stratum) & (groups == group_code)
            )
            rng.shuffle(candidates)
            removed = None
            for candidate in candidates:
                if minimum_size is None or active_rating_count - rating_counts[candidate] >= minimum_size:
                    removed = int(candidate)
                    break
            if removed is None:
                raise RuntimeError("No removable user can reach the requested size.")
            active[removed] = False
            counts[strata[removed], groups[removed]] -= 1
            active_rating_count -= int(rating_counts[removed])
            progress.update(1)
    finally:
        progress.close()

    balanced = ratings[ratings["user_id"].isin(users.loc[active, "user_id"])].copy()
    final_counts = users.loc[active, "group"].value_counts().reindex(
        ["A", "B"], fill_value=0
    )
    return balanced.reset_index(drop=True), {
        "size": len(balanced),
        "users": int(active.sum()),
        "removed_users": int((~active).sum()),
        "group_counts": final_counts,
        "ratio": final_counts["A"] / max(final_counts["B"], 1),
        "target_ratio": target_ratio,
        "maximum_deviation": maximum_deviation,
        "iterations": iteration,
        "condition": condition.name,
    }


def balance_users(
    ratings,
    condition,
    target_size=None,
    target_size_tolerance=0,
    activity_quantiles=10,
    eccentricity_quantiles=10,
    ratio_tolerance=1.001,
    random_state=42,
    max_iterations=None,
):
    """Create a subset by removing users under a selected condition.

    If ``target_size`` is omitted, removal stops at the largest size that
    satisfies the condition. If supplied, removal continues until the
    result is within ``target_size_tolerance`` ratings of that target, or
    raises when complete-user removal cannot reach the requested range.
    """

    return _balance_user_records(
        ratings,
        condition,
        target_size,
        target_size_tolerance,
        activity_quantiles,
        eccentricity_quantiles,
        ratio_tolerance,
        random_state,
        max_iterations,
    )

    if set(ratings["group"].dropna().unique()) != {"A", "B"}:
        raise ValueError("ratings must contain both groups A and B.")
    if ratio_tolerance < 1:
        raise ValueError("ratio_tolerance must be at least 1.")
    if target_size_tolerance < 0:
        raise ValueError("target_size_tolerance must be non-negative.")
    original = ratings.reset_index(drop=True).copy()
    original_features = _rating_features(original)
    activity_edges = _quantile_edges(
        original_features["activity"], activity_quantiles
    )
    eccentricity_edges = _quantile_edges(
        original_features["eccentricity"], eccentricity_quantiles
    )
    target_ratio = _target_ratio(original, condition.equal_groups)
    rng = np.random.default_rng(random_state)
    state = _make_cached_state(original, activity_edges, eccentricity_edges)
    state["condition_dimensions"] = condition.dimensions
    strata, number_of_strata = _cached_strata(
        state, condition, activity_edges, eccentricity_edges
    )
    groups = state["groups"]
    counts = np.zeros((number_of_strata, 2), dtype=np.int64)
    np.add.at(counts.ravel(), strata * 2 + groups, 1)
    user_ids = np.arange(len(state["user_rows"]))
    active_user = np.ones(len(user_ids), dtype=bool)
    active_rating_count = len(original)
    limit = len(user_ids) if max_iterations is None else max_iterations
    minimum_size = (
        None
        if target_size is None
        else max(0, target_size - target_size_tolerance)
    )
    maximum_size = (
        None
        if target_size is None
        else target_size + target_size_tolerance
    )

    progress = tqdm(
        total=limit,
        desc=f"Balancing {condition.name}",
        unit="user",
    )
    try:
        for iteration in range(limit + 1):
            shared, _, deviations = _deviations(counts, target_ratio)
            global_ratio = counts[:, 0].sum() / max(counts[:, 1].sum(), 1)
            global_deviation = max(
                global_ratio / target_ratio,
                target_ratio / global_ratio,
            )
            maximum_deviation = max(float(np.max(deviations)), global_deviation)
            progress.set_postfix(
                ratings=f"{active_rating_count:,}",
                deviation=f"{maximum_deviation:.4f}",
            )

            if target_size is None:
                if maximum_deviation <= ratio_tolerance:
                    break
            elif (
                minimum_size <= active_rating_count <= maximum_size
                and maximum_deviation <= ratio_tolerance
            ):
                break
            elif active_rating_count < minimum_size:
                raise RuntimeError("User removal crossed the requested target size.")

            if iteration == limit:
                raise RuntimeError("User removal did not converge.")
            used = (counts[:, 0] > 0) | (counts[:, 1] > 0)
            worst_stratum = int(np.argmax(deviations))
            global_ratio = counts[:, 0].sum() / max(counts[:, 1].sum(), 1)
            global_deviation = max(
                global_ratio / target_ratio,
                target_ratio / global_ratio,
            )
            shared_deviation = float(np.max(deviations[shared])) if shared.any() else 0.0
            if np.isinf(deviations[worst_stratum]):
                group_code = 0 if counts[worst_stratum, 0] > 0 else 1
                candidate_rows = np.flatnonzero(
                    state["active"]
                    & (strata == worst_stratum)
                    & (groups == group_code)
                )
            elif global_deviation >= shared_deviation:
                group_code = 0 if global_ratio > target_ratio else 1
                candidate_rows = np.flatnonzero(
                    state["active"] & (groups == group_code)
                )
            else:
                group_code = 0 if _deviations(counts, target_ratio)[1][worst_stratum] > target_ratio else 1
                candidate_rows = np.flatnonzero(
                    state["active"]
                    & (strata == worst_stratum)
                    & (groups == group_code)
                )
            candidate_users = np.unique(state["user_codes"][candidate_rows])
            rng.shuffle(candidate_users)
            removed_user = None
            minimum = minimum_size or 0
            for candidate_user in candidate_users:
                user_size = len(state["user_rows"][candidate_user])
                if active_rating_count - user_size >= minimum:
                    removed_user = int(candidate_user)
                    break
            if removed_user is None:
                raise RuntimeError("No complete-user removal can reach the requested size.")
            active_user[removed_user] = False
            active_rating_count -= len(state["user_rows"][removed_user])
            _remove_cached_user(state, removed_user, strata, counts)
            progress.update(1)
    finally:
        progress.close()

    if target_size is not None and maximum_deviation > ratio_tolerance:
        raise RuntimeError(
            "The requested size cannot satisfy the selected balance condition."
        )
    balanced = original[state["active"]].reset_index(drop=True)
    final_counts = balanced["group"].value_counts().reindex(["A", "B"], fill_value=0)
    return balanced, {
        "size": len(balanced),
        "users": balanced["user_id"].nunique(),
        "removed_users": int((~active_user).sum()),
        "group_counts": final_counts,
        "ratio": final_counts["A"] / max(final_counts["B"], 1),
        "target_ratio": target_ratio,
        "maximum_deviation": maximum_deviation,
        "iterations": iteration,
        "condition": condition.name,
    }


def find_reference_size(ratings, **kwargs):
    """Find the largest strict all-criteria subset for one fold."""

    strict = next(condition for condition in CONDITIONS if condition.name == "all")
    return balance_users(ratings, strict, **kwargs)


def trim_users_to_size(ratings, target_size):
    """Remove complete users to reach an exact size when a subset-sum exists."""

    if target_size > len(ratings) or target_size < 0:
        raise ValueError("target_size must be within the rating count.")
    removal_target = len(ratings) - target_size
    if removal_target == 0:
        return ratings.copy()

    user_sizes = ratings.groupby("user_id").size()
    parents = {0: None}
    for user_id, user_size in user_sizes.items():
        previous = list(parents.items())
        for total, _ in previous:
            new_total = total + int(user_size)
            if new_total > removal_target or new_total in parents:
                continue
            parents[new_total] = (total, user_id)
        if removal_target in parents:
            break

    if removal_target not in parents:
        raise RuntimeError("No complete-user subset has the requested size.")
    removed_users = set()
    total = removal_target
    while total:
        previous_total, user_id = parents[total]
        removed_users.add(user_id)
        total = previous_total
    return ratings[~ratings["user_id"].isin(removed_users)].copy().reset_index(drop=True)
