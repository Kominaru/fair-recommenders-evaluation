"""User-removal samplers for fixed-size, rating-level balanced subsets."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from tqdm.auto import tqdm

from user_balancing import add_user_statistics


@dataclass(frozen=True)
class BalanceCondition:
    """Constraints applied to one subset experiment."""

    name: str

    dimensions: tuple[str, ...]

    equal_groups: bool

    ratio_mode: str = "multiplicative"

    global_ratio_constraint: bool = False

    balance_user_counts: bool = False

    global_rating_ratio: bool = False


CONDITIONS = (
    BalanceCondition("none", (), False),
    BalanceCondition("all", ("activity", "eccentricity"), True),
    BalanceCondition(
        "activity_eccentricity",
        ("activity", "eccentricity"),
        False,
        "multiplicative",
        False,
        True,
        True,
    ),
    # BalanceCondition("activity_group", ("activity",), True),
    # BalanceCondition("eccentricity_group", ("eccentricity",), True),
    # BalanceCondition("activity", ("activity",), False),
    # BalanceCondition("eccentricity", ("eccentricity",), False),
    BalanceCondition("group", (), True),
)


def _quantile_edges(values, quantiles):
    return np.unique(np.quantile(values, np.linspace(0, 1, quantiles + 1)))


def _make_fixed_bins(values, edges):
    bins = np.searchsorted(edges[1:-1], values, side="right")
    return np.clip(bins, 0, max(0, len(edges) - 2))


def _target_ratio(ratings, equal_groups):
    counts = ratings["group"].value_counts()
    return 1.0 if equal_groups else counts["A"] / counts["B"]


def _deviations(counts, target_ratio, ratio_mode):
    shared = (counts[:, 0] > 0) & (counts[:, 1] > 0)
    used = (counts[:, 0] > 0) | (counts[:, 1] > 0)
    ratios = counts[:, 0] / np.maximum(counts[:, 1], 1)
    deviations = np.zeros(len(counts), dtype=float)
    deviations[used & ~shared] = np.inf
    if ratio_mode == "absolute":
        deviations[shared] = np.abs(ratios[shared] - target_ratio)
    else:
        deviations[shared] = np.maximum(
            ratios[shared] / target_ratio,
            target_ratio / ratios[shared],
        )
    return shared, ratios, deviations


def _dynamic_swap_users(
    ratings,
    users,
    active,
    target_size,
    target_size_ratio_tolerance,
    global_target_ratio,
    ratio_tolerance,
    ks_tolerance,
    random_state,
    max_swaps=300,
):
    """Improve a fixed-size subset with complete-user add/remove swaps."""

    rng = np.random.default_rng(random_state)

    user_ids = users["user_id"].to_numpy()

    groups = users["group"].to_numpy()

    rating_counts = ratings.groupby("user_id").size()

    active = active.copy()

    minimum_size = int(
        np.ceil(target_size * (1 - target_size_ratio_tolerance)))

    maximum_size = int(
        np.floor(target_size * (1 + target_size_ratio_tolerance)))

    def evaluate():

        selected_ids = set(user_ids[active])

        subset = ratings[ratings["user_id"].isin(selected_ids)].copy()

        statistics = add_user_statistics(subset).set_index("user_id")

        counts = subset["group"].value_counts()

        global_ratio = counts.get("A", 0) / max(counts.get("B", 0), 1)

        group_a = statistics.loc[statistics["group"] == "A"]

        group_b = statistics.loc[statistics["group"] == "B"]

        activity_ks = ks_2samp(
            group_a["activity"],
            group_b["activity"]).statistic

        eccentricity_ks = ks_2samp(

            group_a["eccentricity"], group_b["eccentricity"]

        ).statistic

        ratio_deviation = max(

            global_ratio / global_target_ratio,

            global_target_ratio / max(global_ratio, np.finfo(float).eps),

        )

        score = max(ratio_deviation, activity_ks, eccentricity_ks)

        return subset, global_ratio, activity_ks, eccentricity_ks, score

    subset, global_ratio, activity_ks, eccentricity_ks, score = evaluate()

    for _ in range(max_swaps):

        if (

            minimum_size <= len(subset) <= maximum_size

            and global_ratio / global_target_ratio <= ratio_tolerance

            and global_target_ratio / max(global_ratio, np.finfo(float).eps) <= ratio_tolerance

            and activity_ks <= ks_tolerance

            and eccentricity_ks <= ks_tolerance

        ):

            return subset, global_ratio, activity_ks, eccentricity_ks

        add_group = "A" if global_ratio < global_target_ratio else "B"

        remove_group = "B" if add_group == "A" else "A"

        add_candidates = np.flatnonzero(~active & (groups == add_group))

        remove_candidates = np.flatnonzero(active & (groups == remove_group))

        if not len(add_candidates) or not len(remove_candidates):

            break

        add_candidates = rng.choice(
            add_candidates, min(
                100, len(add_candidates)), replace=False)

        remove_candidates = rng.choice(
            remove_candidates, min(
                100, len(remove_candidates)), replace=False)

        current_size = len(subset)

        pairs = [

            (add_user, remove_user)

            for add_user in add_candidates

            for remove_user in remove_candidates

            if minimum_size

            <= current_size

            + int(rating_counts.get(user_ids[add_user], 0))

            - int(rating_counts.get(user_ids[remove_user], 0))

            <= maximum_size

        ]

        if not pairs:

            break

        current_counts = subset["group"].value_counts()

        add_user, remove_user = min(

            pairs,

            key=lambda pair: abs(

                (

                    current_counts.get("A", 0)

                    + (rating_counts.get(user_ids[pair[0]], 0) if add_group == "A" else 0)

                    - (rating_counts.get(user_ids[pair[1]], 0) if remove_group == "A" else 0)

                )

                / max(

                    current_counts.get("B", 0)

                    + (rating_counts.get(user_ids[pair[0]], 0) if add_group == "B" else 0)

                    - (rating_counts.get(user_ids[pair[1]], 0) if remove_group == "B" else 0),

                    1,

                )

                - global_target_ratio

            ),

        )

        active[add_user] = True

        active[remove_user] = False

        candidate_subset, candidate_ratio, candidate_activity_ks, candidate_eccentricity_ks, candidate_score = evaluate()

        if candidate_score <= score or candidate_ratio / \
                global_target_ratio <= ratio_tolerance:

            subset = candidate_subset

            global_ratio = candidate_ratio

            activity_ks = candidate_activity_ks

            eccentricity_ks = candidate_eccentricity_ks

            score = candidate_score

        else:

            active[add_user] = False

            active[remove_user] = True

    raise RuntimeError(
        "Swap phase could not satisfy the requested constraints.")


def _balance_cached_users(
    ratings,
    condition,
    target_size,
    target_size_ratio_tolerance,
    activity_quantiles,
    eccentricity_quantiles,
    ratio_tolerance,
    ks_tolerance,
    random_state,
    max_iterations,
    protected_user_ids=None,
):
    """Balance one cached record per user while removing complete users."""

    users = add_user_statistics(ratings).reset_index(drop=True)

    user_ids = users["user_id"].to_numpy()

    protected_user_ids = set() if protected_user_ids is None else set(protected_user_ids)

    unknown_protected = protected_user_ids - set(user_ids)

    if unknown_protected:

        raise ValueError("Protected users must occur in the ratings frame.")

    protected = np.isin(user_ids, list(protected_user_ids))

    rng = np.random.default_rng(random_state)

    user_index = pd.Index(user_ids)

    rating_user_codes = user_index.get_indexer(ratings["user_id"])

    rating_values = ratings["rating"].to_numpy(dtype=float)

    item_codes, item_ids = pd.factorize(ratings["item_id"], sort=True)

    item_count = len(item_ids)

    user_rating_counts = np.bincount(

        rating_user_codes, minlength=len(users)

    ).astype(np.int64)

    user_rating_sums = np.bincount(

        rating_user_codes, weights=rating_values, minlength=len(users)

    )

    user_means = user_rating_sums / user_rating_counts

    item_sums = np.bincount(
        item_codes,
        weights=rating_values,
        minlength=item_count)

    item_counts = np.bincount(
        item_codes,
        minlength=item_count).astype(
        np.int64)

    user_rating_positions = [[] for _ in range(len(users))]

    item_user_codes = [[] for _ in range(item_count)]

    item_rating_positions = [[] for _ in range(item_count)]

    for position, (user_code, item_code) in enumerate(

        zip(rating_user_codes, item_codes)

    ):

        user_rating_positions[user_code].append(position)

        item_user_codes[item_code].append(user_code)

        item_rating_positions[item_code].append(position)

    user_rating_positions = [

        np.asarray(positions, dtype=np.int64)

        for positions in user_rating_positions

    ]

    item_user_codes = [

        np.asarray(codes, dtype=np.int64) for codes in item_user_codes

    ]

    item_rating_positions = [

        np.asarray(positions, dtype=np.int64)

        for positions in item_rating_positions

    ]

    groups = (users["group"].to_numpy() == "B").astype(np.int64)

    active = np.ones(len(users), dtype=bool)

    initial_item_means = item_sums[item_codes] / item_counts[item_codes]

    initial_deviations = np.abs(

        rating_values

        - (user_means[rating_user_codes] + initial_item_means) / 2

    )

    deviation_sums = np.bincount(

        rating_user_codes, weights=initial_deviations, minlength=len(users)

    )

    eccentricities = deviation_sums / user_rating_counts

    activity_edges = _quantile_edges(users["activity"], activity_quantiles)

    eccentricity_edges = _quantile_edges(
        eccentricities, eccentricity_quantiles)

    activity_bins = _make_fixed_bins(
        users["activity"].to_numpy(), activity_edges)

    eccentricity_bins = _make_fixed_bins(eccentricities, eccentricity_edges)

    activity_count = max(1, len(activity_edges) - 1)

    eccentricity_count = max(1, len(eccentricity_edges) - 1)

    if condition.dimensions == ():

        strata = np.zeros(len(users), dtype=np.int64)

        number_of_strata = 1

    elif condition.dimensions == ("activity",):

        strata = activity_bins.copy()

        number_of_strata = activity_count

    elif condition.dimensions == ("eccentricity",):

        strata = eccentricity_bins.copy()

        number_of_strata = eccentricity_count

    else:

        strata = activity_bins * eccentricity_count + eccentricity_bins

        number_of_strata = activity_count * eccentricity_count

    counts = np.zeros((number_of_strata, 2), dtype=np.int64)

    active_indices = np.arange(len(users))

    balance_user_counts = condition.equal_groups or condition.balance_user_counts

    if balance_user_counts:

        np.add.at(counts.ravel(), strata * 2 + groups, 1)

        target_ratio = 1.0

    else:

        np.add.at(counts.ravel(), strata * 2 + groups, user_rating_counts)

        target_ratio = _target_ratio(ratings, False)

    rating_counts = user_rating_counts

    global_target_ratio = (

        _target_ratio(ratings, False)

        if condition.global_rating_ratio

        else target_ratio

    )

    active_rating_count = int(rating_counts.sum())

    minimum_size = None if target_size is None else int(

        np.ceil(target_size * (1 - target_size_ratio_tolerance))

    )

    maximum_size = None if target_size is None else int(

        np.floor(target_size * (1 + target_size_ratio_tolerance))

    )

    protected_rating_count = int(rating_counts[protected].sum())

    if maximum_size is not None and protected_rating_count > maximum_size:

        raise RuntimeError("Protected users exceed the requested target size.")

    limit = len(users) if max_iterations is None else max_iterations

    progress = tqdm(
        total=limit,
        desc=f"Balancing {
            condition.name}",
        unit="user")

    try:

        for iteration in range(limit + 1):

            shared, ratios, deviations = _deviations(

                counts, target_ratio, condition.ratio_mode

            )
            used = (counts[:, 0] > 0) | (counts[:, 1] > 0)

            global_counts = (

                np.bincount(
                    groups[active],
                    weights=rating_counts[active],
                    minlength=2)

                if condition.global_rating_ratio

                else counts.sum(axis=0)

            )

            global_ratio = global_counts[0] / max(global_counts[1], 1)

            if condition.ratio_mode == "absolute":

                global_deviation = abs(global_ratio - global_target_ratio)

            else:

                global_deviation = max(

                    global_ratio / global_target_ratio,

                    global_target_ratio / global_ratio,

                )

            stratum_deviation = (

                float(np.max(deviations[used])) if used.any() else 0.0

            )

            maximum_deviation = (

                global_deviation

                if condition.global_ratio_constraint

                else max(stratum_deviation, global_deviation)

            )

            progress.set_postfix(
                ratings=f"{
                    active_rating_count:,}", deviation=f"{
                    maximum_deviation:.4f}")

            size_ok = target_size is None or minimum_size <= active_rating_count <= maximum_size

            if size_ok and maximum_deviation <= ratio_tolerance:

                break

            if not used.any() and size_ok and global_deviation <= ratio_tolerance:

                maximum_deviation = global_deviation

                break

            if minimum_size is not None and active_rating_count < minimum_size:

                raise RuntimeError(
                    "User removal crossed the requested target size.")

            if iteration == limit:

                raise RuntimeError("User removal did not converge.")

            candidate_weights = (

                np.ones(len(users), dtype=np.int64)

                if balance_user_counts

                else rating_counts

            )

            removed_user = None

            if (

                (condition.global_ratio_constraint or condition.global_rating_ratio)

                and global_deviation > ratio_tolerance

            ):

                group_code = 0 if global_ratio > global_target_ratio else 1

                candidate_mask = active & ~protected & (groups == group_code)

                candidate_indices = np.flatnonzero(candidate_mask)

                candidate_indices = candidate_indices[

                    np.lexsort(

                        (

                            rating_counts[candidate_indices],

                            -deviations[strata[candidate_indices]],

                        )

                    )

                ]

                stratum = None

                for candidate in candidate_indices:

                    candidate_stratum = strata[candidate]

                    opposite_group = 1 - group_code

                    if (

                        shared[candidate_stratum]

                        and (

                            counts[candidate_stratum, group_code] <= 1

                            or counts[candidate_stratum, opposite_group] <= 0

                        )

                    ):

                        continue

                    if minimum_size is None or active_rating_count - \
                            rating_counts[candidate] >= minimum_size:

                        removed_user = int(candidate)

                        stratum = int(candidate_stratum)

                        break

                if removed_user is None:

                    fallback_candidates = np.flatnonzero(candidate_mask)

                    fallback_candidates = fallback_candidates[

                        np.argsort(rating_counts[fallback_candidates])

                    ]

                    for candidate in fallback_candidates:

                        if minimum_size is None or active_rating_count - \
                                rating_counts[candidate] >= minimum_size:

                            removed_user = int(candidate)

                            stratum = int(strata[candidate])

                            break

            else:

                stratum_order = np.argsort(

                    np.where(used, deviations, -np.inf)

                )[::-1]

                for candidate_stratum in stratum_order:

                    if not used[candidate_stratum]:

                        continue

                    group_code = (

                        0

                        if ratios[candidate_stratum] > target_ratio

                        else 1

                    )

                    candidates = np.flatnonzero(

                        active

                        & ~protected

                        & (strata == candidate_stratum)

                        & (groups == group_code)

                    )

                    for candidate in candidates:

                        if minimum_size is None or active_rating_count - \
                                rating_counts[candidate] >= minimum_size:

                            stratum = int(candidate_stratum)

                            removed_user = int(candidate)

                            break

                    if removed_user is not None:

                        break

            if removed_user is None:

                # Dynamic eccentricity updates can leave the cached cell
                # counts out of sync with the active users. Rebuild them
                # before declaring that no singleton stratum is removable.
                actual_counts = np.zeros_like(counts)
                np.add.at(
                    actual_counts.ravel(),
                    strata[active] * 2 + groups[active],
                    1 if balance_user_counts else rating_counts[active],
                )
                actual_used = actual_counts.sum(axis=1) > 0
                actual_shared = (
                    (actual_counts[:, 0] > 0)
                    & (actual_counts[:, 1] > 0)
                )
                singleton_order = np.flatnonzero(
                    actual_used & ~actual_shared
                )
                for candidate_stratum in singleton_order:
                    group_code = int(actual_counts[candidate_stratum, 1] > 0)
                    candidates = np.flatnonzero(
                        active
                        & ~protected
                        & (strata == candidate_stratum)
                        & (groups == group_code)
                    )
                    if candidates.size:
                        removed_user = int(candidates[0])
                        stratum = int(candidate_stratum)
                        counts = actual_counts
                        break

            if removed_user is None:

                if target_size is not None and condition.name == "activity_eccentricity":

                    swapped, swap_ratio, activity_ks, eccentricity_ks = _dynamic_swap_users(

                        ratings,

                        users,

                        active,

                        target_size,

                        target_size_ratio_tolerance,

                        global_target_ratio,

                        ratio_tolerance,

                        ks_tolerance,

                        random_state + iteration,

                    )

                    balanced = swapped.reset_index(drop=True)

                    final_counts = balanced["group"].value_counts().reindex(

                        ["A", "B"], fill_value=0

                    )

                    return balanced, {

                        "size": len(balanced),

                        "users": int(balanced["user_id"].nunique()),

                        "removed_users": int(len(users) - balanced["user_id"].nunique()),

                        "group_counts": final_counts,

                        "ratio": final_counts["A"] / max(final_counts["B"], 1),

                        "target_ratio": target_ratio,

                        "global_target_ratio": global_target_ratio,

                        "global_ratio": swap_ratio,

                        "maximum_deviation": max(

                            activity_ks, eccentricity_ks

                        ),

                        "activity_ks": activity_ks,

                        "eccentricity_ks": eccentricity_ks,

                        "iterations": iteration,

                        "condition": condition.name,

                    }

                raise RuntimeError(
                    "No complete-user removal can reach the requested size.")

            active[removed_user] = False

            removed_positions = user_rating_positions[removed_user]

            removed_items = item_codes[removed_positions]

            affected_mask = np.zeros(len(users), dtype=bool)

            item_sums[removed_items] -= rating_values[removed_positions]

            item_counts[removed_items] -= 1

            affected_item_users = np.concatenate(

                [item_user_codes[item_code] for item_code in removed_items]

            )

            affected_mask[affected_item_users] = True

            affected_mask[removed_user] = False

            removed_count = 1 if balance_user_counts else rating_counts[removed_user]

            counts[strata[removed_user], groups[removed_user]] -= removed_count

            affected_array = np.flatnonzero(affected_mask & active)

            for item_code in removed_items:

                item_positions = item_rating_positions[item_code]

                active_positions = item_positions[active[rating_user_codes[item_positions]]]

                if len(active_positions) == 0:

                    continue

                item_users = rating_user_codes[active_positions]

                removed_position = removed_positions[removed_items == item_code][0]

                old_item_mean = (

                    item_sums[item_code] + rating_values[removed_position]

                ) / (item_counts[item_code] + 1)

                new_item_mean = item_sums[item_code] / item_counts[item_code]

                old_deviations = np.abs(

                    rating_values[active_positions]

                    - (user_means[item_users] + old_item_mean) / 2

                )

                new_deviations = np.abs(

                    rating_values[active_positions]

                    - (user_means[item_users] + new_item_mean) / 2

                )

                deviation_sums[item_users] += new_deviations - old_deviations

            eccentricities[affected_array] = (

                deviation_sums[affected_array] / rating_counts[affected_array]

            )

            affected_eccentricity_bins = _make_fixed_bins(

                eccentricities[affected_array], eccentricity_edges

            )

            for affected_user, eccentricity_bin in zip(

                affected_array, affected_eccentricity_bins

            ):

                old_stratum = strata[affected_user]

                if condition.dimensions == ():

                    new_stratum = 0

                elif condition.dimensions == ("activity",):

                    new_stratum = activity_bins[affected_user]

                elif condition.dimensions == ("eccentricity",):

                    new_stratum = eccentricity_bin

                else:

                    new_stratum = (

                        activity_bins[affected_user] * eccentricity_count

                        + eccentricity_bin

                    )

                if new_stratum != old_stratum:

                    weight = 1 if balance_user_counts else rating_counts[affected_user]

                    counts[old_stratum, groups[affected_user]] -= weight

                    counts[new_stratum, groups[affected_user]] += weight

                    strata[affected_user] = new_stratum

            active_rating_count -= int(rating_counts[removed_user])

            progress.update(1)

    finally:

        progress.close()

    if target_size is not None and maximum_deviation > ratio_tolerance:

        raise RuntimeError(
            "The requested size cannot satisfy the selected balance condition.")

    users["eccentricity"] = eccentricities

    retained_users = set(users.loc[active, "user_id"])

    balanced = ratings[ratings["user_id"].isin(
        retained_users)].reset_index(drop=True)

    final_counts = users.loc[active, "group"].value_counts().reindex([
        "A", "B"], fill_value=0)

    return balanced, {

        "size": len(balanced),

        "users": int(active.sum()),

        "removed_users": int((~active).sum()),

        "group_counts": final_counts,

        "ratio": final_counts["A"] / max(final_counts["B"], 1),

        "target_ratio": target_ratio,

        "global_target_ratio": global_target_ratio,

        "global_ratio": global_ratio,

        "maximum_deviation": maximum_deviation,

        "iterations": iteration,

        "condition": condition.name,

    }


def balance_users(
    ratings,
    condition,
    target_size=None,
    target_size_ratio_tolerance=0.001,
    activity_quantiles=10,
    eccentricity_quantiles=10,
    ratio_tolerance=1.001,
    ks_tolerance=0.05,
    random_state=42,
    max_iterations=None,
    protected_user_ids=None,
):
    """Create a subset by removing users under a selected condition.

    If ``target_size`` is omitted, removal stops at the largest size that

    satisfies the condition. If supplied, removal continues until that

    exact rating count is reached.

    """

    if set(ratings["group"].dropna().unique()) != {"A", "B"}:

        raise ValueError("ratings must contain both groups A and B.")

    if condition.ratio_mode == "absolute" and ratio_tolerance < 0:

        raise ValueError("ratio_tolerance must be non-negative.")

    if condition.ratio_mode == "multiplicative" and ratio_tolerance < 1:

        raise ValueError("ratio_tolerance must be at least 1.")

    if target_size_ratio_tolerance < 0:

        raise ValueError("target_size_ratio_tolerance must be non-negative.")

    return _balance_cached_users(

        ratings,

        condition,

        target_size,

        target_size_ratio_tolerance,

        activity_quantiles,

        eccentricity_quantiles,

        ratio_tolerance,

        ks_tolerance,

        random_state,

        max_iterations,

        protected_user_ids,

    )


def hybrid_balance_users(
    ratings,
    condition,
    target_size,
    target_size_ratio_tolerance=0.001,
    activity_quantiles=10,
    eccentricity_quantiles=10,
    ratio_tolerance=1.001,
    random_state=42,
    max_passes=8,
    max_swaps_per_pass=40,
    protected_user_ids=None,
):
    """Balance users with frozen passes and complete-user reactivation swaps.

    Each pass freezes activity/eccentricity bins for the current subset. A
    bounded set of remove/add swaps is then scored using freshly recomputed
    statistics. Only strictly improving swaps are retained, so accepted
    objective values cannot increase, although global optimality is not
    guaranteed.
    """

    if condition.name not in {"none", "all", "activity_eccentricity"}:
        raise ValueError("Hybrid balancing supports none, all, and activity_eccentricity.")
    if target_size < 0 or target_size > len(ratings):
        raise ValueError("target_size must be within the rating count.")
    if max_passes < 1 or max_swaps_per_pass < 1:
        raise ValueError("max_passes and max_swaps_per_pass must be positive.")
    if ratio_tolerance < 1:
        raise ValueError("ratio_tolerance must be at least 1.")

    protected_user_ids = set() if protected_user_ids is None else set(protected_user_ids)
    all_user_ids = pd.Index(ratings["user_id"].unique())
    unknown = protected_user_ids - set(all_user_ids)
    if unknown:
        raise ValueError("Protected users must occur in the ratings frame.")

    minimum_size = int(np.ceil(target_size * (1 - target_size_ratio_tolerance)))
    maximum_size = int(np.floor(target_size * (1 + target_size_ratio_tolerance)))
    user_sizes = ratings.groupby("user_id").size()
    protected_size = int(user_sizes.reindex(list(protected_user_ids), fill_value=0).sum())
    if protected_size > maximum_size:
        raise RuntimeError("Protected users exceed the requested target size.")

    rng = np.random.default_rng(random_state)
    retained = set(all_user_ids)
    while len(ratings[ratings["user_id"].isin(retained)]) > maximum_size:
        removable = list(retained - protected_user_ids)
        if not removable:
            raise RuntimeError("No removable users remain for the target size.")
        removable.sort(key=lambda user_id: (user_sizes[user_id], rng.random()))
        retained.remove(removable[0])

    target_ratio = _target_ratio(ratings, condition.equal_groups)

    def evaluate(user_ids):
        subset = ratings[ratings["user_id"].isin(user_ids)].copy()
        users = add_user_statistics(subset).reset_index(drop=True)
        if condition.name == "none":
            score = 0.0
            stratum_deviation = 0.0
            global_deviation = 0.0
            singleton_count = 0
        else:
            activity_edges = _quantile_edges(users["activity"], activity_quantiles)
            eccentricity_edges = _quantile_edges(users["eccentricity"], eccentricity_quantiles)
            activity_bins = _make_fixed_bins(users["activity"].to_numpy(), activity_edges)
            eccentricity_bins = _make_fixed_bins(users["eccentricity"].to_numpy(), eccentricity_edges)
            if condition.name == "all":
                strata = activity_bins * max(1, len(eccentricity_edges) - 1) + eccentricity_bins
                target = 1.0
                weights = np.ones(len(users), dtype=np.int64)
            else:
                strata = activity_bins * max(1, len(eccentricity_edges) - 1) + eccentricity_bins
                target = target_ratio
                weights = users["activity"].to_numpy(dtype=np.int64)
            counts = np.zeros((int(strata.max()) + 1, 2), dtype=np.int64)
            np.add.at(counts.ravel(), strata * 2 + (users["group"].to_numpy() == "B"), weights)
            _, _, deviations = _deviations(counts, target, "multiplicative")
            used = counts.sum(axis=1) > 0
            shared = (counts[:, 0] > 0) & (counts[:, 1] > 0)
            singleton_count = int(np.sum(used & ~shared))
            finite_deviations = deviations[used & shared]
            stratum_deviation = (
                float(np.max(finite_deviations))
                if len(finite_deviations)
                else 0.0
            )
            group_counts = (
                users["group"].value_counts()
                if condition.name == "all"
                else subset["group"].value_counts()
            )
            global_ratio = group_counts.get("A", 0) / max(group_counts.get("B", 0), 1)
            global_target = 1.0 if condition.name == "all" else target_ratio
            global_deviation = max(
                global_ratio / global_target,
                global_target / max(global_ratio, np.finfo(float).eps),
            )
            score = singleton_count + max(stratum_deviation, global_deviation)
        return subset, score, stratum_deviation, global_deviation

    subset, score, stratum_deviation, global_deviation = evaluate(retained)
    best = (score, retained.copy(), subset, stratum_deviation, global_deviation)
    seen = {frozenset(retained)}
    total_swaps = 0
    passes_used = 0

    for pass_number in range(max_passes):
        passes_used = pass_number + 1
        current_users = add_user_statistics(subset).reset_index(drop=True)
        active_ids = set(retained)
        excluded_ids = set(all_user_ids) - active_ids
        if condition.name == "none" or not excluded_ids:
            break

        activity_edges = _quantile_edges(current_users["activity"], activity_quantiles)
        eccentricity_edges = _quantile_edges(current_users["eccentricity"], eccentricity_quantiles)
        activity_bins = _make_fixed_bins(current_users["activity"].to_numpy(), activity_edges)
        eccentricity_bins = _make_fixed_bins(current_users["eccentricity"].to_numpy(), eccentricity_edges)
        bin_count = max(1, len(eccentricity_edges) - 1)
        strata = activity_bins * bin_count + eccentricity_bins
        target = 1.0 if condition.name == "all" else target_ratio
        weights = np.ones(len(current_users), dtype=np.int64) if condition.name == "all" else current_users["activity"].to_numpy(dtype=np.int64)
        counts = np.zeros((int(strata.max()) + 1, 2), dtype=np.int64)
        np.add.at(counts.ravel(), strata * 2 + (current_users["group"].to_numpy() == "B"), weights)
        _, _, deviations = _deviations(counts, target, "multiplicative")
        worst_bins = np.argsort(np.where(counts.sum(axis=1), deviations, -np.inf))[::-1]
        remove_ids = [
            user_id for user_id, group, user_stratum in zip(
                current_users["user_id"], current_users["group"], strata
            )
            if user_stratum in worst_bins[: min(10, len(worst_bins))]
            and group == ("A" if counts[user_stratum, 0] > counts[user_stratum, 1] else "B")
            and user_id not in protected_user_ids
        ]
        add_ids = list(excluded_ids)
        rng.shuffle(remove_ids)
        rng.shuffle(add_ids)
        remove_ids = remove_ids[:20]
        add_ids = add_ids[:20]
        improved = False
        for remove_id in remove_ids:
            for add_id in add_ids:
                candidate = active_ids - {remove_id} | {add_id}
                candidate_size = int(user_sizes.reindex(list(candidate), fill_value=0).sum())
                if not minimum_size <= candidate_size <= maximum_size:
                    continue
                candidate_subset, candidate_score, candidate_stratum, candidate_global = evaluate(candidate)
                if candidate_score < score - 1e-12:
                    retained = candidate
                    subset = candidate_subset
                    score = candidate_score
                    stratum_deviation = candidate_stratum
                    global_deviation = candidate_global
                    active_ids = candidate
                    total_swaps += 1
                    improved = True
                    if score < best[0]:
                        best = (score, retained.copy(), subset, stratum_deviation, global_deviation)
                    break
            if improved:
                break
        if not improved or frozenset(retained) in seen:
            break
        seen.add(frozenset(retained))
        if total_swaps >= max_swaps_per_pass * max_passes:
            break

    score, retained, subset, stratum_deviation, global_deviation = best
    final_counts = subset["group"].value_counts().reindex(["A", "B"], fill_value=0)
    return subset.reset_index(drop=True), {
        "size": len(subset),
        "users": int(subset["user_id"].nunique()),
        "removed_users": int(len(all_user_ids) - subset["user_id"].nunique()),
        "group_counts": final_counts,
        "ratio": final_counts["A"] / max(final_counts["B"], 1),
        "target_ratio": target_ratio,
        "maximum_deviation": score,
        "stratum_deviation": stratum_deviation,
        "global_deviation": global_deviation,
        "iterations": passes_used,
        "swaps": total_swaps,
        "converged": total_swaps == 0 or passes_used < max_passes,
        "condition": condition.name,
        "method": "hybrid_frozen_swap",
    }


def find_reference_size(ratings, **kwargs):
    """Find the largest strict all-criteria subset for one fold."""

    strict = next(
        condition for condition in CONDITIONS if condition.name == "all")

    return balance_users(ratings, strict, **kwargs)


def trim_users_to_size(
    ratings,
    target_size,
    random_state=42,
    protected_user_ids=None,
    target_size_ratio_tolerance=0.001,
):
    """Remove complete users until the requested rating count is reached."""

    if target_size > len(ratings) or target_size < 0:

        raise ValueError("target_size must be within the rating count.")

    if target_size_ratio_tolerance < 0:

        raise ValueError("target_size_ratio_tolerance must be non-negative.")

    minimum_size = int(
        np.ceil(target_size * (1 - target_size_ratio_tolerance)))

    maximum_size = min(

        len(ratings),

        int(np.floor(target_size * (1 + target_size_ratio_tolerance))),

    )

    minimum_removal = len(ratings) - maximum_size

    maximum_removal = len(ratings) - minimum_size

    protected_user_ids = set() if protected_user_ids is None else set(protected_user_ids)

    frame_users = set(ratings["user_id"])

    unknown_protected = protected_user_ids - frame_users

    if unknown_protected:

        raise ValueError("Protected users must occur in the ratings frame.")

    protected_rating_count = int(

        ratings.loc[ratings["user_id"].isin(protected_user_ids)].shape[0]

    )

    if protected_rating_count > maximum_size:

        raise RuntimeError("Protected users exceed the requested size window.")

    if minimum_removal == 0 and maximum_removal == 0:

        return ratings.copy()

    rng = np.random.default_rng(random_state)

    user_sizes = ratings.loc[

        ~ratings["user_id"].isin(protected_user_ids)

    ].groupby("user_id").size()

    users = user_sizes.index.to_numpy(copy=True)

    sizes = user_sizes.to_numpy(dtype=np.int64, copy=True)

    order = rng.permutation(len(users))

    users = users[order]

    sizes = sizes[order]

    mask = (1 << (maximum_removal + 1)) - 1

    reachable = 1

    block_size = 64

    block_states = [reachable]

    for block_start in range(0, len(users), block_size):

        block_end = min(block_start + block_size, len(users))

        for user_size in sizes[block_start:block_end]:

            reachable |= (reachable << int(user_size)) & mask

        block_states.append(reachable)

    reachable_targets = [

        removal

        for removal in range(minimum_removal, maximum_removal + 1)

        if (reachable >> removal) & 1

    ]

    if not reachable_targets:

        raise RuntimeError(
            "No complete-user subset fits the requested size window.")

    removal_target = min(

        reachable_targets,

        key=lambda removal: (
            abs(len(ratings) - removal - target_size), removal),

    )

    removed_users = set()

    total = removal_target

    for block_index in range(len(block_states) - 2, -1, -1):

        block_start = block_index * block_size

        block_end = min(block_start + block_size, len(users))

        local_states = [block_states[block_index]]

        for user_size in sizes[block_start:block_end]:

            local_states.append(

                local_states[-1] | (local_states[-1] << int(user_size)) & mask

            )

        for offset in range(block_end - block_start - 1, -1, -1):

            if (local_states[offset] >> total) & 1:

                continue

            total -= int(sizes[block_start + offset])

            removed_users.add(users[block_start + offset])

    if total != 0:

        raise RuntimeError("Could not reconstruct the complete-user subset.")

    return ratings[~ratings["user_id"].isin(
        removed_users)].copy().reset_index(drop=True)
