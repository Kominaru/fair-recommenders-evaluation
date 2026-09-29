"""Run the MF balancing experiment with fixed user strata.

This wrapper preserves the existing runner and replaces only its subset
construction functions with the frozen-strata implementations.
"""

import numpy as np

import run_mf_balanced_conditions as experiment
from balanced_subset_frozen import (
    balance_users,
    find_reference_size,
    trim_users_to_size,
)


def select_protected_users(
    reference_train,
    test_frame,
    fraction,
    activity_quantiles,
    eccentricity_quantiles,
    random_state,
):
    """Sample one global cohort from users common to train and test."""

    if not 0 < fraction <= 1:
        raise ValueError("protected fraction must be in (0, 1].")
    test_user_ids = set(test_frame["user_id"])
    users = experiment.add_user_statistics(reference_train)
    users = users[users["user_id"].isin(test_user_ids)].copy()
    if users.empty:
        raise RuntimeError("No fully balanced users are present in the test fold.")
    groups = set(users["group"])
    if groups != {"A", "B"}:
        raise RuntimeError("The common user pool must contain both groups.")

    sample_size = max(2, int(np.floor(fraction * len(users))))
    sample_size = min(sample_size, len(users))
    rng = np.random.default_rng(random_state)
    selected = set(
        users.iloc[rng.choice(len(users), size=sample_size, replace=False)]["user_id"]
    )
    selected_groups = set(users[users["user_id"].isin(selected)]["group"])
    for group in ("A", "B"):
        if group not in selected_groups:
            replacement_pool = users[
                (users["group"] == group) & ~users["user_id"].isin(selected)
            ]
            replacement = replacement_pool.iloc[
                rng.integers(len(replacement_pool))
            ]["user_id"]
            remove_pool = users[
                (users["user_id"].isin(selected))
                & (users["group"] != group)
            ]
            selected.remove(remove_pool.iloc[0]["user_id"])
            selected.add(replacement)
    return selected


experiment.balance_users = balance_users
experiment.find_reference_size = find_reference_size
experiment.trim_users_to_size = trim_users_to_size
experiment.select_protected_users = select_protected_users


if __name__ == "__main__":
    experiment.run(experiment.parse_args())
