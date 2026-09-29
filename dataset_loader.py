"""Dataset loading utilities used by the experiment runners."""

from pathlib import Path

import numpy as np
import pandas as pd


def _normal_logpdf(values, mean, standard_deviation):
    if standard_deviation <= 0:
        raise ValueError("Artificial population standard deviations must be positive.")
    return (
        -0.5 * ((values - mean) / standard_deviation) ** 2
        - np.log(standard_deviation)
        - 0.5 * np.log(2 * np.pi)
    )


def _assign_artificial_groups(
    users,
    feature,
    population_means,
    population_standard_deviations,
    random_state,
):
    """Sample A/B labels from equal-prior Normal population likelihoods."""

    if len(population_means) != 2 or len(population_standard_deviations) != 2:
        raise ValueError("Artificial populations require exactly two means and standard deviations.")
    values = users[feature].to_numpy(dtype=float)
    log_likelihood_a = _normal_logpdf(
        values,
        population_means[0],
        population_standard_deviations[0],
    )
    log_likelihood_b = _normal_logpdf(
        values,
        population_means[1],
        population_standard_deviations[1],
    )
    maximum = np.maximum(log_likelihood_a, log_likelihood_b)
    likelihood_a = np.exp(log_likelihood_a - maximum)
    likelihood_b = np.exp(log_likelihood_b - maximum)
    probability_a = likelihood_a / (likelihood_a + likelihood_b)
    rng = np.random.default_rng(random_state)
    users["group"] = np.where(
        rng.random(len(users)) < probability_a,
        "A",
        "B",
    )
    return users


def load_movielens_ratings(
    data_dir: Path,
    group_by: str = "gender",
    age_threshold: int = 45,
    artificial_activity_means=(200.0, 400.0),
    artificial_activity_standard_deviations=(50.0, 50.0),
    artificial_eccentricity_means=(0.65, 0.85),
    artificial_eccentricity_standard_deviations=(0.15, 0.15),
    random_state: int = 42,
) -> pd.DataFrame:
    """Load MovieLens ratings and map the selected user attribute to A/B."""

    data_dir = Path(data_dir)
    valid_groupings = {"gender", "age", "artificial_activity", "artificial_eccentricity"}
    if group_by not in valid_groupings:
        raise ValueError(f"group_by must be one of {sorted(valid_groupings)}.")

    ratings = pd.read_csv(
        data_dir / "ratings.dat",
        sep="::",
        engine="python",
        names=["user_id", "item_id", "rating", "timestamp"],
        encoding="latin-1",
    )
    users = pd.read_csv(
        data_dir / "users.dat",
        sep="::",
        engine="python",
        names=["user_id", "gender", "age", "occupation", "zip_code"],
        encoding="latin-1",
    )
    ratings = ratings.merge(
        users[["user_id", "gender", "age"]],
        on="user_id",
        how="inner",
    )
    if group_by == "gender":
        ratings["group"] = ratings["gender"].map({"M": "A", "F": "B"})
    elif group_by == "age":
        ratings["group"] = np.where(
            ratings["age"] > age_threshold,
            "A",
            "B",
        )
    else:
        user_means = ratings.groupby("user_id")["rating"].mean()
        item_means = ratings.groupby("item_id")["rating"].mean()
        ratings["user_mean"] = ratings["user_id"].map(user_means)
        ratings["item_mean"] = ratings["item_id"].map(item_means)
        ratings["rating_eccentricity"] = (
            ratings["rating"]
            - (ratings["user_mean"] + ratings["item_mean"]) / 2
        ).abs()
        artificial_users = ratings.groupby("user_id", as_index=False).agg(
            activity=("rating", "size"),
            eccentricity=("rating_eccentricity", "mean"),
        )
        if group_by == "artificial_activity":
            artificial_users = _assign_artificial_groups(
                artificial_users,
                "activity",
                artificial_activity_means,
                artificial_activity_standard_deviations,
                random_state,
            )
        else:
            artificial_users = _assign_artificial_groups(
                artificial_users,
                "eccentricity",
                artificial_eccentricity_means,
                artificial_eccentricity_standard_deviations,
                random_state,
            )
        ratings = ratings.drop(
            columns=["user_mean", "item_mean", "rating_eccentricity"]
        ).merge(artificial_users[["user_id", "group"]], on="user_id")
    ratings = ratings[ratings["group"].isin(["A", "B"])].copy()
    return ratings[["user_id", "item_id", "rating", "group"]]
