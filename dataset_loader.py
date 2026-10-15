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
    if isinstance(random_state, np.ndarray):
        uniforms = random_state
    else:
        uniforms = np.random.default_rng(random_state).random(len(users))
    users["group"] = np.where(uniforms < probability_a, "A", "B")
    return users


def induced_user_features(ratings):
    """Per-user activity and mean rating eccentricity computed from ``ratings``."""

    user_means = ratings.groupby("user_id")["rating"].mean()
    item_means = ratings.groupby("item_id")["rating"].mean()
    eccentricity = (
        ratings["rating"]
        - (ratings["user_id"].map(user_means) + ratings["item_id"].map(item_means)) / 2
    ).abs()
    return pd.DataFrame(
        {
            "activity": ratings.groupby("user_id")["rating"].size(),
            "eccentricity": eccentricity.groupby(ratings["user_id"]).mean(),
        }
    )


def induced_groups(
    ratings,
    user_ids,
    group_by,
    means,
    standard_deviations,
    random_state=42,
):
    """Induced A/B labels from features computed on ``ratings``.

    Each user keeps one uniform draw, indexed by sorted user id, so labels
    computed from different rating subsets (e.g. each fold's training data)
    only differ for users whose assignment probability changes.
    """

    feature = {"artificial_activity": "activity", "artificial_eccentricity": "eccentricity"}[
        group_by
    ]
    rng = np.random.default_rng(random_state)
    user_ids = np.sort(np.unique(user_ids))
    uniforms = pd.Series(rng.random(len(user_ids)), index=user_ids)
    users = induced_user_features(ratings).reset_index(names="user_id")
    users = _assign_artificial_groups(
        users, feature, means, standard_deviations, uniforms.loc[users["user_id"]].to_numpy()
    )
    return users.set_index("user_id")["group"]


def _load_grouped_ratings_file(path, group_by, random_state):
    """Ratings that already carry an A/B group: columns user_id,item_id,rating,group.

    ``group_by="gender"`` (or any real construction) uses the file's labels;
    ``group_by="placebo"`` permutes the labels across users, keeping the group sizes.
    """

    ratings = pd.read_csv(path)
    missing = {"user_id", "item_id", "rating", "group"} - set(ratings.columns)
    if missing:
        raise ValueError(f"{path} lacks columns {sorted(missing)}.")
    if group_by == "placebo":
        users = ratings.drop_duplicates("user_id")[["user_id", "group"]]
        rng = np.random.default_rng(random_state)
        permuted = pd.Series(rng.permutation(users["group"].to_numpy()), index=users["user_id"])
        ratings["group"] = ratings["user_id"].map(permuted)
    elif group_by not in {"gender", "age"}:
        raise ValueError("A grouped ratings file supports group_by 'gender' (its own labels) or 'placebo'.")
    ratings = ratings[ratings["group"].isin(["A", "B"])].copy()
    return ratings[["user_id", "item_id", "rating", "group"]]


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
    if data_dir.is_file():
        return _load_grouped_ratings_file(data_dir, group_by, random_state)
    valid_groupings = {
        "gender",
        "age",
        "artificial_activity",
        "artificial_eccentricity",
        "placebo",
        "placebo_age",
    }
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
    elif group_by == "placebo":
        # Gender labels permuted across users: same A:B ratio as gender, but
        # unrelated to anything about the user.
        rng = np.random.default_rng(random_state)
        labels = users["gender"].map({"M": "A", "F": "B"}).to_numpy()
        placebo = pd.Series(rng.permutation(labels), index=users["user_id"])
        ratings["group"] = ratings["user_id"].map(placebo)
    elif group_by == "placebo_age":
        # Age labels permuted across users: the group sizes of the age
        # construction (older users are the small group A), unrelated to
        # anything about the user.
        rng = np.random.default_rng(random_state)
        labels = np.where(users["age"] > age_threshold, "A", "B")
        placebo = pd.Series(rng.permutation(labels), index=users["user_id"])
        ratings["group"] = ratings["user_id"].map(placebo)
    elif group_by == "age":
        ratings["group"] = np.where(
            ratings["age"] > age_threshold,
            "A",
            "B",
        )
    else:
        if group_by == "artificial_activity":
            means = artificial_activity_means
            standard_deviations = artificial_activity_standard_deviations
        else:
            means = artificial_eccentricity_means
            standard_deviations = artificial_eccentricity_standard_deviations
        groups = induced_groups(
            ratings,
            ratings["user_id"],
            group_by,
            means,
            standard_deviations,
            random_state,
        )
        ratings["group"] = ratings["user_id"].map(groups)
    ratings = ratings[ratings["group"].isin(["A", "B"])].copy()
    return ratings[["user_id", "item_id", "rating", "group"]]
