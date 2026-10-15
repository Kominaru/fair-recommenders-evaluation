"""Score every train-side condition on every test population for the joint EO split.

Per fold this builds the eight matched-size training conditions of
decompose_train_results.LATTICE plus the full training fold, fits each
predictor on each of them, and evaluates every fit on the four fixed test
populations of balanced_test.joint_test_reference. decompose_joint_results.py
turns the resulting long table into the five-player Shapley decomposition.

Each training set loses a random ``--validation-fraction`` of its ratings to a
validation split used only for MF early stopping, so no test rating is used
for model selection. Every fit is scored under three test variants:
``rating10`` (rating-level eccentricity, the main analysis), ``user10``
(label-free user-level eccentricity) and, for the full fold only,
``rating50`` (50 bins). The full model is also scored on the evaluable test
ratings of every other condition (``panel``), which measures how much the
change of evaluated users alone moves the EO.

``--predictors user_item_average,random_uniform`` needs no training; ``mf`` uses
a CUDA GPU when available and the CPU otherwise.
"""

import argparse
import logging
from pathlib import Path
import shutil
import warnings

logging.getLogger("torch.utils.flop_counter").setLevel(logging.ERROR)

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from balanced_subset import CONDITIONS, user_strata
from balanced_test import (
    TEST_POPULATIONS,
    baseline_predictions,
    evaluate_predictions,
    joint_test_reference,
    weight_test_population,
)
from dataset_loader import induced_groups, load_movielens_ratings
from decompose_train_results import LATTICE
from run_mf_balanced_conditions import balanced_bases, build_training_subset

warnings.filterwarnings(
    "ignore",
    message=r"`isinstance\(treespec, LeafSpec\)` is deprecated.*",
)
warnings.filterwarnings(
    "ignore",
    message="Please migrate to LitLogger.*",
    category=UserWarning,
)

PREDICTORS = ("mf", "user_item_average", "random_uniform")
# Opt-in non-factorization models (alt_models.py); not part of the default set.
ALT_PREDICTORS = ("glocal_k", "gcmc")
# name -> (eccentricity definition, bins, scored for every training set)
TEST_VARIANTS = {
    "rating10": ("rating", 10, True),
    "user10": ("user", 10, True),
    "rating50": ("rating", 50, False),
}
CONSTRUCTIONS = (
    "gender",
    "age",
    "artificial_activity",
    "artificial_eccentricity",
    "placebo",
    "placebo_age",
)


def parse_list(choices):
    def parse(value):
        names = tuple(name.strip() for name in value.split(",") if name.strip())
        unknown = set(names) - set(choices)
        if unknown:
            raise argparse.ArgumentTypeError(
                f"Unknown values {sorted(unknown)}; choose from {list(choices)}"
            )
        return names

    return parse


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--original-data-dir", type=Path, default=Path("movielens_data/ml-1m"))
    parser.add_argument("--group-by", choices=CONSTRUCTIONS, default="gender")
    parser.add_argument("--age-threshold", type=int, default=45)
    # Defaults are the induced constructions described in report.tex.
    for feature, means, stds in (
        ("activity", (300.0, 400.0), (200.0, 200.0)),
        ("eccentricity", (0.65, 0.85), (0.15, 0.15)),
    ):
        parser.add_argument(
            f"--artificial-{feature}-means", type=float, nargs=2, default=means
        )
        parser.add_argument(
            f"--artificial-{feature}-stds", type=float, nargs=2, default=stds
        )
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--fold-limit", type=int, default=None)
    parser.add_argument(
        "--only-folds",
        type=lambda value: {int(fold) for fold in value.split(",")},
        default=None,
        help="Comma-separated folds to run, e.g. to split folds across jobs.",
    )
    parser.add_argument("--seed", type=int, default=727)
    parser.add_argument(
        "--label-seed",
        type=int,
        default=42,
        help="Seed of the induced and placebo group assignments.",
    )
    parser.add_argument(
        "--induced-labels",
        choices=("train", "all"),
        default="train",
        help="Compute induced-group features from each fold's training "
        "ratings (no test leakage) or from all ratings.",
    )
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument(
        "--plant-noise-train",
        type=float,
        default=0.0,
        help="Benchmark: add Gaussian noise with this standard deviation to the "
        "training ratings of group B only.",
    )
    parser.add_argument(
        "--plant-noise-test",
        type=float,
        default=0.0,
        help="Benchmark: add Gaussian noise with this standard deviation to the "
        "test ratings of group B only.",
    )
    parser.add_argument(
        "--plant-item-offset",
        type=float,
        default=0.0,
        help="Benchmark: add an item-specific offset drawn from N(0, sigma) to "
        "the training and test ratings of group B only (a group that rates "
        "items differently).",
    )
    parser.add_argument(
        "--user-fraction",
        type=float,
        default=1.0,
        help="Uncertainty check: keep this random share of the users (all their ratings) "
        "before splitting. With 0.5, the spread of an estimate across subsamples "
        "approximates its standard error on the full sample (delete-d jackknife).",
    )
    parser.add_argument("--subsample-seed", type=int, default=0)
    parser.add_argument(
        "--test-side-only",
        action="store_true",
        help="Fit only the full-fold models: enough for the test-side split.",
    )
    parser.add_argument("--activity-quantiles", type=int, default=10)
    parser.add_argument("--eccentricity-quantiles", type=int, default=10)
    parser.add_argument(
        "--test-activity-quantiles",
        type=int,
        default=10,
        help="Bins for the test populations (the report's test side used 50).",
    )
    parser.add_argument("--test-eccentricity-quantiles", type=int, default=10)
    parser.add_argument("--ratio-tolerance", type=float, default=1.05)
    parser.add_argument("--target-size-ratio-tolerance", type=float, default=0.001)
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument(
        "--eccentricity-updates", choices=("frozen", "dynamic"), default="frozen"
    )
    parser.add_argument(
        "--predictors",
        type=parse_list(PREDICTORS + ALT_PREDICTORS),
        default=PREDICTORS,
        help="Comma-separated predictors to fit on every training set.",
    )
    parser.add_argument(
        "--noise-seeds",
        type=int,
        default=3,
        help="MF seeds for the full fold and the resized control (noise floor).",
    )
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2**14)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("mf_joint_checkpoints"))
    parser.add_argument("--results-file", type=Path, default=Path("mf_joint_results.csv"))
    parser.add_argument(
        "--keep-checkpoints",
        action="store_true",
        help="Keep MF checkpoints after scoring (about 30 MB each).",
    )
    parser.add_argument("--glocal-pretrain-epochs", type=int, default=3)
    parser.add_argument("--glocal-finetune-epochs", type=int, default=6)
    parser.add_argument("--gcmc-max-epochs", type=int, default=150)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    args.conditions = tuple(LATTICE.values())
    return args


def load_ratings(args):
    ratings = load_movielens_ratings(
        args.original_data_dir,
        group_by=args.group_by,
        age_threshold=args.age_threshold,
        artificial_activity_means=args.artificial_activity_means,
        artificial_activity_standard_deviations=args.artificial_activity_stds,
        artificial_eccentricity_means=args.artificial_eccentricity_means,
        artificial_eccentricity_standard_deviations=args.artificial_eccentricity_stds,
        random_state=args.label_seed,
    )
    ratings["user_index"], user_ids = pd.factorize(ratings["user_id"], sort=True)
    ratings["item_index"], item_ids = pd.factorize(ratings["item_id"], sort=True)
    return ratings, len(user_ids), len(item_ids)


def relabel_from_training(args, ratings, train_frame, test_frame):
    """Recompute induced groups from this fold's training ratings only."""

    if args.group_by == "artificial_activity":
        means, stds = args.artificial_activity_means, args.artificial_activity_stds
    else:
        means, stds = args.artificial_eccentricity_means, args.artificial_eccentricity_stds
    groups = induced_groups(
        train_frame, ratings["user_id"], args.group_by, means, stds, args.label_seed
    )
    train_frame["group"] = train_frame["user_id"].map(groups)
    test_frame["group"] = test_frame["user_id"].map(groups)


def plant_label_noise(args, fold, train_frame, test_frame, item_ids=None):
    """Add noise or an item-specific shift to group B's ratings.

    Noise on the test ratings changes the expected squared error of group B by
    exactly its variance, which gives a ground truth for the test side. Noise
    on the training ratings changes what the model learns, which gives a
    ground truth through the EO of a model trained without it. An item offset
    makes group B rate items differently from group A, a taste difference that
    a model fitted mostly to group A learns badly.
    """

    rng = np.random.default_rng(args.seed + 10_000 + fold)
    for frame, sigma in (
        (train_frame, args.plant_noise_train),
        (test_frame, args.plant_noise_test),
    ):
        if not sigma:
            continue
        frame["rating"] = frame["rating"].astype(float)
        mask = (frame["group"] == "B").to_numpy()
        frame.loc[mask, "rating"] = frame.loc[mask, "rating"].to_numpy() + rng.normal(
            0.0, sigma, int(mask.sum())
        )
    if args.plant_item_offset:
        # One offset per item, the same in every fold.
        offsets = pd.Series(
            np.random.default_rng(args.seed + 20_000).normal(0.0, args.plant_item_offset, len(item_ids)),
            index=item_ids,
        )
        for frame in (train_frame, test_frame):
            frame["rating"] = frame["rating"].astype(float)
            mask = (frame["group"] == "B").to_numpy()
            frame.loc[mask, "rating"] = frame.loc[mask, "rating"].to_numpy() + offsets.reindex(
                frame.loc[mask, "item_id"]
            ).to_numpy()


def split_validation(args, fold, subset):
    """Hold out a random share of a training set for early stopping."""

    validation = subset.sample(
        frac=args.validation_fraction, random_state=args.seed + fold
    )
    fit = subset.drop(index=validation.index)
    validation = validation[
        validation["user_id"].isin(set(fit["user_id"]))
        & validation["item_id"].isin(set(fit["item_id"]))
    ].copy()
    validation["test_weight"] = 1.0
    return fit, validation


def total_variation(strata, users, column):
    shares = (
        strata.loc[users].groupby("group")[column].value_counts(normalize=True)
        .unstack(fill_value=0)
        .reindex(["A", "B"], fill_value=0)
    )
    return 0.5 * float((shares.loc["A"] - shares.loc["B"]).abs().sum())


def describe_training(subset, strata):
    """Group sizes and remaining factor disparities of one training set."""

    users = subset.groupby("group")["user_id"].nunique()
    ratings = subset["group"].value_counts()
    kept = subset["user_id"].unique()
    return {
        "users_A": int(users.get("A", 0)),
        "users_B": int(users.get("B", 0)),
        "ratings_A": int(ratings.get("A", 0)),
        "ratings_B": int(ratings.get("B", 0)),
        "tvd_activity": total_variation(strata, kept, "activity_bin"),
        "tvd_eccentricity": total_variation(strata, kept, "eccentricity_bin"),
        "tvd_joint": total_variation(strata, kept, "joint_bin"),
    }


def evaluation_frames(training, fit, references, panel_fits):
    """Weighted test frames for every variant, population and panel."""

    frames = []
    for variant, (_, _, everyone) in TEST_VARIANTS.items():
        if not everyone and training != "full":
            continue
        populations = TEST_POPULATIONS + (("both_own",) if variant == "rating10" else ())
        for population in populations:
            frame, info = weight_test_population(fit, references[variant], population)
            frames.append((variant, population, training, frame, info))
        if training == "full" and variant != "rating50":
            for panel, panel_fit in panel_fits.items():
                for population in TEST_POPULATIONS:
                    frame, info = weight_test_population(
                        panel_fit, references[variant], population
                    )
                    frames.append((variant, population, panel, frame, info))
    return frames


def score(frames, predict):
    rows = []
    for variant, population, panel, frame, info in frames:
        metrics = predict(frame)
        rows.append(
            {"test_variant": variant, "population": population, "panel": panel,
             "evaluated_ratings": len(frame), **info, **metrics}
        )
    return rows


def score_baselines(args, fold, fit, frames):
    """Score the closed-form and training-free baselines."""

    rows = []
    for predictor in ("user_item_average", "random_uniform"):
        if predictor not in args.predictors:
            continue

        def predict(frame):
            predictions = baseline_predictions(fit, frame, random_state=args.seed + fold)
            return evaluate_predictions(frame, predictions[predictor])

        rows += [{"predictor": predictor, "seed": 0, **row} for row in score(frames, predict)]
    return rows


def score_mf(args, fold, training, fit, validation, frames, num_users, num_items):
    """Train MF on one training set and score it on every frame."""

    import lightning as L
    import torch

    from mf_model import MatrixFactorizationModel
    from mf_train_eval import evaluate, make_dataloader, train_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = args.noise_seeds if training in {"full", "none"} else 1
    checkpoint_dir = (
        args.checkpoint_dir / args.group_by / f"labels_{args.label_seed}"
        / f"fold_{fold}" / training
    )
    rows = []
    for seed_index in range(seeds):
        L.seed_everything(args.seed + 1000 * fold + seed_index, workers=True)
        checkpoint = train_model(
            train_frame=fit,
            validation_frame=validation,
            num_users=num_users,
            num_items=num_items,
            embedding_dim=args.embedding_dim,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            checkpoint_dir=checkpoint_dir / f"seed_{seed_index}",
            early_stopping_patience=5,
            quiet=args.quiet,
        )
        model = MatrixFactorizationModel.load_from_checkpoint(checkpoint).to(device)

        def predict(frame):
            return evaluate(
                model,
                make_dataloader(frame, args.batch_size, False, include_weights=True),
                frame["group"].to_numpy(),
            )

        rows += [{"predictor": "mf", "seed": seed_index, **row} for row in score(frames, predict)]
    if not args.keep_checkpoints:
        shutil.rmtree(checkpoint_dir, ignore_errors=True)
    return rows


def score_alt(args, fold, name, training, fit, validation, frames, num_users, num_items):
    """Train a non-factorization model (GLocal-K or GC-MC) and score every frame."""

    import lightning as L
    import torch

    from alt_models import train_alt_model
    from mf_train_eval import evaluate, make_dataloader

    options = {
        "glocal_k": dict(
            pretrain_epochs=args.glocal_pretrain_epochs,
            finetune_epochs=args.glocal_finetune_epochs,
        ),
        "gcmc": dict(max_epochs=args.gcmc_max_epochs),
    }[name]
    seeds = args.noise_seeds if training in {"full", "none"} else 1
    rows = []
    for seed_index in range(seeds):
        L.seed_everything(args.seed + 1000 * fold + seed_index, workers=True)
        model = train_alt_model(
            name, fit, validation, num_users, num_items, quiet=args.quiet, **options
        )

        def predict(frame):
            return evaluate(
                model,
                make_dataloader(frame, args.batch_size, False, include_weights=True),
                frame["group"].to_numpy(),
            )

        rows += [{"predictor": name, "seed": seed_index, **row} for row in score(frames, predict)]
    return rows


def run(args):
    if "mf" in args.predictors:
        import torch

        if not torch.cuda.is_available():
            print("No CUDA GPU found; training MF on CPU.")
    ratings, num_users, num_items = load_ratings(args)
    if args.user_fraction < 1.0:
        users = np.sort(ratings["user_id"].unique())
        keep = np.random.default_rng(args.subsample_seed).choice(
            users, int(round(args.user_fraction * len(users))), replace=False
        )
        ratings = ratings[ratings["user_id"].isin(keep)].reset_index(drop=True)
    condition_map = {condition.name: condition for condition in CONDITIONS}
    splitter = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    results = []
    for fold, (train_indices, test_indices) in enumerate(splitter.split(ratings), start=1):
        if args.fold_limit is not None and fold > args.fold_limit:
            break
        if args.only_folds is not None and fold not in args.only_folds:
            continue
        train_frame = ratings.iloc[train_indices].copy()
        test_frame = ratings.iloc[test_indices].copy()
        if args.group_by.startswith("artificial") and args.induced_labels == "train":
            relabel_from_training(args, ratings, train_frame, test_frame)
        plant_label_noise(args, fold, train_frame, test_frame, item_ids=np.sort(ratings["item_id"].unique()))
        strata = user_strata(
            train_frame,
            activity_quantiles=args.activity_quantiles,
            eccentricity_quantiles=args.eccentricity_quantiles,
        )
        if args.test_side_only:
            bases, reference_condition, size = {}, "full", 0
        else:
            bases = balanced_bases(args, fold, train_frame)
            reference_condition = min(bases, key=lambda name: len(bases[name]))
            size = len(bases[reference_condition])
        references = {
            variant: joint_test_reference(
                train_frame, test_frame, activity_quantiles=bins,
                eccentricity_quantiles=bins, eccentricity=eccentricity,
            )
            for variant, (eccentricity, bins, _) in TEST_VARIANTS.items()
        }
        print(f"Fold {fold:2d}: N_ref={size:,} set by {reference_condition}")
        trainings = {"full": train_frame}
        for name in () if args.test_side_only else args.conditions:
            trainings[name] = build_training_subset(
                args, condition_map[name], fold, bases[name], size, strata
            )
        splits = {name: split_validation(args, fold, subset) for name, subset in trainings.items()}
        panel_fits = {name: fit for name, (fit, _) in splits.items() if name != "full"}
        for training, train_subset in trainings.items():
            fit, validation = splits[training]
            frames = evaluation_frames(training, fit, references, panel_fits)
            rows = score_baselines(args, fold, fit, frames)
            if "mf" in args.predictors:
                rows += score_mf(
                    args, fold, training, fit, validation, frames, num_users, num_items
                )
            for name in ALT_PREDICTORS:
                if name in args.predictors:
                    rows += score_alt(
                        args, fold, name, training, fit, validation, frames, num_users, num_items
                    )
            diagnostics = describe_training(train_subset, strata)
            for row in rows:
                row.update(
                    {
                        "construction": args.group_by,
                        "label_seed": args.label_seed,
                        "plant_noise_train": args.plant_noise_train,
                        "plant_noise_test": args.plant_noise_test,
                        "plant_item_offset": args.plant_item_offset,
                        "user_fraction": args.user_fraction,
                        "subsample_seed": args.subsample_seed,
                        "fold": fold,
                        "training": training,
                        "train_ratings": len(fit),
                        "reference_size": size,
                        **diagnostics,
                    }
                )
            main = [row for row in rows if row["test_variant"] == "rating10"
                    and row["panel"] == training]
            for row in main:
                print(
                    f"Fold {fold:2d} {training:30s} {row['predictor']:17s} "
                    f"seed {row['seed']} {row['population']:12s} "
                    f"EO={row['eo_rmse']:+.4f}"
                )
            results += rows
        pd.DataFrame(results).to_csv(args.results_file, index=False)
    print(f"Saved {len(results):,} rows to {args.results_file}")


if __name__ == "__main__":
    run(parse_args())
