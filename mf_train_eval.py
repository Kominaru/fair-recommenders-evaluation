"""Training, data loading, and evaluation helpers for matrix factorization."""

import logging
import warnings

logging.getLogger("torch.utils.flop_counter").setLevel(logging.ERROR)

import numpy as np
import torch
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader, Dataset

from mf_model import MatrixFactorizationModel


for _logger_name in (
    "lightning_utilities.core.rank_zero",
    "lightning.pytorch.utilities.rank_zero",
):
    logging.getLogger(_logger_name).setLevel(logging.ERROR)
warnings.filterwarnings(
    "ignore",
    message=r"Checkpoint directory .* exists and is not empty.*",
    category=UserWarning,
)


class RatingsDataset(Dataset):
    """Expose indexed ratings as contiguous NumPy-backed samples."""

    def __init__(self, frame):
        self.user_ids = frame["user_index"].to_numpy(dtype=np.int64, copy=False)
        self.item_ids = frame["item_index"].to_numpy(dtype=np.int64, copy=False)
        self.ratings = frame["rating"].to_numpy(dtype=np.float32, copy=False)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, index):
        return self.user_ids[index], self.item_ids[index], self.ratings[index]


def make_dataloader(frame, batch_size, shuffle):
    """Create a data loader for an indexed ratings frame."""

    return DataLoader(
        RatingsDataset(frame),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )


def evaluate(model, data_loader, groups):
    """Return overall RMSE/MAE and the RMSE gap between groups."""

    model.eval()
    squared_error_sum = 0.0
    absolute_error_sum = 0.0
    rating_count = 0
    group_squared_error = {group: 0.0 for group in ("A", "B")}
    group_rating_count = {group: 0 for group in ("A", "B")}
    batch_start = 0

    with torch.no_grad():
        for user_ids, item_ids, batch_ratings in data_loader:
            user_ids = user_ids.to(model.device, non_blocking=True)
            item_ids = item_ids.to(model.device, non_blocking=True)
            batch_ratings = batch_ratings.to(model.device, non_blocking=True)
            errors = model(user_ids, item_ids) - batch_ratings
            squared_error_sum += errors.square().sum().item()
            absolute_error_sum += errors.abs().sum().item()
            rating_count += len(batch_ratings)
            batch_groups = groups[batch_start:batch_start + len(batch_ratings)]
            for group in ("A", "B"):
                group_mask = torch.as_tensor(batch_groups == group, device=errors.device)
                group_squared_error[group] += errors[group_mask].square().sum().item()
                group_rating_count[group] += int(group_mask.sum().item())
            batch_start += len(batch_ratings)

    group_rmse = {
        group: (group_squared_error[group] / group_rating_count[group]) ** 0.5
        for group in ("A", "B")
    }
    return {
        "rmse": (squared_error_sum / rating_count) ** 0.5,
        "mae": absolute_error_sum / rating_count,
        "rmse_A": group_rmse["A"],
        "rmse_B": group_rmse["B"],
        "eo_rmse": group_rmse["A"] - group_rmse["B"],
    }


def train_model(
    train_frame,
    validation_frame,
    num_users,
    num_items,
    embedding_dim,
    learning_rate,
    batch_size,
    max_epochs,
    checkpoint_dir,
    early_stopping_patience=5,
    quiet=False,
):
    """Train one MF model and return the best checkpoint path."""

    model = MatrixFactorizationModel(
        num_users, num_items, embedding_dim, learning_rate
    )
    checkpoint = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="best",
        monitor="test_rmse",
        mode="min",
        save_top_k=1,
    )
    trainer = L.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=max_epochs,
        callbacks=[
            checkpoint,
            EarlyStopping(
                monitor="test_rmse",
                mode="min",
                patience=early_stopping_patience,
            ),
        ],
        logger=False,
        enable_model_summary=False,
        enable_progress_bar=not quiet,
        num_sanity_val_steps=0,
    )
    trainer.fit(
        model,
        train_dataloaders=make_dataloader(train_frame, batch_size, True),
        val_dataloaders=make_dataloader(validation_frame, batch_size, False),
    )
    return checkpoint.best_model_path


def train_and_evaluate(
    train_frame,
    test_frame,
    num_users,
    num_items,
    embedding_dim,
    learning_rate,
    batch_size,
    max_epochs,
    checkpoint_dir,
    early_stopping_patience=5,
    quiet=False,
):
    """Train one model and evaluate it on the supplied test frame."""

    checkpoint_path = train_model(
        train_frame=train_frame,
        validation_frame=test_frame,
        num_users=num_users,
        num_items=num_items,
        embedding_dim=embedding_dim,
        learning_rate=learning_rate,
        batch_size=batch_size,
        max_epochs=max_epochs,
        checkpoint_dir=checkpoint_dir,
        early_stopping_patience=early_stopping_patience,
        quiet=quiet,
    )
    return evaluate_checkpoint(checkpoint_path, test_frame, batch_size), checkpoint_path


def evaluate_checkpoint(
    checkpoint_path,
    test_frame,
    batch_size,
):
    """Evaluate a saved MF checkpoint on an alternate test subset."""

    model = MatrixFactorizationModel.load_from_checkpoint(checkpoint_path).to(
        torch.device("cuda")
    )
    return evaluate(
        model,
        make_dataloader(test_frame, batch_size, False),
        test_frame["group"].to_numpy(),
    )