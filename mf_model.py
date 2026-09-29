"""Matrix-factorization model used by the experiment runners."""

import logging

logging.getLogger("torch.utils.flop_counter").setLevel(logging.ERROR)

import lightning as L
import torch
from torch import nn


class MatrixFactorizationModel(L.LightningModule):
    """Predict ratings with user and item embedding vectors."""

    def __init__(self, num_users, num_items, embedding_dim, learning_rate):
        super().__init__()
        self.save_hyperparameters()
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)
        nn.init.xavier_uniform_(self.user_embeddings.weight)
        nn.init.xavier_uniform_(self.item_embeddings.weight)
        self.learning_rate = learning_rate
        self.register_buffer("train_squared_error", torch.tensor(0.0), persistent=False)
        self.register_buffer("train_rating_count", torch.tensor(0), persistent=False)
        self.register_buffer("test_squared_error", torch.tensor(0.0), persistent=False)
        self.register_buffer("test_rating_count", torch.tensor(0), persistent=False)

    def forward(self, user_ids, item_ids):
        user_vectors = self.user_embeddings(user_ids)
        item_vectors = self.item_embeddings(item_ids)
        logits = (user_vectors * item_vectors).sum(dim=1)
        return 1 + 4 * torch.sigmoid(logits)

    def training_step(self, batch, batch_index):
        user_ids, item_ids, ratings = batch
        predictions = self(user_ids, item_ids)
        squared_errors = (predictions - ratings).square()
        self.train_squared_error += squared_errors.detach().sum()
        self.train_rating_count += ratings.numel()
        self.log("train_mse", squared_errors.mean(), on_epoch=True)
        return squared_errors.mean()

    def on_train_epoch_end(self):
        train_rmse = torch.sqrt(self.train_squared_error / self.train_rating_count)
        self.log("train_rmse", train_rmse, prog_bar=True)
        self.train_squared_error.zero_()
        self.train_rating_count.zero_()

    def validation_step(self, batch, batch_index):
        user_ids, item_ids, ratings = batch
        predictions = self(user_ids, item_ids)
        self.test_squared_error += (predictions - ratings).square().detach().sum()
        self.test_rating_count += ratings.numel()

    def on_validation_epoch_end(self):
        test_rmse = torch.sqrt(self.test_squared_error / self.test_rating_count)
        self.log("test_rmse", test_rmse, prog_bar=True)
        self.test_squared_error.zero_()
        self.test_rating_count.zero_()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)