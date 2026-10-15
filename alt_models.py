"""Non-factorization rating predictors: GLocal-K and GC-MC.

Both models expose the interface used by ``mf_train_eval.evaluate``: a
``torch.nn.Module`` with a ``device`` attribute whose ``forward(user_ids,
item_ids)`` returns rating predictions in [1, 5].  ``train_alt_model`` fits one
of them on a ratings frame (columns ``user_index``, ``item_index``, ``rating``),
early-stops on a validation frame, and returns the best model.

* GLocal-K (Han et al., CIKM 2021, https://github.com/usydnlp/Glocal_K): a
  kernelised autoencoder over the item-by-user rating matrix, pre-trained with
  local kernels and fine-tuned with a global convolution kernel.  Ported from
  the authors' TensorFlow 1 notebook; hyperparameters are theirs for ML-1M.
* GC-MC (Berg, Kipf, Welling, 2017, https://arxiv.org/abs/1706.02263):
  relational graph-convolutional encoder over the bipartite rating graph with
  one edge type per rating value and a bilinear decoder over rating classes.
"""

import copy
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _tensor(frame, column, dtype):
    return torch.as_tensor(frame[column].to_numpy().copy(), dtype=dtype, device=DEVICE)


def _rating_matrices(frame, num_users, num_items):
    ratings = torch.zeros(num_items, num_users, device=DEVICE)
    ratings[_tensor(frame, "item_index", torch.long), _tensor(frame, "user_index", torch.long)] = (
        _tensor(frame, "rating", torch.float32)
    )
    return ratings


# ----------------------------------------------------------------------------
# GLocal-K
# ----------------------------------------------------------------------------


class KernelLayer(nn.Module):
    """Dense layer whose weights are gated by a learned local kernel."""

    def __init__(self, n_in, n_hid, n_dim, activation, lambda_s, lambda_2):
        super().__init__()
        self.weight = nn.Parameter(nn.init.xavier_uniform_(torch.empty(n_in, n_hid)))
        self.u = nn.Parameter(torch.randn(n_in, 1, n_dim) * 1e-3)
        self.v = nn.Parameter(torch.randn(1, n_hid, n_dim) * 1e-3)
        self.bias = nn.Parameter(torch.zeros(n_hid))
        self.activation = activation
        self.lambda_s = lambda_s
        self.lambda_2 = lambda_2

    def forward(self, x):
        distance = torch.linalg.vector_norm(self.u - self.v, dim=2)
        local_kernel = torch.clamp(1.0 - distance**2, min=0.0)
        out = self.activation(x @ (self.weight * local_kernel) + self.bias)
        # TF's l2_regularizer(scale) is scale * sum(w^2) / 2.
        reg = 0.5 * self.lambda_s * local_kernel.square().sum()
        reg = reg + 0.5 * self.lambda_2 * self.weight.square().sum()
        return out, reg


class GLocalK(nn.Module):
    def __init__(self, num_users, num_items, n_hid=500, n_dim=5, n_layers=2,
                 gk_size=3, lambda_2=70.0, lambda_s=0.018, dot_scale=0.5):
        super().__init__()
        self.num_users, self.num_items = num_users, num_items
        self.gk_size, self.dot_scale = gk_size, dot_scale
        sigmoid = torch.sigmoid
        self.hidden = nn.ModuleList(
            KernelLayer(num_users if i == 0 else n_hid, n_hid, n_dim, sigmoid, lambda_s, lambda_2)
            for i in range(n_layers)
        )
        self.out = KernelLayer(n_hid, num_users, n_dim, nn.Identity(), lambda_s, lambda_2)
        self.conv_kernel = nn.Parameter(torch.randn(num_items, gk_size**2) * 0.1)
        self.register_buffer("train_matrix", torch.zeros(num_items, num_users), persistent=False)
        self.cached = None
        self.pretraining = True

    @property
    def device(self):
        return self.conv_kernel.device

    def _encode(self, x, regularize):
        reg = 0.0
        for layer in self.hidden:
            x, r = layer(x)
            reg = reg + r
        x, r = self.out(x)
        return x, reg + r

    def matrix(self):
        """Predicted rating matrix (items x users) given the training matrix."""

        data = self.train_matrix
        if self.pretraining:
            return self._encode(data, True)
        y_dash, _ = self._encode(data, False)
        pooled = y_dash.mean(dim=1).reshape(1, -1)
        kernel = (pooled @ self.conv_kernel * self.dot_scale).reshape(1, 1, self.gk_size, self.gk_size)
        smoothed = F.relu(F.conv2d(data[None, None], kernel, padding="same"))[0, 0]
        return self._encode(smoothed, True)

    def loss(self):
        prediction, reg = self.matrix()
        mask = (self.train_matrix > 1e-12).float()
        return 0.5 * (mask * (self.train_matrix - prediction)).square().sum() + reg

    @torch.no_grad()
    def refresh(self):
        self.cached = self.matrix()[0].clamp(1.0, 5.0)

    def forward(self, user_ids, item_ids):
        if self.cached is None:
            self.refresh()
        return self.cached[item_ids, user_ids]


def _lbfgs_chunk(model, parameters, iterations):
    optimizer = torch.optim.LBFGS(
        parameters, lr=1.0, max_iter=iterations, history_size=10,
        tolerance_grad=1e-9, tolerance_change=1e-12, line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad()
        value = model.loss()
        value.backward()
        return value

    optimizer.step(closure)


def _validation_rmse(model, frame):
    model.eval()
    model.cached = None
    users = _tensor(frame, "user_index", torch.long)
    items = _tensor(frame, "item_index", torch.long)
    ratings = _tensor(frame, "rating", torch.float32)
    with torch.no_grad():
        return (model(users, items) - ratings).square().mean().sqrt().item()


def train_glocal_k(train_frame, validation_frame, num_users, num_items, *,
                   pretrain_epochs=20, finetune_epochs=30, pretrain_iters=50,
                   finetune_iters=10, patience=5, quiet=True, **kwargs):
    model = GLocalK(num_users, num_items, **kwargs).to(DEVICE)
    model.train_matrix = _rating_matrices(train_frame, num_users, num_items)
    best_rmse, best_state, stale = float("inf"), None, 0
    for stage, epochs, iters in (("pre", pretrain_epochs, pretrain_iters),
                                 ("fine", finetune_epochs, finetune_iters)):
        model.pretraining = stage == "pre"
        if stage == "fine":
            best_rmse, best_state, stale = float("inf"), None, 0
        for epoch in range(epochs):
            tic = time.time()
            model.train()
            _lbfgs_chunk(model, list(model.parameters()), iters)
            rmse = _validation_rmse(model, validation_frame)
            if not quiet:
                print(f"GLocal-K {stage} epoch {epoch + 1}: val rmse {rmse:.4f} ({time.time() - tic:.1f}s)")
            if rmse < best_rmse - 1e-5:
                best_rmse, stale = rmse, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                stale += 1
                if stale >= patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state, strict=False)
    model.eval()
    model.cached = None
    return model


# ----------------------------------------------------------------------------
# GC-MC
# ----------------------------------------------------------------------------


class GCMC(nn.Module):
    """Graph auto-encoder for rating matrices (one-hot node features)."""

    def __init__(self, num_users, num_items, num_levels=5, hidden=256, out_dim=32,
                 num_basis=2, dropout=0.7):
        super().__init__()
        self.num_users, self.num_items, self.num_levels = num_users, num_items, num_levels
        self.dropout = dropout
        # One-hot inputs make the first-layer weights per-node embeddings per
        # rating level, built from a small set of shared basis matrices.
        self.basis_user = nn.Parameter(nn.init.xavier_uniform_(torch.empty(num_basis, num_items, hidden)))
        self.basis_item = nn.Parameter(nn.init.xavier_uniform_(torch.empty(num_basis, num_users, hidden)))
        self.coeff = nn.Parameter(torch.randn(num_levels, num_basis) * 0.5)
        self.dense_user = nn.Linear(hidden, out_dim)
        self.dense_item = nn.Linear(hidden, out_dim)
        self.decoder_basis = nn.Parameter(nn.init.orthogonal_(torch.empty(num_basis, out_dim * out_dim)).reshape(num_basis, out_dim, out_dim))
        self.decoder_coeff = nn.Parameter(torch.randn(num_levels, num_basis) * 0.5)
        self.cached = None

    @property
    def device(self):
        return self.coeff.device

    def set_graph(self, frame):
        users = _tensor(frame, "user_index", torch.long)
        items = _tensor(frame, "item_index", torch.long)
        levels = _tensor(frame, "rating", torch.float32).round().long() - 1
        user_degree = torch.bincount(users, minlength=self.num_users).clamp(min=1).float()
        item_degree = torch.bincount(items, minlength=self.num_items).clamp(min=1).float()
        # Symmetric normalisation sqrt(d_u d_i), one sparse matrix per level.
        weight = 1.0 / torch.sqrt(user_degree[users] * item_degree[items])
        self.edges = (users, items, levels, weight)

    def _propagate(self):
        users, items, levels, weight = self.edges
        w_user = torch.einsum("lb,bih->lih", self.coeff, self.basis_user)  # levels x items x hid
        w_item = torch.einsum("lb,buh->luh", self.coeff, self.basis_item)  # levels x users x hid
        hidden_users = torch.zeros(self.num_users, w_user.shape[-1], device=w_user.device)
        hidden_items = torch.zeros(self.num_items, w_user.shape[-1], device=w_user.device)
        # Messages from items to users and users to items (one-hot input => row lookup).
        hidden_users.index_add_(0, users, w_user[levels, items] * weight[:, None])
        hidden_items.index_add_(0, items, w_item[levels, users] * weight[:, None])
        return hidden_users, hidden_items

    def encode(self, training):
        hidden_users, hidden_items = self._propagate()
        hidden_users = F.dropout(F.relu(hidden_users), self.dropout, training)
        hidden_items = F.dropout(F.relu(hidden_items), self.dropout, training)
        return self.dense_user(hidden_users), self.dense_item(hidden_items)

    def logits(self, users_repr, items_repr, user_ids, item_ids):
        # Bilinear decoder per rating level, built from shared basis matrices:
        # logit_l = sum_b a_lb * (u P_b) . i, so only num_basis products are formed.
        left = users_repr[user_ids]
        right = items_repr[item_ids]
        per_basis = torch.einsum("ni,bij,nj->nb", left, self.decoder_basis, right)
        return per_basis @ self.decoder_coeff.T

    def forward(self, user_ids, item_ids):
        if self.cached is None:
            with torch.no_grad():
                self.cached = self.encode(False)
        out = self.logits(*self.cached, user_ids, item_ids)
        probabilities = out.softmax(dim=1)
        return (probabilities * torch.arange(1, self.num_levels + 1, device=out.device)).sum(dim=1)


def train_gcmc(train_frame, validation_frame, num_users, num_items, *, max_epochs=300,
               learning_rate=0.05, weight_decay=0.0, patience=10,
               quiet=True, **kwargs):
    model = GCMC(num_users, num_items, **kwargs).to(DEVICE)
    model.set_graph(train_frame)
    users = _tensor(train_frame, "user_index", torch.long)
    items = _tensor(train_frame, "item_index", torch.long)
    targets = _tensor(train_frame, "rating", torch.float32).round().long() - 1
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    best_rmse, best_state, stale = float("inf"), None, 0
    for epoch in range(max_epochs):
        tic = time.time()
        model.train()
        # Full-graph encoder pass, decoder loss over the whole edge set.
        optimizer.zero_grad()
        user_repr, item_repr = model.encode(True)
        loss = F.cross_entropy(model.logits(user_repr, item_repr, users, items), targets)
        loss.backward()
        optimizer.step()
        model.cached = None
        model.eval()
        rmse = _validation_rmse(model, validation_frame)
        if not quiet:
            print(f"GC-MC epoch {epoch + 1}: loss {loss.item():.4f} val rmse {rmse:.4f} ({time.time() - tic:.1f}s)")
        if rmse < best_rmse - 1e-5:
            best_rmse, stale = rmse, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state, strict=False)
    model.eval()
    model.cached = None
    return model


TRAINERS = {"glocal_k": train_glocal_k, "gcmc": train_gcmc}


def train_alt_model(name, train_frame, validation_frame, num_users, num_items, quiet=True, **kwargs):
    return TRAINERS[name](train_frame, validation_frame, num_users, num_items, quiet=quiet, **kwargs)
