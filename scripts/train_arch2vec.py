#!/usr/bin/env python3
"""Train arch2vec GIN VAE on NAS-Bench-101 and extract embeddings.

Self-contained script that replicates the arch2vec pretraining pipeline
(Yan et al., NeurIPS 2020) without requiring the nasbench TF API at runtime.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/train_arch2vec.py

Produces:
    weights/arch2vec/arch2vec-nasbench101.pt
    data/nasbench/nasbench101_accuracy.json
"""

import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

# ---------------------------------------------------------------------------
# arch2vec model components: inlined from github.com/MSU-MLSys-Lab/arch2vec (Apache-2.0) and MODIFIED; see THIRD_PARTY_NOTICES.md
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """MLP with linear output (from gin/models/mlp.py)."""

    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.linear_or_not = True
        self.num_layers = num_layers
        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            self.linear_or_not = False
            self.linears = nn.ModuleList()
            self.batch_norms = nn.ModuleList()
            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for _ in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))
            for _ in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, x):
        if self.linear_or_not:
            return self.linear(x)
        h = x
        for layer in range(self.num_layers - 1):
            h = F.relu(self.batch_norms[layer](self.linears[layer](h)))
        return self.linears[self.num_layers - 1](h)


class Decoder(nn.Module):
    """arch2vec decoder."""

    def __init__(self, embedding_dim, input_dim, dropout,
                 activation_adj=torch.sigmoid, activation_ops=torch.sigmoid,
                 adj_hidden_dim=None, ops_hidden_dim=None):
        super().__init__()
        self.activation_adj = activation_adj
        self.activation_ops = activation_ops
        self.weight = nn.Linear(embedding_dim, input_dim)
        self.dropout = dropout

    def forward(self, embedding):
        embedding = F.dropout(embedding, p=self.dropout, training=self.training)
        ops = self.weight(embedding)
        adj = torch.matmul(embedding, embedding.permute(0, 2, 1))
        return self.activation_adj(ops), self.activation_adj(adj)


class Arch2VecVAE(nn.Module):
    """GIN VAE model for architecture encoding (from models/model.py: Model)."""

    def __init__(self, input_dim=5, hidden_dim=128, latent_dim=16,
                 num_hops=5, num_mlp_layers=2, dropout=0.3, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_hops
        self.eps = nn.Parameter(torch.zeros(self.num_layers - 1))
        self.mlps = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for layer in range(self.num_layers - 1):
            if layer == 0:
                self.mlps.append(MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim))
            else:
                self.mlps.append(MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        self.fc1 = nn.Linear(self.hidden_dim, self.latent_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.latent_dim)
        self.decoder = Decoder(self.latent_dim, self.input_dim, dropout, **kwargs)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def _encoder(self, ops, adj):
        batch_size, node_num, opt_num = ops.shape
        x = ops
        for l in range(self.num_layers - 1):
            neighbor = torch.matmul(adj.float(), x)
            agg = ((1 + self.eps[l]) * x.view(batch_size * node_num, -1)
                   + neighbor.view(batch_size * node_num, -1))
            x = F.relu(self.batch_norms[l](self.mlps[l](agg)).view(
                batch_size, node_num, -1))
        mu = self.fc1(x)
        logvar = self.fc2(x)
        return mu, logvar

    def forward(self, ops, adj):
        mu, logvar = self._encoder(ops, adj)
        z = self.reparameterize(mu, logvar)
        ops_recon, adj_recon = self.decoder(z)
        return ops_recon, adj_recon, mu, logvar


# ---------------------------------------------------------------------------
# Loss and preprocessing (from utils/utils.py and models/model.py)
# ---------------------------------------------------------------------------

class VAEReconstructedLoss:
    def __init__(self, w_ops=1.0, w_adj=1.0, loss_ops=None, loss_adj=None):
        self.w_ops = w_ops
        self.w_adj = w_adj
        self.loss_ops = loss_ops
        self.loss_adj = loss_adj

    def __call__(self, inputs, targets, mu, logvar):
        ops_recon, adj_recon = inputs
        ops, adj = targets
        loss_ops = self.loss_ops(ops_recon, ops)
        loss_adj = self.loss_adj(adj_recon, adj)
        loss = self.w_ops * loss_ops + self.w_adj * loss_adj
        KLD = (-0.5 / (ops.shape[0] * ops.shape[1])
               * torch.mean(torch.sum(
                   1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 2)))
        return loss + KLD


def preprocessing_method4(A, H):
    """Preprocessing method 4: symmetrize adjacency."""
    A = A + A.triu(1).transpose(-1, -2)

    def prep_reverse(A, H):
        return A.triu(1), H

    return A, H, prep_reverse


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def build_dataset(dataset, indices):
    X_adj = []
    X_ops = []
    for ind in indices:
        X_adj.append(torch.Tensor(dataset[str(ind)]['module_adjacency']))
        X_ops.append(torch.Tensor(dataset[str(ind)]['module_operations']))
    return torch.stack(X_adj), torch.stack(X_ops), torch.Tensor(indices)


def train(data_path, epochs=8, bs=32, latent_dim=16, seed=1, device='cuda'):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    print(f'Loading dataset from {data_path}...')
    with open(data_path, 'r') as f:
        dataset = json.load(f)
    n = len(dataset)
    print(f'Dataset size: {n}')

    # Config 4 from arch2vec
    cfg_gae = {
        'activation_adj': torch.sigmoid,
        'activation_ops': torch.softmax,
        'adj_hidden_dim': 128,
        'ops_hidden_dim': 128,
    }
    cfg_loss = {
        'loss_ops': nn.BCELoss(),
        'loss_adj': nn.BCELoss(),
    }

    train_ind = list(range(int(n * 0.9)))
    val_ind = list(range(int(n * 0.9), n))

    X_adj_train, X_ops_train, indices_train = build_dataset(dataset, train_ind)
    X_adj_val, X_ops_val, indices_val = build_dataset(dataset, val_ind)

    model = Arch2VecVAE(
        input_dim=5, hidden_dim=128, latent_dim=latent_dim,
        num_hops=5, num_mlp_layers=2, dropout=0.3, **cfg_gae
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08)
    loss_fn = VAEReconstructedLoss(**cfg_loss)

    for epoch in range(epochs):
        model.train()
        chunks = len(train_ind) // bs + (1 if len(train_ind) % bs > 0 else 0)
        X_adj_split = torch.split(X_adj_train, bs, dim=0)
        X_ops_split = torch.split(X_ops_train, bs, dim=0)
        loss_epoch = []

        for i, (adj, ops) in enumerate(zip(X_adj_split, X_ops_split)):
            optimizer.zero_grad()
            adj, ops = adj.to(device), ops.to(device)
            adj, ops, prep_reverse = preprocessing_method4(adj, ops)
            ops_recon, adj_recon, mu, logvar = model(ops, adj.to(torch.long))
            adj_recon, ops_recon = prep_reverse(adj_recon, ops_recon)
            adj, ops = prep_reverse(adj, ops)
            loss = loss_fn((ops_recon, adj_recon), (ops, adj), mu, logvar)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            loss_epoch.append(loss.item())
            if i % 2000 == 0:
                print(f'  epoch {epoch}: batch {i}/{chunks}: loss: {loss.item():.5f}')

        avg_loss = sum(loss_epoch) / len(loss_epoch)
        print(f'epoch {epoch}: average loss {avg_loss:.5f}')

    # Save model checkpoint
    checkpoint_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                   'weights', 'arch2vec')
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, 'model-nasbench-101.pt')
    torch.save({
        'epoch': epochs - 1,
        'loss': avg_loss,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
    }, checkpoint_path)
    print(f'Saved model checkpoint to {checkpoint_path}')

    return model, dataset


def extract_embeddings(model, dataset, device='cuda'):
    """Extract arch2vec embeddings for all architectures."""
    model.eval()
    embedding = {}
    n = len(dataset)

    print(f'Extracting embeddings for {n} architectures...')
    with torch.no_grad():
        for ind in range(n):
            adj = torch.Tensor(dataset[str(ind)]['module_adjacency']).unsqueeze(0).to(device)
            ops = torch.Tensor(dataset[str(ind)]['module_operations']).unsqueeze(0).to(device)
            adj, ops, _ = preprocessing_method4(adj, ops)
            # Get mu (mean) from encoder -- this is the "feature" vector
            mu, _ = model._encoder(ops, adj)
            # Mean pool over nodes to get a single 16D vector per architecture
            feature = mu.squeeze(0).mean(dim=0).cpu()

            embedding[ind] = {
                'feature': feature,
                'valid_accuracy': float(dataset[str(ind)]['validation_accuracy']),
                'test_accuracy': float(dataset[str(ind)]['test_accuracy']),
                'time': float(dataset[str(ind)]['training_time']),
            }

            if ind % 50000 == 0:
                print(f'  Extracted {ind}/{n}')

    return embedding


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(project_root, 'data', 'nasbench', 'data.json')
    weights_dir = os.path.join(project_root, 'weights', 'arch2vec')
    data_dir = os.path.join(project_root, 'data', 'nasbench')

    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    # Train
    model, dataset = train(data_path, epochs=8, bs=32, latent_dim=16,
                           seed=1, device=device)

    # Extract embeddings
    embeddings = extract_embeddings(model, dataset, device=device)

    # Save in arch2vec format: {index: {'feature': tensor_16d, 'valid_accuracy': float, ...}}
    emb_path = os.path.join(weights_dir, 'arch2vec-nasbench101.pt')
    torch.save(embeddings, emb_path)
    print(f'Saved embeddings to {emb_path}')
    print(f'  Total architectures: {len(embeddings)}')
    print(f'  Sample embedding shape: {embeddings[0]["feature"].shape}')

    # Also create accuracy lookup JSON
    accuracy = {}
    for idx, info in embeddings.items():
        accuracy[str(idx)] = info['valid_accuracy']

    acc_path = os.path.join(data_dir, 'nasbench101_accuracy.json')
    with open(acc_path, 'w') as f:
        json.dump(accuracy, f)
    print(f'Saved accuracy lookup to {acc_path} ({len(accuracy)} entries)')


if __name__ == '__main__':
    main()
