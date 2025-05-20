#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class GNNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, embedding_dim):
        super().__init__()
        self.conv1   = GCNConv(in_channels, hidden_channels)
        self.conv2   = GCNConv(hidden_channels, embedding_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return global_mean_pool(x, batch)

class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z   = self.encoder(x)
        x̂   = self.decoder(z)
        return x̂, z

class CombinedModel(nn.Module):
    def __init__(self, text_input_dim, gnn_in_channels):
        super().__init__()
        # drastically smaller
        ae_hid, ae_emb = 8, 4
        gnn_hid, gnn_emb = 8, 4

        self.text_ae    = Autoencoder(text_input_dim, ae_hid, ae_emb)
        self.gnn        = GNNEncoder(gnn_in_channels, gnn_hid, gnn_emb)
        self.classifier = nn.Linear(ae_emb + gnn_emb, 1)
        self.dropout    = nn.Dropout(0.5)

    def forward(self, text_x, node_x, edge_index, batch):
        x̂, z_text = self.text_ae(text_x)
        z_graph    = self.gnn(node_x, edge_index, batch)
        z          = torch.cat([z_text, z_graph], dim=1)
        z          = self.dropout(z)
        logits     = self.classifier(z)
        return x̂, logits, z

def build_model(text_input_dim: int, gnn_in_channels: int) -> nn.Module:
    return CombinedModel(text_input_dim, gnn_in_channels)
