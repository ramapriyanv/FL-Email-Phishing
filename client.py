#!/usr/bin/env python3
import argparse
import pandas as pd
import torch
import torch.nn as nn
import flwr as fl
from torch.optim import Adam
from torch_geometric.data import Data
from sklearn.feature_extraction.text import HashingVectorizer
from model import build_model

# ——— CONFIG ———
TEXT_DIM     = 200   # must match server
LOCAL_EPOCHS = 1
LR            = 1e-3

VECT = HashingVectorizer(n_features=TEXT_DIM, alternate_sign=False, norm="l2")

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, samples, device):
        self.model     = model.to(device)
        self.samples   = samples
        self.device    = device
        self.crit      = nn.BCEWithLogitsLoss()

    def get_parameters(self):
        return [w.cpu().numpy() for w in self.model.state_dict().values()]

    def set_parameters(self, params):
        sd = {k: torch.tensor(v) for k, v in zip(self.model.state_dict().keys(), params)}
        self.model.load_state_dict(sd, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        opt = Adam(self.model.parameters(), lr=LR)
        for _ in range(LOCAL_EPOCHS):
            for text_x, node_x, edge_index, y in self.samples:
                text_x     = text_x.to(self.device).unsqueeze(0)  # [1,200]
                node_x     = node_x.to(self.device)               # [2,1]
                edge_index = edge_index.to(self.device)           # [2,2]
                y          = y.to(self.device)                    # [1]
                batch_idx  = torch.zeros(node_x.size(0), dtype=torch.long, device=self.device)

                opt.zero_grad()
                _, logits, _ = self.model(text_x, node_x, edge_index, batch_idx)
                loss = self.crit(logits.view(-1), y)
                loss.backward()
                opt.step()

        return self.get_parameters(), len(self.samples), {}

    def evaluate(self, parameters, config):
        return 0.0, len(self.samples), {}

def load_samples(cid: int):
    df = pd.read_csv(f"clients/client_{cid}.csv")
    df["subject"] = df["subject"].fillna("").astype(str)
    df["body"]    = df["body"].fillna("").astype(str)

    samples = []
    for _, row in df.iterrows():
        txt = row["subject"] + " " + row["body"]
        vec = VECT.transform([txt]).toarray()[0]
        text_x = torch.tensor(vec, dtype=torch.float)
        node_x     = torch.ones((2,1), dtype=torch.float)
        edge_index = torch.tensor([[0,1],[1,0]], dtype=torch.long)
        y          = torch.tensor([row["label"]], dtype=torch.float)
        samples.append((text_x, node_x, edge_index, y))
    return samples

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cid", type=int, required=True)
    cid = parser.parse_args().cid

    model   = build_model(text_input_dim=TEXT_DIM, gnn_in_channels=1)
    samples = load_samples(cid)
    client  = FlowerClient(model, samples, torch.device("cpu")).to_client()
    fl.client.start_client(server_address="localhost:8080", client=client)

if __name__ == "__main__":
    main()
