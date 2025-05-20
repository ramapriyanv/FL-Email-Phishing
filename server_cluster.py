#!/usr/bin/env python3
import sys
import pandas as pd
import torch
import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server import ServerConfig
from model import build_model

# CONFIG
TEXT_DIM   = 200
NUM_ROUNDS = 2

def get_initial_parameters():
    model = build_model(text_input_dim=TEXT_DIM, gnn_in_channels=1)
    return ndarrays_to_parameters([w.cpu().numpy() for w in model.state_dict().values()])

class SaveFedAvg(FedAvg):
    def aggregate_fit(self, rnd, results, failures):
        params, metrics = super().aggregate_fit(rnd, results, failures)
        # On last round, save cluster model
        if params is not None and rnd == NUM_ROUNDS:
            nds = parameters_to_ndarrays(params)
            model = build_model(text_input_dim=TEXT_DIM, gnn_in_channels=1)
            sd = {k: torch.tensor(v) for k, v in zip(model.state_dict().keys(), nds)}
            model.load_state_dict(sd)
            out = f"cluster_{cluster_id}_model.pth"
            torch.save(model.state_dict(), out)
            print(f"[Server] Saved → {out}")
        return params, metrics

if __name__ == "__main__":
    # 1) Which cluster are we training?
    cluster_id = int(sys.argv[1])  # pass 0, 1, or 2

    # 2) Load cluster sizes to set min_fit_clients
    df = pd.read_csv("client_clusters.csv")
    clients = df[df.cluster == cluster_id].cid.tolist()
    n_clients = len(clients)

    # 3) Start Flower server
    strategy = SaveFedAvg(
        initial_parameters=get_initial_parameters(),
        fraction_fit=1.0,
        min_fit_clients=n_clients,
        min_available_clients=n_clients,
    )
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )
