#!/usr/bin/env python3
import flwr as fl
import torch
from flwr.server.strategy import FedAvg
from flwr.server import ServerConfig
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from model import build_model

# ——— CONFIG ———
TEXT_DIM    = 200     # cut TF-IDF to 200 features
NUM_ROUNDS  = 2
OUT_MODEL   = "gnn_ae_global.pth"

def get_initial_parameters():
    model = build_model(text_input_dim=TEXT_DIM, gnn_in_channels=1)
    return ndarrays_to_parameters([w.cpu().numpy() for w in model.state_dict().values()])

class SaveFedAvg(FedAvg):
    def aggregate_fit(self, rnd, results, failures):
        params, metrics = super().aggregate_fit(rnd, results, failures)
        if params is not None and rnd == NUM_ROUNDS:
            nds = parameters_to_ndarrays(params)
            model = build_model(TEXT_DIM, 1)
            sd = {k: torch.tensor(v) for k, v in zip(model.state_dict().keys(), nds)}
            model.load_state_dict(sd)
            torch.save(model.state_dict(), OUT_MODEL)
            print(f"[Server] Saved global model → {OUT_MODEL}")
        return params, metrics

def main():
    strategy = SaveFedAvg(
        initial_parameters=get_initial_parameters(),
        fraction_fit=1.0, min_fit_clients=10, min_available_clients=10
    )
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()
