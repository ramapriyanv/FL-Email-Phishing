#!/usr/bin/env python3
import pandas as pd
import torch
from model import build_model

# CONFIG
TEXT_DIM = 200

# 1) Load client→cluster map
df = pd.read_csv("client_clusters.csv")
counts = df.cluster.value_counts().to_dict()
total  = len(df)

# 2) Load each cluster’s checkpoint
models = {}
for cid in counts:
    models[cid] = torch.load(f"cluster_{cid}_model.pth")

# 3) Weighted average by cluster size
avg_state = {}
keys = models[0].keys()
for k in keys:
    # sum(cluster_i * (n_i/total))
    avg_state[k] = sum(
        models[c][k] * (counts[c] / total) for c in counts
    )

# 4) Save final “clustered global” model
torch.save(avg_state, "clustered_global_model.pth")
print("Saved → clustered_global_model.pth")
