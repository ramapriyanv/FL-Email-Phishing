#!/usr/bin/env python3
import pandas as pd
from sklearn.cluster import KMeans

# 1) Compute phishing‐rate per client
rows = []
for cid in range(1, 11):
    df = pd.read_csv(f"clients/client_{cid}.csv")
    rate = df["label"].mean()  # fraction of phishing
    rows.append({"cid": cid, "phishing_ratio": rate})

stats = pd.DataFrame(rows)

# 2) K-Means with k=3
kmeans = KMeans(n_clusters=3, random_state=0)
stats["cluster"] = kmeans.fit_predict(stats[["phishing_ratio"]])

# 3) Save
stats.to_csv("client_clusters.csv", index=False)
print(stats)
