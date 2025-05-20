#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ——— CONFIG ———
INPUT_CSV   = 'CEAS_08.csv'  # your downloaded file
CLIENT_DIR  = 'clients'      # output folder
TEST_CSV    = 'test.csv'
NUM_CLIENTS = 10
TEST_SIZE   = 0.1            # 10% held out for final evaluation
RANDOM_SEED = 42

def main():
    # 1) load & stratify split
    df = pd.read_csv(INPUT_CSV)
    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        stratify=df['label'],
        random_state=RANDOM_SEED
    )

    # 2) shuffle & split train into N shards
    shards = np.array_split(
        train_df.sample(frac=1, random_state=RANDOM_SEED),
        NUM_CLIENTS
    )

    # 3) write out client files
    os.makedirs(CLIENT_DIR, exist_ok=True)
    for i, shard in enumerate(shards, start=1):
        path = os.path.join(CLIENT_DIR, f'client_{i}.csv')
        shard.to_csv(path, index=False)
        print(f"Wrote {len(shard)} records → {path}")

    # 4) write the test set
    test_df.to_csv(TEST_CSV, index=False)
    print(f"Wrote test set ({len(test_df)} records) → {TEST_CSV}")

if __name__ == '__main__':
    main()
