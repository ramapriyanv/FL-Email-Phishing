#!/usr/bin/env python3
import pandas as pd
import torch
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from model import build_model

# CONFIG
TEXT_DIM    = 200
CHECKPOINT  = "clustered_global_model.pth"
TEST_CSV    = "test.csv"
THRESHOLD   = 0.55

# Text extractor
VECT = HashingVectorizer(n_features=TEXT_DIM, alternate_sign=False, norm="l2")

# Load model
model = build_model(text_input_dim=TEXT_DIM, gnn_in_channels=1)
model.load_state_dict(torch.load(CHECKPOINT))
model.eval()

# Load test data
df = pd.read_csv(TEST_CSV)
texts = (df["subject"].fillna("") + " " + df["body"].fillna("")).tolist()
X_txt = VECT.transform(texts).toarray()
y_true = df["label"].values

# Inference
y_prob, y_pred = [], []
for vec in X_txt:
    text_x     = torch.tensor(vec, dtype=torch.float).unsqueeze(0)
    node_x     = torch.ones((2,1), dtype=torch.float)
    edge_index = torch.tensor([[0,1],[1,0]], dtype=torch.long)
    batch_idx  = torch.zeros(2, dtype=torch.long)
    with torch.no_grad():
        _, logits, _ = model(text_x, node_x, edge_index, batch_idx)
        prob = torch.sigmoid(logits).item()
    y_prob.append(prob)
    y_pred.append(int(prob > THRESHOLD))

# Metrics
acc    = accuracy_score(y_true, y_pred)
prec   = precision_score(y_true, y_pred)
rec    = recall_score(y_true, y_pred)
f1     = f1_score(y_true, y_pred)
roc    = roc_auc_score(y_true, y_prob)

print("Final Clustered Aggregation Evaluation")
print(f"  Accuracy : {acc*100:.2f}%")
print(f"  Precision: {prec*100:.2f}%")
print(f"  Recall   : {rec*100:.2f}%")
print(f"  F1-score : {f1*100:.2f}%")
print(f"  ROC-AUC  : {roc*100:.2f}%")
