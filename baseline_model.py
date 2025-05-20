#!/usr/bin/env python3
import os
import pandas as pd
import joblib
from glob import glob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

# ——— CONFIG ———
CLIENT_DIR      = 'clients'                  # folder with client_1.csv … client_10.csv
TEST_CSV        = 'test.csv'
MODEL_PATH      = 'baseline_lr_tuned.pkl'
VECTORIZER_PATH = 'tfidf_vectorizer_tuned.pkl'

def load_train_data():
    """Concatenate all client_i.csv into one DataFrame."""
    paths = glob(os.path.join(CLIENT_DIR, 'client_*.csv'))
    dfs = [pd.read_csv(p) for p in paths]
    return pd.concat(dfs, ignore_index=True)

def preprocess(df):
    """Create a single text field from subject + body, fill NaNs."""
    df = df.copy()
    df['subject'] = df['subject'].fillna('')
    df['body']    = df['body'].fillna('')
    df['text']    = df['subject'] + ' ' + df['body']
    return df

def main():
    # 1) Load data
    train_df = load_train_data()
    test_df  = pd.read_csv(TEST_CSV)

    # 2) Preprocess
    train_df = preprocess(train_df)
    test_df  = preprocess(test_df)

    # 3) Vectorize with aggressive pruning + stop words
    vect = TfidfVectorizer(
        max_features=500,
        stop_words='english',
        min_df=5
    )
    vect.fit(train_df['text'])
    
    # 4) Subsample 50% of the training set
    train_sub = train_df.sample(frac=0.5, random_state=42)
    X_train   = vect.transform(train_sub['text'])
    y_train   = train_sub['label']
    
    X_test = vect.transform(test_df['text'])
    y_test = test_df['label']

    # 5) Train with strong regularization
    clf = LogisticRegression(max_iter=1000, C=0.01)
    clf.fit(X_train, y_train)

    # 6) Evaluate
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:,1]
    acc     = accuracy_score(y_test, y_pred)
    prec    = precision_score(y_test, y_pred)
    rec     = recall_score(y_test, y_pred)
    f1      = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    print("\nTuned Baseline Logistic Regression Results:")
    print(f"  Accuracy : {acc*100:5.2f}%")
    print(f"  Precision: {prec*100:5.2f}%")
    print(f"  Recall   : {rec*100:5.2f}%")
    print(f"  F1-score : {f1*100:5.2f}%")
    print(f"  ROC-AUC  : {roc_auc*100:5.2f}%\n")

    # 7) Save model & vectorizer
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(vect, VECTORIZER_PATH)
    print(f"Saved tuned model → {MODEL_PATH}")
    print(f"Saved tuned vectorizer → {VECTORIZER_PATH}")

if __name__ == '__main__':
    main()