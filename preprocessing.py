"""
preprocessing.py
================
Canary — Decoy-Based Data Breach Detection System
Layer 0: Data Loading and Preprocessing for PaySim dataset

Dataset : PaySim — Synthetic Financial Dataset for Fraud Detection
Source  : Lopez-Rojas, Elmir & Axelsson (EMSS 2016)
Kaggle  : https://www.kaggle.com/datasets/ealaxi/paysim1

PaySim columns:
    step            - hour of simulation (1-744)
    type            - transaction type (CASH-IN, CASH-OUT, DEBIT, PAYMENT, TRANSFER)
    amount          - transaction amount
    nameOrig        - sender ID (dropped - identifier, not a feature)
    oldbalanceOrg   - sender balance before transaction
    newbalanceOrig  - sender balance after transaction
    nameDest        - receiver ID (dropped - identifier)
    oldbalanceDest  - receiver balance before transaction
    newbalanceDest  - receiver balance after transaction
    isFraud         - ground truth label
    isFlaggedFraud  - naive system flag (dropped - leakage)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import pickle


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
RANDOM_SEED   = 42
TEST_SIZE     = 0.2
VAL_SIZE      = 0.1          # fraction of training set

TYPE_COLUMNS  = ['type_CASH_IN', 'type_CASH_OUT', 'type_DEBIT',
                 'type_PAYMENT', 'type_TRANSFER']

NUMERIC_COLS  = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
                 'oldbalanceDest', 'newbalanceDest']

DROP_COLS     = ['nameOrig', 'nameDest', 'isFlaggedFraud']


# ─────────────────────────────────────────────
# Feature Engineering helpers
# ─────────────────────────────────────────────
def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived numeric features that carry semantic meaning.
    All derived from raw columns — no leakage.
    """
    df = df.copy()

    # Balance error features: captures zero-out fraud pattern
    # In PaySim, fraudulent TRANSFER/CASH-OUT often zero out sender balance
    df['orig_balance_diff'] = df['newbalanceOrig'] - df['oldbalanceOrg'] + df['amount']
    df['dest_balance_diff'] = df['newbalanceDest'] - df['oldbalanceDest'] - df['amount']

    # Ratio features: relative transaction size
    df['amount_to_orig_ratio'] = df['amount'] / (df['oldbalanceOrg'] + 1e-9)
    df['amount_to_dest_ratio'] = df['amount'] / (df['oldbalanceDest'] + 1e-9)

    # Flag: sender completely emptied their account
    df['orig_zeroed'] = (df['newbalanceOrig'] == 0).astype(int)

    # Flag: only TRANSFER and CASH-OUT contain fraud in PaySim
    df['is_risky_type'] = df['type'].isin(['TRANSFER', 'CASH_OUT']).astype(int)

    return df


def load_and_preprocess(filepath: str,
                        sample_frac: float = 1.0,
                        save_scaler: bool = True,
                        scaler_path: str = 'models/scaler.pkl') -> dict:
    """
    Full preprocessing pipeline for PaySim.

    Parameters
    ----------
    filepath     : path to PS_20174392719_1491204439457_log.csv
    sample_frac  : fraction of dataset to use (use <1.0 for quick testing)
    save_scaler  : whether to persist the fitted scaler
    scaler_path  : where to save the scaler

    Returns
    -------
    dict with keys:
        X_train, X_val, X_test   : scaled feature arrays (numpy)
        y_train, y_val, y_test   : label arrays (numpy)
        feature_names            : list of feature column names
        scaler                   : fitted StandardScaler
        df_train_raw             : unscaled training DataFrame (for injection layer)
    """
    print("=" * 60)
    print("CANARY — Preprocessing Pipeline")
    print("=" * 60)

    # ── 1. Load ──────────────────────────────────────────────
    print(f"\n[1/6] Loading dataset from: {filepath}")
    df = pd.read_csv(filepath)
    print(f"      Raw shape : {df.shape}")
    print(f"      Fraud rate: {df['isFraud'].mean()*100:.4f}%")

    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=RANDOM_SEED).reset_index(drop=True)
        print(f"      Sampled   : {df.shape} ({sample_frac*100:.0f}%)")

    # ── 2. Drop identifiers & leaky columns ──────────────────
    print("\n[2/6] Dropping identifier and leaky columns...")
    df.drop(columns=DROP_COLS, inplace=True)

    # ── 3. Feature engineering ───────────────────────────────
    print("[3/6] Engineering features...")
    df = _engineer_features(df)

    # ── 4. One-hot encode transaction type ───────────────────
    print("[4/6] Encoding transaction type...")
    # rename spaces to underscores for consistent column names
    df['type'] = df['type'].str.replace('-', '_')
    type_dummies = pd.get_dummies(df['type'], prefix='type')
    df = pd.concat([df.drop(columns=['type']), type_dummies], axis=1)

    # ensure all 5 type columns exist even if a type is absent in sample
    for col in TYPE_COLUMNS:
        if col not in df.columns:
            df[col] = 0

    # ── 5. Train / Val / Test split ──────────────────────────
    print("[5/6] Splitting into train / val / test sets...")
    label_col   = 'isFraud'
    feature_cols = [c for c in df.columns if c != label_col]

    X = df[feature_cols]
    y = df[label_col].values

    # stratify to preserve fraud ratio in each split
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y)

    val_relative = VAL_SIZE / (1 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_relative, random_state=RANDOM_SEED, stratify=y_trainval)

    print(f"      Train : {X_train.shape}  | Fraud: {y_train.mean()*100:.3f}%")
    print(f"      Val   : {X_val.shape}    | Fraud: {y_val.mean()*100:.3f}%")
    print(f"      Test  : {X_test.shape}   | Fraud: {y_test.mean()*100:.3f}%")

    # keep unscaled training df for injection layer (needs interpretable values)
    df_train_raw = X_train.copy()
    df_train_raw['isFraud'] = y_train

    # ── 6. Normalise ─────────────────────────────────────────
    print("[6/6] Fitting StandardScaler on training set...")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)

    if save_scaler:
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"      Scaler saved → {scaler_path}")

    print("\n✓ Preprocessing complete.")
    print("=" * 60)

    return {
        'X_train'      : X_train_s,
        'X_val'        : X_val_s,
        'X_test'       : X_test_s,
        'y_train'      : y_train,
        'y_val'        : y_val,
        'y_test'       : y_test,
        'feature_names': list(feature_cols),
        'scaler'       : scaler,
        'df_train_raw' : df_train_raw,
    }


def print_class_distribution(y: np.ndarray, label: str = "Set"):
    n_fraud     = y.sum()
    n_legit     = len(y) - n_fraud
    fraud_pct   = n_fraud / len(y) * 100
    print(f"{label}: {len(y)} records | Legit: {n_legit} | Fraud: {n_fraud} ({fraud_pct:.4f}%)")


# ─────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────
if __name__ == '__main__':
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else 'data/paysim.csv'
    data = load_and_preprocess(path, sample_frac=0.1)
    print(f"\nFeature count : {len(data['feature_names'])}")
    print(f"Feature names : {data['feature_names']}")
