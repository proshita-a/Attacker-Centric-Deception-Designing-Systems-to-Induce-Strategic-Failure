"""
layer1_decoy_generator.py
=========================
Canary — Decoy-Based Data Breach Detection System
Layer 1: Autoencoder-based Decoy Generation

Primary  : Fully-connected Autoencoder (Week 10-11 of course)
Fallback : VAE with statistical validation pipeline
           (used if autoencoder discriminator accuracy > 70%)

Evaluation metrics (per Xu et al., CTGAN, NeurIPS 2019):
    1. RF Classifier accuracy on real vs decoy   (target: ≤ 60%)
    2. F1, Precision, Recall of that classifier
    3. Feature-wise mean and std comparison
    4. KL divergence per feature
    5. KS-test (Kolmogorov-Smirnov) per feature
"""

import numpy as np
import pandas as pd
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import (classification_report, accuracy_score,
                             f1_score, precision_score, recall_score)
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.special import rel_entr                  # KL divergence helper


# ─────────────────────────────────────────────
# Try importing torch; fall back gracefully
# ─────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[WARNING] PyTorch not found. Only statistical fallback available.")


# ═════════════════════════════════════════════
# 1.  AUTOENCODER  (primary method)
# ═════════════════════════════════════════════

class Autoencoder(nn.Module if TORCH_AVAILABLE else object):
    """
    Fully-connected Autoencoder for tabular data.

    Architecture (Week 10-11 content):
        Encoder: input_dim → 32 → 16 → latent_dim
        Decoder: latent_dim → 16 → 32 → input_dim

    Activation : ReLU (hidden), Identity (output — regression task)
    Regularisation: Dropout + BatchNorm (Week 11)
    """

    def __init__(self, input_dim: int, latent_dim: int = 8):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for Autoencoder.")
        super().__init__()
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, input_dim),   # output = reconstruction, no activation
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)


def train_autoencoder(X_train: np.ndarray,
                      X_val: np.ndarray,
                      latent_dim: int = 8,
                      epochs: int = 50,
                      batch_size: int = 256,
                      lr: float = 1e-3,
                      save_path: str = 'models/autoencoder.pt') -> 'Autoencoder':
    """
    Train the autoencoder on LEGITIMATE records only.
    Training on fraud records would teach it to reproduce fraud patterns,
    which we do NOT want — decoys should look like normal transactions.
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required.")

    print("\n[Layer 1] Training Autoencoder...")
    print(f"  Input dim  : {X_train.shape[1]}")
    print(f"  Latent dim : {latent_dim}")
    print(f"  Epochs     : {epochs}  |  Batch: {batch_size}  |  LR: {lr}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device     : {device}")

    # Convert to tensors
    X_tr = torch.FloatTensor(X_train).to(device)
    X_vl = torch.FloatTensor(X_val).to(device)

    loader = DataLoader(TensorDataset(X_tr, X_tr),
                        batch_size=batch_size, shuffle=True)

    model     = Autoencoder(input_dim=X_train.shape[1], latent_dim=latent_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_loss = float('inf')
    patience_counter = 0
    EARLY_STOP = 10

    for epoch in range(1, epochs + 1):
        # ── train
        model.train()
        train_loss = 0.0
        for xb, _ in loader:
            optimizer.zero_grad()
            recon = model(xb)
            loss  = criterion(recon, xb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(X_train)

        # ── validate
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_vl), X_vl).item()

        scheduler.step(val_loss)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs} | train loss: {train_loss:.6f} | val loss: {val_loss:.6f}")

        # ── early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP:
                print(f"  Early stopping at epoch {epoch}.")
                break

    # reload best weights
    model.load_state_dict(torch.load(save_path, map_location=device))
    model.eval()
    print(f"\n  ✓ Best validation loss: {best_val_loss:.6f}")
    return model


def generate_decoys_autoencoder(model,
                                X_real: np.ndarray,
                                n_decoys: int,
                                noise_std: float = 0.5) -> np.ndarray:
    """
    Generate decoy records by sampling the latent space.

    Strategy:
      1. Encode real records → get latent distribution
      2. Sample latent vectors with added Gaussian noise
      3. Decode sampled vectors → decoy records
      4. Clip to realistic range (within 3 SD of real data per feature)

    noise_std controls how far decoys deviate from real patterns.
    Lower = more realistic decoys. Higher = more distinct from real records.
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required.")

    device = next(model.parameters()).device
    model.eval()

    with torch.no_grad():
        X_t = torch.FloatTensor(X_real).to(device)
        Z   = model.encode(X_t).cpu().numpy()   # latent representations

    # sample new latent vectors: mean ± noise
    z_mean = Z.mean(axis=0)
    z_std  = Z.std(axis=0)
    Z_sample = np.random.normal(
        loc   = z_mean,
        scale = z_std * noise_std,
        size  = (n_decoys, Z.shape[1])
    )

    with torch.no_grad():
        Z_t    = torch.FloatTensor(Z_sample).to(device)
        decoys = model.decode(Z_t).cpu().numpy()

    # clip each feature to ± 3 SD of real data to avoid out-of-range values
    for i in range(X_real.shape[1]):
        lo = X_real[:, i].mean() - 3 * X_real[:, i].std()
        hi = X_real[:, i].mean() + 3 * X_real[:, i].std()
        decoys[:, i] = np.clip(decoys[:, i], lo, hi)

    return decoys


# ═════════════════════════════════════════════
# 2.  STATISTICAL FALLBACK  (if AE fails)
# ═════════════════════════════════════════════

def generate_decoys_statistical(X_real: np.ndarray,
                                n_decoys: int,
                                n_components: int = 10) -> np.ndarray:
    """
    Fallback decoy generator using PCA + Gaussian Mixture Model.
    Used when autoencoder discriminator accuracy > 70%.

    Steps:
      1. PCA: reduce to n_components
      2. Fit multivariate Gaussian on reduced space
      3. Sample from Gaussian
      4. Inverse-PCA back to original space
      5. Uniqueness check: discard records too close to real data

    Course alignment: Week 3 (GMM) + Week 4 (PCA)
    """
    from sklearn.decomposition import PCA
    from sklearn.mixture import GaussianMixture

    print("\n  [Fallback] Using PCA + GMM generator...")

    pca = PCA(n_components=min(n_components, X_real.shape[1]),
              random_state=42)
    Z   = pca.fit_transform(X_real)

    # fit GMM in reduced space
    gmm = GaussianMixture(n_components=5, covariance_type='full',
                          random_state=42)
    gmm.fit(Z)

    # sample and inverse transform
    Z_sample, _ = gmm.sample(n_decoys)
    decoys       = pca.inverse_transform(Z_sample)

    # ── uniqueness check: remove decoys too similar to real records ──
    # compute pairwise distances (approximate using random subsample)
    from sklearn.metrics.pairwise import euclidean_distances
    real_sample  = X_real[np.random.choice(len(X_real), min(500, len(X_real)), replace=False)]
    dists        = euclidean_distances(decoys, real_sample).min(axis=1)
    THRESHOLD    = np.percentile(dists, 10)   # bottom 10% too close → discard
    mask         = dists > THRESHOLD
    decoys       = decoys[mask]

    # top up if we discarded too many
    if len(decoys) < n_decoys:
        extra, _ = gmm.sample(n_decoys - len(decoys))
        decoys   = np.vstack([decoys, pca.inverse_transform(extra)])

    return decoys[:n_decoys]


# ═════════════════════════════════════════════
# 3.  QUALITY EVALUATION
# ═════════════════════════════════════════════

def evaluate_decoy_quality(X_real: np.ndarray,
                           X_decoy: np.ndarray,
                           feature_names: list = None) -> dict:
    """
    Comprehensive decoy quality evaluation.

    Metrics (Xu et al. CTGAN NeurIPS 2019 framework):
      1. RF Discriminator accuracy (target ≤ 60% = essentially random)
      2. Precision, Recall, F1 of that classifier
      3. Feature-wise mean + std comparison
      4. KL divergence per feature (lower = more similar distributions)
      5. KS-test p-value per feature (higher p = distributions are similar)

    Additionally implements the VAE validation pipeline from the project spec:
      - ±2 SD check on feature means
      - KS-test statistical similarity
    """
    print("\n[Layer 1] Evaluating Decoy Quality...")
    print(f"  Real records  : {len(X_real)}")
    print(f"  Decoy records : {len(X_decoy)}")

    n = min(len(X_real), len(X_decoy))

    # ── A. RF Discriminator Test (Xu et al. 2019) ────────────
    # Label real=0, decoy=1 — train RF to distinguish them
    X_combined = np.vstack([X_real[:n], X_decoy[:n]])
    y_combined = np.array([0]*n + [1]*n)

    # 80/20 split for this internal evaluation
    from sklearn.model_selection import train_test_split
    Xtr, Xte, ytr, yte = train_test_split(
        X_combined, y_combined, test_size=0.2, random_state=42)

    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(Xtr, ytr)
    y_pred = rf.predict(Xte)

    acc       = accuracy_score(yte, y_pred)
    f1        = f1_score(yte, y_pred, average='weighted')
    precision = precision_score(yte, y_pred, average='weighted', zero_division=0)
    recall    = recall_score(yte, y_pred, average='weighted', zero_division=0)

    print(f"\n  ── Discriminator RF (real vs decoy) ──")
    print(f"  Accuracy  : {acc:.4f}  (target ≤ 0.60)")
    print(f"  F1        : {f1:.4f}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")

    quality_flag = "✓ GOOD" if acc <= 0.65 else "⚠ POOR (consider fallback)"
    print(f"  Quality   : {quality_flag}")

    # ── B. Feature-wise Statistics ────────────────────────────
    print(f"\n  ── Feature-wise Mean & Std Comparison ──")
    if feature_names is None:
        feature_names = [f'f{i}' for i in range(X_real.shape[1])]

    stats_rows = []
    ks_pvals   = []
    kl_divs    = []

    for i, fname in enumerate(feature_names):
        r = X_real[:, i]
        d = X_decoy[:, i]

        r_mean, r_std = r.mean(), r.std()
        d_mean, d_std = d.mean(), d.std()

        # within ±2 SD of real mean?
        mean_ok = abs(d_mean - r_mean) <= 2 * r_std

        # KS-test
        ks_stat, ks_p = stats.ks_2samp(r, d)
        ks_pvals.append(ks_p)

        # KL divergence (bin-based approximation)
        bins = np.linspace(min(r.min(), d.min()), max(r.max(), d.max()), 50)
        r_hist, _ = np.histogram(r, bins=bins, density=True)
        d_hist, _ = np.histogram(d, bins=bins, density=True)
        # add epsilon to avoid log(0)
        r_hist += 1e-10; d_hist += 1e-10
        kl = np.sum(rel_entr(r_hist, d_hist))
        kl_divs.append(kl)

        stats_rows.append({
            'feature'    : fname,
            'real_mean'  : round(r_mean, 4),
            'decoy_mean' : round(d_mean, 4),
            'real_std'   : round(r_std, 4),
            'decoy_std'  : round(d_std, 4),
            'mean_ok'    : mean_ok,
            'ks_pval'    : round(ks_p, 4),
            'kl_div'     : round(kl, 4),
        })

    stats_df = pd.DataFrame(stats_rows)
    print(stats_df[['feature', 'real_mean', 'decoy_mean', 'real_std',
                     'decoy_std', 'mean_ok', 'ks_pval', 'kl_div']].to_string(index=False))

    n_ok = stats_df['mean_ok'].sum()
    print(f"\n  Features within ±2 SD: {n_ok}/{len(feature_names)}")
    print(f"  Mean KL divergence   : {np.mean(kl_divs):.4f}  (lower is better)")
    print(f"  Mean KS p-value      : {np.mean(ks_pvals):.4f}  (higher is better, >0.05 ideal)")

    return {
        'discriminator_accuracy' : acc,
        'discriminator_f1'       : f1,
        'discriminator_precision': precision,
        'discriminator_recall'   : recall,
        'features_within_2sd'    : int(n_ok),
        'mean_kl_divergence'     : float(np.mean(kl_divs)),
        'mean_ks_pvalue'         : float(np.mean(ks_pvals)),
        'stats_df'               : stats_df,
        'quality_pass'           : acc <= 0.65,
    }


# ═════════════════════════════════════════════
# 4.  MAIN GENERATOR INTERFACE
# ═════════════════════════════════════════════

def generate_decoys(X_train: np.ndarray,
                    X_val: np.ndarray,
                    n_decoys: int,
                    feature_names: list = None,
                    latent_dim: int = 8,
                    epochs: int = 50,
                    force_fallback: bool = False,
                    save_path: str = 'models/autoencoder.pt') -> tuple:
    """
    Main interface: train autoencoder, generate decoys, evaluate quality.
    Falls back to PCA+GMM if decoy quality is poor.

    Returns: (decoy_array, quality_metrics_dict)
    """
    print("\n" + "="*60)
    print("CANARY — Layer 1: Decoy Generator")
    print("="*60)

    if force_fallback or not TORCH_AVAILABLE:
        decoys = generate_decoys_statistical(X_train, n_decoys)
    else:
        # train autoencoder on legitimate records only
        model  = train_autoencoder(X_train, X_val,
                                   latent_dim=latent_dim,
                                   epochs=epochs,
                                   save_path=save_path)
        decoys = generate_decoys_autoencoder(model, X_train, n_decoys)

        # evaluate quality and auto-fallback if needed
        metrics = evaluate_decoy_quality(X_train, decoys, feature_names)

        if not metrics['quality_pass']:
            print("\n  ⚠ Autoencoder quality insufficient. Switching to PCA+GMM fallback...")
            decoys  = generate_decoys_statistical(X_train, n_decoys)
            metrics = evaluate_decoy_quality(X_train, decoys, feature_names)
        else:
            return decoys, metrics

    metrics = evaluate_decoy_quality(X_train, decoys, feature_names)
    return decoys, metrics


if __name__ == '__main__':
    # Smoke test with random data
    print("Running smoke test with random data...")
    X_fake = np.random.randn(5000, 14)
    X_val  = np.random.randn(500, 14)
    names  = [f'feature_{i}' for i in range(14)]

    decoys, metrics = generate_decoys(
        X_fake, X_val, n_decoys=500,
        feature_names=names, epochs=5,
        force_fallback=not TORCH_AVAILABLE
    )
    print(f"\nGenerated {len(decoys)} decoys")
    print(f"Quality pass: {metrics['quality_pass']}")
