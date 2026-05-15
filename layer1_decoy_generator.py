"""
layer1_decoy_generator.py
=========================
DecoyNet — Decoy Data Injection for Threat Reduction
Layer 1: Autoencoder-guided Decoy Generation

Primary  : Autoencoder-guided latent-neighborhood generator
Fallback : Local-neighborhood generator
           (used if autoencoder-guided discriminator accuracy > 65%)

Evaluation metrics inspired by Xu et al. (CTGAN, NeurIPS 2019):
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


def _type_column_indices(feature_names: list, n_features: int) -> list:
    """Find one-hot transaction-type columns from names, with a PaySim fallback."""
    if feature_names:
        cols = [i for i, name in enumerate(feature_names) if str(name).startswith('type_')]
        if cols:
            return cols
    return list(range(n_features - 5, n_features)) if n_features >= 5 else []


def _continuous_column_indices(n_features: int, type_cols: list) -> list:
    type_set = set(type_cols)
    return [i for i in range(n_features) if i not in type_set]


def _repair_type_columns(decoys: np.ndarray,
                         X_real: np.ndarray,
                         type_cols: list,
                         categories: np.ndarray = None) -> np.ndarray:
    """
    Force generated one-hot columns onto the exact scaled values seen in real data.

    StandardScaler turns raw one-hot 0/1 values into two non-binary values per
    column. Writing raw 0/1 into scaled data creates impossible records and makes
    the RF discriminator nearly perfect.
    """
    if not type_cols:
        return decoys

    lows = np.array([np.min(X_real[:, c]) for c in type_cols])
    highs = np.array([np.max(X_real[:, c]) for c in type_cols])

    if categories is None:
        categories = np.argmax(decoys[:, type_cols], axis=1)

    decoys[:, type_cols] = lows
    for row_idx, cat_idx in enumerate(categories):
        decoys[row_idx, type_cols[int(cat_idx)]] = highs[int(cat_idx)]

    return decoys


def _clip_continuous_to_real_range(decoys: np.ndarray,
                                   X_real: np.ndarray,
                                   continuous_cols: list,
                                   q_low: float = 0.001,
                                   q_high: float = 0.999) -> np.ndarray:
    """Clip continuous generated values to robust real-data quantile bounds."""
    for col in continuous_cols:
        lo, hi = np.quantile(X_real[:, col], [q_low, q_high])
        if lo == hi:
            lo, hi = X_real[:, col].min(), X_real[:, col].max()
        decoys[:, col] = np.clip(decoys[:, col], lo, hi)
    return decoys


def _hash_key_like_lookup(row: np.ndarray) -> str:
    """Match SecureLookupTable's six-decimal row representation."""
    return ','.join([f'{v:.6f}' for v in row])


def _ensure_lookup_unique(decoys: np.ndarray,
                          X_real: np.ndarray,
                          continuous_cols: list,
                          rng: np.random.Generator) -> np.ndarray:
    """
    Avoid false positives by ensuring no decoy hashes exactly like a real row.

    The lookup table stores rows rounded to six decimals. Tiny continuous nudges
    are enough to make a decoy unique without changing its distribution.
    """
    if not continuous_cols:
        return decoys

    real_keys = {_hash_key_like_lookup(row) for row in X_real}
    seen_keys = set()
    jitter_cols = np.array(continuous_cols)
    col_scales = np.std(X_real[:, jitter_cols], axis=0)
    col_scales = np.where(col_scales > 0, col_scales, 1.0)

    for row_idx in range(len(decoys)):
        key = _hash_key_like_lookup(decoys[row_idx])
        attempts = 0
        while (key in real_keys or key in seen_keys) and attempts < 20:
            pos = int(rng.integers(0, len(jitter_cols)))
            col = int(jitter_cols[pos])
            direction = -1.0 if rng.random() < 0.5 else 1.0
            decoys[row_idx, col] += direction * max(1e-4, 1e-4 * col_scales[pos])
            key = _hash_key_like_lookup(decoys[row_idx])
            attempts += 1
        seen_keys.add(key)

    return decoys


def generate_decoys_neighborhood(X_real: np.ndarray,
                                 n_decoys: int,
                                 feature_names: list = None,
                                 neighbor_pool: int = 25,
                                 random_state: int = 42) -> np.ndarray:
    """
    High-fidelity fallback: create unique decoys by tiny local interpolation
    between legitimate transactions of the same type.

    This preserves PaySim's empirical distribution much better than sampling a
    global Gaussian in scaled feature space, especially for zero-heavy balance
    columns and scaled one-hot transaction types.
    """
    from sklearn.neighbors import NearestNeighbors

    print("  Generator: local-neighborhood fallback")

    rng = np.random.default_rng(random_state)
    n_features = X_real.shape[1]
    type_cols = _type_column_indices(feature_names, n_features)
    continuous_cols = _continuous_column_indices(n_features, type_cols)

    if type_cols:
        real_categories = np.argmax(X_real[:, type_cols], axis=1)
        category_values, category_counts = np.unique(real_categories, return_counts=True)
        category_probs = category_counts / category_counts.sum()
        requested_counts = rng.multinomial(n_decoys, category_probs)
    else:
        category_values = np.array([0])
        requested_counts = np.array([n_decoys])
        real_categories = np.zeros(len(X_real), dtype=int)

    decoy_parts = []
    category_parts = []

    for category, n_category in zip(category_values, requested_counts):
        if n_category == 0:
            continue

        group_idx = np.where(real_categories == category)[0]
        if len(group_idx) == 0:
            continue

        pool_size = min(len(group_idx), 50000)
        pool_idx = rng.choice(group_idx, size=pool_size, replace=False)
        pool = X_real[pool_idx][:, continuous_cols]

        anchors_idx = rng.choice(group_idx, size=n_category, replace=True)
        anchors = X_real[anchors_idx].copy()

        if pool_size >= 2 and continuous_cols:
            k = min(neighbor_pool, pool_size)
            nn = NearestNeighbors(n_neighbors=k, algorithm='auto')
            nn.fit(pool)
            _, neighbor_pos = nn.kneighbors(anchors[:, continuous_cols])

            chosen_rank = rng.integers(1, k, size=n_category) if k > 1 else np.zeros(n_category, dtype=int)
            partner_idx = pool_idx[neighbor_pos[np.arange(n_category), chosen_rank]]
            partners = X_real[partner_idx]

            # Keep decoys close to real local structure while avoiding exact copies.
            lam = rng.beta(0.35, 8.0, size=(n_category, 1))
            anchors[:, continuous_cols] = (
                anchors[:, continuous_cols]
                + lam * (partners[:, continuous_cols] - anchors[:, continuous_cols])
            )

        decoy_parts.append(anchors)
        category_parts.append(np.full(n_category, int(category), dtype=int))

    decoys = np.vstack(decoy_parts)
    categories = np.concatenate(category_parts) if category_parts else None

    if len(decoys) > n_decoys:
        keep = rng.choice(len(decoys), size=n_decoys, replace=False)
        decoys = decoys[keep]
        categories = categories[keep] if categories is not None else None

    decoys = _clip_continuous_to_real_range(decoys, X_real, continuous_cols)
    decoys = _repair_type_columns(decoys, X_real, type_cols, categories=categories)
    decoys = _ensure_lookup_unique(decoys, X_real, continuous_cols, rng)

    return decoys[:n_decoys]


def _encode_in_batches(model,
                       X: np.ndarray,
                       batch_size: int = 8192) -> np.ndarray:
    """Encode a large numpy array without materialising a giant torch tensor."""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required.")

    device = next(model.parameters()).device
    encoded = []
    model.eval()

    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            xb = torch.FloatTensor(X[start:start + batch_size]).to(device)
            encoded.append(model.encode(xb).cpu().numpy())

    return np.vstack(encoded)


def _decode_in_batches(model,
                       Z: np.ndarray,
                       batch_size: int = 8192) -> np.ndarray:
    """Decode latent vectors in batches."""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required.")

    device = next(model.parameters()).device
    decoded = []
    model.eval()

    with torch.no_grad():
        for start in range(0, len(Z), batch_size):
            zb = torch.FloatTensor(Z[start:start + batch_size]).to(device)
            decoded.append(model.decode(zb).cpu().numpy())

    return np.vstack(decoded)


def generate_decoys_latent_neighborhood(model,
                                        X_real: np.ndarray,
                                        n_decoys: int,
                                        feature_names: list = None,
                                        neighbor_pool: int = 25,
                                        decoder_weight: float = 0.0,
                                        random_state: int = 42) -> np.ndarray:
    """
    Autoencoder-guided local latent decoy generator.

    The autoencoder learns a compact legitimate-transaction manifold. We use
    that latent space to find semantically nearby legitimate transactions, then
    create tiny local interpolations. decoder_weight can optionally blend in a
    small amount of decoder output, but defaults to 0 because scaled tabular
    constraints are better preserved in input space.
    This keeps the ML contribution while respecting tabular constraints.
    """
    from sklearn.neighbors import NearestNeighbors

    print("  Generator: autoencoder-guided latent neighborhood")

    rng = np.random.default_rng(random_state)
    n_features = X_real.shape[1]
    type_cols = _type_column_indices(feature_names, n_features)
    continuous_cols = _continuous_column_indices(n_features, type_cols)

    Z_real = _encode_in_batches(model, X_real)

    if type_cols:
        real_categories = np.argmax(X_real[:, type_cols], axis=1)
        category_values, category_counts = np.unique(real_categories, return_counts=True)
        category_probs = category_counts / category_counts.sum()
        requested_counts = rng.multinomial(n_decoys, category_probs)
    else:
        category_values = np.array([0])
        requested_counts = np.array([n_decoys])
        real_categories = np.zeros(len(X_real), dtype=int)

    decoy_parts = []
    category_parts = []

    for category, n_category in zip(category_values, requested_counts):
        if n_category == 0:
            continue

        group_idx = np.where(real_categories == category)[0]
        if len(group_idx) == 0:
            continue

        pool_size = min(len(group_idx), 50000)
        pool_idx = rng.choice(group_idx, size=pool_size, replace=False)
        latent_pool = Z_real[pool_idx]

        anchors_idx = rng.choice(group_idx, size=n_category, replace=True)
        anchors = X_real[anchors_idx].copy()
        z_anchors = Z_real[anchors_idx]

        if pool_size >= 2:
            k = min(neighbor_pool, pool_size)
            nn = NearestNeighbors(n_neighbors=k, algorithm='auto')
            nn.fit(latent_pool)
            _, neighbor_pos = nn.kneighbors(z_anchors)

            chosen_rank = rng.integers(1, k, size=n_category) if k > 1 else np.zeros(n_category, dtype=int)
            partner_idx = pool_idx[neighbor_pos[np.arange(n_category), chosen_rank]]
            partners = X_real[partner_idx]
            z_partners = Z_real[partner_idx]

            lam = rng.beta(0.35, 8.0, size=(n_category, 1))
            z_synth = z_anchors + lam * (z_partners - z_anchors)

            latent_scale = np.std(Z_real[pool_idx], axis=0, keepdims=True)
            latent_scale = np.where(latent_scale > 0, latent_scale, 1.0)
            z_synth += rng.normal(0.0, 0.005, size=z_synth.shape) * latent_scale

            input_interp = anchors.copy()
            input_interp[:, continuous_cols] = (
                anchors[:, continuous_cols]
                + lam * (partners[:, continuous_cols] - anchors[:, continuous_cols])
            )
        else:
            z_synth = z_anchors
            input_interp = anchors

        decoys = input_interp.copy()
        if decoder_weight > 0 and continuous_cols:
            decoded = _decode_in_batches(model, z_synth)
            decoys[:, continuous_cols] = (
                (1.0 - decoder_weight) * input_interp[:, continuous_cols]
                + decoder_weight * decoded[:, continuous_cols]
            )

        decoy_parts.append(decoys)
        category_parts.append(np.full(n_category, int(category), dtype=int))

    decoys = np.vstack(decoy_parts)
    categories = np.concatenate(category_parts) if category_parts else None

    if len(decoys) > n_decoys:
        keep = rng.choice(len(decoys), size=n_decoys, replace=False)
        decoys = decoys[keep]
        categories = categories[keep] if categories is not None else None

    decoys = _clip_continuous_to_real_range(decoys, X_real, continuous_cols)
    decoys = _repair_type_columns(decoys, X_real, type_cols, categories=categories)
    decoys = _ensure_lookup_unique(decoys, X_real, continuous_cols, rng)

    return decoys[:n_decoys]


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
    print("[WARNING] PyTorch not found. Only local fallback available.")


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
            nn.Dropout(0.05),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.05),
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
                      batch_size: int = 64,
                      lr: float = 1e-3,
                      save_path: str = 'models/autoencoder.pt') -> 'Autoencoder':
    """
    Train the autoencoder on LEGITIMATE records only.
    Training on fraud records would teach it to reproduce fraud patterns,
    which we do NOT want — decoys should look like normal transactions.
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required.")
    
    print("  Training autoencoder")
    print(f"    input_dim={X_train.shape[1]} latent_dim={latent_dim} epochs={epochs}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"    device={device}")

    # Convert to tensors
    X_tr = torch.FloatTensor(X_train).to(device)
    X_vl = torch.FloatTensor(X_val).to(device)

    loader = DataLoader(
    TensorDataset(X_tr, X_tr),
    batch_size=batch_size,
    shuffle=True,
    drop_last=True)

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
            print(f"    epoch {epoch:3d}/{epochs} | train={train_loss:.6f} | val={val_loss:.6f}")

        # ── early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP:
                print(f"    early stopping at epoch {epoch}")
                break

    # reload best weights
    model.load_state_dict(torch.load(save_path, map_location=device))
    model.eval()
    print(f"    best_val_loss={best_val_loss:.6f}")
    return model


def generate_decoys_autoencoder(model,
                                X_real: np.ndarray,
                                n_decoys: int,
                                noise_std: float = 0.10,
                                feature_names: list = None) -> np.ndarray:
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
    # sample REAL latent vectors instead of random Gaussian vectors
    indices = np.random.choice(len(Z), n_decoys, replace=True)

    # choose existing latent vectors
    Z_base = Z[indices]

    # add very small perturbation noise
    noise = np.random.normal(
        loc=0.0,
        scale=noise_std,
        size=Z_base.shape
    )

    # perturbed latent vectors
    Z_sample = Z_base + noise

    with torch.no_grad():
        Z_t    = torch.FloatTensor(Z_sample).to(device)
        decoys = model.decode(Z_t).cpu().numpy()

    type_cols = _type_column_indices(feature_names, X_real.shape[1])
    continuous_cols = _continuous_column_indices(X_real.shape[1], type_cols)

    # --- std rescaling: match each continuous feature's spread to real data ---
    for i in continuous_cols:
        r_mean = X_real[:, i].mean()
        r_std  = X_real[:, i].std()
        d_mean = decoys[:, i].mean()
        d_std  = decoys[:, i].std() + 1e-8   # avoid divide-by-zero

        # centre and rescale to match real std, then re-centre on real mean
        decoys[:, i] = (decoys[:, i] - d_mean) / d_std * r_std + r_mean

        # clip to ±3 SD AFTER rescaling
        lo = r_mean - 3 * r_std
        hi = r_mean + 3 * r_std
        decoys[:, i] = np.clip(decoys[:, i], lo, hi)

    decoys = _clip_continuous_to_real_range(decoys, X_real, continuous_cols)
    decoys = _repair_type_columns(decoys, X_real, type_cols)
    decoys = _ensure_lookup_unique(decoys, X_real, continuous_cols, np.random.default_rng(42))
    return decoys


# ═════════════════════════════════════════════
# 2.  STATISTICAL FALLBACK  (legacy ablation option)
# ═════════════════════════════════════════════

def generate_decoys_statistical(X_real: np.ndarray,
                                n_decoys: int,
                                n_components: int = 10,
                                feature_names: list = None) -> np.ndarray:
    """
    Legacy decoy generator using PCA + Gaussian Mixture Model.
    Kept for ablation experiments; the production fallback is local-neighborhood.

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

    print("  Generator: legacy PCA + GMM ablation")

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

    type_cols = _type_column_indices(feature_names, X_real.shape[1])
    continuous_cols = _continuous_column_indices(X_real.shape[1], type_cols)
    decoys = _clip_continuous_to_real_range(decoys[:n_decoys], X_real, continuous_cols)
    decoys = _repair_type_columns(decoys, X_real, type_cols)
    decoys = _ensure_lookup_unique(decoys, X_real, continuous_cols, np.random.default_rng(42))

    return decoys


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

    Additionally reports simple validation checks:
      - ±2 SD check on feature means
      - KS-test statistical similarity
    """
    print("  Evaluating decoy quality")
    print(f"    real={len(X_real):,} decoy={len(X_decoy):,}")

    n = min(len(X_real), len(X_decoy), 20000)
    rng = np.random.default_rng(42)
    real_idx = rng.choice(len(X_real), size=n, replace=False)
    decoy_idx = rng.choice(len(X_decoy), size=n, replace=False)

    # ── A. RF Discriminator Test (Xu et al. 2019) ────────────
    # Label real=0, decoy=1 — train RF to distinguish them
    X_combined = np.vstack([X_real[real_idx], X_decoy[decoy_idx]])
    y_combined = np.array([0]*n + [1]*n)

    # 80/20 split for this internal evaluation
    from sklearn.model_selection import train_test_split
    Xtr, Xte, ytr, yte = train_test_split(
        X_combined, y_combined, test_size=0.2, random_state=42)

    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)
    rf.fit(Xtr, ytr)
    y_pred = rf.predict(Xte)

    acc       = accuracy_score(yte, y_pred)
    f1        = f1_score(yte, y_pred, average='weighted')
    precision = precision_score(yte, y_pred, average='weighted', zero_division=0)
    recall    = recall_score(yte, y_pred, average='weighted', zero_division=0)

    quality_flag = "PASS" if acc <= 0.65 else "REVIEW"
    print(
        f"    RF discriminator accuracy={acc:.4f} "
        f"| f1={f1:.4f} | target<=0.60 | {quality_flag}"
    )

    # ── B. Feature-wise Statistics ────────────────────────────
    verbose_features = os.environ.get('DECOYNET_VERBOSE_QUALITY', '').lower() in {'1', 'true', 'yes'}
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

        # categorical one-hot columns
        if fname.startswith("type_"):

        # compare category probabilities directly
            active_value = np.max(r)
            real_prob = np.mean(np.isclose(r, active_value))
            decoy_prob = np.mean(np.isclose(d, active_value))

        # KS not meaningful for one-hot binaries
            ks_p = 1.0

        # simple categorical divergence
            kl = abs(real_prob - decoy_prob)

        else:
        # continuous-feature evaluation

            ks_stat, ks_p = stats.ks_2samp(r, d)

            lo = min(r.min(), d.min())
            hi = max(r.max(), d.max())
            bins = np.linspace(lo, hi, 50) if lo != hi else np.array([lo, hi + 1e-8])

            r_hist, _ = np.histogram(r, bins=bins, density=True)
            d_hist, _ = np.histogram(d, bins=bins, density=True)

            r_hist += 1e-10
            d_hist += 1e-10

            kl = np.sum(rel_entr(r_hist, d_hist))

        ks_pvals.append(ks_p)
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
    if verbose_features:
        print(stats_df[['feature', 'real_mean', 'decoy_mean', 'real_std',
                         'decoy_std', 'mean_ok', 'ks_pval', 'kl_div']].to_string(index=False))

    n_ok = stats_df['mean_ok'].sum()
    print(
        f"    features_within_2sd={n_ok}/{len(feature_names)} "
        f"| mean_KL={np.mean(kl_divs):.4f} | mean_KS_p={np.mean(ks_pvals):.4f}"
    )

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
    Falls back to the local-neighborhood generator if decoy quality is poor.

    Returns: (decoy_array, quality_metrics_dict)
    """
    print("\n[1] Decoy generation")

    if force_fallback or not TORCH_AVAILABLE:
        decoys = generate_decoys_neighborhood(X_train, n_decoys, feature_names)
    else:
        # train autoencoder on legitimate records only
        model  = train_autoencoder(X_train, X_val,
                                   latent_dim=latent_dim,
                                   epochs=epochs,
                                   save_path=save_path)
        decoys = generate_decoys_latent_neighborhood(
            model, X_train, n_decoys, feature_names=feature_names)

        # evaluate quality and auto-fallback if needed
        metrics = evaluate_decoy_quality(X_train, decoys, feature_names)

        if not metrics['quality_pass']:
            print("  Autoencoder-guided decoys missed quality target; using local fallback.")
            decoys  = generate_decoys_neighborhood(X_train, n_decoys, feature_names)
            metrics = evaluate_decoy_quality(X_train, decoys, feature_names)
            return decoys, metrics
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
