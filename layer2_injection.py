"""
layer2_injection.py
===================
Canary — Decoy-Based Data Breach Detection System
Layer 2: Decoy Injection + Cryptographically Secured Lookup Table

Four injection strategies:
  1. Random      — baseline shuffle (random positions)
  2. Edge-case   — decision boundary seeding (SVM/RF boundary records)
  3. Cluster     — k-Means centroid neighbourhood injection
  4. High-value  — targets high transaction amount region (attacker bait)

Security:
  Lookup table uses SHA-256(row_hash + secret_salt) so even if an attacker
  steals the lookup table they cannot reverse-engineer which records are decoys.

Attacker profiling (novel contribution):
  Each injected decoy carries a zone tag. When stolen data is checked,
  we identify WHICH zone was hit → reveals attacker's selection strategy.
"""

import numpy as np
import pandas as pd
import hashlib
import os
import json
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
INJECTION_ZONES = ['random', 'edge_case', 'cluster', 'high_value']
DEFAULT_SALT    = os.urandom(32).hex()   # random salt per session


# ═════════════════════════════════════════════
# SECURE LOOKUP TABLE
# ═════════════════════════════════════════════

class SecureLookupTable:
    """
    Cryptographically hashed lookup table for decoy records.

    Storage format:
        { SHA256(row_string + salt) : zone_tag }

    An attacker who steals this table sees only hashes —
    they cannot determine which records are decoys without the salt.
    The salt is never stored alongside the table.
    """

    def __init__(self, salt: str = None):
        self.salt  = salt or DEFAULT_SALT
        self._table: dict = {}          # hash → zone_tag
        self._count: dict = {z: 0 for z in INJECTION_ZONES}

    def _hash_record(self, row: np.ndarray) -> str:
        """SHA-256(row_values_string + secret_salt)"""
        row_str    = ','.join([f'{v:.6f}' for v in row])
        salted     = (row_str + self.salt).encode('utf-8')
        return hashlib.sha256(salted).hexdigest()

    def register(self, decoy_rows: np.ndarray, zone: str):
        """Add decoy records to the lookup table."""
        for row in decoy_rows:
            h = self._hash_record(row)
            self._table[h] = zone
        self._count[zone] = self._count.get(zone, 0) + len(decoy_rows)

    def is_decoy(self, row: np.ndarray) -> tuple:
        """
        Returns (is_decoy: bool, zone: str or None)
        """
        h = self._hash_record(row)
        if h in self._table:
            return True, self._table[h]
        return False, None

    def check_batch(self, rows: np.ndarray) -> dict:
        """
        Check an entire batch of records (simulates stolen data check).

        Returns:
            alarm       : bool — True if any decoy found
            n_decoys    : int  — number of decoys in batch
            zones_hit   : dict — zone → count of decoys from that zone
            decoy_ratio : float
        """
        zones_hit = {}
        n_decoys  = 0

        for row in rows:
            found, zone = self.is_decoy(row)
            if found:
                n_decoys += 1
                zones_hit[zone] = zones_hit.get(zone, 0) + 1

        return {
            'alarm'      : n_decoys > 0,
            'n_decoys'   : n_decoys,
            'zones_hit'  : zones_hit,
            'decoy_ratio': n_decoys / len(rows) if len(rows) > 0 else 0,
        }

    def summary(self):
        print(f"\n  Lookup table: {len(self._table)} decoy hashes registered")
        for zone, count in self._count.items():
            print(f"    {zone:<12}: {count} decoys")

    def save(self, path: str = 'models/lookup_table.json'):
        """Save table (WITHOUT salt — salt stored separately)."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self._table, f)
        print(f"  Lookup table saved → {path}")
        print(f"  ⚠ Salt NOT saved to file (keep separately): {self.salt[:16]}...")

    @classmethod
    def load(cls, path: str, salt: str) -> 'SecureLookupTable':
        """Reload table with the correct salt."""
        with open(path, 'r') as f:
            table_data = json.load(f)
        obj         = cls(salt=salt)
        obj._table  = table_data
        return obj


# ═════════════════════════════════════════════
# INJECTION STRATEGIES
# ═════════════════════════════════════════════

def _strategy_random(X_real: np.ndarray,
                     decoys: np.ndarray,
                     n_inject: int) -> np.ndarray:
    """
    Strategy 1: Random injection (baseline).
    Decoys are inserted at uniformly random positions.
    Rationale: establishes baseline detection rate.
    """
    idx = np.random.choice(len(decoys), size=min(n_inject, len(decoys)), replace=False)
    return decoys[idx]


def _strategy_edge_case(X_real: np.ndarray,
                        y_real: np.ndarray,
                        decoys: np.ndarray,
                        n_inject: int) -> np.ndarray:
    """
    Strategy 2: Edge-case / decision boundary injection.

    Approach:
      - Train a lightweight RF on real data
      - Find real records with prediction probability close to 0.5
        (these sit on the decision boundary)
      - Generate decoys whose latent representation is close to these
        boundary records → inject there

    Why: Sophisticated attackers targeting ambiguous records
         (fraud boundary) will inevitably collect these decoys.
    Course alignment: Week 5-6 (classifiers, decision boundaries)
    """
    if len(np.unique(y_real)) < 2:
        # Can't find boundary without both classes; fall back to random
        return _strategy_random(X_real, decoys, n_inject)

    rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    rf.fit(X_real, y_real)
    probs = rf.predict_proba(X_real)[:, 1]

    # boundary records: probability closest to 0.5
    boundary_mask = np.argsort(np.abs(probs - 0.5))[:int(len(probs)*0.15)]
    boundary_recs = X_real[boundary_mask]

    # find decoys closest to boundary records in feature space
    from sklearn.metrics.pairwise import euclidean_distances
    if len(boundary_recs) > 200:
        boundary_recs = boundary_recs[np.random.choice(len(boundary_recs), 200, replace=False)]

    dists = euclidean_distances(decoys, boundary_recs).min(axis=1)
    closest_idx = np.argsort(dists)[:n_inject]
    return decoys[closest_idx]


def _strategy_cluster(X_real: np.ndarray,
                      decoys: np.ndarray,
                      n_inject: int,
                      k: int = 8) -> np.ndarray:
    """
    Strategy 3: Cluster-based injection (k-Means centroids).

    Approach:
      - Run k-Means on real data
      - Inject decoys nearest to each centroid
      → An attacker using clustering to sample representative records
        will collect records from each cluster, inevitably hitting decoys.

    Course alignment: Week 3 (k-Means)
    """
    k = min(k, len(X_real) // 10)     # safety
    km = KMeans(n_clusters=k, random_state=42, n_init='auto')
    km.fit(X_real)
    centroids = km.cluster_centers_

    # find decoys closest to each centroid
    from sklearn.metrics.pairwise import euclidean_distances
    dists   = euclidean_distances(decoys, centroids).min(axis=1)
    best_idx = np.argsort(dists)[:n_inject]
    return decoys[best_idx]


def _strategy_high_value(X_real: np.ndarray,
                          decoys: np.ndarray,
                          n_inject: int,
                          amount_col_idx: int = 1) -> np.ndarray:
    """
    Strategy 4: High-value record injection.

    Approach:
      - Identify the top 20% of real records by transaction amount
        (amount is column index 1 in scaled PaySim after preprocessing)
      - Generate decoys that mimic this high-value region
      → Attackers specifically hunting high-value transactions
        will hit these decoys first.

    This is the most effective strategy against financially motivated
    targeted attackers.
    """
    # find high-value real records (top 20% by amount feature)
    amounts     = X_real[:, amount_col_idx]
    threshold   = np.percentile(amounts, 80)
    hv_mask     = amounts >= threshold
    hv_records  = X_real[hv_mask]

    if len(hv_records) == 0:
        return _strategy_random(X_real, decoys, n_inject)

    from sklearn.metrics.pairwise import euclidean_distances
    dists    = euclidean_distances(decoys, hv_records).min(axis=1)
    best_idx = np.argsort(dists)[:n_inject]
    return decoys[best_idx]


# ═════════════════════════════════════════════
# MAIN INJECTION LAYER
# ═════════════════════════════════════════════

def inject_decoys(X_real: np.ndarray,
                  y_real: np.ndarray,
                  decoys: np.ndarray,
                  injection_ratio: float = 0.05,
                  strategy_weights: dict = None,
                  salt: str = None) -> tuple:
    """
    Inject decoys into the real dataset using all four strategies.

    Parameters
    ----------
    X_real           : real feature array (scaled)
    y_real           : real labels
    decoys           : generated decoy array from Layer 1
    injection_ratio  : fraction of final dataset that is decoys (default 5%)
    strategy_weights : proportion of decoys per strategy (must sum to 1)
    salt             : secret salt for lookup table (generate fresh if None)

    Returns
    -------
    X_injected  : combined real + decoy feature array
    y_injected  : labels (real labels preserved, decoys labelled as 0 = legit)
    is_decoy    : boolean array (True = decoy record)
    zone_labels : zone string per record ('' for real records)
    lookup      : SecureLookupTable instance
    """
    print("\n" + "="*60)
    print("CANARY — Layer 2: Decoy Injection")
    print("="*60)

    if strategy_weights is None:
        strategy_weights = {
            'random'    : 0.25,
            'edge_case' : 0.25,
            'cluster'   : 0.25,
            'high_value': 0.25,
        }

    # total number of decoys to inject
    n_total_decoys = int(len(X_real) * injection_ratio / (1 - injection_ratio))
    n_total_decoys = min(n_total_decoys, len(decoys))

    print(f"\n  Real records       : {len(X_real)}")
    print(f"  Injection ratio    : {injection_ratio*100:.1f}%")
    print(f"  Decoys to inject   : {n_total_decoys}")
    print(f"  Strategy weights   : {strategy_weights}")

    lookup = SecureLookupTable(salt=salt)

    injected_decoys = []
    zone_tags       = []

    for zone, weight in strategy_weights.items():
        n_zone = max(1, int(n_total_decoys * weight))

        if zone == 'random':
            batch = _strategy_random(X_real, decoys, n_zone)
        elif zone == 'edge_case':
            batch = _strategy_edge_case(X_real, y_real, decoys, n_zone)
        elif zone == 'cluster':
            batch = _strategy_cluster(X_real, decoys, n_zone)
        elif zone == 'high_value':
            batch = _strategy_high_value(X_real, decoys, n_zone)
        else:
            batch = _strategy_random(X_real, decoys, n_zone)

        # register in lookup table
        lookup.register(batch, zone)
        injected_decoys.append(batch)
        zone_tags.extend([zone] * len(batch))

        print(f"  ✓ {zone:<12}: {len(batch)} decoys injected")

    all_decoys = np.vstack(injected_decoys)

    # ── assemble final injected dataset ──────────────────────
    X_injected  = np.vstack([X_real, all_decoys])
    # decoys are labelled as non-fraud (0) — they look like legitimate records
    y_injected  = np.concatenate([y_real, np.zeros(len(all_decoys), dtype=int)])
    is_decoy    = np.array([False]*len(X_real) + [True]*len(all_decoys))
    zone_labels = np.array(['']*len(X_real) + zone_tags)

    # shuffle
    perm        = np.random.permutation(len(X_injected))
    X_injected  = X_injected[perm]
    y_injected  = y_injected[perm]
    is_decoy    = is_decoy[perm]
    zone_labels = zone_labels[perm]

    lookup.summary()

    print(f"\n  Final injected dataset:")
    print(f"    Total records  : {len(X_injected)}")
    print(f"    Real records   : {len(X_real)}")
    print(f"    Decoy records  : {len(all_decoys)}")
    print(f"    Actual decoy % : {len(all_decoys)/len(X_injected)*100:.2f}%")
    print("\n  ✓ Injection complete.")

    return X_injected, y_injected, is_decoy, zone_labels, lookup


def injection_density_experiment(X_real: np.ndarray,
                                 y_real: np.ndarray,
                                 decoys: np.ndarray,
                                 ratios: list = None) -> pd.DataFrame:
    """
    Sweep across injection densities to find the optimal ratio.
    Returns a DataFrame showing decoy count vs injected dataset size.
    Useful for the paper's tradeoff analysis table.
    """
    if ratios is None:
        ratios = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20]

    rows = []
    for r in ratios:
        n_d = int(len(X_real) * r / (1 - r))
        rows.append({
            'injection_ratio': r,
            'n_real'         : len(X_real),
            'n_decoys'       : min(n_d, len(decoys)),
            'total'          : len(X_real) + min(n_d, len(decoys)),
            'actual_pct'     : min(n_d, len(decoys)) / (len(X_real) + min(n_d, len(decoys))) * 100
        })

    df = pd.DataFrame(rows)
    print("\n  ── Injection Density Options ──")
    print(df.to_string(index=False))
    return df


if __name__ == '__main__':
    print("Running Layer 2 smoke test...")
    X_r  = np.random.randn(1000, 14)
    y_r  = np.random.randint(0, 2, 1000)
    decs = np.random.randn(500, 14)

    Xi, yi, isd, zones, lut = inject_decoys(X_r, y_r, decs, injection_ratio=0.05)

    # test lookup
    result = lut.check_batch(Xi[:50])
    print(f"\nBatch check on 50 records: alarm={result['alarm']}, decoys={result['n_decoys']}")
    print(f"Zones hit: {result['zones_hit']}")
