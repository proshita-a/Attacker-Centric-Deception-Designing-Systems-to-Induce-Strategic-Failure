# Decoy-Based Data Breach Detection System (Honeypot Tech)

**Course**: Machine Learning and Pattern Recognition (2nd Year CSAI)  
**Dataset**: PaySim — Synthetic Financial Dataset (Lopez-Rojas, Elmir & Axelsson, EMSS 2016)  
**Kaggle**: https://www.kaggle.com/datasets/ealaxi/paysim1

---

## System Overview

This is a three-layer honeypot system for detecting data exfiltration:

```
Layer 0: Preprocessing   → clean, encode, scale, split PaySim
Layer 1: Decoy Generator → Autoencoder trains on legit records → generates realistic fake transactions
Layer 2: Injection       → 4 strategies place decoys inside real dataset + secure SHA-256 lookup table
Layer 3: Detection       → 4 attack simulations → lookup check → alarm + flood response
```

---

## Project Structure

```
canary/
├── preprocessing.py          # Layer 0: PaySim loading, feature engineering, scaling
├── layer1_decoy_generator.py # Layer 1: Autoencoder + PCA/GMM fallback + quality evaluation
├── layer2_injection.py       # Layer 2: 4 injection strategies + SecureLookupTable (SHA-256)
├── layer3_detection.py       # Layer 3: 4 attack simulators + baselines + flood response
├── main.py                   # End-to-end pipeline runner
├── requirements.txt
├── models/                   # saved weights, scaler, lookup table
└── outputs/                  # results CSVs, quality reports
```

---

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download PaySim from Kaggle (requires account)
#    https://www.kaggle.com/datasets/ealaxi/paysim1
#    Place the CSV in: data/paysim.csv

# 3. Run full pipeline
python main.py --data data/paysim.csv

# 4. Quick test with 10% of data (faster)
python main.py --data data/paysim.csv --sample 0.1 --epochs 10

# 5. Skip autoencoder, use PCA+GMM fallback only
python main.py --data data/paysim.csv --fallback
```

---

## Key Arguments

| Argument | Default | Description |
|---|---|---|
| `--data` | required | Path to paysim CSV |
| `--sample` | 1.0 | Fraction of dataset (0.1 for quick testing) |
| `--inject` | 0.05 | Injection ratio (5% decoys in final dataset) |
| `--latent` | 8 | Autoencoder latent space dimension |
| `--epochs` | 50 | Autoencoder training epochs |
| `--fallback` | False | Use PCA+GMM instead of autoencoder |
| `--trials` | 20 | Repetitions per attack type (for stable averages) |

---

## Papers Referenced

1. **Lopez-Rojas, Elmir & Axelsson (2016)** — *PaySim: A Financial Mobile Money Simulator for Fraud Detection*, EMSS 2016. → Dataset source.

2. **Xu et al. (2019)** — *Modeling Tabular Data using Conditional GAN*, NeurIPS 2019. → Decoy quality evaluation framework (real-vs-synthetic discriminator test).

3. **Springer JDIM Tree-Based Comparison (2023)** — *Detection of Financial Fraud: Comparisons of Tree-Based ML Approaches on PaySim*. → Fraud detection baseline numbers (AUC-ROC, F1) on same dataset.

---

## Injection Strategies

| Strategy | Target | Catches |
|---|---|---|
| Random | Uniform positions | Opportunistic bulk theft |
| Edge-case | Decision boundary records | Sophisticated targeted theft |
| Cluster | k-Means centroid neighbourhood | Clustering-based sampling |
| High-value | Top 20% by transaction amount | Financially motivated attackers |

**Attacker profiling (novel contribution)**: the zone tag in the lookup table tells you 
*which* strategy caught the attacker, revealing their selection method for free.

---

## Attack Types Simulated

| Attack | Description | Traditional Detection | Canary |
|---|---|---|---|
| Bulk steal | Random 10-30% of dataset | Detects (large anomaly) | Detects |
| Targeted | Top-value records only | Misses (looks normal) | Detects |
| Mimicry | Outlier-filtered theft | Misses (bypasses IsoForest) | Detects |
| Slow theft | 100-record batches × 20 | Misses (each batch normal) | Detects |

---

## Security Note

The lookup table (`models/lookup_table.json`) stores only SHA-256 hashes, never raw decoy values.
The salt is stored separately (`models/KEEP_SECRET_salt.txt`).
An attacker who steals the lookup table cannot determine which records are decoys without the salt.
