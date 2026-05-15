# DecoyNet: Decoy Data Injection for Threat Reduction

**Course**: Machine Learning and Pattern Recognition  
**Dataset**: PaySim synthetic financial dataset (Lopez-Rojas, Elmir & Axelsson, 2016)  
**Dataset source**: https://www.kaggle.com/datasets/ealaxi/paysim1

DecoyNet is a three-layer honeypot-style breach detection pipeline for financial transaction data. It generates realistic decoy records, injects them into a real transaction dataset, and raises an alarm if simulated stolen data contains any registered decoy.

The project is designed for exfiltration settings where traditional anomaly detectors can miss targeted theft, mimicry attacks, or slow incremental access.

## System Overview

```text
Layer 0: Preprocessing
  Clean PaySim, engineer ratio features, one-hot encode transaction type,
  scale features, and split train/validation/test data.

Layer 1: Decoy Generation
  Train an autoencoder on legitimate transactions, use its latent space to
  find local neighborhoods of similar records, generate decoys by small local
  interpolations, repair categorical constraints, and validate quality with
  a Random Forest real-vs-decoy discriminator.

Layer 2: Decoy Injection
  Inject decoys through random, edge-case, cluster, and high-value strategies.
  Register each decoy in a salted SHA-256 secure lookup table.

Layer 3: Detection and Evaluation
  Simulate bulk, targeted, mimicry, and slow-theft attacks. Check stolen
  batches against the lookup table, report detection rates, and compare
  against Random Forest and Isolation Forest baselines.
```

## Project Structure

```text
decoynet/
|-- preprocessing.py             # Layer 0: loading, feature engineering, scaling
|-- layer1_decoy_generator.py    # Layer 1: AE-guided latent-neighborhood decoys
|-- layer2_injection.py          # Layer 2: injection strategies + secure lookup
|-- layer3_detection.py          # Layer 3: attack simulation + baselines
|-- main.py                      # End-to-end runner
|-- requirements.txt
|-- models/                      # saved scaler, autoencoder, lookup table, salt
|-- outputs/                     # result summaries and quality reports
`-- paysim dataset.csv           # local PaySim CSV used by this project
```

## Main ML Method

The primary decoy generator is an **autoencoder-guided latent-neighborhood generator**.

Why this design:

- A naive autoencoder decoder can produce unrealistic tabular rows because PaySim has hard constraints, especially scaled one-hot transaction type columns and zero-heavy balance features.
- DecoyNet instead uses the autoencoder as a representation learner. The encoder learns a compressed latent manifold of legitimate transactions.
- Nearest neighbors are found in the learned latent space, and decoys are generated through tiny local interpolations between similar legitimate records.
- Constraint repair then restores valid scaled transaction-type columns.
- A lookup-hash uniqueness check prevents decoys from colliding with real rows.
- A Random Forest discriminator evaluates whether decoys can be separated from real records.

Quality target:

```text
RF discriminator accuracy <= 0.60
```

An accuracy close to 0.50 means the discriminator is close to random guessing.

## Setup

```bash
pip install -r requirements.txt
```

Download PaySim from Kaggle and place the CSV in the project folder or pass its path with `--data`.

## Recommended Commands

Fast sanity test:

```bash
python -B main.py --data "paysim dataset.csv" --sample 0.01 --n_decoys 1000 --latent 8 --epochs 3 --trials 1
```

Main report-style run on a 5% sample:

```bash
python -B main.py --data "paysim dataset.csv" --sample 0.05 --n_decoys 15000 --latent 16 --epochs 30 --trials 10
```

Stronger run on a 10% sample:

```bash
python -B main.py --data "paysim dataset.csv" --sample 0.10 --latent 16 --epochs 50 --trials 20
```

Full-dataset run:

```bash
python -B main.py --data "paysim dataset.csv" --latent 16 --epochs 50 --trials 20
```

The full dataset is large, so the full run can take a long time on a laptop.

Use the non-neural fallback only for ablation or emergency runtime:

```bash
python -B main.py --data "paysim dataset.csv" --sample 0.05 --fallback --trials 20
```

## Command-Line Arguments

| Argument | Default | Description |
|---|---:|---|
| `--data` | required | Path to PaySim CSV |
| `--sample` | `1.0` | Fraction of dataset to use |
| `--inject` | `0.05` | Target decoy ratio in final injected dataset |
| `--latent` | `8` | Autoencoder latent dimension |
| `--epochs` | `50` | Autoencoder training epochs |
| `--n_decoys` | auto | Number of decoys to generate |
| `--fallback` | `False` | Skip autoencoder and use local-neighborhood fallback |
| `--trials` | `20` | Repetitions per attack type |

## Injection Strategies

| Strategy | Placement logic | Intended attacker caught |
|---|---|---|
| Random | Uniformly selected decoys | Opportunistic bulk theft |
| Edge-case | Near fraud decision-boundary records | Sophisticated targeted theft |
| Cluster | Near k-Means cluster centroids | Representative sampling attacks |
| High-value | Near top-amount transactions | Financially motivated attackers |

Each decoy is stored in the lookup table as:

```text
SHA256(row_values + secret_salt) -> injection_zone
```

The raw decoy rows are not stored in the lookup table.

## Attack Simulations

| Attack | Description | Traditional detection risk | DecoyNet behavior |
|---|---|---|---|
| Bulk steal | Random 10-30% dataset theft | Usually visible only at large volume | Detects if any decoy is stolen |
| Targeted | Steals top-value records | Can look like valid high-value access | High-value decoys increase exposure |
| Mimicry | Filters outliers before theft | Can bypass anomaly detectors | Decoys are designed to look in-distribution |
| Slow theft | Small repeated batches | Often missed per batch | Any decoy hit triggers alarm |

## Output Files

```text
outputs/results_summary.csv       # attack detection rates and decoy ratios
outputs/baseline_comparison.csv   # Random Forest and Isolation Forest metrics
outputs/quality_summary.json      # aggregate decoy quality metrics
outputs/decoy_quality.csv         # per-feature quality report
models/autoencoder.pt             # trained autoencoder weights
models/scaler.pkl                 # fitted StandardScaler
models/lookup_table.json          # salted hash lookup table
models/KEEP_SECRET_salt.txt       # salt for lookup checks
```

## References

1. Lopez-Rojas, E., Elmir, A., & Axelsson, S. (2016). *PaySim: A Financial Mobile Money Simulator for Fraud Detection*.
2. Xu et al. (2019). *Modeling Tabular Data using Conditional GAN*. Used here for the real-vs-synthetic discriminator quality evaluation idea.
3. Tree-based fraud detection baselines on PaySim are used as comparison context for Random Forest and Isolation Forest experiments.

## Security Note

The lookup table stores only salted SHA-256 hashes. An attacker who obtains `models/lookup_table.json` alone cannot identify decoys without the salt. In a real system, `KEEP_SECRET_salt.txt` should be stored in a secret manager, not next to the lookup table.
