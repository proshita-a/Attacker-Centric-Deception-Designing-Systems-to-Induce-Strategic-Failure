"""
main.py
=======
DecoyNet — Decoy Data Injection for Threat Reduction
Main pipeline: runs all three layers end-to-end.

Usage:
    python main.py --data path/to/paysim.csv
    python main.py --data path/to/paysim.csv --sample 0.1   # quick test with 10% data
    python main.py --data path/to/paysim.csv --fallback      # skip autoencoder, use local fallback

Output:
    outputs/results_summary.csv    — attack simulation results table
    outputs/decoy_quality.csv      — per-feature quality metrics
    models/autoencoder.pt          — trained autoencoder weights
    models/scaler.pkl              — fitted scaler
    models/lookup_table.json       — hashed decoy lookup table
"""

import argparse
import os
import json
import numpy as np
import pandas as pd

from preprocessing        import load_and_preprocess
from layer1_decoy_generator import generate_decoys
from layer2_injection     import inject_decoys, injection_density_experiment
from layer3_detection     import run_full_detection_experiment


def parse_args():
    parser = argparse.ArgumentParser(description='DecoyNet: Decoy Data Injection for Threat Reduction')
    parser.add_argument('--data',       type=str,   required=True,
                        help='Path to PaySim CSV file')
    parser.add_argument('--sample',     type=float, default=1.0,
                        help='Fraction of dataset to use (default: 1.0)')
    parser.add_argument('--inject',     type=float, default=0.05,
                        help='Injection ratio (default: 0.05 = 5%%)')
    parser.add_argument('--latent',     type=int,   default=8,
                        help='Autoencoder latent dimension (default: 8)')
    parser.add_argument('--epochs',     type=int,   default=50,
                        help='Autoencoder training epochs (default: 50)')
    parser.add_argument('--n_decoys',   type=int,   default=None,
                        help='Number of decoys to generate (default: enough for target injection ratio)')
    parser.add_argument('--fallback',   action='store_true',
                        help='Skip autoencoder, use local-neighborhood fallback')
    parser.add_argument('--trials',     type=int,   default=20,
                        help='Attack simulation trials per attack type (default: 20)')
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs('models',  exist_ok=True)
    os.makedirs('outputs', exist_ok=True)

    print("\n" + "="*72)
    print("DecoyNet: Decoy Data Injection for Threat Reduction")
    print("="*72)
    print(f"Dataset        : {args.data}")
    print(f"Sample fraction: {args.sample:.2f}")
    print(f"Injection ratio: {args.inject*100:.1f}%")
    print(f"Trials/attack  : {args.trials}")

    # ═══════════════════════════════════════
    # LAYER 0: Preprocessing
    # ═══════════════════════════════════════
    data = load_and_preprocess(
        filepath    = args.data,
        sample_frac = args.sample,
        save_scaler = True,
    )

    X_train      = data['X_train']
    X_val        = data['X_val']
    X_test       = data['X_test']
    y_train      = data['y_train']
    y_val        = data['y_val']
    y_test       = data['y_test']
    feature_names= data['feature_names']

    # Use only legitimate (non-fraud) records to train decoy generator
    # Rationale: decoys should look like normal transactions, not fraud
    legit_mask   = y_train == 0
    X_train_legit = X_train[legit_mask]

    # Generate enough decoys to satisfy the injection ratio
    n_needed = int(len(X_train) * args.inject / (1 - args.inject))
    n_decoys = args.n_decoys or max(n_needed, 500)
    print(f"\nDecoys requested: {n_decoys} (needed for target ratio: {n_needed})")

    # ═══════════════════════════════════════
    # LAYER 1: Decoy Generation
    # ═══════════════════════════════════════
    decoys, quality_metrics = generate_decoys(
        X_train       = X_train_legit,
        X_val         = X_val[y_val == 0],
        n_decoys      = n_decoys,
        feature_names = feature_names,
        latent_dim    = args.latent,
        epochs        = args.epochs,
        force_fallback= args.fallback,
        save_path     = 'models/autoencoder.pt',
    )

    # save quality metrics
    quality_df = quality_metrics.get('stats_df', pd.DataFrame())
    if not quality_df.empty:
        quality_df.to_csv('outputs/decoy_quality.csv', index=False)
        print("Saved decoy quality details: outputs/decoy_quality.csv")

    # ═══════════════════════════════════════
    # LAYER 2: Injection
    # ═══════════════════════════════════════

    # Show injection density tradeoff (for paper)
    injection_density_experiment(X_train, y_train, decoys)

    X_injected, y_injected, is_decoy, zone_labels, lookup = inject_decoys(
        X_real          = X_train,
        y_real          = y_train,
        decoys          = decoys,
        injection_ratio = args.inject,
    )

    # save lookup table
    lookup.save('models/lookup_table.json')
    # save salt separately (in real system this goes to a secrets manager)
    with open('models/KEEP_SECRET_salt.txt', 'w') as f:
        f.write(lookup.salt)
    print("Saved lookup table and salt under models/.")

    # ═══════════════════════════════════════
    # LAYER 3: Detection + Baselines
    # ═══════════════════════════════════════
    results = run_full_detection_experiment(
        X_injected    = X_injected,
        y_injected    = y_injected,
        is_decoy      = is_decoy,
        zone_labels   = zone_labels,
        lookup        = lookup,
        all_decoys    = decoys,
        X_train_clean = X_train,
        y_train_clean = y_train,
        X_test_clean  = X_test,
        y_test_clean  = y_test,
        n_trials      = args.trials,
    )

    # ═══════════════════════════════════════
    # SAVE RESULTS
    # ═══════════════════════════════════════
    print("\nSaving result files...")

    # attack summary table
    attack_rows = []
    for k, v in results.items():
        if isinstance(v, dict) and 'detection_rate' in v:
            attack_rows.append({
                'attack'         : k,
                'detection_rate' : v['detection_rate'],
                'decoy_ratio'    : v['decoy_ratio'],
                'n_trials'       : v.get('n_trials', '-'),
                'zones_hit'      : str(v.get('zones_hit', {}))
            })

    results_df = pd.DataFrame(attack_rows)
    results_df.to_csv('outputs/results_summary.csv', index=False)

    # baseline comparison table
    baselines = ['baseline_rf_clean', 'baseline_rf_injected', 'baseline_iforest']
    bl_rows   = []
    for bl in baselines:
        if bl in results:
            bl_rows.append(results[bl])
    pd.DataFrame(bl_rows).to_csv('outputs/baseline_comparison.csv', index=False)

    # decoy quality summary
    quality_summary = {k: v for k, v in quality_metrics.items() if k != 'stats_df'}
    with open('outputs/quality_summary.json', 'w') as f:
        json.dump(quality_summary, f, indent=2)

    print("  outputs/results_summary.csv")
    print("  outputs/baseline_comparison.csv")
    print("  outputs/quality_summary.json")
    print("  outputs/decoy_quality.csv")

    print("\nDecoyNet run complete.\n")

    return results


if __name__ == '__main__':
    main()
