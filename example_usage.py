"""
Example Usage: Demonstrating the three critical components

This shows how to use:
1. Feature normalization
2. Training set definition
3. Anomaly scoring + ranking

Using synthetic data for demonstration (replace with real Gaia data)
"""

import numpy as np
from src.feature_engineering import GaiaFeatureNormalizer, get_feature_statistics
from src.training_set import TrainingSetCurator, TrainingSetConfig, create_binary_holdout_set
from src.anomaly_scoring import AnomalyScorer, generate_summary_statistics


def generate_synthetic_gaia_data(n_stars=10000, binary_fraction=0.3):
    """
    Generate synthetic Gaia-like data for demonstration
    
    In reality, you'd load this from:
    - Gaia Archive TAP query
    - Local Parquet/CSV file
    - HDF5 cache
    """
    print(f"Generating {n_stars:,} synthetic stars...")
    
    # Most stars are "normal" single stars
    n_single = int(n_stars * (1 - binary_fraction))
    n_binary = n_stars - n_single
    
    # Single stars: well-behaved astrometry
    single_data = {
        'source_id': np.arange(n_single),
        'ra': np.random.uniform(0, 360, n_single),
        'dec': np.random.uniform(-90, 90, n_single),
        'parallax': np.abs(np.random.normal(5, 3, n_single)),  # ~200pc typical
        'parallax_error': np.abs(np.random.normal(0.5, 0.2, n_single)),
        'pmra': np.random.normal(0, 10, n_single),
        'pmdec': np.random.normal(0, 10, n_single),
        'radial_velocity': np.random.normal(0, 30, n_single),
        'ruwe': np.abs(np.random.normal(1.0, 0.1, n_single)),  # Good RUWE
        'astrometric_excess_noise': np.abs(np.random.normal(0.3, 0.2, n_single)),
        'ipd_frac_multi_peak': np.abs(np.random.normal(0.05, 0.03, n_single)),
        'non_single_star': np.zeros(n_single, dtype=int)
    }
    
    # Binary stars: disturbed astrometry
    binary_data = {
        'source_id': np.arange(n_single, n_stars),
        'ra': np.random.uniform(0, 360, n_binary),
        'dec': np.random.uniform(-90, 90, n_binary),
        'parallax': np.abs(np.random.normal(5, 3, n_binary)),
        'parallax_error': np.abs(np.random.normal(0.8, 0.3, n_binary)),  # Worse errors
        'pmra': np.random.normal(0, 15, n_binary),  # More scatter
        'pmdec': np.random.normal(0, 15, n_binary),
        'radial_velocity': np.random.normal(0, 50, n_binary),  # More velocity scatter
        'ruwe': np.abs(np.random.normal(1.8, 0.6, n_binary)),  # High RUWE!
        'astrometric_excess_noise': np.abs(np.random.normal(1.5, 0.8, n_binary)),
        'ipd_frac_multi_peak': np.abs(np.random.normal(0.3, 0.2, n_binary)),
        'non_single_star': np.random.choice([0, 1, 2], n_binary, p=[0.3, 0.5, 0.2])
    }
    
    # Combine
    data = {}
    for key in single_data.keys():
        data[key] = np.concatenate([single_data[key], binary_data[key]])
    
    # Add some missing radial velocities (realistic)
    missing_rv_mask = np.random.random(n_stars) < 0.3
    data['radial_velocity'][missing_rv_mask] = np.nan
    
    print(f"  {n_single:,} single stars, {n_binary:,} binaries")
    
    return data


def main():
    """
    Demonstrate the full pipeline with the three critical components
    """
    print("=" * 70)
    print("GAIA STELLAR REMNANT DETECTION - EXAMPLE USAGE")
    print("=" * 70)
    print()
    
    # =========================================================================
    # STEP 1: Generate/Load Data
    # =========================================================================
    print("STEP 1: Loading data...")
    data = generate_synthetic_gaia_data(n_stars=10000, binary_fraction=0.3)
    
    # =========================================================================
    # STEP 2: CRITICAL COMPONENT #2 - Define Training Set
    # =========================================================================
    print("\nSTEP 2: Defining training set (likely single stars)...")
    print("-" * 70)
    
    config = TrainingSetConfig(
        ruwe_max=1.2,
        ipd_frac_multi_peak_max=0.1,
        astrometric_excess_noise_max=1.0,
        parallax_over_error_min=10.0,
        exclude_nss_flag=True,
        validation_fraction=0.1
    )
    
    curator = TrainingSetCurator(config)
    train_idx, val_idx = curator.select_training_set(data)
    
    print(curator.get_summary())
    
    # Estimate contamination
    contamination = curator.estimate_contamination(data)
    print(f"\nEstimated binary contamination: {contamination['recommended_estimate']:.1%}")
    print(f"  ({contamination['note']})")
    
    # Create binary holdout for validation
    binary_idx = create_binary_holdout_set(data, train_idx)
    
    # =========================================================================
    # STEP 3: CRITICAL COMPONENT #1 - Feature Normalization
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 3: Feature engineering and normalization...")
    print("-" * 70)
    
    # Stack features into array
    feature_names = [
        'ra', 'dec', 'parallax', 'pmra', 'pmdec', 'radial_velocity',
        'ruwe', 'astrometric_excess_noise', 'ipd_frac_multi_peak'
    ]
    
    data_array = np.column_stack([data[name] for name in feature_names])
    
    # Initialize normalizer
    normalizer = GaiaFeatureNormalizer()
    
    # Compute derived features (distance, velocities)
    enhanced_data, enhanced_names = normalizer.compute_derived_features(
        data_array, feature_names
    )
    
    print(f"Features before: {len(feature_names)}")
    print(f"Features after:  {len(enhanced_names)} (added derived features)")
    
    # Get statistics before normalization
    stats_before = get_feature_statistics(enhanced_data, enhanced_names)
    print("\nFeature ranges BEFORE normalization:")
    for name in ['parallax', 'ruwe', 'tangential_velocity']:
        if name in stats_before:
            s = stats_before[name]
            print(f"  {name:25s}: [{s['min']:8.2f}, {s['max']:8.2f}]")
    
    # Fit normalizer on TRAINING SET ONLY (critical!)
    train_data = enhanced_data[train_idx]
    normalizer.fit(train_data, enhanced_names)
    
    # Transform all data
    normalized_data, missing_mask = normalizer.transform(enhanced_data, enhanced_names)
    
    stats_after = get_feature_statistics(normalized_data, enhanced_names)
    print("\nFeature ranges AFTER normalization:")
    for name in ['parallax', 'ruwe', 'tangential_velocity']:
        if name in stats_after:
            s = stats_after[name]
            print(f"  {name:25s}: [{s['min']:8.2f}, {s['max']:8.2f}]")
    
    print("\n✓ All features now on comparable scales!")
    
    # =========================================================================
    # STEP 4: Train Autoencoder (placeholder - would use real model)
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 4: Training autoencoder...")
    print("-" * 70)
    print("(Using fake reconstructions for demo - replace with real model)")
    
    # In real implementation:
    # model = build_autoencoder(input_dim=len(enhanced_names))
    # model.fit(normalized_data[train_idx], ...)
    # reconstructions, latents = model.predict(normalized_data)
    
    # For demo: generate fake reconstructions
    # Single stars: good reconstruction (low error)
    # Binaries: bad reconstruction (high error)
    reconstructions = normalized_data.copy()
    
    # Add reconstruction error (small for singles, large for binaries)
    for i in range(len(reconstructions)):
        if data['ruwe'][i] < 1.2:
            # Single star: small error
            reconstructions[i] += np.random.normal(0, 0.1, len(enhanced_names))
        else:
            # Binary: large error
            reconstructions[i] += np.random.normal(0, 0.5, len(enhanced_names))
    
    # Fake latent representations (8D)
    latents = np.random.randn(len(normalized_data), 8)
    
    # =========================================================================
    # STEP 5: CRITICAL COMPONENT #3 - Anomaly Scoring & Ranking
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 5: Computing anomaly scores and ranking candidates...")
    print("-" * 70)
    
    scorer = AnomalyScorer(enhanced_names)
    
    # Fit scorer on training set
    scorer.fit(
        reconstructions[train_idx],
        normalized_data[train_idx],
        latents[train_idx]
    )
    
    # Score all stars
    all_scores = scorer.score_dataset(
        normalized_data,
        reconstructions,
        latents,
        source_ids=data['source_id']
    )
    
    # Print summary
    print(generate_summary_statistics(all_scores))
    
    # Get top candidates
    top_100 = scorer.get_top_candidates(all_scores, n=100, min_percentile=99.0)
    
    print("\nTOP 10 CANDIDATES:")
    print("-" * 70)
    print(f"{'Source ID':>12} {'Score':>8} {'%ile':>6} {'Type':>20} {'Flags'}")
    print("-" * 70)
    
    for i, score in enumerate(top_100[:10]):
        flags = ', '.join(score.feature_flags[:3]) if score.feature_flags else 'none'
        print(f"{score.source_id:12d} {score.total_score:8.4f} "
              f"{score.percentile_rank:6.2f} {score.candidate_type:>20} {flags}")
    
    # Validation: check if known binaries are recovered
    binary_scores = [s for s in all_scores if s.source_id in data['source_id'][binary_idx]]
    if binary_scores:
        binary_percentiles = [s.percentile_rank for s in binary_scores]
        recall_99 = sum(1 for p in binary_percentiles if p >= 99.0) / len(binary_percentiles)
        print(f"\n✓ Binary recall@99th: {recall_99:.1%} of known binaries in top 1%")
    
    # Export candidates
    scorer.export_candidates(all_scores, 'demo_candidates.csv', top_n=1000)
    
    print("\n" + "=" * 70)
    print("DEMO COMPLETE!")
    print("=" * 70)
    print("\nThe three critical components are working:")
    print("  ✓ Feature normalization handles different scales")
    print("  ✓ Training set defined on likely single stars")
    print("  ✓ Clean scoring + ranking produces candidates")
    print("\nNext steps:")
    print("  1. Replace synthetic data with real Gaia query")
    print("  2. Implement real autoencoder (see autoencoder.py)")
    print("  3. Validate top candidates against known catalogs")
    print("  4. Run physics inference (orbit fitting, mass estimation)")


if __name__ == "__main__":
    main()
