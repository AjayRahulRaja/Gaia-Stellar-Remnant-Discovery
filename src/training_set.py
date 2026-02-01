"""
Training Set Curation for Gaia Stellar Remnant Detection

CRITICAL MODULE #2: Training Set Definition
Identifies "likely single stars" to train on the "normal" distribution
"""

import numpy as np
from typing import Dict, Tuple
from dataclasses import dataclass


@dataclass
class TrainingSetConfig:
    """Configuration for training set selection criteria"""
    
    # Primary criterion: RUWE (Renormalized Unit Weight Error)
    # ruwe ~= 1 means astrometry is consistent with single point source
    # ruwe > 1.2-1.4 suggests unresolved binary, bad fit, or other issues
    ruwe_max: float = 1.2
    
    # Secondary quality filters
    ipd_frac_multi_peak_max: float = 0.1  # PSF shouldn't show multiple peaks
    astrometric_excess_noise_max: float = 1.0  # Low unexplained noise (mas)
    parallax_over_error_min: float = 10.0  # High-quality parallax
    visibility_periods_used_min: int = 8  # Sufficient observational baseline
    
    # Use Gaia's own non-single-star flag
    exclude_nss_flag: bool = True
    exclude_duplicated_source: bool = True
    
    # Optionally exclude stars with detected acceleration
    # (indicates long-period companion)
    exclude_acceleration: bool = False
    
    # Validation split
    validation_fraction: float = 0.1
    random_seed: int = 42


class TrainingSetCurator:
    """
    Curates a training set of "likely single stars"
    
    Strategy:
    1. Start with full Gaia sample
    2. Apply quality filters (good astrometry)
    3. Apply single-star filters (low RUWE, no NSS flag, etc.)
    4. Reserve validation set
    5. Estimate contamination (how many binaries still in set)
    """
    
    def __init__(self, config: TrainingSetConfig = None):
        self.config = config or TrainingSetConfig()
        self.training_indices = None
        self.validation_indices = None
        self.metadata = {}
    
    def select_training_set(self, data: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Select training set indices based on quality criteria
        
        Args:
            data: Dictionary mapping feature names to arrays
                  Required keys: 'ruwe', 'parallax', 'parallax_error'
                  Optional: 'ipd_frac_multi_peak', 'astrometric_excess_noise',
                           'non_single_star', 'acceleration'
        
        Returns:
            training_indices: Indices of likely single stars (training)
            validation_indices: Indices for validation set
        """
        n_stars = len(data['ruwe'])
        
        # Start with all stars
        mask = np.ones(n_stars, dtype=bool)
        
        # Track reasons for exclusion (for debugging)
        exclusion_counts = {}
        
        # Filter 1: RUWE (primary single-star criterion)
        ruwe_mask = data['ruwe'] < self.config.ruwe_max
        excluded = mask & ~ruwe_mask
        exclusion_counts['high_ruwe'] = excluded.sum()
        mask &= ruwe_mask
        
        # Filter 2: Parallax quality
        if 'parallax_error' in data:
            parallax_snr = data['parallax'] / data['parallax_error']
            plx_mask = parallax_snr > self.config.parallax_over_error_min
            excluded = mask & ~plx_mask
            exclusion_counts['low_parallax_snr'] = excluded.sum()
            mask &= plx_mask
        
        # Filter 3: Multi-peak PSF fraction
        if 'ipd_frac_multi_peak' in data:
            # Handle missing values
            ipd = data['ipd_frac_multi_peak']
            ipd_mask = (ipd < self.config.ipd_frac_multi_peak_max) | np.isnan(ipd)
            excluded = mask & ~ipd_mask
            exclusion_counts['multi_peak_psf'] = excluded.sum()
            mask &= ipd_mask
        
        # Filter 4: Astrometric excess noise
        if 'astrometric_excess_noise' in data:
            aen = data['astrometric_excess_noise']
            aen_mask = (aen < self.config.astrometric_excess_noise_max) | np.isnan(aen)
            excluded = mask & ~aen_mask
            exclusion_counts['excess_noise'] = excluded.sum()
            mask &= aen_mask
        
        # Filter 5: Gaia's non-single-star flag
        if self.config.exclude_nss_flag and 'non_single_star' in data:
            nss_mask = data['non_single_star'] == 0
            excluded = mask & ~nss_mask
            exclusion_counts['nss_flag'] = excluded.sum()
            mask &= nss_mask
            
        # Filter 6: Visibility periods
        if 'visibility_periods_used' in data:
            vis_mask = data['visibility_periods_used'] >= self.config.visibility_periods_used_min
            excluded = mask & ~vis_mask
            exclusion_counts['low_visibility'] = excluded.sum()
            mask &= vis_mask

        # Filter 7: Duplicated source
        if self.config.exclude_duplicated_source and 'duplicated_source' in data:
            dup_mask = data['duplicated_source'] == 0
            excluded = mask & ~dup_mask
            exclusion_counts['duplicated_source'] = excluded.sum()
            mask &= dup_mask
        
        # Filter 8: Detected acceleration (optional)
        if self.config.exclude_acceleration and 'acceleration' in data:
            # If acceleration measurement exists and is significant
            acc = data['acceleration']
            acc_mask = np.isnan(acc) | (np.abs(acc) < 0.1)  # Threshold TBD
            excluded = mask & ~acc_mask
            exclusion_counts['acceleration'] = excluded.sum()
            mask &= acc_mask
        
        # Get candidate indices
        candidate_indices = np.where(mask)[0]
        
        # Split into training and validation
        np.random.seed(self.config.random_seed)
        n_candidates = len(candidate_indices)
        n_validation = int(n_candidates * self.config.validation_fraction)
        
        # Shuffle and split
        shuffled = candidate_indices.copy()
        np.random.shuffle(shuffled)
        
        self.validation_indices = shuffled[:n_validation]
        self.training_indices = shuffled[n_validation:]
        
        # Store metadata
        self.metadata = {
            'n_total': n_stars,
            'n_training': len(self.training_indices),
            'n_validation': len(self.validation_indices),
            'selection_fraction': (n_candidates / n_stars),
            'exclusion_counts': exclusion_counts,
            'config': self.config
        }
        
        return self.training_indices, self.validation_indices
    
    def estimate_contamination(self, data: Dict[str, np.ndarray]) -> Dict:
        """
        Estimate binary contamination in training set
        
        Uses known binary fractions and RUWE distributions to estimate
        how many binaries likely remain in the "single star" training set
        
        Returns:
            Dictionary with contamination estimates
        """
        if self.training_indices is None:
            raise ValueError("Must select training set first")
        
        # Get RUWE distribution of training set
        training_ruwe = data['ruwe'][self.training_indices]
        
        # Known binary statistics (from literature):
        # - ~50% of stars are in binary/multiple systems
        # - Of binaries, ~20-30% have RUWE > 1.2 (detectable)
        # - So ~10-15% of all stars have RUWE > 1.2
        # - Our training set has RUWE < 1.2, but some binaries slip through
        
        # Simple estimate: binaries with RUWE close to cutoff
        near_cutoff = (training_ruwe > 1.0) & (training_ruwe < self.config.ruwe_max)
        estimated_binary_fraction = near_cutoff.sum() / len(training_ruwe)
        
        # More sophisticated: use mixture model (future work)
        # For now, use empirical estimate from literature
        # Binaries with RUWE < 1.2: ~30% of all binaries (wide/face-on orbits)
        # Binary fraction overall: ~50%
        # So expected contamination: ~0.5 * 0.3 = 15%
        
        literature_contamination = 0.15  # Conservative estimate
        
        return {
            'empirical_near_cutoff': estimated_binary_fraction,
            'literature_estimate': literature_contamination,
            'recommended_estimate': literature_contamination,
            'note': 'Training set likely contains 10-20% undetected binaries'
        }
    
    def get_summary(self) -> str:
        """
        Generate human-readable summary of training set selection
        """
        if self.training_indices is None:
            return "No training set selected yet"
        
        meta = self.metadata
        
        summary = [
            "=" * 60,
            "TRAINING SET SUMMARY",
            "=" * 60,
            f"Total stars:       {meta['n_total']:,}",
            f"Training set:      {meta['n_training']:,} ({meta['selection_fraction']:.1%})",
            f"Validation set:    {meta['n_validation']:,}",
            "",
            "Selection Criteria:",
            f"  RUWE < {self.config.ruwe_max}",
            f"  Parallax S/N > {self.config.parallax_over_error_min}",
            f"  IPD multi-peak < {self.config.ipd_frac_multi_peak_max}",
            f"  Astrometric noise < {self.config.astrometric_excess_noise_max} mas",
            "",
            "Exclusion Breakdown:"
        ]
        
        for reason, count in meta['exclusion_counts'].items():
            summary.append(f"  {reason}: {count:,} stars")
        
        summary.append("=" * 60)
        
        return "\n".join(summary)


def create_binary_holdout_set(data: Dict[str, np.ndarray], 
                               training_indices: np.ndarray) -> np.ndarray:
    """
    Create a "known binary" holdout set for validation
    
    These are stars we're confident are binaries (high RUWE, NSS flag, etc.)
    We should NOT train on these, but we CAN use them to validate that
    our anomaly detection correctly identifies them as anomalous
    
    Args:
        data: Feature dictionary
        training_indices: Already-selected training indices (to exclude)
    
    Returns:
        binary_indices: Indices of likely binaries
    """
    n_stars = len(data['ruwe'])
    
    # Create mask for likely binaries
    binary_mask = np.zeros(n_stars, dtype=bool)
    
    # High RUWE (strong binary signature)
    binary_mask |= data['ruwe'] > 1.4
    
    # Gaia NSS flag
    if 'non_single_star' in data:
        binary_mask |= data['non_single_star'] > 0
    
    # High multi-peak fraction
    if 'ipd_frac_multi_peak' in data:
        ipd = data['ipd_frac_multi_peak']
        binary_mask |= (ipd > 0.2) & ~np.isnan(ipd)
    
    # Significant acceleration
    if 'acceleration' in data:
        acc = data['acceleration']
        binary_mask |= (np.abs(acc) > 0.5) & ~np.isnan(acc)
    
    # Exclude training set stars
    training_mask = np.zeros(n_stars, dtype=bool)
    training_mask[training_indices] = True
    binary_mask &= ~training_mask
    
    binary_indices = np.where(binary_mask)[0]
    
    print(f"Created binary holdout set: {len(binary_indices):,} stars")
    print(f"  (These should score high anomaly if model works correctly)")
    
    return binary_indices
