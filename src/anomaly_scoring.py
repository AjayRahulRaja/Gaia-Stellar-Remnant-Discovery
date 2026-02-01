"""
Anomaly Scoring and Ranking for Gaia Stellar Remnant Detection

CRITICAL MODULE #3: Clean Scoring + Ranking
Converts autoencoder outputs into actionable candidate lists
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from scipy import stats


@dataclass
class AnomalyScore:
    """Container for a star's anomaly scores"""
    source_id: int
    total_score: float  # Composite anomaly score (0-1, higher = more anomalous)
    percentile_rank: float  # Percentile ranking (99.9 = top 0.1%)
    reconstruction_error: Dict[str, float]  # Per-feature errors
    latent_distance: float  # Mahalanobis distance in latent space
    candidate_type: str  # Classification of anomaly type
    feature_flags: List[str]  # Which features are most anomalous


class AnomalyScorer:
    """
    Multi-metric anomaly scoring system
    
    Combines several signals:
    1. Reconstruction error (how well autoencoder reproduces the star)
    2. Per-feature errors (which features are anomalous)
    3. Latent space distance (how far from "normal" cluster)
    4. Feature-specific thresholds (e.g., extreme velocities)
    """
    
    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names
        
        # Will be fit on training set
        self.reconstruction_stats = None
        self.latent_stats = None
        
    def fit(self, training_reconstructions: np.ndarray,
            training_originals: np.ndarray,
            training_latent: np.ndarray):
        """
        Fit anomaly scoring parameters on training set
        
        Learns the "normal" distribution of:
        - Reconstruction errors per feature
        - Latent space position
        
        Args:
            training_reconstructions: Autoencoder outputs for training set
            training_originals: Original (normalized) features for training set
            training_latent: Latent representations for training set
        """
        # Compute reconstruction errors for training set
        reconstruction_errors = np.abs(training_reconstructions - training_originals)
        
        # Per-feature statistics
        self.reconstruction_stats = {}
        for i, name in enumerate(self.feature_names):
            errors = reconstruction_errors[:, i]
            # Use NaN-safe stats
            self.reconstruction_stats[name] = {
                'mean': np.nanmean(errors) if not np.isnan(errors).all() else 0.0,
                'std': np.nanstd(errors) if not np.isnan(errors).all() else 1.0,
                'median': np.nanmedian(errors) if not np.isnan(errors).all() else 0.0,
                'q95': np.nanpercentile(errors, 95) if not np.isnan(errors).all() else 0.0,
                'q99': np.nanpercentile(errors, 99) if not np.isnan(errors).all() else 0.0
            }
        
        # Latent space statistics (for Mahalanobis distance)
        latent_mean = np.mean(training_latent, axis=0)
        latent_cov = np.cov(training_latent.T)
        
        # Add small regularization to avoid singular matrix
        latent_cov += np.eye(latent_cov.shape[0]) * 1e-6
        
        self.latent_stats = {
            'mean': latent_mean,
            'cov': latent_cov,
            'cov_inv': np.linalg.inv(latent_cov)
        }
        
        print(f"Fitted anomaly scorer on {len(training_originals):,} training stars")
    
    def compute_reconstruction_score(self, original: np.ndarray,
                                     reconstruction: np.ndarray) -> Tuple[float, Dict]:
        """
        Compute anomaly score from reconstruction error
        
        Returns:
            total_score: Overall reconstruction anomaly (0-1)
            feature_errors: Per-feature normalized errors
        """
        errors = np.abs(reconstruction - original)
        
        # Normalize each feature error by training statistics
        normalized_errors = {}
        z_scores = []
        
        for i, name in enumerate(self.feature_names):
            if name not in self.reconstruction_stats:
                continue
            
            stats = self.reconstruction_stats[name]
            # Handle NaN in error
            err = errors[i]
            if np.isnan(err):
                z_scores.append(0.0)  # Neutral score for missing features
                normalized_errors[name] = 0.0
            else:
                # Z-score: how many std deviations from training mean
                z_score = (err - stats['mean']) / (stats['std'] + 1e-10)
                z_scores.append(z_score)
                normalized_errors[name] = float(err)
        
        # Total score: average z-score, clipped and normalized to [0, 1]
        # Use nanmean just in case, though we handled it above
        if not z_scores:
            return 0.0, normalized_errors
            
        mean_z = np.nanmean(z_scores)
        total_score = np.clip(mean_z / 5.0, 0, 1)  # z=5 maps to score=1
        
        return total_score, normalized_errors
    
    def compute_latent_score(self, latent: np.ndarray) -> float:
        """
        Compute Mahalanobis distance in latent space
        
        Measures how far this star is from the "normal" cluster
        """
        if self.latent_stats is None:
            return 0.0
        
        # Mahalanobis distance
        diff = latent - self.latent_stats['mean']
        mahal_dist = np.sqrt(diff @ self.latent_stats['cov_inv'] @ diff)
        
        # Normalize: chi-squared distribution with k degrees of freedom
        # 95th percentile of chi2(k=8) ≈ 15.5, 99th ≈ 20
        k = len(latent)
        score = np.clip(mahal_dist / np.sqrt(2 * k), 0, 1)
        
        return score
    
    def classify_anomaly_type(self, feature_errors: Dict[str, float]) -> Tuple[str, List[str]]:
        """
        Classify what type of anomaly this is
        
        Returns:
            category: Primary anomaly type
            flags: List of specific anomalous features
        """
        flags = []
        
        # Define feature groups
        astrometric_features = {'parallax', 'pmra', 'pmdec', 'ra', 'dec'}
        kinematic_features = {'radial_velocity', 'tangential_velocity', 'total_velocity'}
        quality_features = {'ruwe', 'astrometric_excess_noise', 'ipd_frac_multi_peak'}
        
        # Check which features are highly anomalous (> 95th percentile)
        astrometric_anomaly = False
        kinematic_anomaly = False
        quality_anomaly = False
        
        for name, error in feature_errors.items():
            if name not in self.reconstruction_stats:
                continue
            
            threshold = self.reconstruction_stats[name]['q95']
            
            if error > threshold:
                flags.append(name)
                
                if name in astrometric_features:
                    astrometric_anomaly = True
                elif name in kinematic_features:
                    kinematic_anomaly = True
                elif name in quality_features:
                    quality_anomaly = True
        
        # Determine primary category
        if quality_anomaly and astrometric_anomaly:
            category = 'astrometric_binary'  # Likely binary causing astrometric issues
        elif kinematic_anomaly:
            category = 'kinematic_outlier'  # High velocity, runaway star?
        elif astrometric_anomaly:
            category = 'astrometric_outlier'  # Unusual position/motion
        elif quality_anomaly:
            category = 'quality_issue'  # May be instrumental, not physical
        else:
            category = 'general_outlier'
        
        return category, flags
    
    def score_star(self, original: np.ndarray, 
                   reconstruction: np.ndarray,
                   latent: np.ndarray,
                   source_id: int = None) -> AnomalyScore:
        """
        Compute comprehensive anomaly score for a single star
        
        Args:
            original: Original normalized features
            reconstruction: Autoencoder reconstruction
            latent: Latent representation
            source_id: Gaia source ID (optional)
        
        Returns:
            AnomalyScore object with all metrics
        """
        # Reconstruction-based score
        recon_score, feature_errors = self.compute_reconstruction_score(
            original, reconstruction
        )
        
        # Latent space score
        latent_score = self.compute_latent_score(latent)
        
        # Combine scores (weighted average)
        # Give more weight to reconstruction (more interpretable)
        total_score = 0.7 * recon_score + 0.3 * latent_score
        
        # Classify anomaly type
        category, flags = self.classify_anomaly_type(feature_errors)
        
        return AnomalyScore(
            source_id=source_id,
            total_score=total_score,
            percentile_rank=0.0,  # Will be computed after scoring all stars
            reconstruction_error=feature_errors,
            latent_distance=latent_score,
            candidate_type=category,
            feature_flags=flags
        )
    
    def score_dataset(self, originals: np.ndarray,
                      reconstructions: np.ndarray,
                      latents: np.ndarray,
                      source_ids: np.ndarray = None) -> List[AnomalyScore]:
        """
        Score entire dataset and compute percentile ranks
        
        Returns:
            List of AnomalyScore objects, sorted by total_score (descending)
        """
        n_stars = len(originals)
        
        if source_ids is None:
            source_ids = np.arange(n_stars)
        
        # Score each star
        scores = []
        for i in range(n_stars):
            score = self.score_star(
                originals[i],
                reconstructions[i],
                latents[i],
                source_id=source_ids[i]
            )
            scores.append(score)
        
        # Compute percentile ranks
        total_scores = np.array([s.total_score for s in scores])
        percentiles = stats.rankdata(total_scores, method='average') / len(total_scores) * 100
        
        for i, score in enumerate(scores):
            score.percentile_rank = percentiles[i]
        
        # Sort by total score (highest first)
        scores.sort(key=lambda s: s.total_score, reverse=True)
        
        return scores
    
    def get_top_candidates(self, scores: List[AnomalyScore],
                          n: int = 100,
                          min_percentile: float = 99.0,
                          category_filter: str = None) -> List[AnomalyScore]:
        """
        Get top N candidates with optional filtering
        
        Args:
            scores: List of all anomaly scores
            n: Number of candidates to return
            min_percentile: Minimum percentile rank (e.g., 99 = top 1%)
            category_filter: Only return this category (e.g., 'kinematic_outlier')
        
        Returns:
            Filtered and limited list of top candidates
        """
        # Filter by percentile
        candidates = [s for s in scores if s.percentile_rank >= min_percentile]
        
        # Filter by category if specified
        if category_filter:
            candidates = [s for s in candidates if s.candidate_type == category_filter]
        
        # Limit to top N
        return candidates[:n]
    
    def export_candidates(self, scores: List[AnomalyScore],
                         filepath: str,
                         top_n: int = 1000):
        """
        Export top candidates to CSV for further analysis
        
        Args:
            scores: List of anomaly scores
            filepath: Output CSV path
            top_n: Number of top candidates to export
        """
        import csv
        
        top_scores = scores[:top_n]
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            header = [
                'source_id', 'total_score', 'percentile_rank',
                'latent_distance', 'candidate_type', 'feature_flags'
            ]
            # Add per-feature errors
            header.extend([f'error_{name}' for name in self.feature_names])
            writer.writerow(header)
            
            # Data
            for score in top_scores:
                row = [
                    score.source_id,
                    f'{score.total_score:.4f}',
                    f'{score.percentile_rank:.2f}',
                    f'{score.latent_distance:.4f}',
                    score.candidate_type,
                    '|'.join(score.feature_flags)
                ]
                
                # Add per-feature errors
                for name in self.feature_names:
                    error = score.reconstruction_error.get(name, 0.0)
                    row.append(f'{error:.4f}')
                
                writer.writerow(row)
        
        print(f"Exported {len(top_scores)} candidates to {filepath}")


def generate_summary_statistics(scores: List[AnomalyScore]) -> str:
    """
    Generate human-readable summary of anomaly detection results
    """
    n_total = len(scores)
    
    # Category breakdown
    category_counts = {}
    for score in scores:
        cat = score.candidate_type
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    # Top percentile counts
    top_1pct = sum(1 for s in scores if s.percentile_rank >= 99.0)
    top_5pct = sum(1 for s in scores if s.percentile_rank >= 95.0)
    top_10pct = sum(1 for s in scores if s.percentile_rank >= 90.0)
    
    summary = [
        "=" * 60,
        "ANOMALY DETECTION SUMMARY",
        "=" * 60,
        f"Total stars scored: {n_total:,}",
        "",
        "Percentile Distribution:",
        f"  Top 1%:  {top_1pct:,} stars",
        f"  Top 5%:  {top_5pct:,} stars",
        f"  Top 10%: {top_10pct:,} stars",
        "",
        "Anomaly Type Breakdown:",
    ]
    
    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        pct = count / n_total * 100
        summary.append(f"  {cat}: {count:,} ({pct:.1f}%)")
    
    summary.append("=" * 60)
    
    return "\n".join(summary)
