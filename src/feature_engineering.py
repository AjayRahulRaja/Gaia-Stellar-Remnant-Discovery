"""
Feature Engineering for Gaia Stellar Remnant Detection

CRITICAL MODULE #1: Feature Normalization
Handles the wildly different scales of Gaia features (parallax in mas, velocities in km/s, etc.)
"""

import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
import pickle
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class FeatureConfig:
    """Configuration for how each feature should be normalized"""
    name: str
    normalization: str  # 'standard', 'robust', 'minmax', 'log', 'log_standard'
    required: bool = True  # If False, feature can be missing
    

# Feature groups with their normalization strategies
FEATURE_CONFIGS = [
    # Astrometry - mostly Gaussian-like with some outliers
    FeatureConfig('ra', 'standard', required=True),
    FeatureConfig('dec', 'standard', required=True),
    FeatureConfig('parallax', 'robust', required=True),  # Robust due to outliers
    FeatureConfig('pmra', 'standard', required=True),
    FeatureConfig('pmdec', 'standard', required=True),
    FeatureConfig('radial_velocity', 'standard', required=False),  # Often missing
    
    # Quality metrics - bounded or heavy-tailed
    FeatureConfig('astrometric_excess_noise', 'log_standard', required=True),  # Heavy tail
    FeatureConfig('ruwe', 'minmax', required=True),  # Bounded, ~1-10 range
    FeatureConfig('ipd_frac_multi_peak', 'minmax', required=True),  # Fraction 0-1
    
    # Derived kinematics - use log transform for heavy tails
    FeatureConfig('distance', 'log_standard', required=True),  # Distance from parallax
    FeatureConfig('tangential_velocity', 'log_standard', required=True),
    FeatureConfig('total_velocity', 'log_standard', required=False),  # Needs RV
]


class GaiaFeatureNormalizer:
    """
    Multi-strategy feature normalization for Gaia data
    
    Key insight: Different Gaia features need different normalization:
    - Positions/proper motions: StandardScaler (mostly Gaussian)
    - Quality metrics with outliers: RobustScaler
    - Bounded values: MinMaxScaler
    - Heavy-tailed distributions: Log transform + StandardScaler
    """
    
    def __init__(self):
        self.scalers: Dict[str, any] = {}
        self.feature_configs = {fc.name: fc for fc in FEATURE_CONFIGS}
        self.fitted = False
        
    def compute_derived_features(self, data: np.ndarray, 
                                 feature_names: list) -> Tuple[np.ndarray, list]:
        """
        Compute derived features from Gaia observables
        
        Args:
            data: Raw Gaia data array (N_stars, N_features)
            feature_names: List of feature names in data
            
        Returns:
            Enhanced data array with derived features, updated feature list
        """
        # Create a dictionary for easy access
        feature_dict = {name: data[:, i] for i, name in enumerate(feature_names)}
        
        derived = {}
        
        # Distance from parallax (in parsecs)
        if 'parallax' in feature_dict:
            # Only compute for positive parallax
            parallax_mas = feature_dict['parallax']
            distance = np.where(parallax_mas > 0, 1000.0 / parallax_mas, np.nan)
            derived['distance'] = distance
        
        # Tangential velocity (km/s)
        if 'parallax' in feature_dict and 'pmra' in feature_dict and 'pmdec' in feature_dict:
            parallax_arcsec = feature_dict['parallax'] / 1000.0
            pm_total = np.sqrt(feature_dict['pmra']**2 + feature_dict['pmdec']**2)
            # v_tan = 4.74 * mu * d, where mu is in arcsec/yr, d in pc
            v_tan = np.where(parallax_arcsec > 0,
                           4.74 * pm_total * (1.0 / parallax_arcsec),
                           np.nan)
            derived['tangential_velocity'] = v_tan
        
        # Total space velocity (km/s) - only if we have radial velocity
        if 'radial_velocity' in feature_dict and 'tangential_velocity' in derived:
            rv = feature_dict['radial_velocity']
            v_total = np.sqrt(derived['tangential_velocity']**2 + rv**2)
            derived['total_velocity'] = v_total
        
        # Concatenate original + derived
        if derived:
            derived_array = np.column_stack([derived[k] for k in sorted(derived.keys())])
            enhanced_data = np.column_stack([data, derived_array])
            enhanced_names = feature_names + sorted(derived.keys())
        else:
            enhanced_data = data
            enhanced_names = feature_names
            
        return enhanced_data, enhanced_names
    
    def fit(self, data: np.ndarray, feature_names: list):
        """
        Fit normalization parameters on training data (likely single stars)
        
        Args:
            data: Training data array (N_stars, N_features)
            feature_names: List of feature names
        """
        for i, name in enumerate(feature_names):
            if name not in self.feature_configs:
                continue
                
            config = self.feature_configs[name]
            feature_data = data[:, i]
            
            # Handle missing values
            valid_mask = ~np.isnan(feature_data)
            if not valid_mask.any():
                print(f"Warning: Feature {name} has no valid values, skipping")
                continue
            
            valid_data = feature_data[valid_mask].reshape(-1, 1)
            
            # Apply appropriate normalization
            if config.normalization == 'standard':
                scaler = StandardScaler()
            elif config.normalization == 'robust':
                scaler = RobustScaler()
            elif config.normalization == 'minmax':
                scaler = MinMaxScaler(feature_range=(-1, 1))
            elif config.normalization in ['log', 'log_standard']:
                # Log transform then standardize
                # Add small epsilon to avoid log(0)
                log_data = np.log(valid_data + 1e-10)
                if config.normalization == 'log_standard':
                    scaler = StandardScaler()
                    scaler.fit(log_data)
                else:
                    scaler = None
            else:
                raise ValueError(f"Unknown normalization: {config.normalization}")
            
            if scaler is not None and config.normalization not in ['log', 'log_standard']:
                scaler.fit(valid_data)
            
            self.scalers[name] = {
                'scaler': scaler,
                'config': config
            }
        
        self.fitted = True
        print(f"Fitted normalizers for {len(self.scalers)} features")
    
    def transform(self, data: np.ndarray, feature_names: list) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform features using fitted normalizers
        
        Args:
            data: Data to transform (N_stars, N_features)
            feature_names: List of feature names
            
        Returns:
            normalized_data: Normalized features
            missing_mask: Boolean mask (N_stars, N_features) indicating missing values
        """
        if not self.fitted:
            raise ValueError("Normalizer must be fitted before transform")
        
        normalized = np.zeros_like(data, dtype=np.float32)
        missing_mask = np.isnan(data)
        
        for i, name in enumerate(feature_names):
            feature_data = data[:, i].reshape(-1, 1)
            valid_mask = ~np.isnan(feature_data[:, 0])
            
            if name not in self.scalers:
                # Pass through, but fill NaNs with 0
                transformed = feature_data.copy()
                transformed[~valid_mask] = 0.0
                normalized[:, i] = transformed.flatten()
                continue
            
            scaler_info = self.scalers[name]
            scaler = scaler_info['scaler']
            config = scaler_info['config']
            
            # Transform valid data
            if config.normalization in ['log', 'log_standard']:
                transformed = np.log(feature_data + 1e-10)
                if scaler is not None and valid_mask.any():
                    transformed[valid_mask, 0] = scaler.transform(
                        transformed[valid_mask].reshape(-1, 1)
                    ).flatten()
            else:
                transformed = feature_data.copy()
                if scaler is not None and valid_mask.any():
                    transformed[valid_mask, 0] = scaler.transform(
                        feature_data[valid_mask].reshape(-1, 1)
                    ).flatten()
            
            # Fill missing values with 0 (will be masked anyway)
            transformed[~valid_mask] = 0.0
            normalized[:, i] = transformed.flatten()
        
        return normalized, missing_mask
    
    def fit_transform(self, data: np.ndarray, feature_names: list) -> Tuple[np.ndarray, np.ndarray]:
        """Fit and transform in one step"""
        self.fit(data, feature_names)
        return self.transform(data, feature_names)
    
    def save(self, filepath: str):
        """Save fitted normalizer to disk"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'scalers': self.scalers,
                'fitted': self.fitted
            }, f)
        print(f"Saved normalizer to {filepath}")
    
    def load(self, filepath: str):
        """Load fitted normalizer from disk"""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        self.scalers = state['scalers']
        self.fitted = state['fitted']
        print(f"Loaded normalizer from {filepath}")


def get_feature_statistics(data: np.ndarray, feature_names: list) -> Dict:
    """
    Compute statistics for each feature (useful for understanding distributions)
    
    Returns:
        Dictionary with statistics for each feature
    """
    stats = {}
    for i, name in enumerate(feature_names):
        feature_data = data[:, i]
        valid_mask = ~np.isnan(feature_data)
        
        if not valid_mask.any():
            stats[name] = {'missing': True}
            continue
        
        valid_data = feature_data[valid_mask]
        stats[name] = {
            'mean': np.mean(valid_data),
            'std': np.std(valid_data),
            'median': np.median(valid_data),
            'min': np.min(valid_data),
            'max': np.max(valid_data),
            'missing_frac': 1.0 - (valid_mask.sum() / len(feature_data)),
            'n_valid': valid_mask.sum()
        }
    
    return stats
