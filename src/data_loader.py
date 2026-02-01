"""
Data Loader for Gaia Stellar Remnant Detection

Handles fetching data from Gaia Archive using astroquery and local file management.
"""

import os
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from astroquery.gaia import Gaia


class GaiaDataLoader:
    """
    Handles data ingestion for the Gaia remnant detection pipeline.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_config = config.get('data', {})
        self.features_config = config.get('features', {})
        
        # Configure Gaia login or logout if needed
        # Gaia.login(...)
        
    def fetch_gaia_data(self, 
                        limit: int = 50000, 
                        max_distance_pc: float = 200,
                        use_cache: bool = True) -> pd.DataFrame:
        """
        Fetch real Gaia data using astroquery and a TAP query.
        """
        local_path = self.data_config.get('local_path', 'data/gaia_sample.parquet')
        
        if use_cache and os.path.exists(local_path):
            print(f"Loading cached data from {local_path}")
            return pd.read_parquet(local_path)
            
        print(f"Fetching {limit} stars within {max_distance_pc}pc from Gaia Archive...")
        
        # Collect columns from config
        columns = (
            self.features_config.get('required', []) + 
            self.features_config.get('optional', []) + 
            self.features_config.get('quality', []) + 
            self.features_config.get('photometry', [])
        )
        
        # Ensure source_id is present
        if 'source_id' not in columns:
            columns.insert(0, 'source_id')
            
        select_cols = ", ".join(columns)
        
        # Build ADQL query
        parallax_min = 1000.0 / max_distance_pc
        plx_snr_min = self.data_config.get('quality_filters', {}).get('parallax_over_error_min', 5.0)
        mag_max = self.data_config.get('quality_filters', {}).get('phot_g_mean_mag_max', 18.0)
        
        query = f"""
        SELECT TOP {limit} {select_cols}
        FROM gaiadr3.gaia_source
        WHERE parallax > {parallax_min}
          AND parallax_over_error > {plx_snr_min}
          AND phot_g_mean_mag < {mag_max}
        """
        
        print(f"Executing query via astroquery...")
        try:
            # Main Gaia TAP launch
            job = Gaia.launch_job(query)
            # get_results() returns an astropy table, convert to pandas
            df = job.get_results().to_pandas()
            
            # Cache the result
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            df.to_parquet(local_path)
            print(f"Saved {len(df)} rows to {local_path}.")
            return df
            
        except Exception as e:
            print(f"Error fetching from Gaia via astroquery: {e}")
            print("Falling back to synthetic data for development...")
            return self.generate_synthetic_sample(limit)

    def generate_synthetic_sample(self, n_stars: int) -> pd.DataFrame:
        """Generate synthetic Gaia-like data for pipeline testing."""
        data = {
            'source_id': np.arange(n_stars),
            'ra': np.random.uniform(0, 360, n_stars),
            'dec': np.random.uniform(-90, 90, n_stars),
            'parallax': np.abs(np.random.normal(5, 3, n_stars)),
            'parallax_error': np.abs(np.random.normal(0.5, 0.2, n_stars)),
            'pmra': np.random.normal(0, 10, n_stars),
            'pmdec': np.random.normal(0, 10, n_stars),
            'ruwe': np.abs(np.random.normal(1.0, 0.5, n_stars)),
            'astrometric_excess_noise': np.abs(np.random.normal(0.3, 1.0, n_stars)),
            'astrometric_excess_noise_sig': np.abs(np.random.normal(2.0, 5.0, n_stars)),
            'ipd_frac_multi_peak': np.abs(np.random.normal(0.05, 0.1, n_stars)),
            'ipd_gof_harmonic_amplitude': np.abs(np.random.normal(0.1, 0.2, n_stars)),
            'visibility_periods_used': np.random.randint(5, 30, n_stars),
            'non_single_star': np.random.choice([0, 1], n_stars, p=[0.9, 0.1]),
            'duplicated_source': np.zeros(n_stars, dtype=int),
            'phot_g_mean_mag': np.random.uniform(5, 18, n_stars),
            'bp_rp': np.random.uniform(0.5, 3.0, n_stars),
            'phot_bp_rp_excess_factor': np.random.uniform(1.0, 1.5, n_stars)
        }
        return pd.DataFrame(data)

    def preprocess_df(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Convert pandas DataFrame to dictionary of numpy arrays for the pipeline."""
        data_dict = {}
        for col in df.columns:
            # Convert to float64 to ensure numpy compatibility, handle object types from astropy if any
            if df[col].dtype == object:
                # Some Gaia columns like non_single_star might come as objects
                data_dict[col] = pd.to_numeric(df[col], errors='coerce').values
            else:
                data_dict[col] = df[col].values
        return data_dict
