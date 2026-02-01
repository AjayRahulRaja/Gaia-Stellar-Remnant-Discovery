"""
End-to-End Pipeline for Gaia Stellar Remnant Detection

Orchestrates:
1. Data Loading
2. Training Set Curation
3. Feature Engineering & Normalization
4. Model Training
5. Anomaly Scoring & Ranking
6. Results Export
"""

import os
import yaml
import numpy as np
import pandas as pd
from typing import Dict, Any

from src.data_loader import GaiaDataLoader
from src.training_set import TrainingSetCurator, TrainingSetConfig
from src.feature_engineering import GaiaFeatureNormalizer
from src.autoencoder import GaiaAutoencoder
from src.anomaly_scoring import AnomalyScorer


class DiscoveryPipeline:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.loader = GaiaDataLoader(self.config)
        self.curator = TrainingSetCurator(TrainingSetConfig(**self.config.get('training_set', {})))
        self.normalizer = GaiaFeatureNormalizer()
        self.model = None
        self.scorer = None
        
        # Ensure output directory exists
        os.makedirs(self.config.get('output', {}).get('output_dir', 'results/'), exist_ok=True)
        os.makedirs(os.path.join(self.config.get('output', {}).get('output_dir', 'results/'), 'models'), exist_ok=True)

    def run(self, mode: str = 'full'):
        """
        Run the pipeline.
        Modes: 'full', 'train_only', 'score_only'
        """
        print(f"Starting Discovery Pipeline [Mode: {mode}]...")

        # 1. Load Data
        df = self.loader.fetch_gaia_data(limit=10000)
        if df is None:
            print("No real data fetched. Please ensure you have internet or local cache.")
            return
            
        data_dict = self.loader.preprocess_df(df)
        
        # 2. Training Set Selection
        print("Curating training set...")
        train_idx, val_idx = self.curator.select_training_set(data_dict)
        print(self.curator.get_summary())

        # 3. Feature Engineering & Normalization
        print("Normalizing features...")
        # Get feature names from config groupings
        feature_names = (
            self.config['features']['required'] + 
            self.config['features'].get('optional', []) + 
            self.config['features'].get('quality', [])
        )
        # Filter to only those present in the dataframe
        feature_names = [f for f in feature_names if f in df.columns]
        
        # Convert to numpy array
        data_array = np.column_stack([data_dict[name] for name in feature_names])
        
        # Compute derived features
        enhanced_data, enhanced_names = self.normalizer.compute_derived_features(data_array, feature_names)
        
        # Fit normalizer on training set
        self.normalizer.fit(enhanced_data[train_idx], enhanced_names)
        self.normalizer.save(self.config['output']['normalizer_save_path'])
        
        # Transform all
        normalized_data, missing_mask = self.normalizer.transform(enhanced_data, enhanced_names)

        # 4. Model Training
        input_dim = normalized_data.shape[1]
        model_cfg = self.config['model']
        
        self.model = GaiaAutoencoder(
            input_dim=input_dim,
            encoder_dims=model_cfg['encoder_dims'],
            decoder_dims=model_cfg['decoder_dims'],
            latent_dim=model_cfg['latent_dim'],
            learning_rate=model_cfg['learning_rate'],
            dropout_rate=model_cfg['dropout_rate']
        )

        if mode in ['full', 'train_only']:
            print("Training Autoencoder...")
            history = self.model.train(
                normalized_data[train_idx],
                normalized_data[val_idx],
                epochs=model_cfg['epochs'],
                batch_size=model_cfg['batch_size']
            )
            self.model.save(self.config['output']['model_save_path'])
            print("Model saved.")

        # 5. Anomaly Scoring
        if mode in ['full', 'score_only']:
            print("Scoring candidates...")
            reconstructions, latents = self.model.predict(normalized_data)
            
            self.scorer = AnomalyScorer(enhanced_names)
            # Fit scorer stats on training set
            self.scorer.fit(
                reconstructions[train_idx],
                normalized_data[train_idx],
                latents[train_idx]
            )
            
            # Score everything
            all_scores = self.scorer.score_dataset(
                normalized_data,
                reconstructions,
                latents,
                source_ids=data_dict['source_id']
            )
            
            # 6. Export Results
            self.scorer.export_candidates(
                all_scores, 
                self.config['output']['candidates_csv'],
                top_n=self.config['scoring']['top_n_export']
            )
            
            # 7. Export for 3D Visualization
            self.export_viz_data(df, all_scores)
            
            print("Pipeline Complete.")
            print(f"Top candidates exported to {self.config['output']['candidates_csv']}")

    def export_viz_data(self, df: pd.DataFrame, all_scores: list):
        """Export a JSON file optimized for 3D visualization"""
        import json
        
        print("Exporting data for 3D visualization...")
        
        # Map IDs to scores for easy lookup
        score_map = {s.source_id: s for s in all_scores}
        
        viz_stars = []
        
        # Take all top candidates plus a random sample of normal stars
        top_ids = set([s.source_id for s in all_scores[:self.config['scoring']['top_n_export']]])
        
        # Select stars to include
        mask = df['source_id'].isin(top_ids)
        top_df = df[mask]
        
        # Sample non-top stars for context
        other_df = df[~mask].sample(n=min(len(df)-len(top_df), 5000), random_state=42)
        
        combined_df = pd.concat([top_df, other_df])
        
        for _, star in combined_df.iterrows():
            sid = int(star['source_id'])
            s_obj = score_map.get(sid)
            
            if s_obj is None:
                continue
                
            # Basic properties
            viz_star = {
                'id': sid,
                'ra': float(star['ra']),
                'dec': float(star['dec']),
                'parallax': float(star['parallax']),
                'score': float(s_obj.total_score),
                'type': s_obj.candidate_type,
                'is_candidate': sid in top_ids
            }
            viz_stars.append(viz_star)
            
        output_path = os.path.join(self.config.get('output', {}).get('output_dir', 'results/'), 'viz_data.json')
        with open(output_path, 'w') as f:
            json.dump(viz_stars, f)
            
        print(f"Visualization data exported to {output_path}")


if __name__ == "__main__":
    pipeline = DiscoveryPipeline("config.yaml")
    pipeline.run(mode='full')
