"""
Verification Script for Discovery Pipeline Results
"""

import pandas as pd
import numpy as np
import os

def verify_results(csv_path: str):
    print(f"Verifying results in {csv_path}...")
    
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return
        
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} candidates.")
    
    # Check for required columns
    required_cols = ['source_id', 'total_score', 'percentile_rank', 'candidate_type']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"Error: Missing columns {missing}")
        return
        
    print("\nScore Distribution:")
    print(df['total_score'].describe())
    
    print("\nCandidate Types:")
    print(df['candidate_type'].value_counts())
    
    print("\nTop 5 Most Anomalous Candidates:")
    print(df.sort_values('total_score', ascending=False).head(5)[['source_id', 'total_score', 'percentile_rank', 'candidate_type', 'feature_flags']])
    
    # Check if scores are in reasonable range [0, 1]
    # (Though total_score can exceed 1 depending on z-score normalization)
    print(f"\nMax score: {df['total_score'].max():.2f}")
    
    print("\n✓ Result verification complete.")

if __name__ == "__main__":
    verify_results("results/top_candidates.csv")
