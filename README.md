# Gaia Stellar Remnant Detection System

**A self-supervised machine learning framework that uses Gaia astrometry to infer the hidden population of stellar-mass black holes and neutron stars by detecting gravitational anomalies in stellar motion.**

## 🎯 Project Goal

Detect candidate dark-companion systems (black holes, neutron stars, white dwarfs) in Gaia data by learning the "normal" distribution of single stars and flagging outliers.

## 🏗️ Project Structure

The project is organized into a modular structure:

```
.
├── Plan.md                      # Project vision
├── config.yaml                  # Centralized configuration
├── pipeline.py                  # 🚀 Main entry point
├── example_usage.py             # Demo with synthetic data
├── verify_pipeline.py           # Result verification utility
├── src/                         # Core Logic
│   ├── data_loader.py           # Gaia Archive interface (astroquery)
│   ├── feature_engineering.py    # Normalization & derived features
│   ├── training_set.py          # Training set curation (likely singles)
│   ├── autoencoder.py           # ML Model (Robust scikit-learn implementation)
│   └── anomaly_scoring.py       # Candidate detection & ranking
└── results/                     # Models and candidate catalogs
```

## 🔑 Key Features

### 1. Robust Data Loading (`src/data_loader.py`)
Uses `astroquery` to reliably fetch data from the Gaia Archive. Includes local caching (`data/`) and synthetic data fallbacks for development.

### 2. Multi-Strategy Normalization (`src/feature_engineering.py`)
Handles Gaia's heterogeneous features (parallax, velocities, quality flags) using 4 different scaling strategies matched to their distributions.

### 3. Smart Training Set (`src/training_set.py`)
Automatically curates a training set of "likely single stars" using RUWE (<1.2) and other quality metrics, ensuring the model learns only the baseline stellar manifold.

### 4. Robust Autoencoder (`src/autoencoder.py`)
Uses a scikit-learn MLP-based Autoencoder for maximum compatibility. It handles missing values (NaNs) via internal imprinting and is designed to work even if local DL libraries (TF/PyTorch) are unavailable.

### 5. Multi-Metric Scoring (`src/anomaly_scoring.py`)
Combines reconstruction error and percentile ranking to identify the top 1%, 0.1%, etc., of anomalous stars.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install numpy scipy scikit-learn pandas pyyaml astroquery astropy
```

### 2. Run the Main Discovery Pipeline
This script will fetch real Gaia data, train the model on likely singles, and export top candidates.
```bash
python pipeline.py
```

### 3. Launch the 3D Discovery Map
Visualize the findings in an interactive 3D universe.
```bash
python serve_viz.py
```
Open [http://localhost:8000/viz/](http://localhost:8000/viz/) to explore.

## 📊 Pipeline Modes

You can run the pipeline in different modes by editing the `main` block in `pipeline.py`:
- `full`: Complete flow (load -> train -> score)
- `train_only`: Just fit the normalizers and model
- `score_only`: Load existing models and score a new batch of data

---

**Ready to find some black holes? 🕳️⭐**
Run `python pipeline.py` to start the discovery process!
