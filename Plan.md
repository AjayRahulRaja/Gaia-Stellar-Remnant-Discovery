Requirement:
The True Population of Stellar Remnants❓ The problem

We don’t know how many:

Black holes
Neutron stars
White dwarfs
exist nearby.

Many are invisible.

🧠 How Gaia helps

Detects unseen companions via star wobble
Finds high-velocity runaway stars
Identifies compact-object binaries

🆕 Unsolved questions

How many isolated black holes are in the galaxy?
What are supernova kick velocities really like?

This matters for:

Gravitational waves
Supernova physics
Galactic evolution



Big Picture Architecture (what we’re building):

Gaia DR3/DR4 data
        ↓
Feature engineering (astrometry + kinematics)
        ↓
Self-supervised / anomaly ML model
        ↓
Candidate dark-companion systems
        ↓
Mass inference + validation



Model stack—Core model:

Self-supervised Autoencoder (baseline)
OR Variational Autoencoder (VAE) (better uncertainty)

Advanced (next phase)

Graph Neural Network (GNN) for local phase-space consistency
Physics-informed loss terms

Start simple → then scale.





Step 4 — Gaia feature vector (concrete)

For each star:

Astrometry:
- ra, dec
- parallax
- pmra, pmdec
- radial_velocity (if available)

Quality / dynamics:
- astrometric_excess_noise
- ruwe
- ipd_frac_multi_peak
- non_single_star_flag
- acceleration terms (where available)

Derived:
- distance
- tangential velocity
- total space velocity
