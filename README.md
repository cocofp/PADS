# PSADS: ADS-Optimized Zone-Based SMOGN Data Generator

This repository provides the implementation of **PSADS (Partitioned SMOGN with Adaptive Data Selection)** for synthetic data generation in small-sample, cross-region maize yield prediction.

The script refines zone-based SMOGN-generated candidate samples by selecting synthetic samples with higher authenticity, consistency, and diversity.

## Main Features

- **Zone-level adaptive data selection**  
  Synthetic samples are evaluated and selected independently within each Agricultural Resource and Environment Zone (AREZ), preserving zone-specific feature–yield relationships.

- **Reference ensemble for sample-quality evaluation**  
  A heterogeneous reference ensemble consisting of `KNN`, `ElasticNet`, and `LightGBM` is used to evaluate candidate synthetic samples.

- **Authenticity and consistency scoring**  
  Candidate samples are scored using:
  - authenticity score: \(A_i\)
  - consistency score: \(C_i\)
  - integrated score: \(S_i = A_i \times C_i\)

- **Diversity-aware selection**  
  A K-center greedy strategy with adaptive Top-K selection is used to improve feature-space coverage among selected synthetic samples.

- **Multiple seeds and augmentation factors**  
  The script supports multiple random seeds, data splits, and augmentation ratios from 2× to 10×.

- **Intermediate-file saving**  
  Zone-level temporary files are saved during processing to reduce memory pressure and support interrupted runs.

## Requirements

```bash
pip install pandas numpy scikit-learn scipy lightgbm
Usage

Before running the script, modify the path settings in PSADS.py according to your local directory structure:

ROOT_DIR = Path(r"your/project/path")

Then run:

python PSADS.py
Input Data

The script requires:

real training data split by AREZ;
zone-based SMOGN candidate synthetic datasets;
predictor variables including monthly GCVI and meteorological features.

The 36 input features include:

gcvi_4 to gcvi_9
tmax_4 to tmax_9
tmin_4 to tmin_9
precip_4 to precip_9
rad_4 to rad_9
vpd_4 to vpd_9
Output

The script outputs ADS-refined augmented datasets for different:

data splits;
random seeds;
augmentation factors.

The output files are saved in the directory specified by ADS_OUTPUT_DIR.

Note

The reference ensemble is used only for synthetic sample-quality evaluation in the ADS stage. Downstream prediction models such as RFR, XGB, Ridge, and MLP should be trained independently.

Data Availability

The data supporting this study are available at:

https://doi.org/10.6084/m9.figshare.31417913
Citation

If you use this code, please cite the associated paper:

[Add full citation after publication]
License

This repository is released for academic research purposes.
