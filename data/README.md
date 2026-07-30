# Data Files

This folder contains the datasets required to reproduce the machine-learning analyses presented in the manuscript:

**"Spectral Fingerprints of Neutron Displacement Response Revealed by Interpretable Machine Learning"**

## Files

### ml_dataset_v8.csv
Final machine-learning dataset used for model training and evaluation. This dataset contains the extracted spectral descriptors and elemental properties used as input features for the Random Forest and multilayer perceptron (MLP) models.

### material_features.csv
Material-level features extracted from the processed neutron displacement cross-section data.

### material_damage_features.csv
Additional radiation-damage-related features used during feature engineering.

## Original DXS Data

The original neutron displacement cross-section data are publicly available from the IAEA Nuclear Data Services (DXS database).

The intermediate processed dataset (`dxs_master_dataset_all_v2_clean.csv`) can be regenerated from the publicly available IAEA DXS files using the preprocessing scripts provided in this repository, including:

- `src/batch_parse_dxs_folder_v2.py`
- `src/merge_dxs_csvs.py`
- `src/quality_control.py`

The final machine-learning dataset used in this study (`ml_dataset_v8.csv`) is included in this repository.
