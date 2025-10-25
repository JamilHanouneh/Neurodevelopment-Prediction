# Neurodevelopment-Prediction
A Python machine learning pipeline for predicting neurodevelopmental outcomes in infants using structural brain MRI features from the Zenodo Infant Brain MRI Dataset. Includes preprocessing, feature extraction, model training, and visualizations. Dataset not included—download from Zenodo.

# NeuroDevPredict: Predicting Infant Neurodevelopmental Outcomes from Structural Brain MRI and Myelinisation Status Using Machine Learning

## Overview

This repository contains a complete Python-based machine learning pipeline for predicting neurodevelopmental outcomes (cognitive, motor, and language scores) in infants using structural brain MRI features (T1w and T2w scans). It is inspired by research on preterm infant neurodevelopment and uses volumetric/morphometric features combined with clinical metadata.

Key features:
- Preprocessing of skull-stripped MRI scans.
- Feature extraction (volumetric, morphometric, clinical).
- Proxy score generation for neurodevelopmental outcomes based on myelinisation status.
- Training of ML models (Ridge, Random Forest, XGBoost, Gradient Boosting).
- Evaluation metrics and publication-quality visualizations.
- VSCode integration for easy GUI-based execution.

**Important:** This repo does **not** include the dataset, processed data, or results due to size and licensing considerations. Download the dataset separately.

## Dataset

The pipeline uses the [Zenodo Infant Brain MRI Dataset](https://zenodo.org/records/8055666) (12.1 GB ZIP, CC BY 4.0 license).

### How to Obtain and Prepare the Dataset
1. Download the ZIP file from: [https://zenodo.org/records/8055666](https://zenodo.org/records/8055666).
2. Extract the ZIP contents into the `data/raw/` folder in this repository. The structure should look like:
   - `data/raw/s0001/t1.nii` and `t2.nii`
   - `data/raw/s0002/t1.nii` and `t2.nii`
   - ...
   - `data/raw/s0833/t1.nii` and `t2.nii`
   - `data/raw/meta.csv`
   - `data/raw/atlas/` (with reference NIfTI files)
3. Ensure the folder names are `s0001` to `s0833` and files are in NIfTI format (.nii).
4. Citation: Turk, E. A., et al. (2023). Large dataset of infancy and early childhood brain MRIs (T1w and T2w). Zenodo. [https://doi.org/10.5281/zenodo.8055666](https://doi.org/10.5281/zenodo.8055666)

## Installation

1. Clone the repository:
2. git clone https://github.com/YOUR_USERNAME/neurodevpredict.git
cd neurodevpredict
3. Install dependencies:
4. pip install -r requirements.txt

## Usage

### Configuration
Edit `config.yaml` for paths, parameters, or limits (e.g., number of subjects to process).

### Running the Pipeline
Use VSCode for GUI execution:
- Open the project in VSCode.
- Go to the Run and Debug panel.
- Select and run configurations like "Preprocessing", "Feature Extraction", etc.

Alternatively, run via terminal:
- Preprocessing: `python src/data/preprocess.py`
- Feature Extraction: `python src/data/feature_extraction.py`
- Training: `python src/models/train.py`
- Prediction: `python src/models/predict.py`
- Visualization: `python src/visualization/plots.py`

Full pipeline (example script not included; chain them manually):
1. Download and extract dataset to `data/raw/`.
2. Run preprocessing to generate features in `data/processed/`.
3. Run feature extraction to combine into a CSV.
4. Train models (outputs to `outputs/models/` and `outputs/results/`).
5. Generate plots in `outputs/figures/`.

Expected runtime: ~2 hours for 100 subjects, ~6-8 hours for all 833 on a standard machine.

## Project Structure
- `src/`: Core Python scripts.
- `data/`: For raw (user-provided) and processed data (gitignored).
- `outputs/`: For results, models, figures (gitignored).
- `.vscode/`: Configurations for VSCode debugging.

## Contributing
Pull requests welcome! Please follow standard GitHub flow.

For questions, open an issue on GitHub.
