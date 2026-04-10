# Paddy Soil Analyzer

This project trains separate machine learning models for the two analysis phases of the paddy soil workflow.

Phase 1 uses `N`, `P`, `K`, `ph`, and `EC_uS_cm` to recommend fertilizer and soil amendments. Phase 2 uses `ORP_mV` to recommend flooding water for muddy soil preparation. The rule-based `Health_Status` column is kept in the dataset for reference, but it is not used as a model input.

## Project Layout

- `dataset_data_generator.py` - regenerates the synthetic training dataset
- `KrishiLink_MultiOutput_Training_Data_v4.csv` - generated dataset
- `model_training/` - contains all training and split scripts
- `train_all_models.py` - trains all models in one run
- `predict.py` - loads the saved models and prints sample predictions
- `Models/` - stores all saved `.pkl` model files

### Training Files

- `model_training/split_training_data.py` - splits the CSV into model-specific sub-datasets
- `model_training/npk_model_training.py` - trains the NPK recommendation model
- `model_training/ph_model_training.py` - trains the pH recommendation model
- `model_training/ec_model_training.py` - trains the EC recommendation model
- `model_training/orp_model_training.py` - trains the ORP recommendation model

## Model Breakdown

### Phase 1: NPK Model

Input: `N`, `P`, `K`

Outputs: `Urea_kg_per_acre`, `DAP_kg_per_acre`, `MOP_kg_per_acre`

### Phase 1: pH Model

Input: `ph`

Outputs: `Lime_kg_per_acre`, `Gypsum_kg_per_acre`

Only one of these is normally non-zero per row, depending on soil condition.

### Phase 1: EC Model

Input: `EC_uS_cm`

Outputs: `Low_EC_Fertilizer_Boost_kg`, `Phase1_EC_Flush_Water_Liters`

These outputs are mutually exclusive in the synthetic dataset.

### Phase 2: ORP Model

Input: `ORP_mV`

Output: `Phase2_ORP_Flood_Water_Liters`

## Requirements

- Python 3.10 or newer
- `pandas`
- `numpy`
- `scikit-learn`
- `joblib`

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Training Workflow

1. Generate or confirm the CSV exists.
2. Run all model training with:

```bash
python train_all_models.py
```

3. The trained models are saved in the `Models/` folder as:
	- `Models/npk_model.pkl`
	- `Models/ph_model.pkl`
	- `Models/ec_model.pkl`
	- `Models/orp_model.pkl`

If you want to inspect the split sub-datasets first, run:

```bash
python -m model_training.split_training_data
```

## Prediction Workflow

After training, run:

```bash
python predict.py
```

The script loads all four models from `Models/` and prints example phase-wise recommendations.

## Notes

- The dataset is synthetic and intentionally imbalanced to emphasize rare agronomic conditions.
- `Health_Status` is rule-based and intentionally excluded from ML training.
- The model scripts create the `Models/` directory automatically if it does not already exist.