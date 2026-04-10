# Paddy Soil Analyzer

This project trains and tests a two-phase paddy soil recommendation pipeline.

Phase 1 uses `N`, `P`, `K`, `ph`, and `EC_uS_cm` to recommend fertilizer and soil amendment actions. Phase 2 uses `ORP_mV` to recommend flooding water for muddy soil preparation. The `Health_Status` column is rule-based and is not used as a model input.

## Project Layout

- `KrishiLink_MultiOutput_Training_Data_v4.csv` - synthetic training dataset
- `dataset_data_generator.py` - regenerates the dataset
- `model_training/` - training and split scripts
- `train_all_models.py` - trains all four models from one entry point
- `predict.py` - loads saved models and returns predictions
- `test_orchestrator.py` - runs five end-to-end sample sensor readings
- `Models/` - stores saved `.pkl` model files

### Training Scripts

- `model_training/split_training_data.py` - splits the CSV into model-specific dataframes
- `model_training/npk_model_training.py` - trains the NPK model
- `model_training/ph_model_training.py` - trains the pH model
- `model_training/ec_model_training.py` - trains the EC model
- `model_training/orp_model_training.py` - trains the ORP model

## Model Breakdown

### Phase 1: NPK Model

- Input: `N`, `P`, `K`
- Outputs: `Urea_kg_per_acre`, `DAP_kg_per_acre`, `MOP_kg_per_acre`

### Phase 1: pH Model

- Input: `ph`
- Outputs: `Lime_kg_per_acre`, `Gypsum_kg_per_acre`

Only one of these should be returned for a given recommendation.

### Phase 1: EC Model

- Input: `EC_uS_cm`
- Outputs: `Low_EC_Fertilizer_Boost_kg`, `Phase1_EC_Flush_Water_Liters`

Only one of these should be returned for a given recommendation.

### Phase 2: ORP Model

- Input: `ORP_mV`
- Output: `Phase2_ORP_Flood_Water_Liters`

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

1. Confirm the CSV exists in the project root.
2. Run all model training:

```bash
python train_all_models.py
```

3. The trained models are saved in `Models/` as:
   - `Models/npk_model.pkl`
   - `Models/ph_model.pkl`
   - `Models/ec_model.pkl`
   - `Models/orp_model.pkl`

To inspect the sub-datasets before training, run:

```bash
python -m model_training.split_training_data
```

## Prediction Workflow

After training, run:

```bash
python predict.py
```

This loads all four models from `Models/` and prints sample phase-wise predictions.

## End-to-End Test Workflow

To simulate the full KrishiLink pipeline with five sample sensor readings, run:

```bash
python test_orchestrator.py
```

This script calls `phase1_predict` and `phase2_predict` for each case and prints readable output.

## Input Validation

`predict.py` validates sensor inputs before inference and raises a `ValueError` if any value is out of range.

Valid ranges:

- `N`: 0 to 100
- `P`: 0 to 70
- `K`: 0 to 55
- `ph`: 3.5 to 9.0
- `EC_uS_cm`: 0 to 3500
- `ORP_mV`: -350 to 350

## Notes

- The dataset is synthetic and intentionally imbalanced to emphasize rare agronomic conditions.
- `Health_Status` is derived by rules and excluded from model training.
- The model scripts create the `Models/` directory automatically if it does not already exist.