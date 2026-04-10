# Paddy Soil Analyzer

This workspace contains the data generator and generated training dataset for the Paddy Soil Analyzer project.

The upstream GitHub repository at https://github.com/CharithKalasi/Paddy-Soil-Analyzer was empty when checked, so this setup adds the minimum Python project scaffolding needed to run the generator locally.

## Contents

- dataset_data_generator.py - creates the synthetic multi-output training dataset
- KrishiLink_MultiOutput_Training_Data_v4.csv - generated dataset output
- requirements.txt - Python dependencies

## Setup

1. Create and activate a virtual environment.
2. Install dependencies with pip install -r requirements.txt.
3. Run python dataset_data_generator.py to regenerate the CSV.

## Notes

- The generator is deterministic because it uses a fixed random seed.
- The dataset is intentionally imbalanced to emphasize rare agronomic conditions.