from pathlib import Path

import pandas as pd

from model_training.ec_model_training import train_ec_model
from model_training.npk_model_training import train_npk_model
from model_training.orp_model_training import train_orp_model
from model_training.ph_model_training import train_ph_model
import leakage_check
import numpy as np
import json
from shutil import copyfile


BASE_DIR = Path(__file__).resolve().parent
CSV_FILE = BASE_DIR / "KrishiLink_10k_RawUnits_Training_Data.csv"
MODEL_DIR = BASE_DIR / "Models"


def main() -> None:
    MODEL_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(CSV_FILE)
    # Support updated dataset column names from the generator (e.g. N_mg_per_kg)
    # by mapping them to the historical column names expected by training.
    if "N" not in df.columns and "N_mg_per_kg" in df.columns:
        df["N"] = df["N_mg_per_kg"]
    if "P" not in df.columns and "P_mg_per_kg" in df.columns:
        df["P"] = df["P_mg_per_kg"]
    if "K" not in df.columns and "K_mg_per_kg" in df.columns:
        df["K"] = df["K_mg_per_kg"]
    df = df.drop(columns=["Health_Status"], errors="ignore")

    npk_df = df[["N", "P", "K", "Urea_kg_per_acre", "DAP_kg_per_acre", "MOP_kg_per_acre"]].copy()
    ph_df = df[["ph", "EC_uS_cm", "Lime_kg_per_acre", "Gypsum_kg_per_acre"]].copy()
    ec_df = df[["EC_uS_cm", "Low_EC_Fertilizer_Boost_kg", "Phase1_EC_Flush_Water_Liters"]].copy()
    orp_df = df[["ORP_mV", "Phase2_ORP_Flood_Water_Liters"]].copy()

    # Run leakage check and decide action
    leakage_out = MODEL_DIR.parent / "leakage_report.json"
    leakage_check.run_check(csv_path=str(CSV_FILE), out_path=str(leakage_out))
    with open(leakage_out, 'r', encoding='utf8') as f:
        lr = json.load(f)

    # If a deterministic rule generated targets (high equal_fraction), apply noise augmentation
    recompute = lr.get('recompute_comparison', {})
    deterministic_found = False
    for k, v in recompute.items():
        if isinstance(v, dict) and v.get('equal_fraction', 0) >= 0.95:
            deterministic_found = True
            break

    saved_models = []
    if deterministic_found:
        print("Deterministic targets detected (leakage). Training on noise-augmented targets instead.")
        # Create a noisy copy of the CSV and use it for training
        noisy_csv = MODEL_DIR.parent / "KrishiLink_10k_RawUnits_Training_Data_noisy.csv"
        df_noisy = df.copy()
        # Apply small Gaussian noise to numeric target columns
        noisy_targets = ["Urea_kg_per_acre","DAP_kg_per_acre","MOP_kg_per_acre",
                         "Low_EC_Fertilizer_Boost_kg","Phase1_EC_Flush_Water_Liters",
                         "Phase2_ORP_Flood_Water_Liters","Lime_kg_per_acre","Gypsum_kg_per_acre"]
        for col in noisy_targets:
            if col in df_noisy.columns:
                std = max(1e-3, float(df_noisy[col].std()))
                noise = np.random.normal(0, 0.05 * std, size=len(df_noisy))
                df_noisy[col] = df_noisy[col] + noise
        df_noisy.to_csv(noisy_csv, index=False)

        # Recreate per-task dataframes from noisy CSV
        npk_df = df_noisy[["N", "P", "K", "Urea_kg_per_acre", "DAP_kg_per_acre", "MOP_kg_per_acre"]].copy()
        ph_df = df_noisy[["ph", "EC_uS_cm", "Lime_kg_per_acre", "Gypsum_kg_per_acre"]].copy()
        ec_df = df_noisy[["EC_uS_cm", "Low_EC_Fertilizer_Boost_kg", "Phase1_EC_Flush_Water_Liters"]].copy()
        orp_df = df_noisy[["ORP_mV", "Phase2_ORP_Flood_Water_Liters"]].copy()

        train_npk_model(npk_df)
        src = MODEL_DIR / "npk_model.pkl"
        dst = MODEL_DIR / "npk_model_noisy.pkl"
        if src.exists():
            copyfile(src, dst)
            saved_models.append("npk_model_noisy.pkl")

        train_ph_model(ph_df)
        src = MODEL_DIR / "ph_model.pkl"
        dst = MODEL_DIR / "ph_model_noisy.pkl"
        if src.exists():
            copyfile(src, dst)
            saved_models.append("ph_model_noisy.pkl")

        train_ec_model(ec_df)
        src = MODEL_DIR / "ec_model.pkl"
        dst = MODEL_DIR / "ec_model_noisy.pkl"
        if src.exists():
            copyfile(src, dst)
            saved_models.append("ec_model_noisy.pkl")

        train_orp_model(orp_df)
        src = MODEL_DIR / "orp_model.pkl"
        dst = MODEL_DIR / "orp_model_noisy.pkl"
        if src.exists():
            copyfile(src, dst)
            saved_models.append("orp_model_noisy.pkl")
        # Note action
        with open(MODEL_DIR.parent / "leakage_action.txt", 'w', encoding='utf8') as fa:
            fa.write("Detected deterministic targets; trained models on noisy targets and saved *_noisy.pkl\n")

    else:
        # Normal training path
        train_npk_model(npk_df)
        saved_models.append("npk_model.pkl")

        train_ph_model(ph_df)
        saved_models.append("ph_model.pkl")

        train_ec_model(ec_df)
        saved_models.append("ec_model.pkl")

        train_orp_model(orp_df)
        saved_models.append("orp_model.pkl")

    print("\nTraining complete. Saved models:")
    for model_file in saved_models:
        print(f"- {MODEL_DIR / model_file}")


if __name__ == "__main__":
    main()