from pathlib import Path

import pandas as pd

from model_training.ec_model_training import train_ec_model
from model_training.npk_model_training import train_npk_model
from model_training.orp_model_training import train_orp_model
from model_training.ph_model_training import train_ph_model


BASE_DIR = Path(__file__).resolve().parent
CSV_FILE = BASE_DIR / "KrishiLink_MultiOutput_Training_Data_v4.csv"
MODEL_DIR = BASE_DIR / "Models"


def main() -> None:
    MODEL_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(CSV_FILE)
    df = df.drop(columns=["Health_Status"], errors="ignore")

    npk_df = df[["N", "P", "K", "Urea_kg_per_acre", "DAP_kg_per_acre", "MOP_kg_per_acre"]].copy()
    ph_df = df[["ph", "Lime_kg_per_acre", "Gypsum_kg_per_acre"]].copy()
    ec_df = df[["EC_uS_cm", "Low_EC_Fertilizer_Boost_kg", "Phase1_EC_Flush_Water_Liters"]].copy()
    orp_df = df[["ORP_mV", "Phase2_ORP_Flood_Water_Liters"]].copy()

    saved_models = []

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