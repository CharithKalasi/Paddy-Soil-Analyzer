from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
CSV_FILE = BASE_DIR / "KrishiLink_MultiOutput_Training_Data_v4.csv"


def print_dataset_preview(name: str, dataframe: pd.DataFrame) -> None:
    print(f"{name} shape: {dataframe.shape}")
    print(dataframe.head(3))
    print()


def main() -> None:
    df = pd.read_csv(CSV_FILE)

    if "Health_Status" in df.columns:
        df = df.drop(columns=["Health_Status"])

    npk_df = df[["N", "P", "K", "Urea_kg_per_acre", "DAP_kg_per_acre", "MOP_kg_per_acre"]].copy()
    ph_df = df[["ph", "Lime_kg_per_acre", "Gypsum_kg_per_acre"]].copy()
    ec_df = df[["EC_uS_cm", "Low_EC_Fertilizer_Boost_kg", "Phase1_EC_Flush_Water_Liters"]].copy()
    orp_df = df[["ORP_mV", "Phase2_ORP_Flood_Water_Liters"]].copy()

    print_dataset_preview("npk_df", npk_df)
    print_dataset_preview("ph_df", ph_df)
    print_dataset_preview("ec_df", ec_df)
    print_dataset_preview("orp_df", orp_df)


if __name__ == "__main__":
    main()