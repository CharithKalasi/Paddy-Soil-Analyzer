from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "Models"
MODEL_FILE = MODEL_DIR / "npk_model.pkl"
CSV_FILE = BASE_DIR / "KrishiLink_MultiOutput_Training_Data_v4.csv"


def train_npk_model(npk_df: pd.DataFrame):
    """Train and evaluate a multi-output NPK recommendation model."""
    x = npk_df[["N", "P", "K"]]
    y = npk_df[["Urea_kg_per_acre", "DAP_kg_per_acre", "MOP_kg_per_acre"]]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
    )

    base_model = RandomForestRegressor(n_estimators=100, random_state=42)
    model = MultiOutputRegressor(base_model)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    target_names = ["Urea_kg_per_acre", "DAP_kg_per_acre", "MOP_kg_per_acre"]
    print("NPK Model Evaluation")
    print("-" * 30)
    for index, target_name in enumerate(target_names):
        mae = mean_absolute_error(y_test.iloc[:, index], y_pred[:, index])
        r2 = r2_score(y_test.iloc[:, index], y_pred[:, index])
        print(f"{target_name}:")
        print(f"  MAE: {mae:.4f}")
        print(f"  R2:  {r2:.4f}")

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_FILE)
    print(f"\nSaved trained model to: {MODEL_FILE}")

    return model


if __name__ == "__main__":
    data = pd.read_csv(CSV_FILE)
    data = data.drop(columns=["Health_Status"], errors="ignore")
    npk_df = data[["N", "P", "K", "Urea_kg_per_acre", "DAP_kg_per_acre", "MOP_kg_per_acre"]]
    train_npk_model(npk_df)