from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "Models"
MODEL_FILE = MODEL_DIR / "ec_model.pkl"
CSV_FILE = BASE_DIR / "KrishiLink_MultiOutput_Training_Data_v4.csv"


def train_ec_model(ec_df: pd.DataFrame):
    """Train and evaluate a multi-output EC recommendation model.

    The two targets are mutually exclusive in the dataset: low EC rows use the
    fertilizer boost target, while high EC rows use the flush water target.
    """
    x = ec_df[["EC_uS_cm"]]
    y = ec_df[["Low_EC_Fertilizer_Boost_kg", "Phase1_EC_Flush_Water_Liters"]]

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

    target_names = ["Low_EC_Fertilizer_Boost_kg", "Phase1_EC_Flush_Water_Liters"]
    print("EC Model Evaluation")
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
    ec_df = data[["EC_uS_cm", "Low_EC_Fertilizer_Boost_kg", "Phase1_EC_Flush_Water_Liters"]]
    train_ec_model(ec_df)