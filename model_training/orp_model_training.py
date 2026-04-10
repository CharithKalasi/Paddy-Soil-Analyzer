from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "Models"
MODEL_FILE = MODEL_DIR / "orp_model.pkl"
CSV_FILE = BASE_DIR / "KrishiLink_MultiOutput_Training_Data_v4.csv"


def train_orp_model(orp_df: pd.DataFrame):
    """Train and evaluate the Phase 2 ORP recommendation model.

    Target is binary in the dataset: 0 L or 10000 L flood water.
    We model this as classification (flood needed vs not needed).
    """
    x = orp_df[["ORP_mV"]]
    y = (orp_df["Phase2_ORP_Flood_Water_Liters"] > 0).astype(int)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)

    print("ORP Model Evaluation (Binary Classifier)")
    print("-" * 30)
    print(f"Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred, target_names=["No Flood", "Flood"]))

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_FILE)
    print(f"\nSaved trained model to: {MODEL_FILE}")

    return model


if __name__ == "__main__":
    data = pd.read_csv(CSV_FILE)
    data = data.drop(columns=["Health_Status"], errors="ignore")
    orp_df = data[["ORP_mV", "Phase2_ORP_Flood_Water_Liters"]]
    train_orp_model(orp_df)