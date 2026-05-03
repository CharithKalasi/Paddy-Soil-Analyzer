from pathlib import Path
import json

import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    ExtraTreesRegressor,
    AdaBoostRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score


BASE_DIR = Path(__file__).resolve().parent
CSV_FILE = BASE_DIR / "KrishiLink_10k_RawUnits_Training_Data.csv"


def load_df(path):
    df = pd.read_csv(path)
    if "N" not in df.columns and "N_mg_per_kg" in df.columns:
        df["N"] = df["N_mg_per_kg"]
    if "P" not in df.columns and "P_mg_per_kg" in df.columns:
        df["P"] = df["P_mg_per_kg"]
    if "K" not in df.columns and "K_mg_per_kg" in df.columns:
        df["K"] = df["K_mg_per_kg"]
    return df.drop(columns=["Health_Status"], errors="ignore")


def evaluate_model_cv(model, X, y, cv_splits=5):
    kf = KFold(n_splits=cv_splits, shuffle=True, random_state=42)
    maes = []
    r2s = []
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred = np.asarray(y_pred)
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)
        fold_mae = []
        fold_r2 = []
        for col in range(y_test.shape[1]):
            fold_mae.append(mean_absolute_error(y_test.iloc[:, col], y_pred[:, col]))
            fold_r2.append(r2_score(y_test.iloc[:, col], y_pred[:, col]))
        maes.append(fold_mae)
        r2s.append(fold_r2)
    return np.mean(maes, axis=0).tolist(), np.mean(r2s, axis=0).tolist()


def make_candidates(multi=True):
    base = {
        "Linear": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=100, random_state=42),
        "GradientBoosting": GradientBoostingRegressor(random_state=42),
        "HistGradientBoosting": HistGradientBoostingRegressor(random_state=42),
        "AdaBoost": AdaBoostRegressor(random_state=42),
        "KNN": KNeighborsRegressor(),
        "MLP": MLPRegressor(random_state=42, max_iter=500),
        "SVR": SVR(),
    }
    if multi:
        return {k: MultiOutputRegressor(v) for k, v in base.items()}
    return base


def compare_task(X, y, multi=True):
    candidates = make_candidates(multi=multi)
    results = {}
    for name, model in candidates.items():
        try:
            maes, r2s = evaluate_model_cv(model, X, y, cv_splits=5)
            results[name] = {"MAE_per_target": maes, "R2_per_target": r2s}
        except Exception as e:
            results[name] = {"error": str(e)}
    return results


def main():
    df = load_df(CSV_FILE)

    report = {}

    # NPK
    npk_X = df[["N", "P", "K"]]
    npk_y = df[["Urea_kg_per_acre", "DAP_kg_per_acre", "MOP_kg_per_acre"]]
    report["NPK"] = compare_task(npk_X, npk_y, multi=True)

    # PH
    ph_X = df[["ph", "EC_uS_cm"]]
    ph_y = df[["Lime_kg_per_acre", "Gypsum_kg_per_acre"]]
    report["PH"] = compare_task(ph_X, ph_y, multi=True)

    # EC
    ec_X = df[["EC_uS_cm"]]
    ec_y = df[["Low_EC_Fertilizer_Boost_kg", "Phase1_EC_Flush_Water_Liters"]]
    report["EC"] = compare_task(ec_X, ec_y, multi=True)

    # ORP (single output)
    orp_X = df[["ORP_mV"]]
    orp_y = df[["Phase2_ORP_Flood_Water_Liters"]]
    report["ORP"] = compare_task(orp_X, orp_y, multi=False)

    out = BASE_DIR / "model_selection_extended_report.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved extended model selection report to: {out}")


if __name__ == "__main__":
    main()
