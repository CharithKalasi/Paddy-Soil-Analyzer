from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, r2_score


BASE_DIR = Path(__file__).resolve().parent
CSV_FILE = BASE_DIR / "KrishiLink_10k_RawUnits_Training_Data.csv"
MODEL_DIR = BASE_DIR / "Models"
MODEL_DIR.mkdir(exist_ok=True)


def load_and_normalize_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Map generator column names to expected names
    if "N" not in df.columns and "N_mg_per_kg" in df.columns:
        df["N"] = df["N_mg_per_kg"]
    if "P" not in df.columns and "P_mg_per_kg" in df.columns:
        df["P"] = df["P_mg_per_kg"]
    if "K" not in df.columns and "K_mg_per_kg" in df.columns:
        df["K"] = df["K_mg_per_kg"]
    return df


def evaluate_model_cv(model, X, y, cv_splits=5):
    kf = KFold(n_splits=cv_splits, shuffle=True, random_state=42)
    maes = []
    r2s = []
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        # Ensure 2D
        y_pred = np.asarray(y_pred)
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)

        # Per-target metrics
        fold_mae = []
        fold_r2 = []
        for col in range(y_test.shape[1]):
            fold_mae.append(mean_absolute_error(y_test.iloc[:, col], y_pred[:, col]))
            fold_r2.append(r2_score(y_test.iloc[:, col], y_pred[:, col]))

        maes.append(fold_mae)
        r2s.append(fold_r2)

    maes = np.mean(maes, axis=0)
    r2s = np.mean(r2s, axis=0)
    return maes, r2s


def evaluate_model_nested(model, X, y, outer_splits=5, inner_splits=3, n_iter=6):
    """Perform nested CV: outer fold for evaluation, inner RandomizedSearchCV for tuning when applicable."""
    outer = KFold(n_splits=outer_splits, shuffle=True, random_state=42)
    maes = []
    r2s = []
    rf_best_params = []
    for train_idx, test_idx in outer.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # If model is a MultiOutputRegressor wrapping RandomForest, tune inside
        tuned_model = None
        if isinstance(model, MultiOutputRegressor) and isinstance(model.estimator, RandomForestRegressor):
            param_distributions = {
                "estimator__n_estimators": [50, 100, 200],
                "estimator__max_depth": [None, 10, 20],
                "estimator__min_samples_split": [2, 5, 10],
            }
            rnd = RandomizedSearchCV(model, param_distributions, n_iter=n_iter, cv=inner_splits, scoring="neg_mean_absolute_error", random_state=42, n_jobs=-1)
            rnd.fit(X_train, y_train)
            tuned_model = rnd.best_estimator_
            rf_best_params.append(rnd.best_params_)
        else:
            tuned_model = model
            tuned_model.fit(X_train, y_train)

        y_pred = tuned_model.predict(X_test)
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

    maes = np.mean(maes, axis=0)
    r2s = np.mean(r2s, axis=0)
    return maes, r2s, rf_best_params


def compare_models(X, y, candidates=None):
    if candidates is None:
        candidates = {
            "Linear": LinearRegression(),
            "RandomForest": MultiOutputRegressor(RandomForestRegressor(random_state=42, n_estimators=100)),
            "GradientBoosting": MultiOutputRegressor(GradientBoostingRegressor(random_state=42)),
            "HistGradientBoosting": MultiOutputRegressor(HistGradientBoostingRegressor(random_state=42)),
        }

    results = {}
    for name, model in candidates.items():
        # Use nested CV for RandomForest to avoid optimistic estimates
        if name == "RandomForest":
            maes, r2s, best_params = evaluate_model_nested(model, X, y, outer_splits=5, inner_splits=3, n_iter=6)
            results[name] = {"MAE_per_target": maes.tolist(), "R2_per_target": r2s.tolist(), "inner_best_params": best_params}
        else:
            maes, r2s = evaluate_model_cv(model, X, y, cv_splits=5)
            results[name] = {"MAE_per_target": maes.tolist(), "R2_per_target": r2s.tolist()}
    return results


def tune_random_forest(X, y, output_path: Path, n_iter=20):
    # Tune a MultiOutput RandomForest via RandomizedSearchCV on a wrapper
    base = RandomForestRegressor(random_state=42)
    wrapper = MultiOutputRegressor(base)

    param_distributions = {
        "estimator__n_estimators": [50, 100, 200],
        "estimator__max_depth": [None, 10, 20, 30],
        "estimator__min_samples_split": [2, 5, 10],
    }

    # Use a small CV inside RandomizedSearch; this is tuning, not final evaluation
    rnd = RandomizedSearchCV(wrapper, param_distributions, n_iter=n_iter, cv=3, scoring="neg_mean_absolute_error", random_state=42, n_jobs=-1)
    rnd.fit(X, y)
    best = rnd.best_estimator_
    joblib.dump(best, output_path)
    return rnd.best_params_, rnd.best_score_, best


def fmt_targets_list(cols):
    return ", ".join(cols)


def main():
    df = load_and_normalize_csv(CSV_FILE)
    df = df.drop(columns=["Health_Status"], errors="ignore")

    reports = {}

    # Task: NPK multi-output
    npk_X = df[["N", "P", "K"]]
    npk_y = df[["Urea_kg_per_acre", "DAP_kg_per_acre", "MOP_kg_per_acre"]]
    reports["NPK_compare"] = compare_models(npk_X, npk_y)
    # tune RF
    params, score, best = tune_random_forest(npk_X, npk_y, MODEL_DIR / "best_tuned_npk_model.pkl", n_iter=8)
    reports["NPK_tuned"] = {"best_params": params, "best_cv_neg_mae": float(score)}

    # Task: PH multi-output
    ph_X = df[["ph", "EC_uS_cm"]]
    ph_y = df[["Lime_kg_per_acre", "Gypsum_kg_per_acre"]]
    reports["PH_compare"] = compare_models(ph_X, ph_y)
    params, score, best = tune_random_forest(ph_X, ph_y, MODEL_DIR / "best_tuned_ph_model.pkl", n_iter=8)
    reports["PH_tuned"] = {"best_params": params, "best_cv_neg_mae": float(score)}

    # Task: EC multi-output
    ec_X = df[["EC_uS_cm"]]
    ec_y = df[["Low_EC_Fertilizer_Boost_kg", "Phase1_EC_Flush_Water_Liters"]]
    reports["EC_compare"] = compare_models(ec_X, ec_y)
    params, score, best = tune_random_forest(ec_X, ec_y, MODEL_DIR / "best_tuned_ec_model.pkl", n_iter=8)
    reports["EC_tuned"] = {"best_params": params, "best_cv_neg_mae": float(score)}

    # Task: ORP single-output
    orp_X = df[["ORP_mV"]]
    orp_y = df[["Phase2_ORP_Flood_Water_Liters"]]
    # wrap single-output to keep compare_models logic
    reports["ORP_compare"] = compare_models(orp_X, pd.DataFrame(orp_y))
    params, score, best = tune_random_forest(orp_X, pd.DataFrame(orp_y), MODEL_DIR / "best_tuned_orp_model.pkl", n_iter=8)
    reports["ORP_tuned"] = {"best_params": params, "best_cv_neg_mae": float(score)}

    # Save report
    report_path = BASE_DIR / "model_selection_report.json"
    import json

    with open(report_path, "w") as f:
        json.dump(reports, f, indent=2)

    print(f"Saved model selection report to: {report_path}")
    print("Saved tuned models to Models/best_tuned_*.pkl")


if __name__ == "__main__":
    main()
