from pathlib import Path

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "Models"
NPK_MODEL_FILE = MODEL_DIR / "npk_model.pkl"
PH_MODEL_FILE = MODEL_DIR / "ph_model.pkl"
EC_MODEL_FILE = MODEL_DIR / "ec_model.pkl"
ORP_MODEL_FILE = MODEL_DIR / "orp_model.pkl"


npk_model = joblib.load(NPK_MODEL_FILE)
ph_model = joblib.load(PH_MODEL_FILE)
ec_model = joblib.load(EC_MODEL_FILE)
orp_model = joblib.load(ORP_MODEL_FILE)


def phase1_predict(N, P, K, ph, EC):
    """Run Phase 1 predictions for NPK, pH, and EC inputs."""
    npk_input = pd.DataFrame([[N, P, K]], columns=["N", "P", "K"])
    ph_input = pd.DataFrame([[ph]], columns=["ph"])
    ec_input = pd.DataFrame([[EC]], columns=["EC_uS_cm"])

    npk_prediction = npk_model.predict(npk_input)[0]
    ph_prediction = ph_model.predict(ph_input)[0]
    ec_prediction = ec_model.predict(ec_input)[0]

    lime_kg = float(ph_prediction[0])
    gypsum_kg = float(ph_prediction[1])
    ph_recommendation = {
        "Lime_kg_per_acre": lime_kg if lime_kg > 0 else 0.0,
        "Gypsum_kg_per_acre": gypsum_kg if gypsum_kg > 0 else 0.0,
    }

    ec_boost_kg = float(ec_prediction[0])
    ec_flush_liters = float(ec_prediction[1])
    ec_recommendation = {
        "Low_EC_Fertilizer_Boost_kg": ec_boost_kg if ec_boost_kg > 0 else 0.0,
        "Phase1_EC_Flush_Water_Liters": ec_flush_liters if ec_flush_liters > 0 else 0.0,
    }

    return {
        "NPK": {
            "Urea_kg_per_acre": float(npk_prediction[0]),
            "DAP_kg_per_acre": float(npk_prediction[1]),
            "MOP_kg_per_acre": float(npk_prediction[2]),
        },
        "PH": ph_recommendation,
        "EC": ec_recommendation,
    }


def phase2_predict(ORP):
    """Run Phase 2 prediction for ORP input."""
    orp_input = pd.DataFrame([[ORP]], columns=["ORP_mV"])
    orp_prediction = orp_model.predict(orp_input)[0]

    return {
        "Phase2_ORP_Flood_Water_Liters": float(orp_prediction),
    }


def print_pretty_result(title, result):
    print(title)
    print("=" * len(title))
    for section, values in result.items():
        print(f"{section}:")
        if isinstance(values, dict):
            for key, value in values.items():
                print(f"  {key}: {value:.2f}")
        else:
            print(f"  {values}")
    print()


if __name__ == "__main__":
    phase1_result = phase1_predict(N=40, P=20, K=15, ph=5.5, EC=300)
    phase2_result = phase2_predict(ORP=-100)

    print_pretty_result("Phase 1 Prediction", phase1_result)
    print_pretty_result("Phase 2 Prediction", phase2_result)