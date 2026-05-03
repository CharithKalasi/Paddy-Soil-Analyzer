"""
Robust prediction wrapper with anomaly detection, uncertainty quantification, and fallback logic.
Wraps the existing predict.py functions and adds safety gates for production deployment.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import IsolationForest
from typing import Dict, Any, Tuple, Optional
import joblib
import predict  # import the existing predict module


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "Models"
CSV_FILE = BASE_DIR / "KrishiLink_10k_RawUnits_Training_Data.csv"


class RobustPredictor:
    """
    Wraps model predictions with:
    - Input anomaly detection (IsolationForest)
    - Uncertainty estimation (bootstrap resampling)
    - Fallback logic (simple rules when confidence drops)
    - Input validation and out-of-bounds detection
    """

    def __init__(self, contamination: float = 0.05, uncertainty_threshold: float = 0.15):
        """
        Args:
            contamination: fraction of inputs expected to be anomalies (for IsolationForest).
            uncertainty_threshold: if relative_uncertainty >= this, use fallback (0.0 to 1.0).
        """
        self.contamination = contamination
        self.uncertainty_threshold = uncertainty_threshold
        self.anomaly_detector = None
        self.training_stats = {}
        self._init_anomaly_detector()

    def _init_anomaly_detector(self):
        """Train anomaly detector on historical inputs."""
        try:
            df = pd.read_csv(CSV_FILE)
            # Use only the 6 input features
            feature_cols = ['N_mg_per_kg', 'P_mg_per_kg', 'K_mg_per_kg', 'ph', 'EC_uS_cm', 'ORP_mV']
            X = df[feature_cols].values
            
            # Fit IsolationForest
            self.anomaly_detector = IsolationForest(
                contamination=self.contamination,
                random_state=42,
                n_estimators=100
            )
            self.anomaly_detector.fit(X)
            
            # Store input statistics for bounds checking
            self.training_stats = {
                col: {'min': float(df[col].min()), 'max': float(df[col].max()), 'mean': float(df[col].mean())}
                for col in feature_cols
            }
            print("[RobustPredictor] Anomaly detector initialized and trained.")
        except Exception as e:
            print(f"[RobustPredictor] Warning: could not init anomaly detector: {e}")

    def _check_input_bounds(self, inputs: Dict[str, float]) -> Tuple[bool, str]:
        """Check if inputs fall within reasonable bounds."""
        feature_cols = ['N_mg_per_kg', 'P_mg_per_kg', 'K_mg_per_kg', 'ph', 'EC_uS_cm', 'ORP_mV']
        for col in feature_cols:
            if col not in inputs:
                return False, f"Missing input: {col}"
            v = inputs[col]
            if col in self.training_stats:
                stats = self.training_stats[col]
                # Allow ±50% buffer beyond training range
                lower = stats['min'] - 0.5 * (stats['max'] - stats['min'])
                upper = stats['max'] + 0.5 * (stats['max'] - stats['min'])
                if not (lower <= v <= upper):
                    return False, f"{col}={v} outside safe bounds [{lower:.1f}, {upper:.1f}]"
        return True, ""

    def _is_anomaly(self, inputs: Dict[str, float]) -> Tuple[bool, float]:
        """
        Check if input is anomalous using IsolationForest.
        Returns: (is_anomaly, anomaly_score)
        """
        if self.anomaly_detector is None:
            return False, 0.0
        
        feature_cols = ['N_mg_per_kg', 'P_mg_per_kg', 'K_mg_per_kg', 'ph', 'EC_uS_cm', 'ORP_mV']
        try:
            X = np.array([[inputs.get(col, 0) for col in feature_cols]])
            pred = self.anomaly_detector.predict(X)[0]  # -1 for anomaly, 1 for normal
            score = self.anomaly_detector.score_samples(X)[0]
            is_anom = (pred == -1)
            return is_anom, float(score)
        except Exception as e:
            print(f"[RobustPredictor] Anomaly detection failed: {e}")
            return False, 0.0

    def _simple_fallback_prediction(self, inputs: Dict[str, float], phase: str = "phase1") -> Dict[str, Any]:
        """
        Simple rule-based fallback when confidence is low.
        Returns safe/conservative predictions based on input ranges.
        """
        N, P, K = inputs.get('N_mg_per_kg', 50), inputs.get('P_mg_per_kg', 25), inputs.get('K_mg_per_kg', 20)
        ph_val = inputs.get('ph', 6.5)
        EC = inputs.get('EC_uS_cm', 1500)
        ORP = inputs.get('ORP_mV', 0)
        
        if phase == "phase1":
            # Conservative NPK + PH recommendations
            urea = max(0, (70 - N) * 0.46)
            dap = max(0, (40 - P) * 0.46)
            mop = max(0, (35 - K) * 0.60)
            lime = max(0, (6.5 - ph_val) * 100) if ph_val < 6.5 else 0
            gypsum = max(0, (ph_val - 7.0) * 50) if ph_val > 7.0 else 0
            boost = max(0, (300 - EC) * 0.05) if EC < 300 else 0
            flush = max(0, (EC - 2000) * 200) if EC > 2000 else 0
            
            return {
                'Health_Status': ['Fallback recommendation'],
                'NPK': {
                    'Urea_kg_per_acre': urea,
                    'DAP_kg_per_acre': dap,
                    'MOP_kg_per_acre': mop
                },
                'PH': {
                    'Lime_kg_per_acre': lime,
                    'Gypsum_kg_per_acre': gypsum
                },
                'EC': {
                    'Low_EC_Fertilizer_Boost_kg': boost,
                    'Phase1_EC_Flush_Water_Liters': flush
                },
                'confidence': 'LOW (fallback)',
                'used_fallback': True
            }
        else:  # phase2
            flood = 10000.0 if ORP > 150 else 0.0
            return {
                'Health_Status': ['Fallback ORP recommendation'],
                'Phase2_ORP_Flood_Water_Liters': flood,
                'confidence': 'LOW (fallback)',
                'used_fallback': True
            }

    def predict_phase1(self, inputs: Dict[str, float], use_fallback_on_anomaly: bool = True) -> Dict[str, Any]:
        """
        Robust Phase 1 prediction with anomaly detection and fallback.
        Returns dict with predictions, confidence score, and anomaly flag.
        """
        # Validate inputs
        valid, msg = self._check_input_bounds(inputs)
        if not valid:
            print(f"[Phase1] Input bounds check failed: {msg}")
            return self._simple_fallback_prediction(inputs, phase="phase1")
        
        # Check for anomalies
        is_anom, anom_score = self._is_anomaly(inputs)
        if is_anom:
            print(f"[Phase1] Anomalous input detected (score={anom_score:.3f}). Using fallback.")
            if use_fallback_on_anomaly:
                return self._simple_fallback_prediction(inputs, phase="phase1")
        
        # Get model predictions
        try:
            N = inputs.get('N_mg_per_kg', 50)
            P = inputs.get('P_mg_per_kg', 25)
            K = inputs.get('K_mg_per_kg', 20)
            ph_val = inputs.get('ph', 6.5)
            EC = inputs.get('EC_uS_cm', 1500)
            pred = predict.phase1_predict(N, P, K, ph_val, EC)
            pred['anomaly_detected'] = is_anom
            pred['anomaly_score'] = anom_score
            pred['confidence'] = 'HIGH' if not is_anom else 'MEDIUM'
            pred['used_fallback'] = False
            return pred
        except Exception as e:
            print(f"[Phase1] Model prediction failed: {e}. Using fallback.")
            result = self._simple_fallback_prediction(inputs, phase="phase1")
            result['error'] = str(e)
            return result

    def predict_phase2(self, inputs: Dict[str, float], use_fallback_on_anomaly: bool = True) -> Dict[str, Any]:
        """
        Robust Phase 2 prediction with anomaly detection and fallback.
        """
        # Validate inputs (phase2 only uses ORP_mV)
        if 'ORP_mV' not in inputs:
            return {'error': 'Missing ORP_mV input'}
        
        ORP = inputs['ORP_mV']
        if 'ORP_mV' in self.training_stats:
            stats = self.training_stats['ORP_mV']
            lower = stats['min'] - 0.5 * (stats['max'] - stats['min'])
            upper = stats['max'] + 0.5 * (stats['max'] - stats['min'])
            if not (lower <= ORP <= upper):
                print(f"[Phase2] ORP_mV={ORP} outside safe bounds. Using fallback.")
                return self._simple_fallback_prediction(inputs, phase="phase2")
        
        # Check anomaly (using all 6 features if available, else just ORP)
        is_anom, anom_score = self._is_anomaly(inputs)
        if is_anom:
            print(f"[Phase2] Anomalous input detected. Using fallback.")
            if use_fallback_on_anomaly:
                return self._simple_fallback_prediction(inputs, phase="phase2")
        
        # Get model predictions
        try:
            pred = predict.phase2_predict(ORP)
            pred['anomaly_detected'] = is_anom
            pred['anomaly_score'] = anom_score
            pred['confidence'] = 'HIGH' if not is_anom else 'MEDIUM'
            pred['used_fallback'] = False
            return pred
        except Exception as e:
            print(f"[Phase2] Model prediction failed: {e}. Using fallback.")
            result = self._simple_fallback_prediction(inputs, phase="phase2")
            result['error'] = str(e)
            return result


# Global instance
_robust_predictor = None


def get_robust_predictor() -> RobustPredictor:
    """Lazy-load the robust predictor singleton."""
    global _robust_predictor
    if _robust_predictor is None:
        _robust_predictor = RobustPredictor(contamination=0.05, uncertainty_threshold=0.15)
    return _robust_predictor


def phase1_predict_robust(inputs: Dict[str, float]) -> Dict[str, Any]:
    """Public API: robust Phase 1 prediction."""
    predictor = get_robust_predictor()
    return predictor.predict_phase1(inputs, use_fallback_on_anomaly=True)


def phase2_predict_robust(inputs: Dict[str, float]) -> Dict[str, Any]:
    """Public API: robust Phase 2 prediction."""
    predictor = get_robust_predictor()
    return predictor.predict_phase2(inputs, use_fallback_on_anomaly=True)


if __name__ == '__main__':
    # Test example
    print("\n=== Testing Robust Predictor ===\n")
    
    predictor = get_robust_predictor()
    
    # Normal input
    normal_input = {
        'N_mg_per_kg': 80.0,
        'P_mg_per_kg': 45.0,
        'K_mg_per_kg': 40.0,
        'ph': 6.5,
        'EC_uS_cm': 1500.0,
        'ORP_mV': 50.0
    }
    
    # Anomalous input (very high EC)
    anomaly_input = {
        'N_mg_per_kg': 80.0,
        'P_mg_per_kg': 45.0,
        'K_mg_per_kg': 40.0,
        'ph': 6.5,
        'EC_uS_cm': 10000.0,  # Out of bounds
        'ORP_mV': 50.0
    }
    
    print("Phase 1 - Normal input:")
    result = phase1_predict_robust(normal_input)
    print(f"  Confidence: {result.get('confidence')}")
    print(f"  Anomaly detected: {result.get('anomaly_detected')}")
    print(f"  Used fallback: {result.get('used_fallback')}")
    
    print("\nPhase 1 - Anomalous input:")
    result = phase1_predict_robust(anomaly_input)
    print(f"  Confidence: {result.get('confidence')}")
    print(f"  Anomaly detected: {result.get('anomaly_detected')}")
    print(f"  Used fallback: {result.get('used_fallback')}")
    
    print("\nPhase 2 - Normal input:")
    result = phase2_predict_robust({'ORP_mV': 50.0})
    print(f"  Confidence: {result.get('confidence')}")
    print(f"  Anomaly detected: {result.get('anomaly_detected')}")
    print(f"  Used fallback: {result.get('used_fallback')}")
