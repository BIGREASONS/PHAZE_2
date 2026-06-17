"""ASTraM Command Center — Model Adapter Layer.

Defines the generic ModelInterface that ALL pages consume.
The PlaceholderModel is a dummy implementation returning mock predictions.
Claude will later replace this with the final leaderboard model.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List
from datetime import datetime
import hashlib
import numpy as np


class ModelInterface(ABC):
    """Contract that every model must satisfy.

    Pages call ONLY these methods — they never know which model is behind them.
    """

    @abstractmethod
    def load_model(self) -> None:
        """Load model weights / artifacts into memory (e.g., joblib.load)."""

    @abstractmethod
    def predict(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Run single prediction on an incident payload.
        
        Expected Input:
            payload: Dict containing feature key-values.
            
        Expected Output Schema:
            {
                "probability": float (0.0 to 1.0),
                "risk_level": str ("LOW", "MEDIUM", "HIGH", "CRITICAL"),
                "recommended_action": str (Human readable string)
            }
        """

    @abstractmethod
    def predict_batch(self, payloads: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run batch prediction on multiple incidents.
        
        Expected Input:
            payloads: List of Dicts containing feature key-values.
            
        Expected Output Schema:
            List of dictionaries matching the `predict` Output Schema.
        """

    @abstractmethod
    def explain(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Generate explainability metrics (SHAP/LIME) for a single incident.
        
        Expected Input:
            payload: Dict containing feature key-values.
            
        Expected Output Schema:
            {
                "feature_importances": Dict[str, float] (e.g., {"weather": 0.4, "time": 0.1}),
                "top_positive": List[str] (e.g., ["weather", "road_type"]),
                "top_negative": List[str] (e.g., ["day_of_week"])
            }
        """

    @abstractmethod
    def get_model_metadata(self) -> Dict[str, Any]:
        """Return model metadata for the monitoring dashboard.
        
        Expected Output Schema:
            {
                "name": str (e.g., "CatBoost + LightGBM Ensemble"),
                "version": str,
                "training_date": str,
                "status": str ("ONLINE", "OFFLINE"),
                "metrics": Dict[str, float] (e.g., {"F1": 0.92, "AUC": 0.96})
            }
        """


# ═══════════════════════════════════════════════════════════════════════════
# TO_BE_REPLACED_BY_CLAUDE — Placeholder implementation
# ═══════════════════════════════════════════════════════════════════════════

class PlaceholderModel(ModelInterface):
    """Dummy model returning deterministic mock predictions.

    ⚠️  TO_BE_REPLACED_BY_CLAUDE — This entire class will be swapped out
    once the ML team selects the final leaderboard model.  The dashboard
    does NOT depend on any implementation detail here.
    """

    _SEVERITY = {0: "LOW", 1: "MEDIUM", 2: "HIGH", 3: "CRITICAL"}
    _ACTIONS = {
        "LOW":      "Monitor situation. No immediate action required.",
        "MEDIUM":   "Alert nearest patrol unit. Prepare traffic diversion plan.",
        "HIGH":     "Dispatch response team. Activate alternate corridor routing.",
        "CRITICAL": "Initiate road closure protocol. Deploy full incident command.",
    }

    # Mock global feature importances
    GLOBAL_IMPORTANCES = {
        "corridor":       0.28,
        "event_cause":    0.22,
        "hour":           0.15,
        "zone":           0.12,
        "police_station": 0.08,
        "weekday":        0.06,
        "veh_type":       0.04,
        "latitude":       0.03,
        "longitude":      0.02,
    }

    def load_model(self) -> None:
        """No-op for placeholder."""

    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        seed = int(
            hashlib.md5(str(sorted(features.items())).encode()).hexdigest()[:8],
            16,
        )
        rng = np.random.RandomState(seed)
        prob = float(np.clip(rng.beta(2, 10), 0.02, 0.98))

        # Boost probability for known high-risk causes
        cause = str(features.get("event_cause", "")).lower()
        if cause in ("accident", "tree_fall"):
            prob = min(prob + 0.35, 0.95)
        elif cause in ("water_logging",):
            prob = min(prob + 0.20, 0.90)
        elif cause in ("vip_movement", "road_work"):
            prob = min(prob + 0.15, 0.85)

        if prob < 0.25:
            sev = "LOW"
        elif prob < 0.50:
            sev = "MEDIUM"
        elif prob < 0.75:
            sev = "HIGH"
        else:
            sev = "CRITICAL"

        confidence = float(np.clip(0.6 + rng.normal(0, 0.1), 0.4, 0.95))

        return {
            "probability": round(prob, 4),
            "confidence": round(confidence, 4),
            "severity": sev,
            "recommended_action": self._ACTIONS[sev],
            "model": "PlaceholderModel",
            "timestamp": datetime.utcnow().isoformat(),
        }

    def predict_batch(self, features_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [self.predict(f) for f in features_list]

    def explain(self, features: Dict[str, Any]) -> Dict[str, Any]:
        seed = int(
            hashlib.md5(str(sorted(features.items())).encode()).hexdigest()[:8],
            16,
        )
        rng = np.random.RandomState(seed)

        contributions = {}
        for feat, base_imp in self.GLOBAL_IMPORTANCES.items():
            val = float(rng.normal(0, base_imp))
            contributions[feat] = round(val, 4)

        sorted_c = sorted(contributions.items(), key=lambda x: x[1], reverse=True)
        top_pos = [{"feature": k, "contribution": v} for k, v in sorted_c if v > 0][:3]
        top_neg = [{"feature": k, "contribution": v} for k, v in sorted_c if v < 0][-3:]

        return {
            "feature_contributions": contributions,
            "top_positive_drivers": top_pos,
            "top_negative_drivers": top_neg,
            "method": "placeholder_mock",
            "note": "TO_BE_REPLACED_BY_CLAUDE — Real SHAP values after final model selection",
        }

    def get_model_metadata(self) -> Dict[str, Any]:
        return {
            "name": "PlaceholderModel",
            "version": "0.1.0-dev",
            "training_date": "2024-03-15",
            "status": "OPERATIONAL",
            "metrics": {
                "roc_auc": 0.823,
                "pr_auc": 0.467,
                "f1": 0.42,
                "precision": 0.55,
                "recall": 0.34,
            },
            "note": "TO_BE_REPLACED_BY_CLAUDE",
        }
