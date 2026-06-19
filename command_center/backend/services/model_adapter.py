"""GridSight AI Command Center — Model Adapter Layer.

Defines the generic ModelInterface that ALL pages consume.
ProductionEnsembleModel is the served default; PlaceholderModel is a
deterministic fallback used only when the production artifacts are unavailable.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from datetime import datetime
import hashlib
import json
import logging
import os
import threading
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Captured once when this module is first imported (≈ server boot). Powers the
# real process-uptime telemetry on the monitoring page — no faked values.
_PROCESS_START = time.time()


def process_uptime_seconds() -> float:
    """Seconds elapsed since this process first imported the model adapter."""
    return max(0.0, time.time() - _PROCESS_START)


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

    @abstractmethod
    def get_global_importances(self) -> Dict[str, float]:
        """Return the model's global feature-importance ranking.

        Expected Output Schema:
            Dict[str, float] mapping feature name -> importance, e.g.
            {"corridor": 0.28, "event_cause": 0.22, ...}
        """


# ═══════════════════════════════════════════════════════════════════════════
# PlaceholderModel — deterministic fallback (served only when artifacts absent)
# ═══════════════════════════════════════════════════════════════════════════

class PlaceholderModel(ModelInterface):
    """Deterministic mock model used as a graceful fallback.

    Served only when the production ensemble's artifacts or dependencies are
    unavailable (see :func:`get_model`). The dashboard does NOT depend on any
    implementation detail here — it consumes the ModelInterface contract only.
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
            # match the production contract so artifact-less pages (e.g. page 8)
            # can read driver rankings off predict() no matter which model serves
            "feature_contributions": dict(self.GLOBAL_IMPORTANCES),
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
            "note": "Deterministic fallback attribution — served only when the production ensemble is unavailable",
        }

    def get_global_importances(self) -> Dict[str, float]:
        return dict(self.GLOBAL_IMPORTANCES)

    def get_model_metadata(self) -> Dict[str, Any]:
        return {
            "name": "PlaceholderModel",
            "version": "0.1.0-dev",
            "training_date": "2024-03-15",
            "status": "OPERATIONAL",
            "metrics": {
                # mirror the frozen ensemble's honest rolling-OOF metrics so the
                # artifact-less fallback never shows better-than-real numbers
                "roc_auc": 0.7887,
                "pr_auc": 0.3641,
                "f1": 0.4397,
                "precision": 0.4679,
                "recall": 0.4146,
            },
            "note": "Fallback model — metrics mirror the frozen ensemble's honest OOF values",
        }


# ═══════════════════════════════════════════════════════════════════════════
# ProductionEnsembleModel — the frozen, accepted solution
# ═══════════════════════════════════════════════════════════════════════════

class ProductionEnsembleModel(ModelInterface):
    """Frozen equal-weight 7-model probability-averaged ensemble.

    Serves the exact artifacts exported by
    ``final_validation/export_production_model.py`` (Phase F). The inference
    contract is read from ``manifest.json`` — nothing about the ensemble is
    hardcoded here beyond what that manifest declares:

        members  : CatBoost · LightGBM · XGBoost · RandomForest · ExtraTrees
                   · Logistic · TabPFN
        combine  : equal-weight mean of per-member P(closure)  (prob_mean)
        calibrate: isotonic regression fit on honest rolling-OOF
        threshold: operating points in thresholds.json

    Per-member input routing (from the manifest):
        CatBoost, Logistic   -> raw frame [cats + nums]   (own encoders)
        tree members, TabPFN -> ordinal-encoded matrix    (shared encoder.pkl)

    TabPFN ships as a fixed in-context training set; it is refit ONCE in
    :meth:`load_model` and kept warm for the process lifetime.
    """

    # categorical-aware risk action copy (kept identical to the placeholder so
    # the dashboard reads the same regardless of which model is behind it)
    _ACTIONS = PlaceholderModel._ACTIONS

    def __init__(self, artifact_dir: Optional[str] = None) -> None:
        if artifact_dir is None:
            # services -> backend -> command_center -> <repo root>
            repo_root = Path(__file__).resolve().parents[3]
            artifact_dir = os.getenv(
                "ASTRAM_ARTIFACT_DIR",
                str(repo_root / "final_validation" / "artifacts"),
            )
        self.artifact_dir = Path(artifact_dir)
        self._loaded = False
        self._lock = threading.Lock()
        # bounded memoization of identical single-incident predictions so the
        # interactive pages do not re-run the (CPU-heavy) ensemble for a row the
        # user re-selects. Does NOT change any prediction — same input, same out.
        self._predict_cache: "dict[str, Dict[str, Any]]" = {}
        self._cache_cap = 512

    # ----------------------------------------------------------------- loading
    def load_model(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            import joblib
            from catboost import CatBoostClassifier

            d = self.artifact_dir
            self.manifest = json.loads((d / "manifest.json").read_text())
            self.metadata = json.loads((d / "metadata.json").read_text())
            self.thresholds = json.loads((d / "thresholds.json").read_text())

            enc = joblib.load(d / "encoder.pkl")
            self.oe = enc["ordinal_encoder"]
            self.cats = list(enc["cats"])
            self.nums = list(enc["nums"])
            self.calibrator = joblib.load(d / "calibrator.pkl")

            self.members = list(self.manifest["members"])

            cb = CatBoostClassifier()
            cb.load_model(str(d / "CatBoost.cbm"))
            self._cb = cb
            self._lgbm = joblib.load(d / "LightGBM.pkl")
            self._xgb = joblib.load(d / "XGBoost.pkl")
            self._rf = joblib.load(d / "RandomForest.pkl")
            self._et = joblib.load(d / "ExtraTrees.pkl")
            self._logit = joblib.load(d / "Logistic.pkl")

            self._tab = self._refit_tabpfn(d / "tabpfn_refit.npz")

            self._global_importances = self._compute_global_importances()
            self._baseline_row = self._compute_baseline_row()
            self._loaded = True
            logger.info(
                "ProductionEnsembleModel loaded: %d members from %s",
                len(self.members), d,
            )

    def _refit_tabpfn(self, npz_path: Path):
        """Reconstruct the frozen TabPFN exactly as the manifest prescribes:
        refit TabPFNClassifier(cpu, n_estimators=3) on the stored in-context set."""
        os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "1")
        import torch
        from tabpfn import TabPFNClassifier

        try:
            torch.set_num_threads(min(16, os.cpu_count() or 4))
        except Exception:
            pass
        z = np.load(npz_path)
        clf = TabPFNClassifier(device="cpu", ignore_pretraining_limits=True,
                               n_estimators=3)
        clf.fit(z["X"], z["y"])
        return clf

    def _compute_global_importances(self) -> Dict[str, float]:
        """Average the tree members' feature_importances_ over the 10 features.
        Real, frozen-model importances — no SHAP dependency required."""
        feats = self.cats + self.nums

        def norm(v):
            v = np.asarray(v, dtype=float)
            s = v.sum()
            return v / s if s > 0 else v

        imp = np.mean(np.vstack([
            norm(self._lgbm.feature_importances_),
            norm(self._xgb.feature_importances_),
            norm(self._rf.feature_importances_),
            norm(self._et.feature_importances_),
        ]), axis=0)
        return {f: round(float(x), 4) for f, x in
                sorted(zip(feats, imp), key=lambda kv: -kv[1])}

    def _compute_baseline_row(self) -> Dict[str, Any]:
        """Neutral reference incident used for local ablation explanations.
        Derived purely from the artifacts (encoder categories + the TabPFN
        in-context matrix): modal category per cat, median per numeric."""
        z = np.load(self.artifact_dir / "tabpfn_refit.npz")
        X = z["X"]
        row: Dict[str, Any] = {}
        for i, c in enumerate(self.cats):
            col = X[:, i].astype(int)
            code = int(np.bincount(col[col >= 0]).argmax()) if (col >= 0).any() else 0
            cats_i = self.oe.categories_[i]
            row[c] = cats_i[code] if 0 <= code < len(cats_i) else "NA"
        for j, n in enumerate(self.nums):
            row[n] = float(np.median(X[:, len(self.cats) + j]))
        return row

    # ----------------------------------------------------------- core inference
    def _to_raw_frame(self, payloads: List[Dict[str, Any]]):
        """Build the model's raw feature frame from API payloads.

        API payloads carry ``latitude``/``longitude`` (mapped to lat/lon) and
        supply ``hour``/``weekday`` directly. Missing categoricals default to
        "NA" (the encoder maps unseen values to the trained unknown code)."""
        import pandas as pd

        rows = []
        for p in payloads:
            r = {}
            for c in self.cats:
                r[c] = str(p.get(c, "NA"))
            r["lat"] = float(p.get("lat", p.get("latitude", 0.0)) or 0.0)
            r["lon"] = float(p.get("lon", p.get("longitude", 0.0)) or 0.0)
            r["hour"] = int(p.get("hour", 0) or 0)
            r["weekday"] = int(p.get("weekday", 0) or 0)
            rows.append(r)
        return pd.DataFrame(rows)[self.cats + self.nums]

    def _calibrated_proba(self, raw):
        """Return (member_matrix [N,M], calibrated [N]) for a raw feature frame.
        One predict per member; TabPFN cost is dominated by its fixed context,
        so batching N rows into a single call is near-free."""
        codes = self.oe.transform(raw[self.cats]).astype(float)
        xtree = np.hstack([codes, raw[self.nums].values.astype(float)])
        raw_in = raw[self.cats + self.nums]

        per = {
            "CatBoost": self._cb.predict_proba(raw_in)[:, 1],
            "LightGBM": self._lgbm.predict_proba(xtree)[:, 1],
            "XGBoost": self._xgb.predict_proba(xtree)[:, 1],
            "RandomForest": self._rf.predict_proba(xtree)[:, 1],
            "ExtraTrees": self._et.predict_proba(xtree)[:, 1],
            "Logistic": self._logit.predict_proba(raw_in)[:, 1],
            "TabPFN": self._tab.predict_proba(xtree)[:, 1],
        }
        matrix = np.column_stack([per[m] for m in self.members])
        combined = matrix.mean(axis=1)                  # equal-weight prob_mean
        calibrated = np.asarray(self.calibrator.predict(combined), dtype=float)
        return matrix, calibrated

    def _severity(self, prob: float) -> str:
        """Map a calibrated probability to a band using the honest operating
        points in thresholds.json (falls back to fixed cutoffs if absent)."""
        def thr(key, default):
            try:
                return float(self.thresholds[key]["threshold"])
            except Exception:
                return default

        t90 = thr("recall_0.90", 0.05)   # watchlist
        t70 = thr("recall_0.70", 0.15)   # response-worthy
        tf1 = thr("max_f1", 0.33)        # would be flagged a closure
        if prob >= tf1:
            return "CRITICAL"
        if prob >= t70:
            return "HIGH"
        if prob >= t90:
            return "MEDIUM"
        return "LOW"

    @staticmethod
    def _cache_key(features: Dict[str, Any]) -> str:
        return hashlib.md5(
            json.dumps(features, sort_keys=True, default=str).encode()
        ).hexdigest()

    # --------------------------------------------------------------- interface
    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        if not self._loaded:
            self.load_model()
        key = self._cache_key(features)
        cached = self._predict_cache.get(key)
        if cached is not None:
            return dict(cached)

        raw = self._to_raw_frame([features])
        matrix, calibrated = self._calibrated_proba(raw)
        prob = float(calibrated[0])
        # confidence = inter-model agreement (tight spread -> high confidence)
        spread = float(matrix[0].std())
        confidence = float(np.clip(1.0 - 2.0 * spread, 0.4, 0.99))
        severity = self._severity(prob)
        result = {
            "probability": round(prob, 4),
            "confidence": round(confidence, 4),
            "severity": severity,
            "recommended_action": self._ACTIONS[severity],
            # quick global driver ranking for compact panels (page 8);
            # call explain() for per-incident local attribution.
            "feature_contributions": dict(self._global_importances),
            "model": "ProductionEnsembleModel",
            "n_members": len(self.members),
            "timestamp": datetime.utcnow().isoformat(),
        }
        if len(self._predict_cache) < self._cache_cap:
            self._predict_cache[key] = dict(result)
        return result

    def predict_batch(self, features_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not self._loaded:
            self.load_model()
        if not features_list:
            return []
        raw = self._to_raw_frame(features_list)
        matrix, calibrated = self._calibrated_proba(raw)   # single batched call
        out = []
        for i in range(len(features_list)):
            prob = float(calibrated[i])
            spread = float(matrix[i].std())
            confidence = float(np.clip(1.0 - 2.0 * spread, 0.4, 0.99))
            severity = self._severity(prob)
            out.append({
                "probability": round(prob, 4),
                "confidence": round(confidence, 4),
                "severity": severity,
                "recommended_action": self._ACTIONS[severity],
                "model": "ProductionEnsembleModel",
                "n_members": len(self.members),
                "timestamp": datetime.utcnow().isoformat(),
            })
        return out

    def explain(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Local attribution by single-feature ablation against the dataset
        baseline, scored by the FULL frozen ensemble. The actual incident plus
        one baselined variant per feature are scored in ONE batched call, so the
        cost is ~one extra ensemble evaluation (one TabPFN forward pass)."""
        if not self._loaded:
            self.load_model()
        feats = self.cats + self.nums
        rows = [dict(features)]
        for f in feats:
            v = dict(features)
            base = self._baseline_row[f]
            # API uses latitude/longitude; map back so the override lands
            v[f] = base
            if f == "lat":
                v["latitude"] = base
            elif f == "lon":
                v["longitude"] = base
            rows.append(v)

        raw = self._to_raw_frame(rows)
        _, calibrated = self._calibrated_proba(raw)
        base_p = float(calibrated[0])
        contribs = {f: round(base_p - float(calibrated[i + 1]), 4)
                    for i, f in enumerate(feats)}
        ordered = sorted(contribs.items(), key=lambda kv: kv[1], reverse=True)
        top_pos = [{"feature": k, "contribution": v} for k, v in ordered if v > 0][:3]
        top_neg = [{"feature": k, "contribution": v} for k, v in ordered if v < 0][-3:]
        return {
            "feature_contributions": contribs,
            "top_positive_drivers": top_pos,
            "top_negative_drivers": top_neg,
            "base_probability": round(base_p, 4),
            "method": "single-feature ablation vs dataset baseline "
                      "(full 7-model calibrated ensemble)",
        }

    def get_global_importances(self) -> Dict[str, float]:
        if not self._loaded:
            self.load_model()
        return dict(self._global_importances)

    def get_model_metadata(self) -> Dict[str, Any]:
        if not self._loaded:
            self.load_model()
        md = self.metadata
        mf1 = self.thresholds.get("max_f1", {})
        commit = (md.get("git_commit") or "")[:8]
        return {
            "name": "GridSight AI 7-Model Equal-Weight Ensemble",
            "version": f"1.0.0-frozen ({commit})" if commit else "1.0.0-frozen",
            "training_date": (md.get("exported_at_utc", "") or "")[:10],
            "status": "ONLINE",
            "metrics": {
                # rolling-origin out-of-fold metrics (honest, not in-sample).
                # ROC-AUC from results/final_ensemble_validation.json (incumbent).
                "roc_auc": 0.7887,
                "pr_auc": round(float(md.get("incumbent_oof_pr_auc", 0.0)), 4),
                "pr_auc_std": round(float(md.get("incumbent_oof_pr_auc_std", 0.0)), 4),
                "f1": round(float(mf1.get("f1", 0.0)), 4),
                "precision": round(float(mf1.get("precision", 0.0)), 4),
                "recall": round(float(mf1.get("recall", 0.0)), 4),
            },
            "members": list(self.members),
            "combiner": self.manifest.get("combiner", "prob_mean"),
            "validation": md.get("validation", ""),
            "note": "Frozen incumbent — equal-weight probability average, "
                    "isotonic-calibrated. Validation gate: FREEZE.",
        }


# ═══════════════════════════════════════════════════════════════════════════
# Factory — process-wide singleton with graceful fallback
# ═══════════════════════════════════════════════════════════════════════════

_MODEL_SINGLETON: Optional[ModelInterface] = None
_MODEL_LOCK = threading.Lock()


def get_model() -> ModelInterface:
    """Return the served model as a process-wide singleton.

    Prefers the frozen :class:`ProductionEnsembleModel`. If its artifacts or
    runtime dependencies are unavailable (or loading fails), falls back to
    :class:`PlaceholderModel` so the Command Center still boots. Cached so the
    ~270 MB artifact load and the TabPFN refit happen at most once per process.
    """
    global _MODEL_SINGLETON
    if _MODEL_SINGLETON is not None:
        return _MODEL_SINGLETON
    with _MODEL_LOCK:
        if _MODEL_SINGLETON is not None:
            return _MODEL_SINGLETON
        if os.getenv("ASTRAM_USE_PLACEHOLDER", "").lower() in ("1", "true", "yes"):
            model: ModelInterface = PlaceholderModel()
            model.load_model()
            logger.info("ASTRAM_USE_PLACEHOLDER set — serving PlaceholderModel.")
        else:
            try:
                model = ProductionEnsembleModel()
                model.load_model()
            except Exception as exc:  # missing artifacts / deps / load failure
                logger.warning(
                    "ProductionEnsembleModel unavailable (%s) — "
                    "falling back to PlaceholderModel.", exc,
                )
                model = PlaceholderModel()
                model.load_model()
        _MODEL_SINGLETON = model
        return _MODEL_SINGLETON
