"""THROWAWAY: end-to-end verification of the production adapter contract."""
import sys, os, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "command_center"))
from backend.services.model_adapter import ProductionEnsembleModel, get_model, PlaceholderModel

t = time.time()
m = ProductionEnsembleModel()
m.load_model()
print(f"[load] {time.time()-t:.1f}s  members={m.members}", flush=True)

feat = {"event_type": "unplanned", "event_cause": "accident", "veh_type": "car",
        "corridor": "ORR", "police_station": "Madiwala", "zone": "South",
        "latitude": 12.9716, "longitude": 77.5946, "hour": 18, "weekday": 2}

# page-2 style payload missing veh_type -> must not crash
feat_missing = {"event_type": "planned", "event_cause": "road_work",
                "corridor": "Unknown", "police_station": "Unknown", "zone": "Unknown",
                "latitude": 13.0, "longitude": 77.6, "hour": 3, "weekday": 6}

p = m.predict(feat)
print(f"[predict] {json.dumps(p)}", flush=True)
assert {"probability", "confidence", "severity", "recommended_action",
        "feature_contributions"} <= set(p), "predict keys missing"
assert 0.0 <= p["probability"] <= 1.0
assert p["severity"] in ("LOW", "MEDIUM", "HIGH", "CRITICAL")
assert isinstance(p["feature_contributions"], dict) and p["feature_contributions"]

# cache: identical payload returns instantly + identical value
tc = time.time(); p2 = m.predict(feat)
assert p2["probability"] == p["probability"]
print(f"[cache-hit] {time.time()-tc:.3f}s (must be ~0)", flush=True)

# batch with a missing-column payload (one TabPFN call for both rows)
b = m.predict_batch([feat, feat_missing])
print(f"[predict_batch] {json.dumps(b)}", flush=True)
assert len(b) == 2 and all(0 <= x["probability"] <= 1 for x in b)

e = m.explain(feat)
print(f"[explain] {json.dumps(e)}", flush=True)
assert {"feature_contributions", "top_positive_drivers", "top_negative_drivers"} <= set(e)
for d in e["top_positive_drivers"] + e["top_negative_drivers"]:
    assert "feature" in d and "contribution" in d

md = m.get_model_metadata()
print(f"[metadata] {json.dumps(md)}", flush=True)
assert {"name", "version", "training_date", "status", "metrics"} <= set(md)
for k in ("roc_auc", "pr_auc", "f1", "precision", "recall"):
    assert k in md["metrics"], f"metric {k} missing"

# factory returns a working singleton
fm = get_model()
assert fm.get_model_metadata()["name"]
print(f"[factory] type={type(fm).__name__}", flush=True)
print(f"\nALL CONTRACT CHECKS PASSED  total={time.time()-t:.1f}s", flush=True)
