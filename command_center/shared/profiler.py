import os
import time
import json
import atexit
from functools import wraps
from pathlib import Path

ENABLE_PROFILING = os.getenv("GRIDSIGHT_PROFILE", "0") == "1"
PROFILE_DUMP_PATH = Path(__file__).resolve().parents[1] / "cache" / "profiling_results.json"

# In-memory storage for the run
_profile_data = {
    "timings": {},   # func_name -> {"total_time": 0, "count": 0, "calls": []}
    "caches": {},    # cache_name -> {"hits": 0, "misses": 0}
    "payloads": {}   # endpoint -> {"total_bytes": 0, "count": 0, "calls": []}
}

def profile_time(name=None):
    """Decorator to measure execution time of a function."""
    def decorator(func):
        if not ENABLE_PROFILING:
            return func
        
        op_name = name or func.__name__
        if op_name not in _profile_data["timings"]:
            _profile_data["timings"][op_name] = {"total_time": 0.0, "count": 0, "calls": []}
            
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - start
                _profile_data["timings"][op_name]["total_time"] += elapsed
                _profile_data["timings"][op_name]["count"] += 1
                _profile_data["timings"][op_name]["calls"].append(elapsed)
        return wrapper
    return decorator

class Profiler:
    @staticmethod
    def record_cache_hit(cache_name: str):
        if not ENABLE_PROFILING: return
        if cache_name not in _profile_data["caches"]:
            _profile_data["caches"][cache_name] = {"hits": 0, "misses": 0}
        _profile_data["caches"][cache_name]["hits"] += 1

    @staticmethod
    def record_cache_miss(cache_name: str):
        if not ENABLE_PROFILING: return
        if cache_name not in _profile_data["caches"]:
            _profile_data["caches"][cache_name] = {"hits": 0, "misses": 0}
        _profile_data["caches"][cache_name]["misses"] += 1

    @staticmethod
    def record_payload(endpoint: str, size_bytes: int):
        if not ENABLE_PROFILING: return
        if endpoint not in _profile_data["payloads"]:
            _profile_data["payloads"][endpoint] = {"total_bytes": 0, "count": 0, "calls": []}
        _profile_data["payloads"][endpoint]["total_bytes"] += size_bytes
        _profile_data["payloads"][endpoint]["count"] += 1
        _profile_data["payloads"][endpoint]["calls"].append(size_bytes)

def _dump_stats():
    if not ENABLE_PROFILING:
        return
    PROFILE_DUMP_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # Merge with existing stats if multiple processes run (e.g., Streamlit and FastAPI)
    final_data = _profile_data
    if PROFILE_DUMP_PATH.exists():
        try:
            with open(PROFILE_DUMP_PATH, "r") as f:
                existing = json.load(f)
                # simple merge arrays and counts
                for k, v in existing.get("timings", {}).items():
                    if k not in final_data["timings"]:
                        final_data["timings"][k] = v
                    else:
                        final_data["timings"][k]["total_time"] += v["total_time"]
                        final_data["timings"][k]["count"] += v["count"]
                        final_data["timings"][k]["calls"].extend(v["calls"])
                
                for k, v in existing.get("caches", {}).items():
                    if k not in final_data["caches"]:
                        final_data["caches"][k] = v
                    else:
                        final_data["caches"][k]["hits"] += v["hits"]
                        final_data["caches"][k]["misses"] += v["misses"]

                for k, v in existing.get("payloads", {}).items():
                    if k not in final_data["payloads"]:
                        final_data["payloads"][k] = v
                    else:
                        final_data["payloads"][k]["total_bytes"] += v["total_bytes"]
                        final_data["payloads"][k]["count"] += v["count"]
                        final_data["payloads"][k]["calls"].extend(v["calls"])
        except Exception:
            pass

    with open(PROFILE_DUMP_PATH, "w") as f:
        json.dump(final_data, f, indent=2)

atexit.register(_dump_stats)
