"""MapmyIndia (Mappls) Geospatial Intelligence Service.

Handles reverse geocoding and place intelligence with graceful fallbacks
and file-based caching for offline/repeated demo usage.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import requests
from functools import lru_cache

from backend.config import MAPMYINDIA_API_KEY, BASE_DIR
from shared.profiler import Profiler, profile_time

logger = logging.getLogger(__name__)

CACHE_DIR = BASE_DIR / "cache"
CACHE_FILE = CACHE_DIR / "mapmyindia_cache.json"

class MapmyIndiaService:
    def __init__(self):
        self.api_key = MAPMYINDIA_API_KEY
        self.cache: Dict[str, Any] = self._load_cache()
        self.session = requests.Session()
    
    def _load_cache(self) -> Dict[str, Any]:
        """Load the local JSON cache to prevent duplicate API calls."""
        if not CACHE_FILE.exists():
            return {}
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load MapmyIndia cache: {e}")
            return {}
            
    def _save_cache(self):
        """Persist the cache to disk."""
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            with open(CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save MapmyIndia cache: {e}")
            
    def _make_request(self, cache_key: str, url: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Make a request to MapmyIndia API with caching and error handling."""
        if cache_key in self.cache:
            Profiler.record_cache_hit("MapmyIndiaCache")
            return self.cache[cache_key]
            
        Profiler.record_cache_miss("MapmyIndiaCache")
        if not self.api_key:
            logger.debug("MapmyIndia API key missing. Returning fallback.")
            return None
            
        try:
            resp = self.session.get(url, params=params, timeout=3.0)
            if resp.status_code == 200:
                data = resp.json()
                self.cache[cache_key] = data
                self._save_cache()
                return data
            else:
                logger.warning(f"MapmyIndia API Error {resp.status_code}: {resp.text}")
                return None
        except Exception as e:
            logger.error(f"MapmyIndia API Request Exception: {e}")
            return None

    @profile_time("mapmyindia.reverse_geocode")
    def reverse_geocode(self, lat: float, lon: float) -> Optional[Dict[str, str]]:
        """Convert lat/lon to a human-readable address."""
        # Round coordinates slightly to improve cache hits for nearby points
        rlat, rlon = round(lat, 4), round(lon, 4)
        cache_key = f"rev_{rlat}_{rlon}"
        
        url = f"https://apis.mappls.com/advancedmaps/v1/{self.api_key}/rev_geocode"
        params = {"lat": rlat, "lng": rlon}
        
        data = self._make_request(cache_key, url, params)
        if not data or "results" not in data or len(data["results"]) == 0:
            return None
            
        result = data["results"][0]
        
        # Build a clean address string
        parts = []
        if result.get("houseNumber"): parts.append(result["houseNumber"])
        if result.get("houseName"): parts.append(result["houseName"])
        if result.get("poi"): parts.append(result["poi"])
        if result.get("street"): parts.append(result["street"])
        if result.get("subLocality"): parts.append(result["subLocality"])
        if result.get("locality"): parts.append(result["locality"])
        
        city = result.get("city", "Bengaluru")
        
        return {
            "formatted_address": ", ".join(parts) if parts else city,
            "city": city,
            "district": result.get("district", ""),
            "state": result.get("state", ""),
            "pincode": result.get("pincode", "")
        }

    @profile_time("mapmyindia.nearby_places")
    def nearby_places(self, lat: float, lon: float, radius: int = 1000) -> List[Dict[str, str]]:
        """Find notable landmarks near the location."""
        rlat, rlon = round(lat, 4), round(lon, 4)
        cache_key = f"nearby_{rlat}_{rlon}_{radius}"
        
        url = f"https://apis.mappls.com/advancedmaps/v1/{self.api_key}/nearby_search"
        params = {"lat": rlat, "lng": rlon, "radius": radius}
        
        data = self._make_request(cache_key, url, params)
        if not data or "suggestedLocations" not in data:
            return []
            
        places = []
        for loc in data["suggestedLocations"][:3]: # Keep top 3 for the UI
            places.append({
                "name": loc.get("poi", loc.get("placeName", "Unknown Place")),
                "address": loc.get("placeAddress", ""),
                "distance": loc.get("distance", 0)
            })
            
        return places

# Singleton instance
mapmyindia_service = MapmyIndiaService()
