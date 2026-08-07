"""
Client for the National Weather Service API (api.weather.gov).

No API key required - the NWS API is free and open.
Documentation: https://www.weather.gov/documentation/services-web-api
"""

import hashlib
import re
from typing import Any

import requests

_BASE_URL = "https://api.weather.gov"
_DEFAULT_TIMEOUT = 30
_USER_AGENT = "Databricks-Weather-App/1.0"

# Regex to extract lat/lon from strings like "Chicago, IL" or "41.8781,-87.6298"
_LAT_LON_RE = re.compile(r"^(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)$")


class WeatherClient:
    """Thin wrapper around the National Weather Service API."""

    def __init__(self, base_url: str | None = None, timeout: int = _DEFAULT_TIMEOUT):
        self.base_url = (base_url or _BASE_URL).rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()
        # NWS requires a User-Agent header
        self._session.headers.update(
            {
                "User-Agent": _USER_AGENT,
                "Accept": "application/json",
            }
        )

    def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        """Make a GET request to the NWS API."""
        resp = self._session.get(f"{self.base_url}{path}", params=params, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def resolve_location(self, location: str) -> dict | None:
        """
        Resolve a location string to lat/lon coordinates using Nominatim geocoding.
        
        Accepts:
        - "lat,lon" format (e.g., "41.8781,-87.6298")
        - City/state format (e.g., "Chicago, IL")
        
        Returns dict with 'lat', 'lon', and 'location' keys, or None if resolution fails.
        """
        location = location.strip()
        
        try:
            response = requests.get(
                "https://nominatim.openstreetmap.org/search",
                params={
                    "q": location,
                    "format": "jsonv2",
                    "limit": 1
                },
                headers={
                    "User-Agent": "weather-client/1.0"
                },
                timeout=30
            )
            response.raise_for_status()
            
            results = response.json()
            if not results:
                return None
            
            result = results[0]
            return {
                "lat": float(result["lat"]),
                "lon": float(result["lon"]),
                "location": location
            }
        except (requests.RequestException, KeyError, ValueError, IndexError):
            return None

    def get_grid_point(self, lat: float, lon: float) -> dict | None:
        """
        Resolve lat/lon to a NWS grid point.
        
        Returns dict with 'office', 'gridX', 'gridY' keys needed for forecast calls,
        or None if the location is not covered by NWS (e.g., outside US).
        """
        try:
            data = self.get(f"/points/{lat},{lon}")
            properties = data.get("properties", {})
            
            # Properties already contains gridId, gridX, and gridY - use them directly
            return {
                "office": properties.get("gridId"),
                "gridX": properties.get("gridX"),
                "gridY": properties.get("gridY"),
            }
        except (requests.HTTPError, KeyError, ValueError, IndexError):
            return None

    def get_active_alerts(self, lat: float, lon: float, limit: int = 50) -> list[dict]:
        """
        Fetch active weather alerts for a specific lat/lon point.
        
        Args:
            lat: Latitude
            lon: Longitude
            limit: Maximum number of alerts to return
        
        Returns list of alert dicts formatted for database insertion with:
            - id: Unique alert ID (features.properties.id)
            - location: Area description (features.properties.areaDesc)
            - source_type: Alert type (features.properties.@type)
            - headline: Brief summary (features.properties.headline)
            - narrative_text: Full alert description (features.properties.description)
            - issued_at: When alert becomes effective (features.properties.effective)
            - payload: The entire feature JSON object
            - synced_at: Timestamp when row was fetched (ISO format)
        """
        from datetime import datetime, timezone
        
        params = {
            "point": f"{lat},{lon}"
        }
        
        try:
            data = self.get("/alerts/active", params=params)
            features = data.get("features", [])
            
            alerts = []
            synced_at = datetime.now(timezone.utc).isoformat()
            
            for feature in features[:limit]:
                properties = feature.get("properties", {})
                alerts.append({
                    "id": properties.get("id"),
                    "location": properties.get("areaDesc"),
                    "source_type": "alerts",
                    "headline": properties.get("headline"),
                    "narrative_text": properties.get("description"),
                    "issued_at": properties.get("effective"),
                    "payload": feature,  # Entire feature JSON
                    "synced_at": synced_at,
                })
            
            return alerts
        except requests.HTTPError:
            return []

    def get_forecast(self, office: str, grid_x: int, grid_y: int, location: str, limit: int = 50) -> list[dict]:
        """
        Fetch multi-day forecast for a grid point.
        
        Args:
            office: NWS office code (e.g., "LOT")
            grid_x: Grid X coordinate
            grid_y: Grid Y coordinate
            location: Human-readable location name for the forecast
            limit: Maximum number of forecast periods to return
        
        Returns list of forecast dicts formatted for database insertion with:
            - id: Unique ID (location + startTime)
            - location: Location name
            - source_type: "forecast"
            - headline: Forecast period name (e.g., "Tonight")
            - narrative_text: Detailed forecast text
            - issued_at: When forecast was generated
            - effective_at: When this period starts
            - payload: The entire period JSON object
            - synced_at: Timestamp when row was fetched
        """
        from datetime import datetime, timezone
        
        try:
            data = self.get(f"/gridpoints/{office}/{grid_x},{grid_y}/forecast")
            properties = data.get("properties", {})
            periods = properties.get("periods", [])
            
            forecasts = []
            synced_at = datetime.now(timezone.utc).isoformat()
            
            for period in periods[:limit]:
                forecasts.append({
                    "id": f"{location}_{period.get('startTime')}",
                    "location": location,
                    "source_type": "forecast",
                    "headline": period.get("name"),
                    "narrative_text": period.get("detailedForecast"),
                    "issued_at": properties.get("generatedAt"),
                    "effective_at": period.get("startTime"),
                    "payload": period,
                    "synced_at": synced_at,
                })
            
            return forecasts
        except requests.HTTPError:
            return []

    def get_hourly_forecast(self, office: str, grid_x: int, grid_y: int, location: str, limit: int = 24) -> list[dict]:
        """
        Fetch hourly forecast for a grid point.
        
        Args:
            office: NWS office code (e.g., "LOT")
            grid_x: Grid X coordinate
            grid_y: Grid Y coordinate
            location: Human-readable location name for the forecast
            limit: Maximum number of hourly periods to return (default 24 hours)
        
        Returns list of hourly forecast dicts formatted for database insertion
        (same structure as get_forecast).
        """
        from datetime import datetime, timezone
        
        try:
            data = self.get(f"/gridpoints/{office}/{grid_x},{grid_y}/forecast/hourly")
            properties = data.get("properties", {})
            periods = properties.get("periods", [])
            
            forecasts = []
            synced_at = datetime.now(timezone.utc).isoformat()
            
            for period in periods[:limit]:
                forecasts.append({
                    "id": f"{location}_{period.get('startTime')}",
                    "location": location,
                    "source_type": "hourly_forecast",
                    "headline": period.get("name"),
                    "narrative_text": period.get("detailedForecast"),
                    "issued_at": properties.get("generatedAt"),
                    "effective_at": period.get("startTime"),
                    "payload": period,
                    "synced_at": synced_at,
                })
            
            return forecasts
        except requests.HTTPError:
            return []


def generate_document_id(source_type: str, location: str, issued_at: str, event: str = "") -> str:
    """
    Generate a stable, deterministic document ID for deduplication.
    
    For alerts: hash of (alert_id or event + location + issued_at)
    For forecasts: hash of (location + period_start_time)
    """
    key = f"{source_type}:{location}:{issued_at}:{event}"
    return hashlib.sha256(key.encode()).hexdigest()[:32]
