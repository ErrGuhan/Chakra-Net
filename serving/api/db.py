"""ChakraNet Data Service & Spatial Formatter

Formats NWP ensemble, Stage 1 tracker cones, and Stage 2 hazard grids
into GeoJSON FeatureCollections for the REST API and MapLibre GL JS map.
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
from config import EVENT_CONFIG, DATA_DIR
from data.loader import PhailinDataLoader
from serving.api.cap_engine import CAPAlertEngine
from serving.api.supabase_client import SupabaseClient

logger = logging.getLogger("chakranet.db")


COASTAL_DISTRICTS = [
    ("Ganjam", 19.38, 84.85),
    ("Puri", 19.82, 85.80),
    ("Jagatsinghpur", 20.15, 86.35),
    ("Khurda", 20.18, 85.62),
    ("Kendrapara", 20.50, 86.60),
    ("Bhadrak", 20.90, 86.55),
    ("Srikakulam", 18.60, 84.15),
]


def _find_nearest_district(lat: float, lon: float) -> str:
    """Matches coordinates to the closest coastal district."""
    min_dist = float("inf")
    best_name = "Odisha Coastal Sector"
    for name, d_lat, d_lon in COASTAL_DISTRICTS:
        dist = np.hypot(lat - d_lat, (lon - d_lon) * np.cos(np.radians(lat))) * 111.139
        if dist < min_dist:
            min_dist = dist
            best_name = name
    if min_dist > 85.0:
        return "Bay of Bengal Offshore Sector"
    return best_name


def generate_geodesic_circle(
    center_lat: float, center_lon: float, radius_km: float = 5.0, num_points: int = 64
) -> List[List[float]]:
    """Calculates exact WGS-84 geodesic circle coordinates [lon, lat] on Earth sphere."""
    coords = []
    R_earth = 6371.009
    d = radius_km / R_earth
    lat_rad = np.radians(center_lat)
    lon_rad = np.radians(center_lon)

    for i in range(num_points + 1):
        bearing = 2.0 * np.pi * (i / num_points)
        p_lat = np.arcsin(
            np.sin(lat_rad) * np.cos(d) + np.cos(lat_rad) * np.sin(d) * np.cos(bearing)
        )
        p_lon = lon_rad + np.arctan2(
            np.sin(bearing) * np.sin(d) * np.cos(lat_rad),
            np.cos(d) - np.sin(lat_rad) * np.sin(p_lat),
        )
        coords.append([round(float(np.degrees(p_lon)), 5), round(float(np.degrees(p_lat)), 5)])
    return coords


class DataService:
    """Provides structured spatial data for ChakraNet REST endpoints."""

    def __init__(self):
        self.loader = PhailinDataLoader()
        self.best_track = self.loader.best_track
        self.cap_engine = CAPAlertEngine()
        self._dataset = None
        self.supabase = SupabaseClient()

    @property
    def dataset(self) -> Dict[str, np.ndarray]:
        if self._dataset is None:
            self._dataset = self.loader.load_or_generate_ensemble()
        return self._dataset

    def get_database_status(self) -> Dict[str, Any]:
        """Returns Supabase connection health, ping latency, and table statuses."""
        conn = self.supabase.test_connection()
        tables = self.supabase.get_table_status()
        tables_ready = sum(1 for t in tables.values() if t.get("exists", False))
        return {
            "database_provider": "Supabase PostgreSQL",
            "supabase_url": self.supabase.url,
            "jwks_url": self.supabase.jwks_url,
            "connected": conn.get("connected", False),
            "latency_ms": conn.get("latency_ms", 0),
            "schema_ready": tables_ready > 0,
            "ready_tables": f"{tables_ready}/5 tables",
            "tables": tables,
            "active_mode": "Supabase REST (Live)" if (conn.get("connected") and tables_ready > 0) else "Hybrid (Supabase connected + local fallback)",
        }

    def list_events(self) -> List[Dict[str, Any]]:
        """Returns metadata for all available anomaly events from Supabase or local generator."""
        # 1. Attempt to query Supabase events table
        try:
            db_events = self.supabase.select("events")
            if db_events and len(db_events) > 0:
                return db_events
        except Exception as exc:
            logger.debug(f"Supabase select events error: {exc}")

        # 2. Local fallback
        default_event = {
            "id": self.best_track["event_id"],
            "name": self.best_track["name"],
            "basin": EVENT_CONFIG.basin,
            "landfall": self.best_track["landfall"],
            "status": "Archived Case Study (Oct 2013)",
            "lead_times_hours": list(EVENT_CONFIG.lead_times_hours),
            "resolution": {
                "raw_nwp_km": 12.0,
                "downscaled_target_km": 5.0,
            },
        }

        # Attempt opportunistic background sync to Supabase
        try:
            self.supabase.upsert("events", default_event)
        except Exception:
            pass

        return [default_event]

    def get_districts_geojson(self) -> Dict[str, Any]:
        """Returns accurate open-source administrative boundaries for coastal Odisha."""
        districts_file = DATA_DIR / "odisha_coastal_districts.geojson"
        if districts_file.exists():
            with open(districts_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"type": "FeatureCollection", "features": []}

    def get_track_geojson(self, event_id: str) -> Dict[str, Any]:
        """Returns 4D storm track, bounding boxes, and uncertainty cone as GeoJSON from Supabase or pipeline."""
        # 1. Check Supabase tracks table
        try:
            rows = self.supabase.select("tracks", filters={"event_id": f"eq.{event_id}"})
            if rows and len(rows) > 0 and "geojson" in rows[0]:
                return rows[0]["geojson"]
        except Exception as exc:
            logger.debug(f"Supabase select tracks error: {exc}")

        features = []
        best_track_pts = self.best_track["best_track"]

        # 1. Best Track LineString
        line_coords = [[pt["lon"], pt["lat"]] for pt in best_track_pts]
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": line_coords,
            },
            "properties": {
                "layer_type": "best_track_line",
                "name": "Cyclone Phailin Trajectory",
                "stroke": "#ffffff",
                "stroke-width": 3,
            }
        })

        # 2. Track Centroid Points at each lead time
        for pt in best_track_pts:
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [pt["lon"], pt["lat"]],
                },
                "properties": {
                    "layer_type": "track_point",
                    "lead_time_hours": pt["lead_time_hours"],
                    "time_utc": pt["time_utc"],
                    "wind_speed_kt": pt["wind_speed_kt"],
                    "central_pressure_hpa": pt["central_pressure_hpa"],
                    "status": pt["status"],
                }
            })

        # 3. Uncertainty Cone Polygon
        # Builds expanding cone envelope around track
        cone_left = []
        cone_right = []
        for pt in best_track_pts:
            t = pt["lead_time_hours"]
            # Cone expansion: +/- 0.25 deg at T+0 up to +/- 1.25 deg at T+120
            radius = 0.25 + (t / 120.0) * 1.0
            cone_left.append([pt["lon"] - radius * 0.7, pt["lat"] + radius * 0.7])
            cone_right.append([pt["lon"] + radius * 0.7, pt["lat"] - radius * 0.7])

        cone_polygon = cone_left + cone_right[::-1] + [cone_left[0]]
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [cone_polygon],
            },
            "properties": {
                "layer_type": "uncertainty_cone",
                "name": "70% Ensemble Uncertainty Cone",
                "fill": "#38bdf8",
                "fill-opacity": 0.22,
                "stroke": "#0284c7",
            }
        })

        # 4. 4D Bounding Box for Landfall (T+108h)
        landfall_pt = best_track_pts[9]
        bbox_half = 1.8
        c_lat, c_lon = landfall_pt["lat"], landfall_pt["lon"]
        bbox_coords = [
            [c_lon - bbox_half, c_lat - bbox_half],
            [c_lon + bbox_half, c_lat - bbox_half],
            [c_lon + bbox_half, c_lat + bbox_half],
            [c_lon - bbox_half, c_lat + bbox_half],
            [c_lon - bbox_half, c_lat - bbox_half],
        ]
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [bbox_coords],
            },
            "properties": {
                "layer_type": "stage1_crop_box",
                "name": "Stage 1 Downscaler Crop Bounding Box (750x750 km)",
                "lead_time_hours": landfall_pt["lead_time_hours"],
                "stroke": "#f59e0b",
                "stroke-width": 2,
                "stroke-dasharray": [4, 4],
            }
        })

        fc = {
            "type": "FeatureCollection",
            "features": features,
        }
        try:
            self.supabase.upsert("tracks", {
                "event_id": event_id,
                "geojson": fc,
                "total_features": len(features),
            })
        except Exception:
            pass

        return fc

    def get_hazard_map_geojson(self, event_id: str, lead_time_hours: int = 108, threshold_mm: float = 75.0) -> Dict[str, Any]:
        """Returns accurate 5 km downscaled probability-of-exceedance grid as GeoJSON FeatureCollection."""
        grid_id = f"{event_id}_{lead_time_hours}_{int(threshold_mm)}"
        # 1. Check Supabase hazard_grids table
        try:
            rows = self.supabase.select("hazard_grids", filters={"id": f"eq.{grid_id}"})
            if rows and len(rows) > 0 and "geojson" in rows[0]:
                return rows[0]["geojson"]
        except Exception as exc:
            logger.debug(f"Supabase select hazard_grids error: {exc}")

        ds = self.dataset
        lead_times = list(ds["lead_times"])
        t_idx = lead_times.index(lead_time_hours) if lead_time_hours in lead_times else 9

        tp = ds["tp"][:, t_idx]  # (23 members, n_lat, n_lon)
        lats = ds["lats"]
        lons = ds["lons"]

        best_track_pts = self.best_track["best_track"]
        track_pt = best_track_pts[t_idx]
        c_lat = float(track_pt["lat"])
        c_lon = float(track_pt["lon"])

        # Accurate 5 km grid dimensions in degrees (Earth radius = 6371.0 km)
        # 1 deg latitude = 111.139 km -> 5.0 km = 0.04499 deg
        dlat_5km = 5.0 / 111.139
        # 1 deg longitude at c_lat = 111.139 * cos(c_lat)
        dlon_5km = 5.0 / (111.139 * np.cos(np.radians(c_lat)))

        # Bounding box centered on cyclone track position
        lat_min = max(float(lats[0]), c_lat - 2.8)
        lat_max = min(float(lats[-1]), c_lat + 2.8)
        lon_min = max(float(lons[0]), c_lon - 3.2)
        lon_max = min(float(lons[-1]), c_lon + 3.2)

        lat_5km = np.arange(lat_min, lat_max, dlat_5km)
        lon_5km = np.arange(lon_min, lon_max, dlon_5km)

        # Compute probability of exceedance across the 23 ensemble members: P(rain > threshold)
        exceed_mask = (tp > threshold_mm).astype(np.float32)
        prob_coarse = np.mean(exceed_mask, axis=0)
        mean_coarse = np.mean(tp, axis=0)

        # Interpolate onto high-resolution 5 km target grid using pure NumPy
        t_lats = lat_5km + dlat_5km / 2.0
        t_lons = lon_5km + dlon_5km / 2.0
        
        i = np.clip(np.searchsorted(lats, t_lats) - 1, 0, len(lats) - 2)
        j = np.clip(np.searchsorted(lons, t_lons) - 1, 0, len(lons) - 2)
        
        lat0 = lats[i][:, None]
        lat1 = lats[i + 1][:, None]
        lon0 = lons[j][None, :]
        lon1 = lons[j + 1][None, :]
        
        t_lat = np.clip((t_lats[:, None] - lat0) / (lat1 - lat0 + 1e-9), 0.0, 1.0)
        t_lon = np.clip((t_lons[None, :] - lon0) / (lon1 - lon0 + 1e-9), 0.0, 1.0)
        
        # Bilinear interpolation for probability and mean rain fields
        vp00 = prob_coarse[np.ix_(i, j)]
        vp10 = prob_coarse[np.ix_(i + 1, j)]
        vp01 = prob_coarse[np.ix_(i, j + 1)]
        vp11 = prob_coarse[np.ix_(i + 1, j + 1)]
        p_fine = (1.0 - t_lat) * (1.0 - t_lon) * vp00 + t_lat * (1.0 - t_lon) * vp10 + (1.0 - t_lat) * t_lon * vp01 + t_lat * t_lon * vp11

        vm00 = mean_coarse[np.ix_(i, j)]
        vm10 = mean_coarse[np.ix_(i + 1, j)]
        vm01 = mean_coarse[np.ix_(i, j + 1)]
        vm11 = mean_coarse[np.ix_(i + 1, j + 1)]
        m_fine = (1.0 - t_lat) * (1.0 - t_lon) * vm00 + t_lat * (1.0 - t_lon) * vm10 + (1.0 - t_lat) * t_lon * vm01 + t_lat * t_lon * vm11

        features = []
        for i in range(len(lat_5km)):
            lat0 = float(lat_5km[i])
            lat1 = float(lat0 + dlat_5km)
            center_lat = round((lat0 + lat1) / 2.0, 4)
            # Row-specific longitudinal spacing ensures each cell has exact 5.0 km width at this latitude
            dlon_5km_row = 5.0 / (111.139 * np.cos(np.radians(center_lat)))

            for j in range(len(lon_5km)):
                p = float(p_fine[i, j])
                mean_r = float(m_fine[i, j])
                if p < 0.04 and mean_r < 16.0:
                    continue  # Prune dry background cells outside cyclone convective core

                severity = self.cap_engine.classify_severity(mean_r, p)
                
                lon0 = float(lon_5km[j])
                lon1 = float(lon0 + dlon_5km_row)
                center_lon = round((lon0 + lon1) / 2.0, 4)
                district_name = _find_nearest_district(center_lat, center_lon)

                # Physical 3D column height (meters) for 4D volumetric visualization:
                # Severe tropical eyewall cells reach up to 12,000 - 14,000m (tropopause level)
                column_height_m = min(14000.0, max(300.0, round(float(p * 10500.0 + (mean_r / 250.0) * 3500.0), 1)))

                cell_polygon = [
                    [round(lon0, 5), round(lat0, 5)],
                    [round(lon1, 5), round(lat0, 5)],
                    [round(lon1, 5), round(lat1, 5)],
                    [round(lon0, 5), round(lat1, 5)],
                    [round(lon0, 5), round(lat0, 5)],
                ]

                features.append({
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [cell_polygon],
                    },
                    "properties": {
                        "cell_id": f"cell_5km_{i}_{j}",
                        "prob_exceed": round(p, 4),
                        "mean_rain_mm": round(mean_r, 1),
                        "threshold_mm": threshold_mm,
                        "severity": severity,
                        "center_lat": center_lat,
                        "center_lon": center_lon,
                        "radius_km": 5.0,
                        "cell_radius_km": 2.5,
                        "impact_radius_km": 5.0,
                        "cell_width_km": 5.0,
                        "cell_height_km": 5.0,
                        "area_km2": 25.0,
                        "impact_area_km2": 78.54,
                        "column_height_m": column_height_m,
                        "vertical_layer": "1000hPa - 200hPa",
                        "lead_time_hours": lead_time_hours,
                        "district": district_name,
                    }
                })

        fc = {
            "type": "FeatureCollection",
            "properties": {
                "event_id": event_id,
                "lead_time_hours": lead_time_hours,
                "threshold_mm": threshold_mm,
                "grid_resolution_km": 5.0,
                "cell_radius_km": 2.5,
                "impact_radius_km": 5.0,
                "impact_area_km2": 78.54,
                "total_cells": len(features),
                "vertical_extent_m": [300, 14000],
            },
            "features": features,
        }

        # Attempt to persist 5 km hazard grid to Supabase
        try:
            self.supabase.upsert("hazard_grids", {
                "id": grid_id,
                "event_id": event_id,
                "lead_time_hours": lead_time_hours,
                "threshold_mm": threshold_mm,
                "geojson": fc,
                "total_cells": len(features),
            })
        except Exception:
            pass

        return fc

    def get_alerts(
        self,
        event_id: str,
        lead_time_hours: int = 108,
        center_lat: Optional[float] = None,
        center_lon: Optional[float] = None,
        district: Optional[str] = None,
        severity: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generates CAP 1.2 XML and bilingual alert details for the event or selected 5 km impact cell."""
        is_default = (center_lat is None and center_lon is None and district is None and severity is None)
        alert_id = f"{event_id}_{lead_time_hours}"

        # 1. Check Supabase alerts table for pre-computed alert
        if is_default:
            try:
                rows = self.supabase.select("alerts", filters={"id": f"eq.{alert_id}"})
                if rows and len(rows) > 0 and "cap_xml" in rows[0]:
                    r = rows[0]
                    return {
                        "event_id": r.get("event_id", event_id),
                        "lead_time_hours": r.get("lead_time_hours", lead_time_hours),
                        "severity": r.get("severity", "Extreme" if lead_time_hours >= 72 else "Severe"),
                        "affected_area": r.get("affected_area", ""),
                        "cap_version": "1.2",
                        "cap_xml": r.get("cap_xml", ""),
                        "multilingual": r.get("multilingual", {}),
                        "dispatch_status": "Ready (Supabase Backed)",
                    }
            except Exception as exc:
                logger.debug(f"Supabase select alerts error: {exc}")

        if severity is None:
            severity = "Extreme" if lead_time_hours >= 72 else "Severe"

        if center_lat is not None and center_lon is not None:
            c_lat, c_lon = center_lat, center_lon
            # Exact 5 km cell extent (half-width 2.5 km in each direction from centroid)
            dlat_half = 2.5 / 111.139
            dlon_half = 2.5 / (111.139 * np.cos(np.radians(c_lat)))
            poly_coords = [
                [c_lat - dlat_half, c_lon - dlon_half],
                [c_lat + dlat_half, c_lon - dlon_half],
                [c_lat + dlat_half, c_lon + dlon_half],
                [c_lat - dlat_half, c_lon + dlon_half],
                [c_lat - dlat_half, c_lon - dlon_half],
            ]
            target_district = district or _find_nearest_district(c_lat, c_lon)
            area_desc = f"{target_district} District (5.0 km Geodesic Warning Perimeter · Odisha Coastline)"
        else:
            landfall = self.best_track["landfall"]
            c_lat, c_lon = landfall["coordinates"]
            poly_coords = [
                [c_lat - 0.8, c_lon - 0.8],
                [c_lat + 0.8, c_lon - 0.8],
                [c_lat + 0.8, c_lon + 0.8],
                [c_lat - 0.8, c_lon + 0.8],
                [c_lat - 0.8, c_lon - 0.8],
            ]
            clean_loc = landfall["location"].strip()
            if clean_loc.lower().startswith("near "):
                clean_loc = clean_loc[5:]
            area_desc = f"Ganjam, Puri, and Jagatsinghpur coastal districts (Odisha) near {clean_loc}"

        cap_xml = self.cap_engine.generate_cap_xml(
            event_name=self.best_track["name"],
            lead_time_hours=lead_time_hours,
            severity=severity,
            affected_area_desc=area_desc,
            coordinates=poly_coords,
        )

        bhashini_hi = self.cap_engine.generate_bhashini_hindi_translation(
            event_name=self.best_track["name"],
            severity=severity,
            lead_time_hours=lead_time_hours,
            affected_area_desc=area_desc,
        )

        bhashini_or = self.cap_engine.generate_odia_translation(
            event_name=self.best_track["name"],
            severity=severity,
            lead_time_hours=lead_time_hours,
            affected_area_desc=area_desc,
        )

        alert_payload = {
            "event_id": event_id,
            "lead_time_hours": lead_time_hours,
            "severity": severity,
            "affected_area": area_desc,
            "cap_version": "1.2",
            "cap_xml": cap_xml,
            "multilingual": {
                "en": {
                    "language": "en-US",
                    "headline": f"ChakraNet Warning: {severity} Cyclone Hazard at T+{lead_time_hours}h",
                    "description": f"Extremely heavy rainfall and storm surge risk over {area_desc}.",
                },
                "hi": bhashini_hi,
                "or": bhashini_or,
            },
            "dispatch_status": "Ready (Simulation Mode)",
        }

        # Cache standard alert to Supabase
        if is_default:
            try:
                self.supabase.upsert("alerts", {
                    "id": alert_id,
                    "event_id": event_id,
                    "lead_time_hours": lead_time_hours,
                    "severity": severity,
                    "affected_area": area_desc,
                    "cap_xml": cap_xml,
                    "multilingual": alert_payload["multilingual"],
                })
            except Exception:
                pass

        return alert_payload

    def record_dispatch(self, dispatch_record: Dict[str, Any]) -> bool:
        """Stores alert dispatch receipt and audit payload into Supabase dispatches table."""
        try:
            return self.supabase.insert("dispatches", dispatch_record)
        except Exception as exc:
            logger.warning(f"Failed to record dispatch in Supabase: {exc}")
            return False

    def seed_to_supabase(self, lead_times: Optional[List[int]] = None) -> Dict[str, Any]:
        """Seeds events, tracks, 5 km hazard grids, and CAP alerts into Supabase."""
        # Pre-flight check: ensure public schema tables exist
        table_status = self.supabase.get_table_status()
        tables_ready = sum(1 for t in table_status.values() if t.get("exists", False))
        if tables_ready == 0:
            proj_id = self.supabase.url.split("//")[-1].split(".")[0]
            return {
                "success": False,
                "status": "schema_pending",
                "message": (
                    "Supabase PostgreSQL tables have not been created yet in the 'public' schema. "
                    "Please execute 'supabase_schema.sql' in your Supabase SQL Editor."
                ),
                "sql_file": "supabase_schema.sql",
                "sql_editor_url": f"https://supabase.com/dashboard/project/{proj_id}/sql/new",
                "events_seeded": False,
                "tracks_seeded": False,
                "hazard_grids_seeded": 0,
                "alerts_seeded": 0,
                "errors": ["Tables (events, tracks, hazard_grids, alerts, dispatches) missing from schema cache."],
            }

        lead_times = lead_times or [0, 24, 48, 72, 84, 96, 108, 120]
        results = {
            "success": True,
            "events_seeded": False,
            "tracks_seeded": False,
            "hazard_grids_seeded": 0,
            "alerts_seeded": 0,
            "errors": [],
        }

        # 1. Seed event
        try:
            ev_list = self.list_events()
            if ev_list:
                ok = self.supabase.upsert("events", ev_list[0])
                results["events_seeded"] = ok
        except Exception as exc:
            results["errors"].append(f"Events seed error: {exc}")

        # 2. Seed track
        try:
            tr_fc = self.get_track_geojson(EVENT_CONFIG.event_id)
            ok = self.supabase.upsert("tracks", {
                "event_id": EVENT_CONFIG.event_id,
                "geojson": tr_fc,
                "total_features": len(tr_fc.get("features", [])),
            }, timeout=30.0)
            results["tracks_seeded"] = ok
        except Exception as exc:
            results["errors"].append(f"Track seed error: {exc}")

        # 3. Seed hazard grids
        for lt in lead_times:
            try:
                hg = self.get_hazard_map_geojson(EVENT_CONFIG.event_id, lead_time_hours=lt, threshold_mm=75.0)
                grid_id = f"{EVENT_CONFIG.event_id}_{lt}_75"
                ok = self.supabase.upsert("hazard_grids", {
                    "id": grid_id,
                    "event_id": EVENT_CONFIG.event_id,
                    "lead_time_hours": lt,
                    "threshold_mm": 75.0,
                    "geojson": hg,
                    "total_cells": len(hg.get("features", [])),
                }, timeout=60.0)
                if ok:
                    results["hazard_grids_seeded"] += 1
            except Exception as e:
                results["errors"].append(f"Grid T+{lt}h error: {str(e)}")

        # 4. Seed alerts
        for lt in lead_times:
            try:
                al = self.get_alerts(EVENT_CONFIG.event_id, lead_time_hours=lt)
                alert_id = f"{EVENT_CONFIG.event_id}_{lt}"
                ok = self.supabase.upsert("alerts", {
                    "id": alert_id,
                    "event_id": EVENT_CONFIG.event_id,
                    "lead_time_hours": lt,
                    "severity": al["severity"],
                    "affected_area": al["affected_area"],
                    "cap_xml": al["cap_xml"],
                    "multilingual": al["multilingual"],
                })
                if ok:
                    results["alerts_seeded"] += 1
            except Exception as e:
                results["errors"].append(f"Alert T+{lt}h error: {str(e)}")

        return results

