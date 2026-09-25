"""ChakraNet FastAPI REST API Endpoints

Implements the official Stage 3 serving layer endpoints:
- GET  /events                     -> List tracked anomaly events
- GET  /events/{id}/track          -> Tracker output (4D box + cone) as GeoJSON
- GET  /events/{id}/hazard-map     -> Downscaled P(exceed) grid as GeoJSON
- GET  /events/{id}/alerts         -> Generated CAP 1.2 XML alerts & JSON metadata
- POST /events/{id}/alerts/dispatch -> MOCKED dispatch to SACHET / Bhashini (clearly labeled SIMULATED)
"""

import logging
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, Query, Response
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

from serving.api.db import DataService

logger = logging.getLogger("chakranet.api")
router = APIRouter()
data_service = DataService()


class DispatchRequest(BaseModel):
    cell_id: Optional[str] = Field(None, description="Target cell identifier")
    channels: List[str] = Field(default=["SACHET_SMS", "BHASHINI_VOICE", "CAP_BROADCAST"])
    target_districts: Optional[List[str]] = Field(default=["Ganjam", "Puri", "Khurda"])
    simulation_mode: bool = Field(default=True, description="Strictly simulated mock dispatch flag")


class DispatchResponse(BaseModel):
    status: str
    dispatch_id: str
    timestamp_utc: str
    channels_triggered: List[str]
    sachet_status: str
    bhashini_status: str
    disclaimer: str
    audit_log: Dict[str, Any]


@router.get("/db/status")
def get_database_status():
    """Returns Supabase connection health, ping latency, and table statuses."""
    return data_service.get_database_status()


@router.post("/db/seed")
def seed_database():
    """Populates Supabase database with Cyclone Phailin tracks, hazard maps, and alerts."""
    return data_service.seed_to_supabase()


@router.get("/events")
def list_events():
    """Returns list of tracked extreme-weather anomaly events."""
    return data_service.list_events()


@router.get("/events/{event_id}/districts")
def get_event_districts(event_id: str):
    """Returns accurate open-source administrative boundaries for coastal Odisha & vulnerable zones."""
    if event_id != "phailin_2013":
        raise HTTPException(status_code=404, detail=f"Event '{event_id}' not found")
    return data_service.get_districts_geojson()


@router.get("/events/{event_id}/track")
def get_event_track(event_id: str):
    """Returns the 4D tracker output (trajectory, centroid points, uncertainty cone, crop bounding box) as GeoJSON."""
    if event_id != "phailin_2013":
        raise HTTPException(status_code=404, detail=f"Event '{event_id}' not found")
    return data_service.get_track_geojson(event_id)


@router.get("/events/{event_id}/hazard-map")
def get_hazard_map(
    event_id: str,
    lead_time: int = Query(default=108, description="Forecast lead time in hours (0, 12, ..., 120)"),
    threshold_mm: float = Query(default=75.0, description="Precipitation exceedance threshold in mm/24h"),
):
    """Returns the downscaled 5 km probability-of-exceedance grid as GeoJSON polygons."""
    if event_id != "phailin_2013":
        raise HTTPException(status_code=404, detail=f"Event '{event_id}' not found")
    return data_service.get_hazard_map_geojson(event_id, lead_time_hours=lead_time, threshold_mm=threshold_mm)


@router.get("/events/{event_id}/alerts")
def get_alerts(
    event_id: str,
    lead_time: int = Query(default=108, description="Forecast lead time in hours"),
    format: str = Query(default="json", description="Response format: 'json' or 'xml'"),
    lat: Optional[float] = Query(default=None, description="Optional center latitude for cell drilldown"),
    lon: Optional[float] = Query(default=None, description="Optional center longitude for cell drilldown"),
    district: Optional[str] = Query(default=None, description="Optional district name"),
    severity: Optional[str] = Query(default=None, description="Optional alert severity override"),
):
    """Returns generated CAP 1.2 XML and bilingual alert details."""
    if event_id != "phailin_2013":
        raise HTTPException(status_code=404, detail=f"Event '{event_id}' not found")
    alert_data = data_service.get_alerts(
        event_id,
        lead_time_hours=lead_time,
        center_lat=lat,
        center_lon=lon,
        district=district,
        severity=severity,
    )

    if format.lower() == "xml":
        return Response(content=alert_data["cap_xml"], media_type="application/xml")
    return alert_data


@router.post("/events/{event_id}/alerts/dispatch", response_model=DispatchResponse)
def dispatch_alert(event_id: str, req: DispatchRequest):
    """MOCKED dispatch to NDMA SACHET and Bhashini.
    
    CRITICAL NOTICE: This endpoint does NOT connect to live government alerting infrastructure.
    It generates a signed simulation receipt and logs the transmission payload.
    """
    if event_id != "phailin_2013":
        raise HTTPException(status_code=404, detail=f"Event '{event_id}' not found")

    import uuid
    dispatch_id = f"SIM-SACHET-{uuid.uuid4().hex[:10].upper()}"
    ts = datetime.now(timezone.utc).isoformat()

    logger.info(f"[SIMULATED DISPATCH] Event: {event_id}, ID: {dispatch_id}, Channels: {req.channels}")

    audit_data = {
        "event_id": event_id,
        "target_districts": req.target_districts,
        "cell_id": req.cell_id,
        "mode": "PROTOTYPE_DEMO",
        "compliance": "OASIS CAP 1.2 XML Standard",
    }

    # Record dispatch receipt into Supabase database
    data_service.record_dispatch({
        "dispatch_id": dispatch_id,
        "event_id": event_id,
        "cell_id": req.cell_id,
        "channels": req.channels,
        "target_districts": req.target_districts,
        "simulation_mode": req.simulation_mode,
        "status": "SIMULATED_TRANSMISSION_SUCCESS",
        "sachet_status": "MOCKED: Payload logged to simulated NDMA SACHET gateway",
        "bhashini_status": "MOCKED: Hindi audio text synthesized via templated Bhashini interface",
        "disclaimer": "SIMULATED — Not connected to live SACHET or Bhashini production endpoints.",
        "audit_log": audit_data,
        "created_at": ts,
    })

    return DispatchResponse(
        status="SIMULATED_TRANSMISSION_SUCCESS",
        dispatch_id=dispatch_id,
        timestamp_utc=ts,
        channels_triggered=req.channels,
        sachet_status="MOCKED: Payload logged to simulated NDMA SACHET gateway",
        bhashini_status="MOCKED: Hindi audio text synthesized via templated Bhashini interface",
        disclaimer="SIMULATED — Not connected to live SACHET or Bhashini production endpoints.",
        audit_log=audit_data,
    )
