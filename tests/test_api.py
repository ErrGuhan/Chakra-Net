"""Unit & Integration Tests for ChakraNet Serving Layer & API Endpoints"""

import pytest
from fastapi.testclient import TestClient
from serving.api.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_root_serves_html(client):
    """Checks that root endpoint GET / serves the MapLibre frontend."""
    res = client.get("/")
    assert res.status_code == 200
    assert "ChakraNet" in res.text
    assert "MapLibre" in res.text or "map-view" in res.text


def test_list_events(client):
    """Validates GET /events returns Cyclone Phailin metadata."""
    res = client.get("/events")
    assert res.status_code == 200
    events = res.json()
    assert len(events) >= 1
    assert events[0]["id"] == "phailin_2013"
    assert "landfall" in events[0]


def test_get_districts_geojson(client):
    """Validates GET /events/{id}/districts returns accurate open-source administrative boundaries."""
    res = client.get("/events/phailin_2013/districts")
    assert res.status_code == 200
    data = res.json()
    assert data["type"] == "FeatureCollection"
    assert len(data["features"]) >= 6
    names = [f["properties"].get("district") or f["properties"].get("name") for f in data["features"]]
    assert "Ganjam" in names
    assert "Puri" in names


def test_get_track_geojson(client):
    """Validates GET /events/{id}/track returns valid GeoJSON FeatureCollection."""
    res = client.get("/events/phailin_2013/track")
    assert res.status_code == 200
    geojson = res.json()
    assert geojson["type"] == "FeatureCollection"
    assert len(geojson["features"]) >= 4
    
    layer_types = [f["properties"].get("layer_type") for f in geojson["features"]]
    assert "best_track_line" in layer_types
    assert "uncertainty_cone" in layer_types
    assert "stage1_crop_box" in layer_types


def test_get_hazard_map_geojson(client):
    """Validates GET /events/{id}/hazard-map returns downscaled grid cells."""
    res = client.get("/events/phailin_2013/hazard-map?lead_time=108&threshold_mm=75")
    assert res.status_code == 200
    geojson = res.json()
    assert geojson["type"] == "FeatureCollection"
    assert len(geojson["features"]) > 0
    
    first_cell = geojson["features"][0]
    assert "prob_exceed" in first_cell["properties"]
    assert "severity" in first_cell["properties"]
    assert first_cell["properties"]["severity"] in ["Minor", "Moderate", "Severe", "Extreme"]


def test_get_alerts_json_and_xml(client):
    """Validates GET /events/{id}/alerts produces CAP 1.2 XML and bilingual details."""
    # 1. JSON endpoint
    res_json = client.get("/events/phailin_2013/alerts?lead_time=108")
    assert res_json.status_code == 200
    data = res_json.json()
    assert data["cap_version"] == "1.2"
    assert "cap_xml" in data
    assert "multilingual" in data
    assert "hi" in data["multilingual"]
    assert "SIMULATED Bhashini" in data["multilingual"]["hi"]["provider"]

    # 2. Raw XML endpoint
    res_xml = client.get("/events/phailin_2013/alerts?lead_time=108&format=xml")
    assert res_xml.status_code == 200
    assert "application/xml" in res_xml.headers["content-type"]
    assert "<alert xmlns=\"urn:oasis:names:tc:emergency:cap:1.2\">" in res_xml.text
    assert "<severity>Extreme</severity>" in res_xml.text


def test_mock_dispatch_endpoint(client):
    """Validates POST /events/{id}/alerts/dispatch creates simulation receipt."""
    payload = {
        "cell_id": "cell_19_84",
        "channels": ["SACHET_SMS", "BHASHINI_VOICE"],
        "target_districts": ["Ganjam", "Puri"],
        "simulation_mode": True,
    }
    res = client.post("/events/phailin_2013/alerts/dispatch", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "SIMULATED_TRANSMISSION_SUCCESS"
    assert "SIM-SACHET-" in data["dispatch_id"]
    assert "SIMULATED" in data["disclaimer"]
