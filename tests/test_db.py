"""Unit & Integration Tests for ChakraNet Supabase Server API Database Integration"""

import pytest
from fastapi.testclient import TestClient
from config import SUPABASE_CONFIG
from serving.api.supabase_client import SupabaseClient
from serving.api.db import DataService
from serving.api.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_supabase_config():
    """Validates Supabase configuration is loaded properly from environment / .env."""
    if SUPABASE_CONFIG.url:
        assert "supabase.co" in SUPABASE_CONFIG.url
        assert len(SUPABASE_CONFIG.publishable_key) > 0
        assert len(SUPABASE_CONFIG.secret_key) > 0
        assert "jwks.json" in SUPABASE_CONFIG.jwks_url


def test_supabase_client_connection():
    """Validates Supabase REST API connection and ping latency."""
    sc = SupabaseClient()
    conn = sc.test_connection()
    assert conn["connected"] is True
    assert conn["status_code"] == 200
    assert conn["latency_ms"] > 0
    assert "supabase.co" in conn["supabase_url"]


def test_supabase_client_table_status():
    """Validates table status inspection detects expected table names."""
    sc = SupabaseClient()
    statuses = sc.get_table_status()
    for table_name in ["events", "tracks", "hazard_grids", "alerts", "dispatches"]:
        assert table_name in statuses
        assert "exists" in statuses[table_name]
        assert "row_count" in statuses[table_name]


def test_data_service_get_database_status():
    """Validates DataService.get_database_status() returns complete diagnostic metadata."""
    ds = DataService()
    status = ds.get_database_status()
    assert status["database_provider"] == "Supabase PostgreSQL"
    assert "supabase_url" in status
    assert "connected" in status
    assert "tables" in status


def test_api_db_status_endpoints(client):
    """Validates GET /db/status and GET /api/db/status HTTP endpoints."""
    for path in ["/db/status", "/api/db/status", "/api/v1/db/status"]:
        res = client.get(path)
        assert res.status_code == 200
        data = res.json()
        assert data["database_provider"] == "Supabase PostgreSQL"
        assert data["connected"] is True
        assert "tables" in data


def test_api_dispatch_saves_record(client):
    """Validates that POST /events/{id}/alerts/dispatch executes and records receipt."""
    payload = {
        "cell_id": "cell_5km_19_84",
        "channels": ["SACHET_SMS", "BHASHINI_VOICE", "CAP_BROADCAST"],
        "target_districts": ["Ganjam", "Puri"],
        "simulation_mode": True,
    }
    res = client.post("/events/phailin_2013/alerts/dispatch", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "SIMULATED_TRANSMISSION_SUCCESS"
    assert "SIM-SACHET-" in data["dispatch_id"]
    assert "SIMULATED" in data["disclaimer"]
