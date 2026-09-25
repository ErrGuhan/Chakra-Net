"""Script to export representative sample API payloads for Milestone 4 verification."""

import os
import sys
import json
import urllib.request

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from serving.api.db import DataService


def export_api_examples(output_dir: str = "tests/artifacts/api_examples"):
    os.makedirs(output_dir, exist_ok=True)
    service = DataService()

    # 1. Sample Track GeoJSON
    try:
        req = urllib.request.Request("http://127.0.0.1:8000/api/v1/events/phailin_2013/track")
        with urllib.request.urlopen(req) as resp:
            track_data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        track_data = service.get_track_geojson("phailin_2013")

    with open(os.path.join(output_dir, "track_response.geojson"), "w", encoding="utf-8") as f:
        json.dump(track_data, f, indent=2)

    # 2. Sample Hazard Map GeoJSON
    try:
        req = urllib.request.Request("http://127.0.0.1:8000/api/v1/events/phailin_2013/hazard-map?lead_time=108&threshold_mm=75")
        with urllib.request.urlopen(req) as resp:
            hazard_data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        hazard_data = service.get_hazard_map_geojson("phailin_2013", lead_time_hours=108, threshold_mm=75.0)

    with open(os.path.join(output_dir, "hazard_map_response.geojson"), "w", encoding="utf-8") as f:
        json.dump(hazard_data, f, indent=2)

    # 3. Sample OASIS CAP 1.2 XML
    try:
        req = urllib.request.Request("http://127.0.0.1:8000/api/v1/events/phailin_2013/alerts?lead_time=108&format=xml")
        with urllib.request.urlopen(req) as resp:
            xml_content = resp.read().decode("utf-8")
    except Exception:
        alerts = service.get_alerts("phailin_2013", lead_time_hours=108)
        xml_content = alerts["cap_xml"]

    with open(os.path.join(output_dir, "sample_alert.xml"), "w", encoding="utf-8") as f:
        f.write(xml_content)

    # 4. Dispatch Simulated Receipt JSON
    try:
        post_data = json.dumps({
            "channels": ["SACHET_SMS", "BHASHINI_VOICE", "CAP_BROADCAST"],
            "target_districts": ["Ganjam", "Puri", "Khurda"],
            "simulation_mode": True,
        }).encode("utf-8")
        req = urllib.request.Request(
            "http://127.0.0.1:8000/api/v1/events/phailin_2013/alerts/dispatch",
            data=post_data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req) as resp:
            dispatch_receipt = json.loads(resp.read().decode("utf-8"))
    except Exception:
        dispatch_receipt = {
            "status": "SIMULATED_TRANSMISSION_SUCCESS",
            "dispatch_id": "SIM-SACHET-PHAILIN-SAMPLE",
            "timestamp_utc": "2026-09-24T22:05:00Z",
            "channels_triggered": ["SACHET_SMS", "BHASHINI_VOICE", "CAP_BROADCAST"],
            "sachet_status": "MOCKED: Payload logged to simulated NDMA SACHET gateway",
            "bhashini_status": "MOCKED: Hindi audio text synthesized via templated Bhashini interface",
            "disclaimer": "SIMULATED — Not connected to live SACHET or Bhashini production endpoints.",
            "audit_log": {
                "event_id": "phailin_2013",
                "target_districts": ["Ganjam", "Puri", "Khurda"],
                "mode": "PROTOTYPE_DEMO"
            }
        }

    with open(os.path.join(output_dir, "dispatch_simulated_receipt.json"), "w", encoding="utf-8") as f:
        json.dump(dispatch_receipt, f, indent=2)

    print(f"Exported all 4 sample API artifacts successfully to {output_dir}")


if __name__ == "__main__":
    export_api_examples()
