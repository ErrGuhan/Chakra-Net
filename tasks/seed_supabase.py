"""ChakraNet Supabase Database Seeding & Verification Script

Populates the Supabase PostgreSQL database with:
- Cyclone Phailin event metadata
- 4D trajectory, uncertainty cone, and bounding boxes
- 5 km downscaled probability-of-exceedance grids
- OASIS CAP 1.2 alerts and Bhashini multilingual translations

Usage:
    uv run python tasks/seed_supabase.py
"""

import sys
import json
import logging
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config import SUPABASE_CONFIG
from serving.api.supabase_client import SupabaseClient
from serving.api.db import DataService

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("chakranet.seed")


def main():
    print("=" * 70)
    print(" ChakraNet Supabase PostgreSQL Database Setup & Seeder")
    print(" SIH 2026 PS 26078: Extreme Weather Anomaly Pipeline")
    print("=" * 70)
    print(f"Target Supabase URL : {SUPABASE_CONFIG.url}")
    print(f"JWKS Endpoint       : {SUPABASE_CONFIG.jwks_url}")
    print("-" * 70)

    client = SupabaseClient()
    conn_info = client.test_connection()
    print(f"Connection Status   : {'[ONLINE]' if conn_info['connected'] else '[OFFLINE]'}")
    print(f"Ping Latency        : {conn_info.get('latency_ms', 0)} ms")

    if not conn_info["connected"]:
        print(f"\n[ERROR] Unable to connect to Supabase: {conn_info.get('message')}")
        sys.exit(1)

    table_statuses = client.get_table_status()
    print("\nTable Statuses in Supabase:")
    ready_count = 0
    for tbl, st in table_statuses.items():
        exists = st.get("exists", False)
        if exists:
            ready_count += 1
            print(f"  - {tbl:<15} : [EXISTS] {st.get('row_count', 0)} rows")
        else:
            print(f"  - {tbl:<15} : [MISSING] {st.get('hint', st.get('status'))}")

    if ready_count == 0:
        proj_id = SUPABASE_CONFIG.url.split("//")[-1].split(".")[0]
        print("\n" + "!" * 70)
        print(" ACTION REQUIRED: Database tables not yet created in Supabase.")
        print(" To enable live Supabase persistence:")
        print(f" 1. Open Supabase SQL Editor: https://supabase.com/dashboard/project/{proj_id}/sql/new")
        print(" 2. Paste and run the contents of 'supabase_schema.sql'")
        print(" 3. Re-run: uv run python tasks/seed_supabase.py")
        print("\n [NOTE] ChakraNet API & Frontend currently operate in resilient Hybrid Mode")
        print("        (serving 100% of spatial tracks, hazard grids & alerts via local pipeline).")
        print("!" * 70 + "\n")
        return

    print("\nInitializing ChakraNet DataService...")
    ds = DataService()

    print("Executing database seeding routine (lead times 0h, 24h, 48h, 72h, 84h, 96h, 108h, 120h)...")
    res = ds.seed_to_supabase()

    print("\nSeeding Results:")
    print(f"  - Events Seeded       : {res['events_seeded']}")
    print(f"  - Tracks Seeded       : {res['tracks_seeded']}")
    print(f"  - Hazard Grids Seeded : {res['hazard_grids_seeded']} grids")
    print(f"  - Alerts Seeded       : {res['alerts_seeded']} alerts")
    if res.get("errors"):
        print(f"  - Warnings/Errors     : {len(res['errors'])} errors encountered")
        for err in res["errors"][:3]:
            print(f"      * {err}")

    # Final table check
    updated_statuses = client.get_table_status()
    print("\nUpdated Table Row Counts:")
    for tbl, st in updated_statuses.items():
        print(f"  - {tbl:<15} : {st.get('row_count', 0)} rows (accessible: {st.get('accessible', False)})")

    print("\nDone! ChakraNet Server API is configured with Supabase.")


if __name__ == "__main__":
    main()
