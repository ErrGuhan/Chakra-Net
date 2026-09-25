"""ChakraNet Supabase REST Client

Provides an asynchronous and synchronous HTTP interface to Supabase PostgREST,
Auth JWKS, and Database tables (events, tracks, hazard_grids, alerts, dispatches).
"""

import time
import logging
from typing import Dict, Any, List, Optional, Union
import requests
from config import SUPABASE_CONFIG, SupabaseConfig

logger = logging.getLogger("chakranet.supabase")


class SupabaseClient:
    """Lightweight, robust Supabase PostgREST client for ChakraNet Server API."""

    def __init__(self, config: Optional[SupabaseConfig] = None):
        self.config = config or SUPABASE_CONFIG
        self.url = self.config.url.rstrip("/")
        self.rest_url = f"{self.url}/rest/v1"
        self.secret_key = self.config.secret_key
        self.publishable_key = self.config.publishable_key
        self.jwks_url = self.config.jwks_url
        self.timeout = 10.0

    @property
    def headers(self) -> Dict[str, str]:
        """Headers authenticated with the service secret key."""
        return {
            "apikey": self.secret_key,
            "Authorization": f"Bearer {self.secret_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def test_connection(self) -> Dict[str, Any]:
        """Tests connectivity and latency to Supabase PostgREST endpoint."""
        t0 = time.perf_counter()
        try:
            res = requests.get(f"{self.rest_url}/", headers=self.headers, timeout=self.timeout)
            latency_ms = round((time.perf_counter() - t0) * 1000, 2)
            is_ok = res.status_code == 200
            return {
                "connected": is_ok,
                "status_code": res.status_code,
                "latency_ms": latency_ms,
                "supabase_url": self.url,
                "jwks_url": self.jwks_url,
                "message": "Connected to Supabase PostgreSQL REST Engine" if is_ok else res.text,
            }
        except Exception as exc:
            latency_ms = round((time.perf_counter() - t0) * 1000, 2)
            logger.warning(f"Supabase connection test failed: {exc}")
            return {
                "connected": False,
                "status_code": 0,
                "latency_ms": latency_ms,
                "supabase_url": self.url,
                "jwks_url": self.jwks_url,
                "message": f"Connection error: {str(exc)}",
            }

    def check_table_exists(self, table: str) -> bool:
        """Returns True if the specified table exists in the schema and is accessible."""
        try:
            res = requests.get(
                f"{self.rest_url}/{table}?select=id&limit=1",
                headers=self.headers,
                timeout=self.timeout,
            )
            # PGRST205 indicates table does not exist
            if res.status_code == 404 and "PGRST205" in res.text:
                return False
            return res.status_code in (200, 206)
        except Exception as exc:
            logger.debug(f"Error checking table '{table}': {exc}")
            return False

    def get_table_status(self) -> Dict[str, Dict[str, Any]]:
        """Checks the existence and row count for all ChakraNet application tables."""
        tables = ["events", "tracks", "hazard_grids", "alerts", "dispatches"]
        status = {}
        for tbl in tables:
            try:
                # Use count=exact header to get row count without fetching records
                headers = {**self.headers, "Prefer": "count=exact"}
                res = requests.get(
                    f"{self.rest_url}/{tbl}?select=*&limit=0",
                    headers=headers,
                    timeout=self.timeout,
                )
                if res.status_code == 200 or res.status_code == 206:
                    content_range = res.headers.get("content-range", "")
                    count = 0
                    if "/" in content_range:
                        try:
                            count = int(content_range.split("/")[-1])
                        except ValueError:
                            count = 0
                    status[tbl] = {
                        "exists": True,
                        "accessible": True,
                        "row_count": count,
                        "status": "ready",
                    }
                elif res.status_code == 404 and "PGRST205" in res.text:
                    status[tbl] = {
                        "exists": False,
                        "accessible": False,
                        "row_count": 0,
                        "status": "missing_table",
                        "hint": "Run supabase_schema.sql in Supabase SQL editor",
                    }
                else:
                    status[tbl] = {
                        "exists": False,
                        "accessible": False,
                        "row_count": 0,
                        "status": f"http_{res.status_code}",
                        "detail": res.text[:120],
                    }
            except Exception as exc:
                status[tbl] = {
                    "exists": False,
                    "accessible": False,
                    "row_count": 0,
                    "status": "connection_error",
                    "detail": str(exc),
                }
        return status

    def select(
        self,
        table: str,
        filters: Optional[Dict[str, str]] = None,
        select_cols: str = "*",
        limit: Optional[int] = None,
        order: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Queries rows from a table with PostgREST query parameters."""
        params = {"select": select_cols}
        if filters:
            params.update(filters)
        if limit is not None:
            params["limit"] = str(limit)
        if order:
            params["order"] = order

        res = requests.get(
            f"{self.rest_url}/{table}",
            headers=self.headers,
            params=params,
            timeout=self.timeout,
        )
        if res.status_code == 200:
            return res.json()
        elif res.status_code == 404 and "PGRST205" in res.text:
            logger.debug(f"Table '{table}' does not exist in Supabase.")
            return []
        else:
            logger.warning(f"Supabase select error on '{table}': {res.status_code} {res.text}")
            return []

    def upsert(
        self,
        table: str,
        data: Union[Dict[str, Any], List[Dict[str, Any]]],
        on_conflict: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> bool:
        """Upserts rows into the target table (inserts or updates on primary key)."""
        headers = {
            **self.headers,
            "Prefer": "resolution=merge-duplicates,return=representation",
        }
        params = {}
        if on_conflict:
            params["on_conflict"] = on_conflict

        payload = data if isinstance(data, list) else [data]
        req_timeout = timeout or 45.0
        try:
            res = requests.post(
                f"{self.rest_url}/{table}",
                headers=headers,
                params=params,
                json=payload,
                timeout=req_timeout,
            )
            if res.status_code in (200, 201):
                return True
            if res.status_code == 404 and "PGRST205" in res.text:
                logger.debug(f"Supabase table '{table}' not yet created in schema cache (PGRST205).")
                return False
            logger.warning(f"Supabase upsert failed on '{table}' [{res.status_code}]: {res.text[:200]}")
            return False
        except Exception as exc:
            logger.warning(f"Supabase upsert exception on '{table}': {exc}")
            return False

    def insert(
        self,
        table: str,
        data: Union[Dict[str, Any], List[Dict[str, Any]]],
        timeout: Optional[float] = None,
    ) -> bool:
        """Inserts row(s) into the target table."""
        payload = data if isinstance(data, list) else [data]
        req_timeout = timeout or 30.0
        try:
            res = requests.post(
                f"{self.rest_url}/{table}",
                headers=self.headers,
                json=payload,
                timeout=req_timeout,
            )
            if res.status_code in (200, 201):
                return True
            if res.status_code == 404 and "PGRST205" in res.text:
                logger.debug(f"Supabase table '{table}' not yet created in schema cache (PGRST205).")
                return False
            logger.warning(f"Supabase insert failed on '{table}' [{res.status_code}]: {res.text[:200]}")
            return False
        except Exception as exc:
            logger.warning(f"Supabase insert exception on '{table}': {exc}")
            return False
