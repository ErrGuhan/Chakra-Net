"""Vercel Serverless Function Entry Point for ChakraNet

Exposes the FastAPI ASGI application for Vercel deployment.
"""

import os
import sys
from pathlib import Path

# Add project root directory to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from fastapi import Request
from serving.api.main import app

@app.middleware("http")
async def vercel_rewrite_middleware(request: Request, call_next):
    """Translates Vercel destination path (/api/index.py) back to original matched route."""
    if request.scope.get("path") in ("/api/index.py", "/api/index", "/api"):
        orig_path = request.headers.get("x-matched-path") or request.query_params.get("__path")
        if orig_path:
            clean_path = orig_path.split("?")[0]
            if clean_path and clean_path not in ("/api/index.py", "/api/index"):
                request.scope["path"] = clean_path
    return await call_next(request)
