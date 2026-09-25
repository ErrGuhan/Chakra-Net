"""ChakraNet Serving Application Main Entry Point

FastAPI web service integrating:
- REST API endpoints for tracked anomalies, 5 km hazard maps, and CAP alerts
- Static file serving for MapLibre GL JS interactive dashboard
- CORS middleware for seamless local browser interaction
"""

import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from serving.api.routes import router as api_router
from config import SERVING_DIR

app = FastAPI(
    title="ChakraNet Weather Tracking & Downscaling Alert API",
    description=(
        "Operational dual-stage atmospheric physics pipeline for tracking extreme weather anomalies in "
        "medium-range NWP ensemble forecasts, downscaling to 5 km hazard maps, and "
        "generating CAP 1.2 XML alerts (SIH 2026 PS 26078, NCMRWF/MoES)."
    ),
    version="1.0.0",
)

# Enable CORS for browser MapLibre clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes (at root, /api, and /api/v1 for convenience)
app.include_router(api_router, tags=["ChakraNet Core Endpoints"])
app.include_router(api_router, prefix="/api", tags=["ChakraNet Namespaced API"])
app.include_router(api_router, prefix="/api/v1", tags=["ChakraNet Versioned API"])

# Frontend directory
frontend_dir = SERVING_DIR / "frontend"
try:
    frontend_dir.mkdir(parents=True, exist_ok=True)
except (OSError, PermissionError):
    pass

# Mount frontend static assets
app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")


@app.get("/", include_in_schema=False)
async def serve_dashboard():
    """Serves the main MapLibre GL JS dashboard."""
    index_file = frontend_dir / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {
        "message": "ChakraNet API operational. Frontend index.html not yet built.",
        "docs": "/docs",
        "events": "/events",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("serving.api.main:app", host="127.0.0.1", port=8000, reload=True)
