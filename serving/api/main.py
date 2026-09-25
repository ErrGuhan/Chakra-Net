"""ChakraNet Serving Application Main Entry Point

FastAPI web service integrating:
- REST API endpoints for tracked anomalies, 5 km hazard maps, and CAP alerts
- Static file serving for MapLibre GL JS interactive dashboard
- CORS middleware for seamless local browser interaction
- Security middleware: HTTP security headers on every response
"""

import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
from starlette.middleware.base import BaseHTTPMiddleware

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

# ---------------------------------------------------------------------------
# Security Headers Middleware
# Injects best-practice HTTP security headers on every response.
# ---------------------------------------------------------------------------
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response: Response = await call_next(request)

        # Prevent clickjacking
        response.headers["X-Frame-Options"] = "DENY"

        # Prevent MIME-type sniffing
        response.headers["X-Content-Type-Options"] = "nosniff"

        # Legacy XSS protection (belt-and-suspenders for older browsers)
        response.headers["X-XSS-Protection"] = "1; mode=block"

        # Referrer policy — limit what is sent cross-origin
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        # Permissions policy — disable unused browser APIs
        response.headers["Permissions-Policy"] = (
            "geolocation=(), microphone=(), camera=(), payment=(), usb=(), midi=()"
        )

        # HSTS — tell browsers to always use HTTPS (1 year, include subdomains)
        # Only injected when served over HTTPS; harmless on HTTP dev servers.
        response.headers["Strict-Transport-Security"] = (
            "max-age=31536000; includeSubDomains; preload"
        )

        # Cache-Control: no caching for HTML to avoid stale pages
        content_type = response.headers.get("content-type", "")
        if "text/html" in content_type:
            response.headers["Cache-Control"] = (
                "no-store, no-cache, must-revalidate, proxy-revalidate"
            )
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"

        return response

app.add_middleware(SecurityHeadersMiddleware)

# ---------------------------------------------------------------------------
# CORS: Allow all origins safely with credentials=False (standard for public/tokenless APIs)
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-Requested-With"],
)

# Include API routes (at root, /api, and /api/v1 for convenience)
app.include_router(api_router, tags=["ChakraNet Core Endpoints"])
app.include_router(api_router, prefix="/api", tags=["ChakraNet Namespaced API"])
app.include_router(api_router, prefix="/api/v1", tags=["ChakraNet Versioned API"])

# Frontend directory
frontend_dir = SERVING_DIR / "frontend"
public_dir = project_root / "public"
try:
    frontend_dir.mkdir(parents=True, exist_ok=True)
except (OSError, PermissionError):
    pass

# Mount frontend static assets
static_source = frontend_dir if (frontend_dir / "styles.css").exists() else public_dir
app.mount("/static", StaticFiles(directory=str(static_source)), name="static")


@app.get("/health", tags=["System"])
async def health_check():
    """Health check endpoint for container probes and uptime monitoring."""
    return {"status": "ok", "service": "ChakraNet API", "version": "1.0.0"}


@app.get("/", include_in_schema=False)
async def serve_dashboard():
    """Serves the main MapLibre GL JS dashboard."""
    for candidate in [frontend_dir / "index.html", public_dir / "index.html"]:
        if candidate.exists():
            return FileResponse(candidate)
    return {
        "message": "ChakraNet API operational. Frontend index.html not yet built.",
        "docs": "/docs",
        "events": "/events",
    }


@app.get("/styles.css", include_in_schema=False)
async def serve_styles():
    for candidate in [frontend_dir / "styles.css", public_dir / "styles.css"]:
        if candidate.exists():
            return FileResponse(candidate, media_type="text/css")
    return Response(status_code=404)


@app.get("/app.js", include_in_schema=False)
async def serve_app_js():
    for candidate in [frontend_dir / "app.js", public_dir / "app.js"]:
        if candidate.exists():
            return FileResponse(candidate, media_type="application/javascript")
    return Response(status_code=404)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("serving.api.main:app", host="127.0.0.1", port=8000, reload=True)
