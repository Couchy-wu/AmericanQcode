"""FastAPI application factory for the web dashboard."""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

from src.engine.scanner import Scanner

TEMPLATES_DIR = Path(__file__).parent / "templates"
STATIC_DIR = Path(__file__).parent / "static"

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup/shutdown events."""
    # Startup
    from src.data.database import init_db
    await init_db()
    yield
    # Shutdown
    pass


def create_app(scanner: Optional[Scanner] = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        scanner: Optional Scanner instance for live scanning.
                 If provided, the API can trigger scans.
    """
    app = FastAPI(
        title="AmericanQcode",
        description="US Stock Quantitative Trading Dashboard",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount static files
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # Store scanner in app state
    app.state.scanner = scanner
    app.state.latest_signals = []

    # Register routers
    from src.web.routers import dashboard, api_signals, api_charts, api_screener, ws

    app.include_router(dashboard.router)
    app.include_router(api_signals.router, prefix="/api")
    app.include_router(api_charts.router, prefix="/api")
    app.include_router(api_screener.router, prefix="/api")
    app.include_router(ws.router)

    return app
