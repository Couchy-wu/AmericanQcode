"""Dashboard page routes."""

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from src.web.app import templates

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard page."""
    return templates.TemplateResponse("dashboard.html", {"request": request})


@router.get("/chart/{ticker}", response_class=HTMLResponse)
async def chart_page(request: Request, ticker: str):
    """Single ticker detailed chart page."""
    return templates.TemplateResponse("chart.html", {
        "request": request,
        "ticker": ticker.upper(),
    })


@router.get("/screener", response_class=HTMLResponse)
async def screener_page(request: Request):
    """Stock screener page."""
    return templates.TemplateResponse("screener.html", {"request": request})


@router.get("/signals", response_class=HTMLResponse)
async def signals_page(request: Request):
    """Signal history page."""
    return templates.TemplateResponse("signals.html", {"request": request})
