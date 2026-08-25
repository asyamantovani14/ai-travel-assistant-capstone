"""FastAPI entrypoint for the Atlas travel planning experience."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from travel_service import TravelFilters, get_travel_service


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = PROJECT_ROOT / "web"

app = FastAPI(title="Atlas Travel Assistant", version="1.0.0")
app.mount("/assets", StaticFiles(directory=WEB_ROOT / "assets"), name="assets")


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=20_000)


class FilterRequest(BaseModel):
    countries: list[str] = Field(default_factory=list, max_length=12)
    activities: list[str] = Field(default_factory=list, max_length=12)
    min_days: Optional[int] = Field(default=None, ge=1, le=365)
    max_budget: Optional[int] = Field(default=None, ge=1, le=10_000_000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=2, max_length=4_000)
    history: list[ChatMessage] = Field(default_factory=list, max_length=20)
    filters: FilterRequest = Field(default_factory=FilterRequest)


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(WEB_ROOT / "index.html")


@app.get("/app.js", include_in_schema=False)
def javascript():
    return FileResponse(WEB_ROOT / "app.js", media_type="text/javascript")


@app.get("/styles.css", include_in_schema=False)
def stylesheet():
    return FileResponse(WEB_ROOT / "styles.css", media_type="text/css")


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "atlas"}


@app.post("/api/chat")
async def chat(request: ChatRequest):
    filters = TravelFilters(
        countries=tuple(value.strip().lower() for value in request.filters.countries if value.strip()),
        activities=tuple(value.strip().lower() for value in request.filters.activities if value.strip()),
        min_days=request.filters.min_days,
        max_budget=request.filters.max_budget,
    )
    history = [message.model_dump() for message in request.history[-8:]]
    try:
        service = await run_in_threadpool(get_travel_service)
        return await run_in_threadpool(service.plan, request.message, history, filters)
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=502, detail="Travel planning is temporarily unavailable") from error
