"""OMNISCOPE API Server."""
from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from .models import TraceEvent, IngestBatch, TraceOverview, TraceGraph, TimelineEntry, RiskScores
from .store import TraceStore
from .risk import RiskEngine


# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
store = TraceStore()
risk_engine = RiskEngine()
ws_connections: dict[str, list[WebSocket]] = {}  # trace_id -> websockets
global_ws: list[WebSocket] = []  # for trace list updates


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(title="OMNISCOPE", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# REST API
# ---------------------------------------------------------------------------

@app.post("/api/v1/traces")
async def ingest_events(batch: IngestBatch) -> dict[str, Any]:
    """Ingest a batch of trace events."""
    store.ingest_batch(batch.events)

    # Notify WebSocket subscribers
    for event in batch.events:
        trace_id = event.trace_id
        if trace_id in ws_connections:
            msg = json.dumps({"type": "event", "data": event.model_dump()})
            for ws in ws_connections[trace_id]:
                try:
                    await ws.send_text(msg)
                except Exception:
                    pass

    # Notify global subscribers
    if batch.events:
        for ws in global_ws:
            try:
                await ws.send_text(json.dumps({"type": "update"}))
            except Exception:
                pass

    return {"ingested": len(batch.events)}


@app.post("/api/v1/event")
async def ingest_single(event: TraceEvent) -> dict[str, Any]:
    """Ingest a single trace event."""
    store.ingest(event)

    trace_id = event.trace_id
    if trace_id in ws_connections:
        msg = json.dumps({"type": "event", "data": event.model_dump()})
        for ws in ws_connections[trace_id]:
            try:
                await ws.send_text(msg)
            except Exception:
                pass

    for ws in global_ws:
        try:
            await ws.send_text(json.dumps({"type": "update"}))
        except Exception:
            pass

    return {"ingested": 1}


@app.post("/api/v1/traces/{trace_id}/complete")
@app.post("/api/v1/traces/{trace_id}/completed")
async def complete_trace(trace_id: str) -> dict[str, str]:
    store.complete_trace(trace_id)
    return {"status": "completed"}


@app.post("/api/v1/traces/{trace_id}/fail")
@app.post("/api/v1/traces/{trace_id}/failed")
async def fail_trace(trace_id: str) -> dict[str, str]:
    store.fail_trace(trace_id)
    return {"status": "failed"}


@app.get("/api/v1/traces")
async def list_traces(limit: int = Query(default=50)) -> list[TraceOverview]:
    return store.list_traces(limit)


@app.get("/api/v1/traces/{trace_id}")
async def get_trace(trace_id: str) -> dict[str, Any]:
    events = store.get_events(trace_id)
    if not events:
        return {"error": "trace not found"}

    total_tokens = sum(e.input_tokens + e.output_tokens for e in events)
    total_cost = sum(e.cost_usd or 0 for e in events)
    agents = list(set(e.agent_id for e in events if e.agent_id))
    frameworks = list(set(e.framework for e in events if e.framework != "generic"))

    return {
        "trace_id": trace_id,
        "event_count": len(events),
        "total_tokens": total_tokens,
        "total_cost": round(total_cost, 4),
        "agents": agents,
        "frameworks": frameworks or ["generic"],
        "events": [e.model_dump() for e in events],
    }


@app.get("/api/v1/traces/{trace_id}/graph")
async def get_graph(trace_id: str) -> TraceGraph:
    return store.get_graph(trace_id)


@app.get("/api/v1/traces/{trace_id}/timeline")
async def get_timeline(trace_id: str) -> list[TimelineEntry]:
    return store.get_timeline(trace_id)


@app.get("/api/v1/traces/{trace_id}/risk")
async def get_risk(trace_id: str) -> dict[str, RiskScores]:
    events = store.get_events(trace_id)
    return risk_engine.compute_all(events)


@app.get("/api/v1/risk/{trace_id}/{agent_id}")
async def get_agent_risk(trace_id: str, agent_id: str) -> RiskScores:
    events = store.get_events(trace_id)
    return risk_engine.compute(events, agent_id)


@app.get("/api/v1/events/{event_id}")
async def get_event(event_id: str) -> dict[str, Any]:
    event = store.get_event(event_id)
    if event is None:
        return {"error": "event not found"}
    return event.model_dump()


@app.get("/api/v1/events/{event_id}/causal_chain")
async def get_causal_chain(event_id: str) -> list[dict[str, Any]]:
    chain = store.causal_chain(event_id)
    results = []
    for eid in chain:
        evt = store.get_event(eid)
        if evt:
            results.append(evt.model_dump())
    return results


@app.get("/api/v1/events/{event_id}/downstream")
async def get_downstream(event_id: str) -> list[dict[str, Any]]:
    impacted = store.downstream_impact(event_id)
    results = []
    for eid in impacted:
        evt = store.get_event(eid)
        if evt:
            results.append(evt.model_dump())
    return results


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------

@app.websocket("/ws/v1/traces/{trace_id}/live")
async def trace_live(websocket: WebSocket, trace_id: str):
    await websocket.accept()
    if trace_id not in ws_connections:
        ws_connections[trace_id] = []
    ws_connections[trace_id].append(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws_connections[trace_id].remove(websocket)


@app.websocket("/ws/v1/updates")
async def global_updates(websocket: WebSocket):
    await websocket.accept()
    global_ws.append(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        global_ws.remove(websocket)


# ---------------------------------------------------------------------------
# Static file serving for dashboard
# ---------------------------------------------------------------------------

dashboard_dist = Path(__file__).parent.parent.parent / "dashboard" / "dist"

if dashboard_dist.exists():
    app.mount("/assets", StaticFiles(directory=str(dashboard_dist / "assets")), name="assets")

    @app.get("/{full_path:path}")
    async def serve_dashboard(full_path: str):
        file_path = dashboard_dist / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(dashboard_dist / "index.html")
