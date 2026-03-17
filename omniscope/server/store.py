"""SQLite-backed trace store with graph query support."""
from __future__ import annotations

import json
import sqlite3
import threading
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from .models import (
    TraceEvent, TraceOverview, GraphNode, GraphEdge, TraceGraph,
    TimelineEntry, RiskScores, AgentState,
)


class TraceStore:
    """Thread-safe SQLite trace store with in-memory graph index."""

    def __init__(self, db_path: str = "omniscope.db"):
        self._db_path = db_path
        self._local = threading.local()
        self._init_db()
        # In-memory indices for fast graph queries
        self._adjacency: dict[str, list[str]] = defaultdict(list)
        self._reverse_adjacency: dict[str, list[str]] = defaultdict(list)
        self._lock = threading.Lock()

    def _get_conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(self._db_path)
            self._local.conn.row_factory = sqlite3.Row
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA foreign_keys=ON")
        return self._local.conn

    def _init_db(self):
        conn = self._get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS events (
                event_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                trace_id TEXT NOT NULL,
                span_id TEXT NOT NULL,
                parent_span_id TEXT,
                event_type TEXT NOT NULL,
                agent_id TEXT NOT NULL DEFAULT '',
                agent_name TEXT NOT NULL DEFAULT '',
                model_name TEXT,
                framework TEXT NOT NULL DEFAULT 'generic',
                input_tokens INTEGER DEFAULT 0,
                output_tokens INTEGER DEFAULT 0,
                latency_ms REAL DEFAULT 0.0,
                cost_usd REAL,
                confidence_score REAL,
                input_preview TEXT DEFAULT '',
                output_preview TEXT DEFAULT '',
                tool_name TEXT,
                tool_input TEXT DEFAULT '{}',
                tool_output TEXT DEFAULT '',
                tool_success INTEGER DEFAULT 1,
                error_message TEXT,
                causal_parents TEXT DEFAULT '[]',
                data_dependencies TEXT DEFAULT '[]',
                tags TEXT DEFAULT '{}',
                metadata TEXT DEFAULT '{}'
            );

            CREATE INDEX IF NOT EXISTS idx_events_trace ON events(trace_id);
            CREATE INDEX IF NOT EXISTS idx_events_agent ON events(agent_id);
            CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
            CREATE INDEX IF NOT EXISTS idx_events_ts ON events(timestamp);

            CREATE TABLE IF NOT EXISTS traces (
                trace_id TEXT PRIMARY KEY,
                trace_name TEXT DEFAULT '',
                framework TEXT DEFAULT 'generic',
                started_at TEXT,
                ended_at TEXT,
                status TEXT DEFAULT 'running'
            );
        """)
        conn.commit()

    def ingest(self, event: TraceEvent) -> None:
        conn = self._get_conn()
        conn.execute("""
            INSERT OR REPLACE INTO events (
                event_id, timestamp, trace_id, span_id, parent_span_id,
                event_type, agent_id, agent_name, model_name, framework,
                input_tokens, output_tokens, latency_ms, cost_usd, confidence_score,
                input_preview, output_preview, tool_name, tool_input, tool_output,
                tool_success, error_message, causal_parents, data_dependencies,
                tags, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            event.event_id, event.timestamp, event.trace_id, event.span_id,
            event.parent_span_id, event.event_type.value if hasattr(event.event_type, 'value') else event.event_type,
            event.agent_id, event.agent_name, event.model_name, event.framework,
            event.input_tokens, event.output_tokens, event.latency_ms, event.cost_usd,
            event.confidence_score, event.input_preview[:2000], event.output_preview[:2000],
            event.tool_name, json.dumps(event.tool_input), event.tool_output[:2000],
            1 if event.tool_success else 0, event.error_message,
            json.dumps(event.causal_parents), json.dumps(event.data_dependencies),
            json.dumps(event.tags), json.dumps(event.metadata),
        ))

        # Upsert trace record
        conn.execute("""
            INSERT INTO traces (trace_id, trace_name, framework, started_at, status)
            VALUES (?, ?, ?, ?, 'running')
            ON CONFLICT(trace_id) DO UPDATE SET
                framework = COALESCE(excluded.framework, framework)
        """, (event.trace_id, event.tags.get("trace_name", ""), event.framework, event.timestamp))
        conn.commit()

        # Update in-memory graph
        with self._lock:
            for parent_id in event.causal_parents:
                self._adjacency[parent_id].append(event.event_id)
                self._reverse_adjacency[event.event_id].append(parent_id)
            for dep_id in event.data_dependencies:
                self._adjacency[dep_id].append(event.event_id)
                self._reverse_adjacency[event.event_id].append(dep_id)
            if event.parent_span_id:
                # Find event with matching span_id
                row = conn.execute(
                    "SELECT event_id FROM events WHERE span_id = ? AND trace_id = ? LIMIT 1",
                    (event.parent_span_id, event.trace_id)
                ).fetchone()
                if row:
                    parent_eid = row["event_id"]
                    if event.event_id not in self._adjacency.get(parent_eid, []):
                        self._adjacency[parent_eid].append(event.event_id)
                        self._reverse_adjacency[event.event_id].append(parent_eid)

    def ingest_batch(self, events: list[TraceEvent]) -> None:
        for event in events:
            self.ingest(event)

    def complete_trace(self, trace_id: str) -> None:
        conn = self._get_conn()
        conn.execute(
            "UPDATE traces SET status = 'completed', ended_at = ? WHERE trace_id = ?",
            (datetime.utcnow().isoformat() + "Z", trace_id)
        )
        conn.commit()

    def fail_trace(self, trace_id: str) -> None:
        conn = self._get_conn()
        conn.execute(
            "UPDATE traces SET status = 'failed', ended_at = ? WHERE trace_id = ?",
            (datetime.utcnow().isoformat() + "Z", trace_id)
        )
        conn.commit()

    def list_traces(self, limit: int = 50) -> list[TraceOverview]:
        conn = self._get_conn()
        rows = conn.execute("""
            SELECT t.trace_id, t.trace_name, t.framework, t.started_at, t.status,
                   COUNT(e.event_id) as event_count,
                   SUM(e.input_tokens + e.output_tokens) as total_tokens,
                   SUM(COALESCE(e.cost_usd, 0)) as total_cost,
                   MAX(e.latency_ms) as max_latency,
                   GROUP_CONCAT(DISTINCT e.agent_id) as agent_ids
            FROM traces t
            LEFT JOIN events e ON t.trace_id = e.trace_id
            GROUP BY t.trace_id
            ORDER BY t.started_at DESC
            LIMIT ?
        """, (limit,)).fetchall()

        results = []
        for row in rows:
            started = row["started_at"] or ""
            agents = (row["agent_ids"] or "").split(",")
            agents = [a for a in agents if a]

            results.append(TraceOverview(
                trace_id=row["trace_id"],
                trace_name=row["trace_name"] or row["trace_id"][:8],
                framework=row["framework"] or "generic",
                started_at=started,
                event_count=row["event_count"] or 0,
                agent_ids=agents,
                total_tokens=row["total_tokens"] or 0,
                total_cost=round(row["total_cost"] or 0, 4),
                status=row["status"] or "running",
            ))
        return results

    def get_events(self, trace_id: str) -> list[TraceEvent]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM events WHERE trace_id = ? ORDER BY timestamp ASC",
            (trace_id,)
        ).fetchall()
        return [self._row_to_event(row) for row in rows]

    def get_graph(self, trace_id: str) -> TraceGraph:
        events = self.get_events(trace_id)
        if not events:
            return TraceGraph(nodes=[], edges=[])

        nodes = []
        edges = []
        seen_edges = set()

        for evt in events:
            label = evt.agent_name or evt.agent_id or evt.event_type
            if evt.tool_name:
                label = f"{label}: {evt.tool_name}"

            risk = 0.0
            if evt.confidence_score is not None:
                risk = max(0.0, 1.0 - evt.confidence_score) * 0.5

            state = AgentState.COMPLETED
            if evt.error_message:
                state = AgentState.FAILED

            nodes.append(GraphNode(
                id=evt.event_id,
                agent_id=evt.agent_id,
                agent_name=evt.agent_name or evt.agent_id,
                event_type=evt.event_type,
                label=label,
                timestamp=evt.timestamp,
                latency_ms=evt.latency_ms,
                confidence=evt.confidence_score,
                risk_score=risk,
                state=state,
                input_preview=evt.input_preview[:200],
                output_preview=evt.output_preview[:200],
                model_name=evt.model_name,
                tokens=evt.input_tokens + evt.output_tokens,
                framework=evt.framework,
            ))

            for parent_id in evt.causal_parents:
                edge_key = (parent_id, evt.event_id, "causal")
                if edge_key not in seen_edges:
                    edges.append(GraphEdge(source=parent_id, target=evt.event_id, edge_type="causal"))
                    seen_edges.add(edge_key)

            for dep_id in evt.data_dependencies:
                edge_key = (dep_id, evt.event_id, "data")
                if edge_key not in seen_edges:
                    edges.append(GraphEdge(source=dep_id, target=evt.event_id, edge_type="data", animated=True))
                    seen_edges.add(edge_key)

        # If no explicit edges, build from parent_span_id
        if not edges:
            span_to_event = {evt.span_id: evt.event_id for evt in events}
            for evt in events:
                if evt.parent_span_id and evt.parent_span_id in span_to_event:
                    parent_eid = span_to_event[evt.parent_span_id]
                    edge_key = (parent_eid, evt.event_id, "causal")
                    if edge_key not in seen_edges:
                        edges.append(GraphEdge(source=parent_eid, target=evt.event_id, edge_type="causal"))
                        seen_edges.add(edge_key)

        return TraceGraph(nodes=nodes, edges=edges)

    def get_timeline(self, trace_id: str) -> list[TimelineEntry]:
        events = self.get_events(trace_id)
        if not events:
            return []

        base_time = datetime.fromisoformat(events[0].timestamp.rstrip("Z"))
        entries = []

        for evt in events:
            evt_time = datetime.fromisoformat(evt.timestamp.rstrip("Z"))
            offset_ms = (evt_time - base_time).total_seconds() * 1000

            risk = 0.0
            if evt.confidence_score is not None:
                risk = max(0.0, 1.0 - evt.confidence_score) * 0.5

            label = evt.agent_name or evt.agent_id or evt.event_type
            if evt.tool_name:
                label = f"{label}: {evt.tool_name}"

            entries.append(TimelineEntry(
                event_id=evt.event_id,
                agent_id=evt.agent_id,
                agent_name=evt.agent_name or evt.agent_id,
                event_type=evt.event_type,
                start_ms=offset_ms,
                duration_ms=evt.latency_ms,
                label=label,
                risk_score=risk,
                confidence=evt.confidence_score,
            ))
        return entries

    def get_agent_events(self, trace_id: str, agent_id: str) -> list[TraceEvent]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM events WHERE trace_id = ? AND agent_id = ? ORDER BY timestamp ASC",
            (trace_id, agent_id)
        ).fetchall()
        return [self._row_to_event(row) for row in rows]

    def get_event(self, event_id: str) -> TraceEvent | None:
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM events WHERE event_id = ?", (event_id,)).fetchone()
        if row:
            return self._row_to_event(row)
        return None

    def causal_chain(self, event_id: str) -> list[str]:
        visited = set()
        chain = []

        def dfs(eid: str):
            if eid in visited:
                return
            visited.add(eid)
            chain.append(eid)
            with self._lock:
                for parent in self._reverse_adjacency.get(eid, []):
                    dfs(parent)

        dfs(event_id)
        return chain

    def downstream_impact(self, event_id: str) -> list[str]:
        visited = set()
        impacted = []

        def dfs(eid: str):
            if eid in visited:
                return
            visited.add(eid)
            impacted.append(eid)
            with self._lock:
                for child in self._adjacency.get(eid, []):
                    dfs(child)

        dfs(event_id)
        return impacted

    def _row_to_event(self, row: sqlite3.Row) -> TraceEvent:
        return TraceEvent(
            event_id=row["event_id"],
            timestamp=row["timestamp"],
            trace_id=row["trace_id"],
            span_id=row["span_id"],
            parent_span_id=row["parent_span_id"],
            event_type=row["event_type"],
            agent_id=row["agent_id"],
            agent_name=row["agent_name"],
            model_name=row["model_name"],
            framework=row["framework"],
            input_tokens=row["input_tokens"],
            output_tokens=row["output_tokens"],
            latency_ms=row["latency_ms"],
            cost_usd=row["cost_usd"],
            confidence_score=row["confidence_score"],
            input_preview=row["input_preview"],
            output_preview=row["output_preview"],
            tool_name=row["tool_name"],
            tool_input=json.loads(row["tool_input"] or "{}"),
            tool_output=row["tool_output"] or "",
            tool_success=bool(row["tool_success"]),
            error_message=row["error_message"],
            causal_parents=json.loads(row["causal_parents"] or "[]"),
            data_dependencies=json.loads(row["data_dependencies"] or "[]"),
            tags=json.loads(row["tags"] or "{}"),
            metadata=json.loads(row["metadata"] or "{}"),
        )
