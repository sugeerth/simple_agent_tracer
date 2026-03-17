"""Core trace collector -- sends events to OMNISCOPE server."""
from __future__ import annotations

import uuid
import time
from datetime import datetime
from typing import Any

import httpx


class OmniscopeCollector:
    """
    Collects trace events and sends them to the OMNISCOPE server.

    Usage:
        collector = OmniscopeCollector("http://localhost:8781")
        trace_id = collector.start_trace("my-agent-workflow")
        collector.emit(trace_id=trace_id, agent_id="planner", event_type="llm_call", ...)
        collector.end_trace(trace_id)

    Or as context manager:
        with collector.trace("my-workflow") as t:
            t.emit(agent_id="planner", event_type="llm_call", ...)
    """

    def __init__(self, server_url: str = "http://localhost:8781", async_mode: bool = False):
        self.server_url = server_url.rstrip("/")
        self._client = httpx.Client(timeout=10.0)
        self._traces: dict[str, list[dict]] = {}
        self._span_stack: dict[str, list[str]] = {}
        self._batch_buffer: list[dict] = []
        self._batch_size = 10

    def start_trace(self, name: str = "", framework: str = "generic") -> str:
        trace_id = str(uuid.uuid4())
        self._traces[trace_id] = []
        self._span_stack[trace_id] = []
        if name:
            self.emit(
                trace_id=trace_id,
                agent_id="system",
                event_type="system_event",
                framework=framework,
                tags={"trace_name": name},
                output_preview=f"Trace started: {name}",
            )
        return trace_id

    def end_trace(self, trace_id: str, status: str = "completed") -> None:
        self._flush_batch()
        try:
            self._client.post(f"{self.server_url}/api/v1/traces/{trace_id}/{status}")
        except Exception:
            pass

    def trace(self, name: str = "", framework: str = "generic") -> _TraceContext:
        return _TraceContext(self, name, framework)

    def emit(
        self,
        trace_id: str,
        agent_id: str = "",
        agent_name: str = "",
        event_type: str = "system_event",
        model_name: str | None = None,
        framework: str = "generic",
        input_tokens: int = 0,
        output_tokens: int = 0,
        latency_ms: float = 0.0,
        cost_usd: float | None = None,
        confidence_score: float | None = None,
        input_preview: str = "",
        output_preview: str = "",
        tool_name: str | None = None,
        tool_input: dict[str, Any] | None = None,
        tool_output: str = "",
        tool_success: bool = True,
        error_message: str | None = None,
        causal_parents: list[str] | None = None,
        data_dependencies: list[str] | None = None,
        tags: dict[str, str] | None = None,
        metadata: dict[str, Any] | None = None,
        span_id: str | None = None,
        parent_span_id: str | None = None,
    ) -> str:
        event_id = str(uuid.uuid4())
        _span_id = span_id or str(uuid.uuid4())

        # Auto-set parent from span stack
        _parent = parent_span_id
        if _parent is None and trace_id in self._span_stack and self._span_stack[trace_id]:
            _parent = self._span_stack[trace_id][-1]

        event = {
            "event_id": event_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "trace_id": trace_id,
            "span_id": _span_id,
            "parent_span_id": _parent,
            "event_type": event_type,
            "agent_id": agent_id,
            "agent_name": agent_name or agent_id,
            "model_name": model_name,
            "framework": framework,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency_ms": latency_ms,
            "cost_usd": cost_usd,
            "confidence_score": confidence_score,
            "input_preview": input_preview[:2000],
            "output_preview": output_preview[:2000],
            "tool_name": tool_name,
            "tool_input": tool_input or {},
            "tool_output": tool_output[:2000],
            "tool_success": tool_success,
            "error_message": error_message,
            "causal_parents": causal_parents or [],
            "data_dependencies": data_dependencies or [],
            "tags": tags or {},
            "metadata": metadata or {},
        }

        self._batch_buffer.append(event)
        if len(self._batch_buffer) >= self._batch_size:
            self._flush_batch()

        return event_id

    def push_span(self, trace_id: str, span_id: str) -> None:
        if trace_id not in self._span_stack:
            self._span_stack[trace_id] = []
        self._span_stack[trace_id].append(span_id)

    def pop_span(self, trace_id: str) -> str | None:
        if trace_id in self._span_stack and self._span_stack[trace_id]:
            return self._span_stack[trace_id].pop()
        return None

    def _flush_batch(self) -> None:
        if not self._batch_buffer:
            return
        batch = self._batch_buffer[:]
        self._batch_buffer.clear()
        try:
            self._client.post(
                f"{self.server_url}/api/v1/traces",
                json={"events": batch},
                timeout=5.0,
            )
        except Exception as e:
            # Silently drop on connection failure -- don't break the instrumented app
            pass


class _TraceContext:
    def __init__(self, collector: OmniscopeCollector, name: str, framework: str):
        self._collector = collector
        self._name = name
        self._framework = framework
        self.trace_id: str = ""

    def __enter__(self) -> _TraceContext:
        self.trace_id = self._collector.start_trace(self._name, self._framework)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._collector._flush_batch()
        status = "failed" if exc_type else "completed"
        self._collector.end_trace(self.trace_id, status)

    def emit(self, **kwargs) -> str:
        kwargs["trace_id"] = self.trace_id
        if "framework" not in kwargs:
            kwargs["framework"] = self._framework
        return self._collector.emit(**kwargs)


# Singleton for convenience
omniscope = OmniscopeCollector()
