"""
Generic adapter for any agentic framework.

Provides a simple, framework-agnostic API for emitting trace events.
Use this when your framework doesn't have a dedicated adapter, or when
building custom agent systems.

Usage:
    from omniscope.sdk.adapters.generic_adapter import OmniscopeTracer

    tracer = OmniscopeTracer(server_url="http://localhost:8781")

    with tracer.trace("my-workflow") as t:
        with t.agent("planner") as planner:
            planner.llm_call(
                model="gpt-4",
                input_text="Plan the task...",
                output_text="Step 1: ...",
                input_tokens=100,
                output_tokens=200,
            )

        with t.agent("executor") as executor:
            executor.tool_call(
                tool_name="web_search",
                tool_input={"query": "latest AI news"},
                tool_output="Results: ...",
            )
            executor.llm_call(
                model="gpt-4",
                input_text="Synthesize results...",
                output_text="Based on the search...",
            )
"""
from __future__ import annotations

import time
import uuid
from typing import Any

from ..collector import OmniscopeCollector


class OmniscopeTracer:
    """Generic framework-agnostic tracer."""

    def __init__(
        self,
        server_url: str = "http://localhost:8781",
        framework: str = "generic",
        collector: OmniscopeCollector | None = None,
    ):
        self._collector = collector or OmniscopeCollector(server_url)
        self._framework = framework

    def trace(self, name: str = "") -> TraceContext:
        return TraceContext(self._collector, name, self._framework)


class TraceContext:
    """Context manager for a full trace."""

    def __init__(self, collector: OmniscopeCollector, name: str, framework: str):
        self._collector = collector
        self._name = name
        self._framework = framework
        self.trace_id: str = ""

    def __enter__(self) -> TraceContext:
        self.trace_id = self._collector.start_trace(self._name, self._framework)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._collector._flush_batch()
        status = "failed" if exc_type else "completed"
        self._collector.end_trace(self.trace_id, status)

    def agent(self, agent_id: str, agent_name: str = "") -> AgentContext:
        return AgentContext(self._collector, self.trace_id, agent_id, agent_name or agent_id, self._framework)

    def emit(self, **kwargs) -> str:
        kwargs["trace_id"] = self.trace_id
        kwargs.setdefault("framework", self._framework)
        return self._collector.emit(**kwargs)


class AgentContext:
    """Context manager for an agent within a trace."""

    def __init__(
        self,
        collector: OmniscopeCollector,
        trace_id: str,
        agent_id: str,
        agent_name: str,
        framework: str,
    ):
        self._collector = collector
        self._trace_id = trace_id
        self._agent_id = agent_id
        self._agent_name = agent_name
        self._framework = framework
        self._span_id = str(uuid.uuid4())
        self._start_time: float = 0.0
        self._last_event_id: str | None = None

    def __enter__(self) -> AgentContext:
        self._start_time = time.time()
        self._last_event_id = self._collector.emit(
            trace_id=self._trace_id,
            agent_id=self._agent_id,
            agent_name=self._agent_name,
            event_type="agent_start",
            framework=self._framework,
            span_id=self._span_id,
        )
        self._collector.push_span(self._trace_id, self._span_id)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        latency = (time.time() - self._start_time) * 1000
        event_type = "error" if exc_type else "agent_end"
        self._collector.emit(
            trace_id=self._trace_id,
            agent_id=self._agent_id,
            agent_name=self._agent_name,
            event_type=event_type,
            framework=self._framework,
            latency_ms=latency,
            error_message=str(exc_val) if exc_type else None,
        )
        self._collector.pop_span(self._trace_id)

    def llm_call(
        self,
        model: str = "",
        input_text: str = "",
        output_text: str = "",
        input_tokens: int = 0,
        output_tokens: int = 0,
        latency_ms: float = 0.0,
        confidence: float | None = None,
        cost_usd: float | None = None,
    ) -> str:
        event_id = self._collector.emit(
            trace_id=self._trace_id,
            agent_id=self._agent_id,
            agent_name=self._agent_name,
            event_type="llm_call",
            model_name=model,
            framework=self._framework,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            confidence_score=confidence,
            input_preview=input_text[:2000],
            output_preview=output_text[:2000],
            causal_parents=[self._last_event_id] if self._last_event_id else [],
        )
        self._last_event_id = event_id
        return event_id

    def tool_call(
        self,
        tool_name: str,
        tool_input: dict[str, Any] | None = None,
        tool_output: str = "",
        success: bool = True,
        latency_ms: float = 0.0,
        error: str | None = None,
    ) -> str:
        event_id = self._collector.emit(
            trace_id=self._trace_id,
            agent_id=self._agent_id,
            agent_name=self._agent_name,
            event_type="tool_call",
            framework=self._framework,
            tool_name=tool_name,
            tool_input=tool_input or {},
            tool_output=tool_output[:2000],
            tool_success=success,
            latency_ms=latency_ms,
            error_message=error,
            causal_parents=[self._last_event_id] if self._last_event_id else [],
        )
        self._last_event_id = event_id
        return event_id

    def decision(
        self,
        description: str = "",
        confidence: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        event_id = self._collector.emit(
            trace_id=self._trace_id,
            agent_id=self._agent_id,
            agent_name=self._agent_name,
            event_type="agent_decision",
            framework=self._framework,
            confidence_score=confidence,
            output_preview=description[:2000],
            metadata=metadata or {},
            causal_parents=[self._last_event_id] if self._last_event_id else [],
        )
        self._last_event_id = event_id
        return event_id

    def message_to(
        self,
        target_agent: str,
        content: str = "",
        message_type: str = "task",
    ) -> str:
        event_id = self._collector.emit(
            trace_id=self._trace_id,
            agent_id=self._agent_id,
            agent_name=self._agent_name,
            event_type="inter_agent_message",
            framework=self._framework,
            output_preview=f"[{message_type}] -> {target_agent}: {content[:200]}",
            metadata={"target_agent": target_agent, "message_type": message_type},
            causal_parents=[self._last_event_id] if self._last_event_id else [],
        )
        self._last_event_id = event_id
        return event_id
