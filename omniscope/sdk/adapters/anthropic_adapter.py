"""
Anthropic SDK adapter for OMNISCOPE.

Wraps the Anthropic Python SDK client to trace:
- messages.create calls
- Tool use responses
- Multi-turn conversations
- Token usage and costs

Usage:
    from anthropic import Anthropic
    from omniscope.sdk.adapters.anthropic_adapter import OmniscopeAnthropicWrapper

    client = Anthropic()
    traced = OmniscopeAnthropicWrapper(client, server_url="http://localhost:8781")

    # Use traced.messages.create() instead of client.messages.create()
    response = traced.create(
        model="claude-sonnet-4-6",
        messages=[{"role": "user", "content": "Hello"}],
        agent_id="my_agent",  # optional: tag which agent this belongs to
    )

    # Or wrap an existing trace:
    with traced.trace("my-workflow") as t:
        response = t.create(
            model="claude-sonnet-4-6",
            messages=[{"role": "user", "content": "Hello"}],
            agent_id="planner",
        )
"""
from __future__ import annotations

import time
from typing import Any

from ..collector import OmniscopeCollector


# Approximate pricing per million tokens (as of 2026)
_PRICING = {
    "claude-opus-4-6": {"input": 15.0, "output": 75.0},
    "claude-sonnet-4-6": {"input": 3.0, "output": 15.0},
    "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.0},
}


class OmniscopeAnthropicWrapper:
    """Wraps the Anthropic client to emit OMNISCOPE trace events."""

    def __init__(
        self,
        client=None,
        server_url: str = "http://localhost:8781",
        collector: OmniscopeCollector | None = None,
        default_trace_id: str | None = None,
    ):
        self._client = client
        self._collector = collector or OmniscopeCollector(server_url)
        self._default_trace_id = default_trace_id

    def trace(self, name: str = "") -> _AnthropicTraceContext:
        return _AnthropicTraceContext(self, name)

    def create(
        self,
        model: str = "claude-sonnet-4-6",
        messages: list[dict[str, Any]] | None = None,
        system: str | None = None,
        max_tokens: int = 4096,
        tools: list[dict] | None = None,
        agent_id: str = "anthropic_agent",
        agent_name: str = "",
        trace_id: str | None = None,
        **kwargs,
    ) -> Any:
        """Call messages.create with tracing."""
        if self._client is None:
            try:
                from anthropic import Anthropic
                self._client = Anthropic()
            except ImportError:
                raise ImportError("Anthropic SDK not installed. pip install anthropic")

        _trace_id = trace_id or self._default_trace_id
        if not _trace_id:
            _trace_id = self._collector.start_trace("anthropic-call", framework="anthropic")

        # Build input preview
        input_preview = ""
        if messages:
            last_msg = messages[-1]
            content = last_msg.get("content", "")
            if isinstance(content, list):
                text_parts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
                input_preview = " ".join(text_parts)[:500]
            else:
                input_preview = str(content)[:500]

        start = time.time()
        try:
            response = self._client.messages.create(
                model=model,
                messages=messages or [],
                system=system or "You are a helpful assistant.",
                max_tokens=max_tokens,
                tools=tools or [],
                **kwargs,
            )
            latency = (time.time() - start) * 1000

            # Extract response data
            output_text = ""
            tool_calls = []
            for block in response.content:
                if hasattr(block, "text"):
                    output_text += block.text
                elif hasattr(block, "type") and block.type == "tool_use":
                    tool_calls.append({
                        "tool_name": block.name,
                        "tool_input": block.input,
                        "tool_use_id": block.id,
                    })

            input_tokens = getattr(response.usage, "input_tokens", 0)
            output_tokens = getattr(response.usage, "output_tokens", 0)
            cost = self._estimate_cost(model, input_tokens, output_tokens)

            # Emit LLM call event
            event_id = self._collector.emit(
                trace_id=_trace_id,
                agent_id=agent_id,
                agent_name=agent_name or agent_id,
                event_type="llm_call",
                model_name=model,
                framework="anthropic",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency,
                cost_usd=cost,
                input_preview=input_preview,
                output_preview=output_text[:2000],
                metadata={
                    "stop_reason": response.stop_reason,
                    "model": response.model,
                },
            )

            # Emit tool call events
            for tc in tool_calls:
                self._collector.emit(
                    trace_id=_trace_id,
                    agent_id=agent_id,
                    event_type="tool_call",
                    framework="anthropic",
                    tool_name=tc["tool_name"],
                    tool_input=tc["tool_input"],
                    causal_parents=[event_id],
                )

            return response

        except Exception as e:
            latency = (time.time() - start) * 1000
            self._collector.emit(
                trace_id=_trace_id,
                agent_id=agent_id,
                event_type="error",
                model_name=model,
                framework="anthropic",
                latency_ms=latency,
                error_message=str(e),
            )
            raise

    def _estimate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        # Match model to pricing
        pricing = None
        for key, p in _PRICING.items():
            if key in model:
                pricing = p
                break
        if not pricing:
            pricing = {"input": 3.0, "output": 15.0}  # default to sonnet pricing

        cost = (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000
        return round(cost, 6)


class _AnthropicTraceContext:
    def __init__(self, wrapper: OmniscopeAnthropicWrapper, name: str):
        self._wrapper = wrapper
        self._name = name
        self.trace_id: str = ""

    def __enter__(self) -> _AnthropicTraceContext:
        self.trace_id = self._wrapper._collector.start_trace(
            self._name or "anthropic-trace",
            framework="anthropic",
        )
        self._wrapper._default_trace_id = self.trace_id
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._wrapper._collector._flush_batch()
        status = "failed" if exc_type else "completed"
        self._wrapper._collector.end_trace(self.trace_id, status)
        self._wrapper._default_trace_id = None

    def create(self, **kwargs) -> Any:
        kwargs["trace_id"] = self.trace_id
        return self._wrapper.create(**kwargs)
