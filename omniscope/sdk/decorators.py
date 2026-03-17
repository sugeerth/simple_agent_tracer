"""Decorators for instrumenting any Python function as a traced agent/tool/LLM call."""
from __future__ import annotations

import functools
import time
from typing import Any, Callable

from .collector import omniscope


def trace_agent(
    name: str = "",
    agent_id: str = "",
    framework: str = "generic",
    collector=None,
):
    """Decorator to trace a function as an agent decision."""
    _collector = collector or omniscope

    def decorator(fn: Callable) -> Callable:
        _name = name or fn.__name__
        _agent_id = agent_id or fn.__name__

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            trace_id = kwargs.pop("_trace_id", None) or ""
            start = time.time()
            try:
                result = fn(*args, **kwargs)
                latency = (time.time() - start) * 1000
                _collector.emit(
                    trace_id=trace_id,
                    agent_id=_agent_id,
                    agent_name=_name,
                    event_type="agent_decision",
                    framework=framework,
                    latency_ms=latency,
                    input_preview=_safe_preview(args, kwargs),
                    output_preview=_safe_preview(result),
                )
                return result
            except Exception as e:
                latency = (time.time() - start) * 1000
                _collector.emit(
                    trace_id=trace_id,
                    agent_id=_agent_id,
                    agent_name=_name,
                    event_type="error",
                    framework=framework,
                    latency_ms=latency,
                    input_preview=_safe_preview(args, kwargs),
                    error_message=str(e),
                )
                raise
        return wrapper
    return decorator


def trace_tool(
    name: str = "",
    agent_id: str = "",
    framework: str = "generic",
    collector=None,
):
    """Decorator to trace a function as a tool call."""
    _collector = collector or omniscope

    def decorator(fn: Callable) -> Callable:
        _name = name or fn.__name__
        _agent_id = agent_id or "tool_executor"

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            trace_id = kwargs.pop("_trace_id", None) or ""
            start = time.time()
            try:
                result = fn(*args, **kwargs)
                latency = (time.time() - start) * 1000
                _collector.emit(
                    trace_id=trace_id,
                    agent_id=_agent_id,
                    event_type="tool_call",
                    framework=framework,
                    latency_ms=latency,
                    tool_name=_name,
                    tool_input=_safe_dict(kwargs),
                    tool_output=str(result)[:2000],
                    tool_success=True,
                )
                return result
            except Exception as e:
                latency = (time.time() - start) * 1000
                _collector.emit(
                    trace_id=trace_id,
                    agent_id=_agent_id,
                    event_type="tool_call",
                    framework=framework,
                    latency_ms=latency,
                    tool_name=_name,
                    tool_input=_safe_dict(kwargs),
                    tool_success=False,
                    error_message=str(e),
                )
                raise
        return wrapper
    return decorator


def trace_llm_call(
    name: str = "",
    agent_id: str = "",
    model_name: str = "",
    framework: str = "generic",
    collector=None,
):
    """Decorator to trace a function as an LLM call."""
    _collector = collector or omniscope

    def decorator(fn: Callable) -> Callable:
        _name = name or fn.__name__
        _agent_id = agent_id or fn.__name__

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            trace_id = kwargs.pop("_trace_id", None) or ""
            start = time.time()
            try:
                result = fn(*args, **kwargs)
                latency = (time.time() - start) * 1000
                _collector.emit(
                    trace_id=trace_id,
                    agent_id=_agent_id,
                    agent_name=_name,
                    event_type="llm_call",
                    model_name=model_name,
                    framework=framework,
                    latency_ms=latency,
                    input_preview=_safe_preview(args, kwargs),
                    output_preview=_safe_preview(result),
                )
                return result
            except Exception as e:
                latency = (time.time() - start) * 1000
                _collector.emit(
                    trace_id=trace_id,
                    agent_id=_agent_id,
                    agent_name=_name,
                    event_type="error",
                    model_name=model_name,
                    framework=framework,
                    latency_ms=latency,
                    error_message=str(e),
                )
                raise
        return wrapper
    return decorator


def _safe_preview(*args) -> str:
    try:
        parts = []
        for a in args:
            s = str(a)
            if len(s) > 500:
                s = s[:500] + "..."
            parts.append(s)
        return " | ".join(parts)
    except Exception:
        return ""


def _safe_dict(d: dict) -> dict:
    try:
        result = {}
        for k, v in d.items():
            if k.startswith("_"):
                continue
            s = str(v)
            result[k] = s[:500] if len(s) > 500 else s
        return result
    except Exception:
        return {}
