"""OMNISCOPE SDK — instrument any agentic framework."""
from .collector import OmniscopeCollector, omniscope
from .decorators import trace_agent, trace_tool, trace_llm_call
from .adapters.generic_adapter import OmniscopeTracer

__all__ = [
    "OmniscopeCollector",
    "OmniscopeTracer",
    "omniscope",
    "trace_agent",
    "trace_tool",
    "trace_llm_call",
]
