"""OMNISCOPE SDK - instrument any agentic framework."""
from .collector import OmniscopeCollector, omniscope
from .decorators import trace_agent, trace_tool, trace_llm_call

__all__ = ["OmniscopeCollector", "omniscope", "trace_agent", "trace_tool", "trace_llm_call"]
