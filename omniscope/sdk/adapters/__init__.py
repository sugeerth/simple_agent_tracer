"""Framework adapters for OMNISCOPE.

Each adapter instruments a specific agent framework with zero config.
All adapters are lazy-imported — the framework dependency is only
required when you actually use the adapter.

Supported frameworks:
    - LangChain / LangGraph
    - CrewAI
    - Anthropic SDK
    - OpenAI Agents SDK
    - AutoGen / AG2
    - Generic (any Python agent system)
"""

# Generic adapter is always available (no external deps)
from .generic_adapter import OmniscopeTracer, TraceContext, AgentContext

__all__ = [
    "OmniscopeTracer",
    "TraceContext",
    "AgentContext",
]
