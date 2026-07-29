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
    - Claude Code (live session transcript tailing)
    - Generic (any Python agent system)
"""

# Generic adapter is always available (no external deps)
from .generic_adapter import OmniscopeTracer, TraceContext, AgentContext

# Claude Code adapter needs only stdlib + the SDK's own httpx, so it imports eagerly too
from .claude_code_adapter import OmniscopeClaudeCodeTracer, ClaudeSessionTailer, discover

__all__ = [
    "OmniscopeTracer",
    "TraceContext",
    "AgentContext",
    "OmniscopeClaudeCodeTracer",
    "ClaudeSessionTailer",
    "discover",
]
