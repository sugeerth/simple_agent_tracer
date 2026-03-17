"""
OMNISCOPE: Multi-Agent Observability Platform.

Quick start:
    import omniscope
    omniscope.init()                       # connect to server

    with omniscope.trace("my-workflow") as t:
        with t.agent("planner") as a:
            a.llm_call(model="gpt-4", input_text="Plan...", output_text="Step 1...")
            a.tool_call(tool_name="search", tool_input={"q": "AI"}, tool_output="Results...")

Framework integrations:
    handler = omniscope.langchain_handler()          # LangChain/LangGraph
    tracer = omniscope.crewai_tracer()               # CrewAI
    client = omniscope.wrap_anthropic(anthropic_client)  # Anthropic SDK
    tracer = omniscope.openai_agents_tracer()         # OpenAI Agents SDK
    tracer = omniscope.autogen_tracer()               # AutoGen/AG2
"""
__version__ = "0.1.0"

from omniscope.sdk.collector import OmniscopeCollector
from omniscope.sdk.adapters.generic_adapter import OmniscopeTracer, TraceContext, AgentContext

# Module-level singleton
_tracer: OmniscopeTracer | None = None
_server_url: str = "http://localhost:8781"


def init(server_url: str = "http://localhost:8781", framework: str = "generic") -> OmniscopeTracer:
    """Initialize OMNISCOPE. Call once at startup.

    Args:
        server_url: OMNISCOPE server URL (default: http://localhost:8781)
        framework: Default framework name for traces

    Returns:
        OmniscopeTracer instance (also stored as module singleton)

    Example:
        import omniscope
        omniscope.init()
        # or
        omniscope.init(server_url="http://my-server:8781")
    """
    global _tracer, _server_url
    _server_url = server_url
    _tracer = OmniscopeTracer(server_url=server_url, framework=framework)
    return _tracer


def trace(name: str = "", framework: str = "") -> TraceContext:
    """Start a traced workflow. Use as a context manager.

    Example:
        with omniscope.trace("research-pipeline") as t:
            with t.agent("planner") as a:
                a.llm_call(model="gpt-4", input_text="...", output_text="...")
    """
    t = _get_tracer()
    if framework:
        return TraceContext(t._collector, name, framework)
    return t.trace(name)


def _get_tracer() -> OmniscopeTracer:
    global _tracer
    if _tracer is None:
        _tracer = OmniscopeTracer(server_url=_server_url)
    return _tracer


# ---------------------------------------------------------------------------
# Framework-specific helpers (lazy imports — no dependency required)
# ---------------------------------------------------------------------------

def langchain_handler(trace_name: str = "", server_url: str = ""):
    """Get a LangChain/LangGraph callback handler.

    Example:
        handler = omniscope.langchain_handler()
        chain.invoke({"input": "..."}, config={"callbacks": [handler]})
    """
    from omniscope.sdk.adapters.langchain_adapter import OmniscopeCallbackHandler
    url = server_url or _server_url
    return OmniscopeCallbackHandler(server_url=url, trace_name=trace_name)


def crewai_tracer(server_url: str = ""):
    """Get a CrewAI tracer.

    Example:
        tracer = omniscope.crewai_tracer()
        result = tracer.trace_crew(crew, inputs={"topic": "AI"})
    """
    from omniscope.sdk.adapters.crewai_adapter import OmniscopeCrewTracer
    url = server_url or _server_url
    return OmniscopeCrewTracer(server_url=url)


def wrap_anthropic(client=None, server_url: str = ""):
    """Wrap an Anthropic client for tracing.

    Example:
        from anthropic import Anthropic
        client = omniscope.wrap_anthropic(Anthropic())
        resp = client.create(model="claude-sonnet-4-6", messages=[...], agent_id="my_agent")
    """
    from omniscope.sdk.adapters.anthropic_adapter import OmniscopeAnthropicWrapper
    url = server_url or _server_url
    return OmniscopeAnthropicWrapper(client=client, server_url=url)


def openai_agents_tracer(server_url: str = ""):
    """Get an OpenAI Agents SDK tracer.

    Example:
        tracer = omniscope.openai_agents_tracer()
        result = await tracer.trace_run(agent, "What is quantum computing?")
    """
    from omniscope.sdk.adapters.openai_agents_adapter import OmniscopeAgentsTracer
    url = server_url or _server_url
    return OmniscopeAgentsTracer(server_url=url)


def autogen_tracer(server_url: str = ""):
    """Get an AutoGen/AG2 tracer.

    Example:
        tracer = omniscope.autogen_tracer()
        result = tracer.trace_chat(manager, "Build me a website")
    """
    from omniscope.sdk.adapters.autogen_adapter import OmniscopeAutoGenTracer
    url = server_url or _server_url
    return OmniscopeAutoGenTracer(server_url=url)
