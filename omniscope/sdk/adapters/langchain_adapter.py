"""
LangChain / LangGraph adapter for OMNISCOPE.

Implements LangChain's BaseCallbackHandler to automatically trace:
- LLM calls (on_llm_start / on_llm_end)
- Chain executions (on_chain_start / on_chain_end)
- Tool invocations (on_tool_start / on_tool_end)
- Agent actions (on_agent_action / on_agent_finish)
- Retrieval (on_retriever_start / on_retriever_end)

Usage with LangChain:
    from omniscope.sdk.adapters.langchain_adapter import OmniscopeCallbackHandler
    handler = OmniscopeCallbackHandler(server_url="http://localhost:8781")

    # With a chain:
    chain.invoke({"input": "..."}, config={"callbacks": [handler]})

    # With an agent:
    agent_executor.invoke({"input": "..."}, config={"callbacks": [handler]})

Usage with LangGraph:
    from langgraph.graph import StateGraph
    graph = StateGraph(...)
    app = graph.compile()
    app.invoke({"input": "..."}, config={"callbacks": [handler]})
"""
from __future__ import annotations

import time
import uuid
from typing import Any

from ..collector import OmniscopeCollector


def _get_base_handler():
    """Dynamically import LangChain's BaseCallbackHandler."""
    try:
        from langchain_core.callbacks import BaseCallbackHandler
        return BaseCallbackHandler
    except ImportError:
        try:
            from langchain.callbacks.base import BaseCallbackHandler
            return BaseCallbackHandler
        except ImportError:
            # Fallback: return object so the class definition still works
            # but won't be usable without langchain installed
            return object


_Base = _get_base_handler()


class OmniscopeCallbackHandler(_Base):
    """LangChain/LangGraph callback handler that sends trace events to OMNISCOPE."""

    def __init__(
        self,
        server_url: str = "http://localhost:8781",
        trace_name: str = "",
        collector: OmniscopeCollector | None = None,
    ):
        if _Base is not object:
            super().__init__()
        self._collector = collector or OmniscopeCollector(server_url)
        self._trace_id = self._collector.start_trace(
            trace_name or "langchain-trace",
            framework="langchain",
        )
        self._run_timers: dict[str, float] = {}
        self._run_agents: dict[str, str] = {}
        self._run_spans: dict[str, str] = {}

    @property
    def trace_id(self) -> str:
        return self._trace_id

    def finish(self) -> None:
        self._collector.end_trace(self._trace_id)

    # --- LLM ---

    def on_llm_start(self, serialized: dict, prompts: list[str], *, run_id, **kwargs):
        rid = str(run_id)
        self._run_timers[rid] = time.time()
        span_id = str(uuid.uuid4())
        self._run_spans[rid] = span_id

        model_name = serialized.get("id", [""])[-1] if serialized.get("id") else ""
        agent_id = kwargs.get("metadata", {}).get("langgraph_node", "") or model_name or "llm"
        self._run_agents[rid] = agent_id

    def on_llm_end(self, response, *, run_id, **kwargs):
        rid = str(run_id)
        latency = (time.time() - self._run_timers.get(rid, time.time())) * 1000
        agent_id = self._run_agents.get(rid, "llm")

        output_text = ""
        input_tokens = 0
        output_tokens = 0
        model_name = None

        if hasattr(response, "generations") and response.generations:
            for gen_list in response.generations:
                for gen in gen_list:
                    output_text += getattr(gen, "text", str(gen))

        if hasattr(response, "llm_output") and response.llm_output:
            usage = response.llm_output.get("token_usage", {})
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)
            model_name = response.llm_output.get("model_name")

        self._collector.emit(
            trace_id=self._trace_id,
            agent_id=agent_id,
            agent_name=agent_id,
            event_type="llm_call",
            model_name=model_name,
            framework="langchain",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency,
            output_preview=output_text[:2000],
            span_id=self._run_spans.get(rid),
        )

    def on_llm_error(self, error, *, run_id, **kwargs):
        rid = str(run_id)
        latency = (time.time() - self._run_timers.get(rid, time.time())) * 1000
        self._collector.emit(
            trace_id=self._trace_id,
            agent_id=self._run_agents.get(rid, "llm"),
            event_type="error",
            framework="langchain",
            latency_ms=latency,
            error_message=str(error),
        )

    # --- Chain ---

    def on_chain_start(self, serialized: dict, inputs: dict, *, run_id, **kwargs):
        rid = str(run_id)
        self._run_timers[rid] = time.time()
        span_id = str(uuid.uuid4())
        self._run_spans[rid] = span_id

        chain_name = serialized.get("id", [""])[-1] if serialized.get("id") else "chain"
        # LangGraph populates langgraph_node in metadata
        node_name = kwargs.get("metadata", {}).get("langgraph_node", "")
        agent_id = node_name or chain_name
        self._run_agents[rid] = agent_id

        self._collector.emit(
            trace_id=self._trace_id,
            agent_id=agent_id,
            agent_name=agent_id,
            event_type="chain_start",
            framework="langchain",
            input_preview=str(inputs)[:2000],
            span_id=span_id,
        )
        self._collector.push_span(self._trace_id, span_id)

    def on_chain_end(self, outputs, *, run_id, **kwargs):
        rid = str(run_id)
        latency = (time.time() - self._run_timers.get(rid, time.time())) * 1000
        agent_id = self._run_agents.get(rid, "chain")

        self._collector.emit(
            trace_id=self._trace_id,
            agent_id=agent_id,
            agent_name=agent_id,
            event_type="chain_end",
            framework="langchain",
            latency_ms=latency,
            output_preview=str(outputs)[:2000],
            span_id=self._run_spans.get(rid),
        )
        self._collector.pop_span(self._trace_id)

    def on_chain_error(self, error, *, run_id, **kwargs):
        rid = str(run_id)
        latency = (time.time() - self._run_timers.get(rid, time.time())) * 1000
        self._collector.emit(
            trace_id=self._trace_id,
            agent_id=self._run_agents.get(rid, "chain"),
            event_type="error",
            framework="langchain",
            latency_ms=latency,
            error_message=str(error),
        )
        self._collector.pop_span(self._trace_id)

    # --- Tool ---

    def on_tool_start(self, serialized: dict, input_str: str, *, run_id, **kwargs):
        rid = str(run_id)
        self._run_timers[rid] = time.time()
        tool_name = serialized.get("name", "unknown_tool")
        self._run_agents[rid] = f"tool:{tool_name}"
        self._run_spans[rid] = str(uuid.uuid4())

    def on_tool_end(self, output, *, run_id, **kwargs):
        rid = str(run_id)
        latency = (time.time() - self._run_timers.get(rid, time.time())) * 1000
        agent_label = self._run_agents.get(rid, "tool:unknown")
        tool_name = agent_label.replace("tool:", "")

        self._collector.emit(
            trace_id=self._trace_id,
            agent_id="tool_executor",
            event_type="tool_call",
            framework="langchain",
            latency_ms=latency,
            tool_name=tool_name,
            tool_output=str(output)[:2000],
            tool_success=True,
            span_id=self._run_spans.get(rid),
        )

    def on_tool_error(self, error, *, run_id, **kwargs):
        rid = str(run_id)
        latency = (time.time() - self._run_timers.get(rid, time.time())) * 1000
        agent_label = self._run_agents.get(rid, "tool:unknown")
        tool_name = agent_label.replace("tool:", "")

        self._collector.emit(
            trace_id=self._trace_id,
            agent_id="tool_executor",
            event_type="tool_call",
            framework="langchain",
            latency_ms=latency,
            tool_name=tool_name,
            tool_success=False,
            error_message=str(error),
        )

    # --- Agent ---

    def on_agent_action(self, action, *, run_id, **kwargs):
        rid = str(run_id)
        self._run_timers[rid] = time.time()
        self._collector.emit(
            trace_id=self._trace_id,
            agent_id="agent",
            event_type="agent_decision",
            framework="langchain",
            output_preview=f"Action: {action.tool} | Input: {str(action.tool_input)[:500]}",
        )

    def on_agent_finish(self, finish, *, run_id, **kwargs):
        rid = str(run_id)
        latency = (time.time() - self._run_timers.get(rid, time.time())) * 1000
        self._collector.emit(
            trace_id=self._trace_id,
            agent_id="agent",
            event_type="agent_end",
            framework="langchain",
            latency_ms=latency,
            output_preview=str(finish.return_values)[:2000],
        )

    # --- Retriever ---

    def on_retriever_start(self, serialized: dict, query: str, *, run_id, **kwargs):
        rid = str(run_id)
        self._run_timers[rid] = time.time()
        self._run_agents[rid] = "retriever"

    def on_retriever_end(self, documents, *, run_id, **kwargs):
        rid = str(run_id)
        latency = (time.time() - self._run_timers.get(rid, time.time())) * 1000
        doc_count = len(documents) if documents else 0
        self._collector.emit(
            trace_id=self._trace_id,
            agent_id="retriever",
            event_type="retrieval",
            framework="langchain",
            latency_ms=latency,
            output_preview=f"Retrieved {doc_count} documents",
            metadata={"document_count": doc_count},
        )
