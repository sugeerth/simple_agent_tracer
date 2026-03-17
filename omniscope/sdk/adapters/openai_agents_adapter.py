"""
OpenAI Agents SDK adapter for OMNISCOPE.

Wraps OpenAI's Agents SDK (agents.run / Runner) to trace:
- Agent executions
- Tool calls
- Handoffs between agents
- Guardrail evaluations

Usage:
    from agents import Agent, Runner
    from omniscope.sdk.adapters.openai_agents_adapter import OmniscopeAgentsTracer

    tracer = OmniscopeAgentsTracer(server_url="http://localhost:8781")

    agent = Agent(name="researcher", instructions="...")
    result = await tracer.trace_run(agent, "What is quantum computing?")

    # Or wrap the Runner directly:
    traced_runner = tracer.wrap_runner(Runner)
    result = await traced_runner.run(agent, "What is quantum computing?")
"""
from __future__ import annotations

import time
import uuid
from typing import Any

from ..collector import OmniscopeCollector


class OmniscopeAgentsTracer:
    """Instruments OpenAI Agents SDK with OMNISCOPE tracing."""

    def __init__(
        self,
        server_url: str = "http://localhost:8781",
        collector: OmniscopeCollector | None = None,
    ):
        self._collector = collector or OmniscopeCollector(server_url)

    async def trace_run(self, agent, input_text: str, **kwargs) -> Any:
        """Run an OpenAI agent with full tracing."""
        try:
            from agents import Runner
        except ImportError:
            raise ImportError(
                "OpenAI Agents SDK not installed. Install with: pip install openai-agents"
            )

        agent_name = getattr(agent, "name", "agent")
        trace_id = self._collector.start_trace(
            f"openai-agents:{agent_name}",
            framework="openai_agents",
        )

        self._collector.emit(
            trace_id=trace_id,
            agent_id=agent_name,
            agent_name=agent_name,
            event_type="agent_start",
            framework="openai_agents",
            input_preview=input_text[:2000],
            metadata={"instructions": getattr(agent, "instructions", "")[:500]},
        )

        start = time.time()
        try:
            result = await Runner.run(agent, input_text, **kwargs)
            latency = (time.time() - start) * 1000

            # Trace individual steps from the result
            self._trace_run_result(trace_id, result, agent_name)

            self._collector.emit(
                trace_id=trace_id,
                agent_id=agent_name,
                agent_name=agent_name,
                event_type="agent_end",
                framework="openai_agents",
                latency_ms=latency,
                output_preview=str(getattr(result, "final_output", result))[:2000],
            )
            self._collector.end_trace(trace_id)
            return result

        except Exception as e:
            latency = (time.time() - start) * 1000
            self._collector.emit(
                trace_id=trace_id,
                agent_id=agent_name,
                event_type="error",
                framework="openai_agents",
                latency_ms=latency,
                error_message=str(e),
            )
            self._collector.end_trace(trace_id, status="failed")
            raise

    def _trace_run_result(self, trace_id: str, result: Any, default_agent: str):
        """Extract and trace individual steps from a RunResult."""
        # OpenAI Agents SDK stores steps in result.new_items or similar
        items = getattr(result, "new_items", []) or []

        for item in items:
            item_type = type(item).__name__

            if "MessageOutput" in item_type:
                self._collector.emit(
                    trace_id=trace_id,
                    agent_id=getattr(item, "agent_name", default_agent) or default_agent,
                    event_type="llm_call",
                    framework="openai_agents",
                    model_name=getattr(item, "model", None),
                    input_tokens=getattr(getattr(item, "usage", None), "input_tokens", 0) or 0,
                    output_tokens=getattr(getattr(item, "usage", None), "output_tokens", 0) or 0,
                    output_preview=str(getattr(item, "content", ""))[:2000],
                )

            elif "ToolCallItem" in item_type or "ToolCall" in item_type:
                tool_name = getattr(item, "name", None) or getattr(item, "tool_name", "unknown")
                self._collector.emit(
                    trace_id=trace_id,
                    agent_id=getattr(item, "agent_name", default_agent) or default_agent,
                    event_type="tool_call",
                    framework="openai_agents",
                    tool_name=tool_name,
                    tool_input={"arguments": str(getattr(item, "arguments", ""))[:500]},
                    tool_output=str(getattr(item, "output", ""))[:2000],
                )

            elif "HandoffItem" in item_type or "Handoff" in item_type:
                target = getattr(item, "target_agent", None) or getattr(item, "agent_name", "unknown")
                self._collector.emit(
                    trace_id=trace_id,
                    agent_id=getattr(item, "source_agent", default_agent) or default_agent,
                    event_type="inter_agent_message",
                    framework="openai_agents",
                    output_preview=f"Handoff to {target}",
                    metadata={"target_agent": str(target)},
                )

    def wrap_runner(self, runner_class):
        """Return a wrapped Runner class that auto-traces all runs."""
        tracer = self

        class TracedRunner:
            @staticmethod
            async def run(agent, input_text: str, **kwargs):
                return await tracer.trace_run(agent, input_text, **kwargs)

        return TracedRunner
