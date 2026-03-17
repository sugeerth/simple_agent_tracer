"""
CrewAI adapter for OMNISCOPE.

Wraps CrewAI's Crew execution to trace:
- Agent task assignments
- LLM calls within each agent
- Tool usage
- Inter-agent delegation
- Final crew output

Usage:
    from crewai import Agent, Task, Crew
    from omniscope.sdk.adapters.crewai_adapter import OmniscopeCrewTracer

    tracer = OmniscopeCrewTracer(server_url="http://localhost:8781")

    researcher = Agent(name="Researcher", ...)
    writer = Agent(name="Writer", ...)
    task1 = Task(description="...", agent=researcher)
    task2 = Task(description="...", agent=writer)

    crew = Crew(agents=[researcher, writer], tasks=[task1, task2])
    result = tracer.trace_crew(crew, inputs={"topic": "AI"})
"""
from __future__ import annotations

import time
import uuid
from typing import Any

from ..collector import OmniscopeCollector


class OmniscopeCrewTracer:
    """Instruments a CrewAI Crew with OMNISCOPE tracing."""

    def __init__(
        self,
        server_url: str = "http://localhost:8781",
        collector: OmniscopeCollector | None = None,
    ):
        self._collector = collector or OmniscopeCollector(server_url)

    def trace_crew(self, crew, inputs: dict[str, Any] | None = None) -> Any:
        """
        Run a CrewAI Crew with full OMNISCOPE tracing.

        Patches the crew's agents to emit trace events for every
        LLM call, tool use, and delegation.
        """
        trace_name = getattr(crew, "name", None) or "crewai-execution"
        trace_id = self._collector.start_trace(trace_name, framework="crewai")

        # Emit crew start
        agent_names = []
        for agent in getattr(crew, "agents", []):
            name = getattr(agent, "role", None) or getattr(agent, "name", "agent")
            agent_names.append(name)

        self._collector.emit(
            trace_id=trace_id,
            agent_id="crew_orchestrator",
            agent_name="Crew Orchestrator",
            event_type="agent_start",
            framework="crewai",
            output_preview=f"Crew started with agents: {', '.join(agent_names)}",
            metadata={"agents": agent_names, "task_count": len(getattr(crew, "tasks", []))},
        )

        # Patch agents
        original_executes = {}
        for agent in getattr(crew, "agents", []):
            agent_id = getattr(agent, "role", None) or getattr(agent, "name", str(id(agent)))
            original_executes[id(agent)] = getattr(agent, "execute_task", None)
            self._patch_agent(agent, agent_id, trace_id)

        start = time.time()
        try:
            result = crew.kickoff(inputs=inputs)
            latency = (time.time() - start) * 1000

            self._collector.emit(
                trace_id=trace_id,
                agent_id="crew_orchestrator",
                agent_name="Crew Orchestrator",
                event_type="agent_end",
                framework="crewai",
                latency_ms=latency,
                output_preview=str(result)[:2000],
            )
            self._collector.end_trace(trace_id)
            return result

        except Exception as e:
            latency = (time.time() - start) * 1000
            self._collector.emit(
                trace_id=trace_id,
                agent_id="crew_orchestrator",
                event_type="error",
                framework="crewai",
                latency_ms=latency,
                error_message=str(e),
            )
            self._collector.end_trace(trace_id, status="failed")
            raise

        finally:
            # Unpatch
            for agent in getattr(crew, "agents", []):
                orig = original_executes.get(id(agent))
                if orig:
                    agent.execute_task = orig

    def _patch_agent(self, agent, agent_id: str, trace_id: str):
        """Monkey-patch a CrewAI agent to emit trace events."""
        original_execute = getattr(agent, "execute_task", None)
        if original_execute is None:
            return

        collector = self._collector

        def traced_execute(task, context=None, tools=None):
            task_desc = getattr(task, "description", str(task))[:500]
            span_id = str(uuid.uuid4())

            collector.emit(
                trace_id=trace_id,
                agent_id=agent_id,
                agent_name=agent_id,
                event_type="agent_start",
                framework="crewai",
                input_preview=task_desc,
                span_id=span_id,
                metadata={"task_description": task_desc},
            )
            collector.push_span(trace_id, span_id)

            start = time.time()
            try:
                result = original_execute(task, context=context, tools=tools)
                latency = (time.time() - start) * 1000

                collector.emit(
                    trace_id=trace_id,
                    agent_id=agent_id,
                    agent_name=agent_id,
                    event_type="agent_end",
                    framework="crewai",
                    latency_ms=latency,
                    output_preview=str(result)[:2000],
                )
                collector.pop_span(trace_id)
                return result
            except Exception as e:
                latency = (time.time() - start) * 1000
                collector.emit(
                    trace_id=trace_id,
                    agent_id=agent_id,
                    event_type="error",
                    framework="crewai",
                    latency_ms=latency,
                    error_message=str(e),
                )
                collector.pop_span(trace_id)
                raise

        agent.execute_task = traced_execute

    def emit_manual(
        self,
        trace_id: str,
        agent_id: str,
        event_type: str = "agent_decision",
        **kwargs,
    ) -> str:
        """Emit a manual trace event for custom CrewAI instrumentation."""
        return self._collector.emit(
            trace_id=trace_id,
            agent_id=agent_id,
            event_type=event_type,
            framework="crewai",
            **kwargs,
        )
