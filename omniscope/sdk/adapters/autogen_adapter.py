"""
AutoGen / AG2 adapter for OMNISCOPE.

Instruments AutoGen group chats and two-agent conversations to trace:
- Agent messages (each reply in the conversation)
- Tool/function calls
- Group chat orchestration (speaker selection)
- Nested chats

Usage:
    from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
    from omniscope.sdk.adapters.autogen_adapter import OmniscopeAutoGenTracer

    tracer = OmniscopeAutoGenTracer(server_url="http://localhost:8781")

    assistant = AssistantAgent("assistant", llm_config={...})
    user_proxy = UserProxyAgent("user_proxy", code_execution_config={...})

    # Trace a two-agent chat
    result = tracer.trace_chat(user_proxy, "Build me a snake game", recipient=assistant)

    # Trace a group chat
    group = GroupChat(agents=[assistant, user_proxy, critic], messages=[])
    manager = GroupChatManager(groupchat=group, llm_config={...})
    result = tracer.trace_chat(user_proxy, "Build me a snake game", recipient=manager)
"""
from __future__ import annotations

import time
import uuid
from typing import Any

from ..collector import OmniscopeCollector


class OmniscopeAutoGenTracer:
    """Instruments AutoGen agents with OMNISCOPE tracing."""

    def __init__(
        self,
        server_url: str = "http://localhost:8781",
        collector: OmniscopeCollector | None = None,
    ):
        self._collector = collector or OmniscopeCollector(server_url)

    def trace_chat(
        self,
        initiator,
        message: str,
        recipient=None,
        trace_name: str = "",
        **chat_kwargs,
    ) -> Any:
        """Run an AutoGen chat with full OMNISCOPE tracing.

        Works with both two-agent chats and GroupChatManager.

        Args:
            initiator: The agent that starts the conversation
            message: The initial message
            recipient: The agent or GroupChatManager to chat with
            trace_name: Optional name for the trace
            **chat_kwargs: Additional kwargs passed to initiate_chat()
        """
        initiator_name = getattr(initiator, "name", "initiator")
        recipient_name = getattr(recipient, "name", "recipient") if recipient else "auto"
        _trace_name = trace_name or f"autogen:{initiator_name}->{recipient_name}"

        trace_id = self._collector.start_trace(_trace_name, framework="autogen")

        # Detect group chat
        is_group = _is_group_chat_manager(recipient)
        if is_group:
            agents = _get_group_agents(recipient)
            agent_names = [getattr(a, "name", str(i)) for i, a in enumerate(agents)]
            self._collector.emit(
                trace_id=trace_id,
                agent_id="group_manager",
                agent_name=recipient_name,
                event_type="agent_start",
                framework="autogen",
                output_preview=f"Group chat with: {', '.join(agent_names)}",
                metadata={"agents": agent_names, "type": "group_chat"},
            )

        # Patch all relevant agents to capture messages
        agents_to_patch = []
        if is_group:
            agents_to_patch = _get_group_agents(recipient)
            agents_to_patch.append(recipient)  # patch the manager too
        else:
            agents_to_patch = [initiator]
            if recipient:
                agents_to_patch.append(recipient)

        originals = {}
        for agent in agents_to_patch:
            originals[id(agent)] = self._patch_agent(agent, trace_id)

        # Emit the initial message
        self._collector.emit(
            trace_id=trace_id,
            agent_id=initiator_name,
            agent_name=initiator_name,
            event_type="inter_agent_message",
            framework="autogen",
            input_preview=message[:2000],
            output_preview=f"-> {recipient_name}: {message[:500]}",
            metadata={"target_agent": recipient_name, "message_type": "initiate"},
        )

        start = time.time()
        try:
            if recipient:
                result = initiator.initiate_chat(recipient, message=message, **chat_kwargs)
            else:
                result = initiator.initiate_chat(message=message, **chat_kwargs)
            latency = (time.time() - start) * 1000

            self._collector.emit(
                trace_id=trace_id,
                agent_id="system",
                agent_name="OMNISCOPE",
                event_type="agent_end",
                framework="autogen",
                latency_ms=latency,
                output_preview=f"Chat completed in {latency:.0f}ms",
            )
            self._collector.end_trace(trace_id)
            return result

        except Exception as e:
            latency = (time.time() - start) * 1000
            self._collector.emit(
                trace_id=trace_id,
                agent_id="system",
                event_type="error",
                framework="autogen",
                latency_ms=latency,
                error_message=str(e),
            )
            self._collector.end_trace(trace_id, status="failed")
            raise

        finally:
            # Unpatch all agents
            for agent in agents_to_patch:
                orig = originals.get(id(agent))
                if orig:
                    self._unpatch_agent(agent, orig)

    def _patch_agent(self, agent, trace_id: str) -> dict[str, Any]:
        """Monkey-patch an AutoGen agent to emit trace events. Returns original methods."""
        originals = {}
        agent_name = getattr(agent, "name", "agent")
        collector = self._collector

        # Patch generate_reply if it exists
        orig_generate = getattr(agent, "generate_reply", None)
        if orig_generate and callable(orig_generate):
            originals["generate_reply"] = orig_generate

            def traced_generate_reply(messages=None, sender=None, **kwargs):
                sender_name = getattr(sender, "name", "unknown") if sender else "unknown"
                span_id = str(uuid.uuid4())

                last_msg = ""
                if messages:
                    last = messages[-1] if isinstance(messages, list) else messages
                    last_msg = str(last.get("content", "") if isinstance(last, dict) else last)[:500]

                collector.emit(
                    trace_id=trace_id,
                    agent_id=agent_name,
                    agent_name=agent_name,
                    event_type="agent_start",
                    framework="autogen",
                    span_id=span_id,
                    input_preview=f"From {sender_name}: {last_msg}",
                )

                start = time.time()
                try:
                    result = orig_generate(messages=messages, sender=sender, **kwargs)
                    latency = (time.time() - start) * 1000

                    reply_text = ""
                    if isinstance(result, dict):
                        reply_text = str(result.get("content", ""))[:2000]
                    elif isinstance(result, str):
                        reply_text = result[:2000]

                    # Check if it's a function/tool call
                    if isinstance(result, dict) and result.get("function_call"):
                        fc = result["function_call"]
                        collector.emit(
                            trace_id=trace_id,
                            agent_id=agent_name,
                            agent_name=agent_name,
                            event_type="tool_call",
                            framework="autogen",
                            tool_name=fc.get("name", "unknown"),
                            tool_input={"arguments": str(fc.get("arguments", ""))[:500]},
                            latency_ms=latency,
                        )
                    elif isinstance(result, dict) and result.get("tool_calls"):
                        for tc in result["tool_calls"]:
                            fn = tc.get("function", {})
                            collector.emit(
                                trace_id=trace_id,
                                agent_id=agent_name,
                                agent_name=agent_name,
                                event_type="tool_call",
                                framework="autogen",
                                tool_name=fn.get("name", "unknown"),
                                tool_input={"arguments": str(fn.get("arguments", ""))[:500]},
                                latency_ms=latency,
                            )
                    else:
                        collector.emit(
                            trace_id=trace_id,
                            agent_id=agent_name,
                            agent_name=agent_name,
                            event_type="llm_call",
                            framework="autogen",
                            latency_ms=latency,
                            output_preview=reply_text,
                        )

                    return result
                except Exception as e:
                    latency = (time.time() - start) * 1000
                    collector.emit(
                        trace_id=trace_id,
                        agent_id=agent_name,
                        event_type="error",
                        framework="autogen",
                        latency_ms=latency,
                        error_message=str(e),
                    )
                    raise

            agent.generate_reply = traced_generate_reply

        # Patch execute_function if it exists (UserProxyAgent)
        orig_exec = getattr(agent, "execute_function", None)
        if orig_exec and callable(orig_exec):
            originals["execute_function"] = orig_exec

            def traced_execute_function(func_call, **kwargs):
                func_name = ""
                if isinstance(func_call, dict):
                    func_name = func_call.get("name", "unknown")
                else:
                    func_name = getattr(func_call, "name", "unknown")

                start = time.time()
                try:
                    result = orig_exec(func_call, **kwargs)
                    latency = (time.time() - start) * 1000
                    collector.emit(
                        trace_id=trace_id,
                        agent_id=agent_name,
                        agent_name=agent_name,
                        event_type="tool_call",
                        framework="autogen",
                        tool_name=func_name,
                        tool_output=str(result)[:2000],
                        tool_success=True,
                        latency_ms=latency,
                    )
                    return result
                except Exception as e:
                    latency = (time.time() - start) * 1000
                    collector.emit(
                        trace_id=trace_id,
                        agent_id=agent_name,
                        event_type="tool_call",
                        framework="autogen",
                        tool_name=func_name,
                        tool_success=False,
                        latency_ms=latency,
                        error_message=str(e),
                    )
                    raise

            agent.execute_function = traced_execute_function

        return originals

    def _unpatch_agent(self, agent, originals: dict[str, Any]):
        for method_name, original in originals.items():
            setattr(agent, method_name, original)


def _is_group_chat_manager(agent) -> bool:
    cls_name = type(agent).__name__ if agent else ""
    return "GroupChatManager" in cls_name


def _get_group_agents(manager) -> list:
    groupchat = getattr(manager, "groupchat", None) or getattr(manager, "_groupchat", None)
    if groupchat:
        return list(getattr(groupchat, "agents", []))
    return []
