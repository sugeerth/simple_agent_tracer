"""Unit tests for the Claude Code live-tracing adapter.

Covers the pieces of omniscope/sdk/adapters/claude_code_adapter.py:

* discover()             -- session transcript discovery
* ClaudeSessionTailer    -- incremental, partial-line-safe JSONL tailing + noise skipping
* _SessionMapper         -- transcript records -> OMNISCOPE events (durations, errors, truncation)
* OmniscopeClaudeCodeTracer -- end-to-end: event sequence, trace naming, subagent spawn linking

All fixture data is SYNTHETIC: it mirrors the Claude Code transcript schema
(envelope fields, assistant message splitting by message.id, tool_use/tool_result
pairing, noise line types) but contains no real transcript content.

No network and no server: the mapper/tracer are driven with a FakeCollector that
mimics OmniscopeCollector's emit()/_batch_buffer contract.

Run: python3 -m pytest tests/test_claude_code_adapter.py -q
"""
from __future__ import annotations

import json
import os
import time
import uuid as uuid_module
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from omniscope.sdk.adapters.claude_code_adapter import (
    ClaudeSessionTailer,
    OmniscopeClaudeCodeTracer,
    _SessionMapper,
    discover,
)

# ---------------------------------------------------------------------------
# Fake collector (mirrors OmniscopeCollector's surface used by the adapter)
# ---------------------------------------------------------------------------


class FakeCollector:
    """Records every emit() as a full event dict; no batching thresholds, no network.

    Mirrors the real collector's contract that the adapter relies on:
    emit(**kwargs) -> event_id, events appended to _batch_buffer (so the
    mapper can patch timestamps), _batch_size, _flush_batch(), end_trace().
    """

    def __init__(self):
        self.events: list[dict] = []  # every event, in emit order
        self.ended: list[tuple[str, str]] = []
        self._batch_buffer: list[dict] = []
        self._batch_size = 10

    def emit(self, trace_id: str = "", **kw: Any) -> str:
        event_id = str(uuid_module.uuid4())
        agent_id = kw.get("agent_id", "")
        event = {
            "event_id": event_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "trace_id": trace_id,
            "span_id": kw.get("span_id") or str(uuid_module.uuid4()),
            "parent_span_id": kw.get("parent_span_id"),
            "event_type": kw.get("event_type", "system_event"),
            "agent_id": agent_id,
            "agent_name": kw.get("agent_name") or agent_id,
            "model_name": kw.get("model_name"),
            "framework": kw.get("framework", "generic"),
            "input_tokens": kw.get("input_tokens", 0),
            "output_tokens": kw.get("output_tokens", 0),
            "latency_ms": kw.get("latency_ms", 0.0),
            "cost_usd": kw.get("cost_usd"),
            "confidence_score": kw.get("confidence_score"),
            "input_preview": (kw.get("input_preview") or "")[:2000],
            "output_preview": (kw.get("output_preview") or "")[:2000],
            "tool_name": kw.get("tool_name"),
            "tool_input": kw.get("tool_input") or {},
            "tool_output": (kw.get("tool_output") or "")[:2000],
            "tool_success": kw.get("tool_success", True),
            "error_message": kw.get("error_message"),
            "causal_parents": kw.get("causal_parents") or [],
            "data_dependencies": kw.get("data_dependencies") or [],
            "tags": kw.get("tags") or {},
            "metadata": kw.get("metadata") or {},
        }
        self.events.append(event)  # same object: timestamp patches stay visible
        self._batch_buffer.append(event)
        return event_id

    def _flush_batch(self) -> None:
        self._batch_buffer.clear()

    def end_trace(self, trace_id: str, status: str = "completed") -> None:
        self._flush_batch()
        self.ended.append((trace_id, status))


# ---------------------------------------------------------------------------
# Synthetic fixture lines (modeled on the Claude Code transcript schema)
# ---------------------------------------------------------------------------

SESSION_ID = "11111111-2222-3333-4444-555555555555"
SYNTH_CWD = "/tmp/omniscope-synthetic-project"
CHILD_AGENT_ID = "abc123def456abc12"
BASH_TOOL_USE_ID = "toolu_synth_bash0001"
TASK_TOOL_USE_ID = "toolu_synth_task0001"
PROMPT_TEXT = "Refactor the synthetic widget module"
ASSISTANT_TEXT = "I will start by listing the synthetic project files."
TRUNC_SUFFIX = "...[truncated]"


def _uuid(i: int) -> str:
    return f"00000000-0000-4000-8000-{i:012d}"


def T(seconds: float) -> str:
    base = datetime(2026, 6, 12, 10, 0, 0)
    return (base + timedelta(seconds=seconds)).isoformat(timespec="milliseconds") + "Z"


def _env(uuid_: str, parent: str | None, ts: str) -> dict:
    return {
        "uuid": uuid_,
        "parentUuid": parent,
        "timestamp": ts,
        "sessionId": SESSION_ID,
        "isSidechain": False,
        "cwd": SYNTH_CWD,
        "gitBranch": "main",
        "version": "2.1.173",
        "userType": "external",
        "entrypoint": "cli",
    }


def user_prompt_line(uuid_: str, parent: str | None, ts: str, text: str,
                     is_meta: bool = False) -> dict:
    line = _env(uuid_, parent, ts)
    line.update({
        "type": "user",
        "promptId": "prompt-" + uuid_[-8:],
        "message": {"role": "user", "content": text},
    })
    if is_meta:
        line["isMeta"] = True
    return line


def _usage(input_tokens: int = 120, output_tokens: int = 45) -> dict:
    return {
        "input_tokens": input_tokens,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
        "output_tokens": output_tokens,
    }


def assistant_line(uuid_: str, parent: str, ts: str, msg_id: str, blocks: list[dict],
                   stop_reason: str | None = None, usage: dict | None = None) -> dict:
    line = _env(uuid_, parent, ts)
    line.update({
        "type": "assistant",
        "requestId": "req_synth" + uuid_[-8:],
        "message": {
            "model": "claude-fable-5",
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "content": blocks,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": usage or _usage(),
        },
    })
    return line


def tool_use_block(tool_use_id: str, name: str, tool_input: dict) -> dict:
    return {"type": "tool_use", "id": tool_use_id, "name": name,
            "input": tool_input, "caller": {"type": "direct"}}


def tool_result_line(uuid_: str, parent: str, ts: str, tool_use_id: str, content,
                     tool_use_result, source_uuid: str, is_error: bool = False) -> dict:
    line = _env(uuid_, parent, ts)
    line.update({
        "type": "user",
        "message": {
            "role": "user",
            "content": [{
                "tool_use_id": tool_use_id,
                "type": "tool_result",
                "content": content,
                "is_error": is_error,
            }],
        },
        "toolUseResult": tool_use_result,
        "sourceToolAssistantUUID": source_uuid,
    })
    return line


def turn_duration_line(uuid_: str, parent: str, ts: str, duration_ms: int) -> dict:
    line = _env(uuid_, parent, ts)
    line.update({
        "type": "system",
        "subtype": "turn_duration",
        "durationMs": duration_ms,
        "messageCount": 6,
        "pendingWorkflowCount": 0,
        "isMeta": False,
    })
    return line


def noise_lines() -> list[dict]:
    """Transcript noise the pipeline must drop."""
    attachment = _env(_uuid(90), _uuid(89), T(0.5))
    attachment.update({"type": "attachment", "attachment": {"type": "task_reminder"}})
    return [
        {"type": "file-history-snapshot", "messageId": "msg_synth_snap",
         "snapshot": {"trackedFileBackups": {}}, "isSnapshotUpdate": False},
        {"type": "mode", "mode": "normal", "sessionId": SESSION_ID},
        {"type": "permission-mode", "permissionMode": "auto"},
        {"type": "last-prompt", "lastPrompt": "synthetic last prompt", "leafUuid": _uuid(91)},
        {"type": "queue-operation", "operation": "enqueue", "sessionId": SESSION_ID,
         "timestamp": T(0.6), "content": "synthetic queued command"},
        {"type": "pr-link", "prNumber": 1, "prUrl": "https://example.invalid/pr/1"},
        user_prompt_line(_uuid(92), _uuid(90), T(0.7),
                         "<local-command-caveat>synthetic meta caveat</local-command-caveat>",
                         is_meta=True),
        attachment,
    ]


def session_lines() -> list[dict]:
    """A coherent synthetic session: prompt -> assistant turn (text + Bash
    tool_use split across two lines sharing one message.id) -> tool_result ->
    Task spawn -> async-launch result -> turn_duration. Noise interleaved."""
    bash_use = assistant_line(
        _uuid(3), _uuid(2), T(2.0), "msg_synth_A",
        [tool_use_block(BASH_TOOL_USE_ID, "Bash",
                        {"command": f"ls {SYNTH_CWD}", "description": "List project files"})],
        stop_reason="tool_use",
    )
    task_use = assistant_line(
        _uuid(5), _uuid(4), T(5.0), "msg_synth_B",
        [tool_use_block(TASK_TOOL_USE_ID, "Task",
                        {"description": "Synthetic child task",
                         "prompt": "Do a synthetic sub-task",
                         "run_in_background": True})],
        stop_reason="tool_use",
        usage=_usage(input_tokens=200, output_tokens=60),
    )
    return [
        {"type": "file-history-snapshot", "messageId": "msg_synth_snap0",
         "snapshot": {"trackedFileBackups": {}}, "isSnapshotUpdate": False},
        {"type": "ai-title", "aiTitle": "Synthetic widget refactor", "sessionId": SESSION_ID},
        user_prompt_line(_uuid(1), None, T(0.0), PROMPT_TEXT),
        assistant_line(_uuid(2), _uuid(1), T(1.0), "msg_synth_A",
                       [{"type": "text", "text": ASSISTANT_TEXT}]),
        bash_use,
        {"type": "mode", "mode": "normal", "sessionId": SESSION_ID},
        tool_result_line(_uuid(4), _uuid(3), T(4.25), BASH_TOOL_USE_ID,
                         "file_a.py\nfile_b.py",
                         {"stdout": "file_a.py\nfile_b.py", "stderr": "",
                          "interrupted": False, "isImage": False},
                         source_uuid=_uuid(3)),
        task_use,
        tool_result_line(_uuid(6), _uuid(5), T(6.0), TASK_TOOL_USE_ID,
                         "Async agent launched",
                         {"isAsync": True, "status": "async_launched",
                          "agentId": CHILD_AGENT_ID,
                          "description": "Synthetic child task",
                          "prompt": "Do a synthetic sub-task"},
                         source_uuid=_uuid(5)),
        turn_duration_line(_uuid(7), _uuid(6), T(7.0), 7000),
        {"type": "last-prompt", "lastPrompt": PROMPT_TEXT, "leafUuid": _uuid(7)},
    ]


def child_lines() -> list[dict]:
    """Synthetic subagent transcript spawned by the Task tool."""
    prompt = user_prompt_line(_uuid(61), None, T(6.5), "Do a synthetic sub-task")
    reply = assistant_line(_uuid(62), _uuid(61), T(7.5), "msg_synth_C",
                           [{"type": "text", "text": "Child task done."}],
                           usage=_usage(input_tokens=10, output_tokens=5))
    for line in (prompt, reply):
        line["isSidechain"] = True
        line["agentId"] = CHILD_AGENT_ID
    return [prompt, reply]


def _write_jsonl(path: Path, lines: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for line in lines:
            handle.write(json.dumps(line) + "\n")


def _append_text(path: Path, text: str) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(text)


def make_tracer(tmp_path: Path, lines: list[dict],
                with_child: bool = False) -> tuple[FakeCollector, OmniscopeClaudeCodeTracer]:
    session_file = tmp_path / f"{SESSION_ID}.jsonl"
    _write_jsonl(session_file, lines)
    if with_child:
        sub_dir = tmp_path / SESSION_ID / "subagents"
        _write_jsonl(sub_dir / f"agent-{CHILD_AGENT_ID}.jsonl", child_lines())
        (sub_dir / f"agent-{CHILD_AGENT_ID}.meta.json").write_text(json.dumps({
            "agentType": "general-purpose",
            "description": "Synthetic child task",
            "toolUseId": TASK_TOOL_USE_ID,
        }))
    fake = FakeCollector()
    tracer = OmniscopeClaudeCodeTracer(session_file, collector=fake)
    return fake, tracer


def drain(tracer: OmniscopeClaudeCodeTracer) -> None:
    while tracer.poll():
        pass


# ---------------------------------------------------------------------------
# discover()
# ---------------------------------------------------------------------------


def test_discover_lists_sessions_newest_first(tmp_path):
    project = tmp_path / "-Users-synthetic-project"
    older = project / "aaaaaaaa-0000-4000-8000-000000000001.jsonl"
    newer = project / "bbbbbbbb-0000-4000-8000-000000000002.jsonl"
    _write_jsonl(older, [user_prompt_line(_uuid(1), None, T(0.0), "old session")])
    _write_jsonl(newer, [user_prompt_line(_uuid(2), None, T(1.0), "new session")])
    past = time.time() - 100
    os.utime(older, (past, past))

    sessions = discover(tmp_path)
    assert [s["session_id"] for s in sessions] == [newer.stem, older.stem]
    assert sessions[0]["path"] == newer
    assert sessions[0]["project_dir"] == project.name
    assert discover(tmp_path / "does-not-exist") == []


# ---------------------------------------------------------------------------
# Tailer
# ---------------------------------------------------------------------------


def test_tailer_reads_incrementally(tmp_path):
    path = tmp_path / f"{SESSION_ID}.jsonl"
    line_a = user_prompt_line(_uuid(31), None, T(0.0), "synthetic prompt one")
    line_b = turn_duration_line(_uuid(32), _uuid(31), T(1.0), 1000)
    _write_jsonl(path, [line_a, line_b])

    tailer = ClaudeSessionTailer(path)
    assert [r["uuid"] for r in tailer.read_new()] == [_uuid(31), _uuid(32)]
    assert tailer.read_new() == []  # nothing new

    # Append: malformed JSON and a non-dict line are skipped, valid line returned.
    _append_text(path, "{{{not json\n")
    _append_text(path, "[1, 2, 3]\n")
    line_c = user_prompt_line(_uuid(33), _uuid(32), T(2.0), "synthetic prompt two")
    _append_text(path, json.dumps(line_c) + "\n")
    assert [r["uuid"] for r in tailer.read_new()] == [_uuid(33)]


def test_tailer_holds_back_partial_line(tmp_path):
    path = tmp_path / f"{SESSION_ID}.jsonl"
    line_a = json.dumps(user_prompt_line(_uuid(41), None, T(0.0), "synthetic prompt"))
    line_b = json.dumps(assistant_line(_uuid(42), _uuid(41), T(1.0), "msg_synth_P",
                                       [{"type": "text", "text": "partial write test"}]))
    assert len(line_b) > 40

    # Simulate Claude Code mid-append: full line A plus the first 40 bytes of B.
    path.write_text(line_a + "\n" + line_b[:40], encoding="utf-8")
    tailer = ClaudeSessionTailer(path)
    assert [r["uuid"] for r in tailer.read_new()] == [_uuid(41)], \
        "tailer must return only complete lines and hold back the partial tail"

    # Complete line B; the tailer must now return exactly B, intact.
    _append_text(path, line_b[40:] + "\n")
    records = tailer.read_new()
    assert [r["uuid"] for r in records] == [_uuid(42)]
    assert records[0]["message"]["content"][0]["text"] == "partial write test"
    assert tailer.read_new() == []


def test_tailer_skips_noise_lines(tmp_path):
    path = tmp_path / f"{SESSION_ID}.jsonl"
    keep_a = user_prompt_line(_uuid(51), None, T(0.0), "real prompt")
    keep_b = turn_duration_line(_uuid(52), _uuid(51), T(1.0), 500)
    _write_jsonl(path, noise_lines()[:4] + [keep_a] + noise_lines()[4:] + [keep_b])

    tailer = ClaudeSessionTailer(path)
    assert [r["uuid"] for r in tailer.read_new()] == [_uuid(51), _uuid(52)]


def test_tailer_handles_empty_or_missing_file(tmp_path):
    missing = ClaudeSessionTailer(tmp_path / "missing.jsonl")
    assert missing.read_new() == []
    empty_path = tmp_path / "empty.jsonl"
    empty_path.write_text("", encoding="utf-8")
    assert ClaudeSessionTailer(empty_path).read_new() == []


# ---------------------------------------------------------------------------
# Tracer end-to-end: event sequence, durations, spawn linking
# ---------------------------------------------------------------------------


def test_event_sequence_from_session(tmp_path):
    fake, tracer = make_tracer(tmp_path, session_lines())
    drain(tracer)
    tracer.finalize()

    assert [e["event_type"] for e in fake.events] == [
        "system_event",  # bootstrap (trace name)
        "system_event",  # user prompt
        "llm_call",      # assistant turn msg_synth_A (text + tool_use lines grouped)
        "tool_call",     # Bash
        "llm_call",      # assistant turn msg_synth_B
        "tool_call",     # Task spawn
        "system_event",  # turn_duration
    ]
    assert all(e["trace_id"] == SESSION_ID for e in fake.events)
    assert all(e["framework"] == "claude_code" for e in fake.events)
    assert fake.ended == [(SESSION_ID, "completed")]

    bootstrap, prompt, llm_a, bash, llm_b, task, turn = fake.events
    assert bootstrap["tags"]["trace_name"] == "Synthetic widget refactor"  # from ai-title
    assert bootstrap["metadata"]["cwd"] == SYNTH_CWD

    assert prompt["agent_id"] == "user"
    assert prompt["input_preview"] == PROMPT_TEXT

    # The two assistant JSONL lines sharing message.id collapse into ONE llm_call.
    assert llm_a["model_name"] == "claude-fable-5"
    assert llm_a["span_id"] == "msg_synth_A"
    assert llm_a["input_tokens"] == 120 and llm_a["output_tokens"] == 45
    assert llm_a["output_preview"] == ASSISTANT_TEXT
    assert llm_a["input_preview"] == PROMPT_TEXT
    assert llm_a["timestamp"] == T(1.0)  # original transcript timestamp preserved

    assert bash["tool_name"] == "Bash"
    assert bash["span_id"] == BASH_TOOL_USE_ID
    assert bash["parent_span_id"] == "msg_synth_A"
    assert bash["causal_parents"] == [llm_a["event_id"]]
    assert "file_a.py" in bash["tool_output"]
    assert bash["tool_success"] is True

    assert task["tool_name"] == "Task"
    assert turn["latency_ms"] == 7000.0
    assert turn["metadata"]["subtype"] == "turn_duration"


def test_tool_duration_pairing(tmp_path):
    fake, tracer = make_tracer(tmp_path, session_lines())
    drain(tracer)

    by_tool = {e["tool_name"]: e for e in fake.events if e["event_type"] == "tool_call"}
    # Bash: tool_use at T+2.000s, tool_result at T+4.250s.
    assert by_tool["Bash"]["latency_ms"] == pytest.approx(2250.0, abs=1.0)
    assert by_tool["Bash"]["timestamp"] == T(2.0)  # event sits at tool start
    # Task: tool_use at T+5.0s, async-launch result at T+6.0s.
    assert by_tool["Task"]["latency_ms"] == pytest.approx(1000.0, abs=1.0)
    # Assistant turn A spans its two lines: T+1.0s -> T+2.0s.
    llm_a = next(e for e in fake.events if e["event_type"] == "llm_call")
    assert llm_a["latency_ms"] == pytest.approx(1000.0, abs=1.0)


def test_replay_is_idempotent_across_reattach(tmp_path):
    """event_ids come from stable transcript ids (message.id, tool_use_id, line
    uuid), so re-attaching to a session re-emits the SAME ids and the server's
    INSERT OR REPLACE dedupes instead of doubling the trace."""
    fake_first, tracer_first = make_tracer(tmp_path, session_lines(), with_child=True)
    drain(tracer_first)
    tracer_first.finalize()

    fake_second = FakeCollector()
    tracer_second = OmniscopeClaudeCodeTracer(
        tmp_path / f"{SESSION_ID}.jsonl", collector=fake_second)
    drain(tracer_second)
    tracer_second.finalize()

    ids_first = [e["event_id"] for e in fake_first.events]
    ids_second = [e["event_id"] for e in fake_second.events]
    assert ids_first == ids_second
    assert len(ids_first) == len(set(ids_first)), "event_ids must be unique within a run"
    # causal links use the same stable ids in both runs
    bash_first = next(e for e in fake_first.events if e["tool_name"] == "Bash")
    bash_second = next(e for e in fake_second.events if e["tool_name"] == "Bash")
    assert bash_first["causal_parents"] == bash_second["causal_parents"]


def test_task_spawn_links_child_agent_transcript(tmp_path):
    fake, tracer = make_tracer(tmp_path, session_lines(), with_child=True)
    drain(tracer)
    tracer.finalize()

    task = next(e for e in fake.events if e["tool_name"] == "Task")
    child_events = [e for e in fake.events if e["agent_id"] == f"agent-{CHILD_AGENT_ID}"]
    assert child_events, "subagent transcript referenced by toolUseResult.agentId was not adopted"
    assert all(e["trace_id"] == SESSION_ID for e in child_events)
    assert [e["event_type"] for e in child_events] == ["system_event", "llm_call"]

    first = child_events[0]
    assert first["agent_name"] == "general-purpose"  # from agent-<id>.meta.json
    assert first["causal_parents"] == [task["event_id"]]  # graph: Task tool -> child agent
    assert first["parent_span_id"] == TASK_TOOL_USE_ID
    assert "sub-task" in first["input_preview"]


# ---------------------------------------------------------------------------
# Mapper unit tests: errors, unfinished tools, truncation
# ---------------------------------------------------------------------------


def test_interleaved_message_lines_resume_as_one_llm_call():
    """Real transcripts interleave tool_result lines between assistant lines
    sharing one message.id. The turn must resume (not restart) so the single
    stable-id llm_call keeps the earlier text, and tools are not re-registered."""
    fake = FakeCollector()
    mapper = _SessionMapper(fake, SESSION_ID)
    mapper.process(assistant_line(_uuid(71), _uuid(70), T(1.0), "msg_synth_I",
                                  [{"type": "text", "text": ASSISTANT_TEXT}]))
    mapper.process(assistant_line(
        _uuid(72), _uuid(71), T(1.2), "msg_synth_I",
        [tool_use_block("toolu_synth_int0001", "Bash",
                        {"command": "true", "description": "first parallel tool"})],
        stop_reason="tool_use"))
    mapper.process(tool_result_line(
        _uuid(73), _uuid(72), T(2.0), "toolu_synth_int0001", "ok",
        {"stdout": "ok", "stderr": "", "interrupted": False, "isImage": False},
        source_uuid=_uuid(72)))
    mapper.process(assistant_line(
        _uuid(74), _uuid(73), T(2.5), "msg_synth_I",
        [tool_use_block("toolu_synth_int0002", "Bash",
                        {"command": "true", "description": "second parallel tool"})],
        stop_reason="tool_use"))
    mapper.process(tool_result_line(
        _uuid(75), _uuid(74), T(3.0), "toolu_synth_int0002", "ok2",
        {"stdout": "ok2", "stderr": "", "interrupted": False, "isImage": False},
        source_uuid=_uuid(74)))
    mapper.finish()

    llm_emits = [e for e in fake.events if e["event_type"] == "llm_call"]
    # Every flush of the turn reuses the SAME stable event_id -> one row upserted
    # server-side -> tokens are counted once per API message.
    assert {e["event_id"] for e in llm_emits} == {"msg_synth_I"}
    # The winning (last) write still contains the first fragment's text.
    assert llm_emits[-1]["output_preview"] == ASSISTANT_TEXT
    # The resumed re-flush keeps its original causal parent (no llm<->tool cycle).
    assert all(e["causal_parents"] == llm_emits[0]["causal_parents"] for e in llm_emits)

    tool_events = [e for e in fake.events if e["event_type"] == "tool_call"]
    assert [t["span_id"] for t in tool_events] == ["toolu_synth_int0001",
                                                   "toolu_synth_int0002"]
    assert all(t["causal_parents"] == ["msg_synth_I"] for t in tool_events)
    # Resuming must not re-register tools as pending (no phantom result_missing).
    assert all("result_missing" not in t["metadata"] for t in tool_events)


def test_failed_tool_result_marks_failure():
    fake = FakeCollector()
    mapper = _SessionMapper(fake, SESSION_ID)
    mapper.process(assistant_line(
        _uuid(12), _uuid(11), T(1.0), "msg_synth_E",
        [tool_use_block("toolu_synth_err0001", "Bash",
                        {"command": "frobnicate", "description": "Run missing command"})],
        stop_reason="tool_use",
    ))
    mapper.process(tool_result_line(
        _uuid(13), _uuid(12), T(2.0), "toolu_synth_err0001",
        "zsh: command not found: frobnicate",
        {"stdout": "", "stderr": "zsh: command not found: frobnicate",
         "interrupted": False, "isImage": False},
        source_uuid=_uuid(12), is_error=True,
    ))
    mapper.finish()

    tool_events = [e for e in fake.events if e["event_type"] == "tool_call"]
    assert len(tool_events) == 1
    failed = tool_events[0]
    assert failed["tool_success"] is False
    assert "command not found" in failed["error_message"]
    assert failed["latency_ms"] == pytest.approx(1000.0, abs=1.0)


def test_unresolved_tool_flushed_on_finish():
    fake = FakeCollector()
    mapper = _SessionMapper(fake, SESSION_ID)
    mapper.process(assistant_line(
        _uuid(14), _uuid(13), T(1.0), "msg_synth_U",
        [tool_use_block("toolu_synth_open0001", "Bash",
                        {"command": "sleep 999", "description": "Still running"})],
        stop_reason="tool_use",
    ))
    mapper.finish()

    tool_events = [e for e in fake.events if e["event_type"] == "tool_call"]
    assert len(tool_events) == 1
    assert tool_events[0]["metadata"] == {"result_missing": True}
    assert tool_events[0]["span_id"] == "toolu_synth_open0001"


def test_truncation_caps_content_fields():
    big = "x" * 5000
    limit = 100 + len(TRUNC_SUFFIX)
    fake = FakeCollector()
    mapper = _SessionMapper(fake, SESSION_ID, truncate=100)
    mapper.process(user_prompt_line(_uuid(21), None, T(0.0), "Process this blob: " + big))
    mapper.process(assistant_line(
        _uuid(22), _uuid(21), T(1.0), "msg_synth_T",
        [{"type": "text", "text": big},
         tool_use_block("toolu_synth_big0001", "Bash",
                        {"command": "cat " + big, "description": "Dump large file"})],
        stop_reason="tool_use",
    ))
    mapper.process(tool_result_line(
        _uuid(23), _uuid(22), T(2.0), "toolu_synth_big0001", big,
        {"stdout": big, "stderr": "", "interrupted": False, "isImage": False},
        source_uuid=_uuid(22),
    ))
    mapper.finish()

    assert fake.events
    # The big payload reached the tool event (truncated, with marker)...
    tool_event = next(e for e in fake.events if e["event_type"] == "tool_call")
    assert tool_event["tool_output"].startswith("xxxx")
    assert tool_event["tool_output"].endswith(TRUNC_SUFFIX)
    assert tool_event["tool_input"]["command"].endswith(TRUNC_SUFFIX)
    # ...and every text field everywhere respects the cap.
    for event in fake.events:
        for field in ("input_preview", "output_preview", "tool_output", "error_message"):
            value = event.get(field) or ""
            assert len(value) <= limit, f"{field} not truncated: {len(value)} chars"
        for value in event["tool_input"].values():
            if isinstance(value, str):
                assert len(value) <= limit
