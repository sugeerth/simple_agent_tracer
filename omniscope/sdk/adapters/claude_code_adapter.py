"""
Claude Code adapter for OMNISCOPE -- live-traces local Claude Code sessions.

How it works:
1. Claude Code appends JSONL transcripts at ~/.claude/projects/<munged-cwd>/<sessionId>.jsonl.
2. discover() lists those files; pick the newest (--latest), one id (--session), or all
   recently-modified ones (--all-active).
3. ClaudeSessionTailer re-reads each file from a saved byte offset, keeping any partial
   trailing line buffered, so each poll parses only new complete lines.
4. Noise line types (file-history-snapshot, mode, queue-operation, ...) are dropped early.
5. _SessionMapper converts the rest to OMNISCOPE events: user prompts -> system_event,
   assistant lines grouped by message.id -> one llm_call, tool_use blocks -> pending tools,
   tool_result lines -> tool_call events with real durations and success/error status.
6. Agent/Task and Workflow spawns are correlated via toolUseResult.agentId / runId; the spawned
   subagents/*.jsonl transcripts are tailed too, as child agent streams in the same trace.
7. Events flow through OmniscopeCollector (batched POST /api/v1/traces); trace_id = session id
   and event_ids are derived from stable transcript ids (message.id, tool_use_id, line uuid),
   so re-attaching/replaying the same session is idempotent (server INSERT OR REPLACE).
   Long text is truncated (default 500 chars) -- observability, not transcript exfiltration.

Usage:
    # one-shot replay of the most recent session
    python3 -m omniscope.sdk.adapters.claude_code_adapter --latest --replay

    # live-follow a specific session
    python3 -m omniscope.sdk.adapters.claude_code_adapter --session <id> --follow

    # follow every session active in the last 5 minutes
    python3 -m omniscope.sdk.adapters.claude_code_adapter --all-active --follow --server http://localhost:8781

    # programmatic
    from omniscope.sdk.adapters.claude_code_adapter import OmniscopeClaudeCodeTracer, discover
    tracer = OmniscopeClaudeCodeTracer(discover()[0]["path"])
    while tracer.poll():
        pass
    tracer.finalize()
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

from ..collector import OmniscopeCollector

FRAMEWORK = "claude_code"
DEFAULT_CLAUDE_DIR = Path.home() / ".claude" / "projects"
DEFAULT_TRUNCATE = 500

# Transcript line types that carry no trace signal (scouted from real sessions).
_SKIP_TYPES = {
    "file-history-snapshot",
    "mode",
    "permission-mode",
    "last-prompt",
    "queue-operation",
    "pr-link",
    "attachment",
}

# Tool names that spawn a subagent transcript we should also tail.
_AGENT_SPAWN_TOOLS = {"Agent", "Task"}


def _clip(text: Any, limit: int) -> str:
    text = "" if text is None else str(text)
    if len(text) <= limit:
        return text
    return text[:limit] + "...[truncated]"


def _clip_obj(value: Any, limit: int) -> Any:
    """Recursively truncate string leaves of a tool_input-style structure."""
    if isinstance(value, str):
        return _clip(value, limit)
    if isinstance(value, dict):
        return {k: _clip_obj(v, limit) for k, v in value.items()}
    if isinstance(value, list):
        return [_clip_obj(v, limit) for v in value]
    return value


def _parse_ts(ts: str) -> float:
    """ISO-8601 (optionally Z-suffixed) -> epoch seconds, 0.0 on failure."""
    try:
        return datetime.fromisoformat(ts.rstrip("Z")).timestamp()
    except (ValueError, AttributeError, TypeError):
        return 0.0


def discover(claude_dir: str | Path = DEFAULT_CLAUDE_DIR) -> list[dict[str, Any]]:
    """List Claude Code session transcripts, newest first.

    Returns dicts with: project_dir, session_id, path, mtime, size.
    """
    root = Path(claude_dir).expanduser()
    sessions: list[dict[str, Any]] = []
    if not root.is_dir():
        return sessions
    for path in root.glob("*/*.jsonl"):
        try:
            stat = path.stat()
        except OSError:
            continue
        sessions.append({
            "project_dir": path.parent.name,
            "session_id": path.stem,
            "path": path,
            "mtime": stat.st_mtime,
            "size": stat.st_size,
        })
    sessions.sort(key=lambda s: s["mtime"], reverse=True)
    return sessions


class ClaudeSessionTailer:
    """Incrementally reads parsed JSONL records from an append-only transcript.

    Tracks a byte offset between read_new() calls; a partial trailing line is
    buffered until its newline arrives. If the file shrinks (rotation/truncation)
    the tailer restarts from offset 0.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._offset = 0
        self._partial = b""

    def read_new(self) -> list[dict[str, Any]]:
        try:
            size = self.path.stat().st_size
        except OSError:
            return []  # file missing (not written yet, or removed)
        if size < self._offset:
            self._offset = 0
            self._partial = b""
        if size == self._offset:
            return []
        with open(self.path, "rb") as f:
            f.seek(self._offset)
            chunk = f.read()
            self._offset = f.tell()
        data = self._partial + chunk
        lines = data.split(b"\n")
        self._partial = lines.pop()  # last piece has no newline yet (often b"")
        records: list[dict[str, Any]] = []
        for raw in lines:
            raw = raw.strip()
            if not raw:
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError:
                continue  # malformed line -- skip, never crash the tail
            if not isinstance(record, dict):
                continue
            if record.get("type") in _SKIP_TYPES or record.get("isMeta") is True:
                continue
            records.append(record)
        return records


class _SessionMapper:
    """Maps parsed transcript records from ONE jsonl stream to OMNISCOPE events.

    The main session and each subagent transcript get their own mapper, all
    emitting into the same trace_id. Child mappers are seeded with the spawning
    tool_call's event_id/span_id so the graph links parent -> child agent.
    """

    def __init__(
        self,
        collector: OmniscopeCollector,
        trace_id: str,
        agent_id: str = "claude-code",
        agent_name: str = "Claude Code",
        prompt_agent_id: str = "user",
        prompt_agent_name: str = "User",
        session_dir: Path | None = None,
        truncate: int = DEFAULT_TRUNCATE,
        parent_event_id: str | None = None,
        parent_span_id: str | None = None,
    ):
        self._collector = collector
        self._trace_id = trace_id
        self._agent_id = agent_id
        self._agent_name = agent_name or agent_id
        self._prompt_agent_id = prompt_agent_id
        self._prompt_agent_name = prompt_agent_name
        self._session_dir = session_dir
        self._truncate = truncate
        self._parent_span_id = parent_span_id
        self._last_event_id = parent_event_id  # causal-chain seed
        self._last_input = ""
        self._turn: dict[str, Any] | None = None  # current assistant message.id group
        self._closed_turn: dict[str, Any] | None = None  # last flushed turn, resumable
        self._pending_tools: dict[str, dict[str, Any]] = {}  # tool_use_id -> info
        # Spawn discoveries for the tracer to adopt: (path, parent_event_id, parent_span_id)
        self.spawned_files: list[tuple[Path, str, str]] = []
        # Workflow run dirs to watch for agent-*.jsonl: (dir, parent_event_id, parent_span_id)
        self.spawned_dirs: list[tuple[Path, str, str]] = []

    # --- emit helper (preserves transcript timestamps + stable event ids) ---

    def _emit(self, timestamp: str | None = None, event_id: str | None = None,
              **kwargs: Any) -> str:
        """event_id, when given, is a stable transcript-derived id so replaying
        the same session overwrites rather than duplicates (server PK upsert)."""
        c = self._collector
        kwargs.setdefault("trace_id", self._trace_id)
        kwargs.setdefault("framework", FRAMEWORK)
        if "causal_parents" not in kwargs and self._last_event_id:
            kwargs["causal_parents"] = [self._last_event_id]
        kwargs.setdefault("parent_span_id", self._parent_span_id)
        if (timestamp or event_id) and len(c._batch_buffer) >= c._batch_size - 1:
            c._flush_batch()  # ensure emit() can't flush before we patch the event
        emitted_id = c.emit(**kwargs)
        if (timestamp or event_id) and c._batch_buffer:
            event = c._batch_buffer[-1]
            if timestamp:
                event["timestamp"] = timestamp
            if event_id:
                event["event_id"] = emitted_id = event_id
        self._last_event_id = emitted_id
        return emitted_id

    # --- record dispatch ---

    def process(self, record: dict[str, Any]) -> None:
        rtype = record.get("type")
        if rtype == "assistant":
            self._on_assistant(record)
        elif rtype == "user":
            self._flush_turn()
            self._on_user(record)
        elif rtype == "system":
            self._flush_turn()
            if record.get("subtype") == "turn_duration":
                self._on_turn_duration(record)
        # ai-title is consumed by the header pre-scan; other types are ignored.

    def finish(self) -> None:
        """Flush the open assistant turn and any tools that never got a result."""
        self._flush_turn()
        for tu_id, pending in self._pending_tools.items():
            self._emit(
                timestamp=pending["ts"],
                event_id=tu_id or None,
                event_type="tool_call",
                agent_id=self._agent_id,
                agent_name=self._agent_name,
                tool_name=pending["name"],
                tool_input=_clip_obj(pending["input"], self._truncate),
                tool_success=True,
                span_id=tu_id,
                parent_span_id=pending["turn_span"],
                causal_parents=[pending["llm_event_id"]],
                metadata={"result_missing": True},
            )
        self._pending_tools.clear()

    # --- assistant turns ---

    def _on_assistant(self, record: dict[str, Any]) -> None:
        message = record.get("message") or {}
        mid = message.get("id") or record.get("uuid") or ""
        if self._turn and self._turn["id"] != mid:
            self._flush_turn()
        ts = record.get("timestamp", "")
        if self._turn is None:
            if self._closed_turn and self._closed_turn["id"] == mid:
                # Same message.id continues after a tool_result interleaved its
                # lines: resume the turn so the re-emitted llm_call (same stable
                # event_id, server-side upsert) keeps the earlier text/usage.
                self._turn = self._closed_turn
            else:
                self._turn = {
                    "id": mid,
                    "model": message.get("model"),
                    "start_ts": ts,
                    "end_ts": ts,
                    "usage": {},
                    "text": [],
                    "thinking": [],
                    "tools": [],
                    "stop_reason": None,
                }
            self._closed_turn = None
        turn = self._turn
        turn["end_ts"] = ts or turn["end_ts"]
        turn["usage"] = message.get("usage") or turn["usage"]
        turn["stop_reason"] = message.get("stop_reason") or turn["stop_reason"]
        for block in message.get("content") or []:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text":
                turn["text"].append(block.get("text", ""))
            elif btype == "thinking":
                turn["thinking"].append(block.get("thinking", ""))
            elif btype == "tool_use":
                turn["tools"].append((block, ts))

    def _flush_turn(self) -> None:
        turn, self._turn = self._turn, None
        if not turn:
            return
        usage = turn["usage"] or {}
        latency = max(0.0, (_parse_ts(turn["end_ts"]) - _parse_ts(turn["start_ts"])) * 1000)
        if "causal_parents" not in turn:  # pin on first flush; a resumed re-flush
            turn["causal_parents"] = (    # must not chain from its own tool_calls
                [self._last_event_id] if self._last_event_id else []
            )
        metadata: dict[str, Any] = {
            "message_id": turn["id"],
            "stop_reason": turn["stop_reason"],
            "cache_read_input_tokens": usage.get("cache_read_input_tokens", 0),
            "cache_creation_input_tokens": usage.get("cache_creation_input_tokens", 0),
        }
        if turn["thinking"]:
            metadata["thinking_preview"] = _clip("".join(turn["thinking"]), self._truncate)
        event_id = self._emit(
            timestamp=turn["start_ts"],
            event_id=turn["id"] or None,
            event_type="llm_call",
            agent_id=self._agent_id,
            agent_name=self._agent_name,
            model_name=turn["model"],
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            latency_ms=latency,
            input_preview=self._last_input,
            output_preview=_clip("".join(turn["text"]), self._truncate),
            span_id=turn["id"],
            causal_parents=turn["causal_parents"],
            metadata=metadata,
        )
        for block, block_ts in turn["tools"]:
            tu_id = block.get("id") or ""
            self._pending_tools[tu_id] = {
                "name": block.get("name", ""),
                "input": block.get("input") or {},
                "ts": block_ts,
                "llm_event_id": event_id,
                "turn_span": turn["id"],
            }
        turn["tools"] = []  # registered; a resumed re-flush must not re-register
        self._closed_turn = turn

    # --- user lines: prompts and tool results ---

    def _on_user(self, record: dict[str, Any]) -> None:
        content = (record.get("message") or {}).get("content")
        if isinstance(content, str):
            self._on_prompt(record, content)
            return
        if isinstance(content, list):
            results = [b for b in content if isinstance(b, dict) and b.get("type") == "tool_result"]
            if results:
                for block in results:
                    self._on_tool_result(record, block)
            else:  # prompt sent as content blocks
                text = " ".join(b.get("text", "") for b in content if isinstance(b, dict))
                if text.strip():
                    self._on_prompt(record, text)

    def _on_prompt(self, record: dict[str, Any], text: str) -> None:
        preview = _clip(text.strip(), self._truncate)
        self._last_input = preview
        self._emit(
            timestamp=record.get("timestamp"),
            event_id=record.get("uuid"),
            event_type="system_event",
            agent_id=self._prompt_agent_id,
            agent_name=self._prompt_agent_name,
            input_preview=preview,
            span_id=record.get("uuid"),
        )

    def _on_tool_result(self, record: dict[str, Any], block: dict[str, Any]) -> None:
        tu_id = block.get("tool_use_id", "")
        pending = self._pending_tools.pop(tu_id, None)
        if pending is None:
            return  # result for a tool_use we never saw (offset started mid-file)
        tool_result = record.get("toolUseResult")
        is_error = bool(block.get("is_error"))
        output = self._tool_output_text(block, tool_result)
        latency = max(0.0, (_parse_ts(record.get("timestamp", "")) - _parse_ts(pending["ts"])) * 1000)
        event_id = self._emit(
            timestamp=pending["ts"],  # tool started here; latency covers the run
            event_id=tu_id or None,
            event_type="tool_call",
            agent_id=self._agent_id,
            agent_name=self._agent_name,
            latency_ms=latency,
            tool_name=pending["name"],
            tool_input=_clip_obj(pending["input"], self._truncate),
            tool_output=_clip(output, self._truncate),
            tool_success=not is_error,
            error_message=_clip(output, self._truncate) if is_error else None,
            input_preview=_clip(json.dumps(pending["input"], default=str), self._truncate),
            output_preview=_clip(output, self._truncate),
            span_id=tu_id,
            parent_span_id=pending["turn_span"],
            causal_parents=[pending["llm_event_id"]],
        )
        self._correlate_spawn(pending["name"], tool_result, event_id, tu_id)

    def _correlate_spawn(
        self, tool_name: str, tool_result: Any, event_id: str, span_id: str
    ) -> None:
        if not isinstance(tool_result, dict) or self._session_dir is None:
            return
        if tool_name in _AGENT_SPAWN_TOOLS and tool_result.get("agentId"):
            path = self._session_dir / "subagents" / f"agent-{tool_result['agentId']}.jsonl"
            self.spawned_files.append((path, event_id, span_id))
        elif tool_name == "Workflow" and tool_result.get("runId"):
            run_dir = Path(
                tool_result.get("transcriptDir")
                or self._session_dir / "subagents" / "workflows" / tool_result["runId"]
            )
            self.spawned_dirs.append((run_dir, event_id, span_id))

    @staticmethod
    def _tool_output_text(block: dict[str, Any], tool_result: Any) -> str:
        content = block.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                p.get("text", "") for p in content if isinstance(p, dict)
            )
        if content:
            return str(content)
        if isinstance(tool_result, dict):
            return str(tool_result.get("stdout") or tool_result.get("stderr") or "")
        return str(tool_result or "")

    # --- turn duration ---

    def _on_turn_duration(self, record: dict[str, Any]) -> None:
        duration = record.get("durationMs", 0)
        self._emit(
            timestamp=record.get("timestamp"),
            event_id=record.get("uuid"),
            event_type="system_event",
            agent_id=self._agent_id,
            agent_name=self._agent_name,
            latency_ms=float(duration),
            output_preview=f"Turn completed: {record.get('messageCount', 0)} messages in {duration} ms",
            metadata={"subtype": "turn_duration"},
        )


class OmniscopeClaudeCodeTracer:
    """Tails one Claude Code session (plus its spawned subagents) into one trace.

    trace_id == the Claude Code session id, so re-attaching maps to the same trace.
    """

    def __init__(
        self,
        session_path: str | Path,
        server_url: str = "http://localhost:8781",
        collector: OmniscopeCollector | None = None,
        truncate: int = DEFAULT_TRUNCATE,
    ):
        self._path = Path(session_path)
        self._collector = collector or OmniscopeCollector(server_url)
        self._truncate = truncate
        self.trace_id = self._path.stem
        self._session_dir = self._path.parent / self._path.stem
        self._main = _SessionMapper(
            self._collector,
            self.trace_id,
            session_dir=self._session_dir,
            truncate=truncate,
        )
        self._streams: list[tuple[ClaudeSessionTailer, _SessionMapper]] = [
            (ClaudeSessionTailer(self._path), self._main)
        ]
        self._known_files = {self._path}
        self._watch_dirs: list[tuple[Path, str, str]] = []
        self._started = False

    def poll(self) -> int:
        """Process all new transcript lines once. Returns number of items handled."""
        if not self._started:
            self._bootstrap()
            self._started = True
        n = 0
        for tailer, mapper in list(self._streams):
            for record in tailer.read_new():
                mapper.process(record)
                n += 1
            n += self._adopt_spawns(mapper)
        n += self._scan_watch_dirs()
        self._collector._flush_batch()
        return n

    def finalize(self, status: str = "completed") -> None:
        for _, mapper in self._streams:
            mapper.finish()
        self._collector.end_trace(self.trace_id, status)

    # --- internals ---

    def _bootstrap(self) -> None:
        """Emit the first event, which fixes the trace name (server only sets it on insert)."""
        title, first_ts, cwd, branch, prompt = self._scan_header()
        name = title or (f"claude-code: {prompt[:60]}" if prompt else f"claude-code: {self.trace_id[:8]}")
        self._main._emit(
            timestamp=first_ts or None,
            event_id=f"{self.trace_id}-bootstrap",
            event_type="system_event",
            agent_id="claude-code",
            agent_name="Claude Code",
            tags={"trace_name": name},
            output_preview=f"Attached to Claude Code session {self.trace_id}",
            metadata={"cwd": cwd, "git_branch": branch, "session_path": str(self._path)},
        )

    def _scan_header(self) -> tuple[str, str, str, str, str]:
        """Cheap pre-scan: latest ai-title, first timestamp/cwd/gitBranch, first prompt."""
        title = first_ts = cwd = branch = prompt = ""
        try:
            with open(self._path, encoding="utf-8", errors="replace") as f:
                for line in f:
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(record, dict):
                        continue
                    if record.get("type") == "ai-title" and record.get("aiTitle"):
                        title = record["aiTitle"]  # keep the latest title
                    if not first_ts and record.get("timestamp"):
                        first_ts = record["timestamp"]
                    if not cwd and record.get("cwd"):
                        cwd = record["cwd"]
                        branch = record.get("gitBranch", "")
                    if (
                        not prompt
                        and record.get("type") == "user"
                        and not record.get("isMeta")
                        and isinstance((record.get("message") or {}).get("content"), str)
                    ):
                        prompt = record["message"]["content"].strip()
        except OSError:
            pass
        return title, first_ts, cwd, branch, prompt

    def _adopt_spawns(self, mapper: _SessionMapper) -> int:
        n = 0
        for path, parent_event_id, parent_span_id in mapper.spawned_files:
            n += self._add_child(path, parent_event_id, parent_span_id)
        mapper.spawned_files.clear()
        for run_dir, parent_event_id, parent_span_id in mapper.spawned_dirs:
            self._watch_dirs.append((run_dir, parent_event_id, parent_span_id))
            n += 1
        mapper.spawned_dirs.clear()
        return n

    def _scan_watch_dirs(self) -> int:
        n = 0
        for run_dir, parent_event_id, parent_span_id in self._watch_dirs:
            if not run_dir.is_dir():
                continue
            for path in sorted(run_dir.glob("agent-*.jsonl")):
                n += self._add_child(path, parent_event_id, parent_span_id)
        return n

    def _add_child(self, path: Path, parent_event_id: str, parent_span_id: str) -> int:
        if path in self._known_files:
            return 0
        self._known_files.add(path)
        agent_id = path.stem  # "agent-<17hex>"
        agent_name = agent_id
        meta_path = path.with_name(path.stem + ".meta.json")
        try:
            agent_name = json.loads(meta_path.read_text()).get("agentType", agent_id)
        except (OSError, json.JSONDecodeError, AttributeError):
            pass
        mapper = _SessionMapper(
            self._collector,
            self.trace_id,
            agent_id=agent_id,
            agent_name=agent_name,
            prompt_agent_id=agent_id,
            prompt_agent_name=agent_name,
            session_dir=self._session_dir,
            truncate=self._truncate,
            parent_event_id=parent_event_id,
            parent_span_id=parent_span_id,
        )
        self._streams.append((ClaudeSessionTailer(path), mapper))
        return 1


# --- CLI ---


def _select_sessions(args: argparse.Namespace) -> list[dict[str, Any]]:
    sessions = discover(args.claude_dir)
    if args.session:
        direct = Path(args.session).expanduser()
        if direct.is_file():
            return [{"session_id": direct.stem, "path": direct}]
        matches = [s for s in sessions if s["session_id"].startswith(args.session)]
        return matches[:1]
    if args.all_active:
        cutoff = time.time() - args.active_minutes * 60
        return [s for s in sessions if s["mtime"] >= cutoff]
    return sessions[:1]  # --latest (default)


def _verify_token(server_url: str, token: str) -> None:
    """Optional: ingest is unauthenticated, but verify a dashboard token if given."""
    try:
        resp = httpx.get(
            f"{server_url.rstrip('/')}/api/v1/auth/me",
            headers={"Authorization": f"Bearer {token}"},
            timeout=5.0,
        )
        info = resp.json()
        if info.get("error"):
            print(f"[omniscope] auth warning: {info['error']} (continuing; ingest needs no auth)", file=sys.stderr)
        else:
            print(f"[omniscope] authenticated as {info.get('username')}", file=sys.stderr)
    except Exception:
        print("[omniscope] auth check failed (server unreachable?); continuing", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python3 -m omniscope.sdk.adapters.claude_code_adapter",
        description="Trace local Claude Code sessions into OMNISCOPE.",
    )
    parser.add_argument("--server", default="http://localhost:8781", help="OMNISCOPE server URL")
    select = parser.add_mutually_exclusive_group()
    select.add_argument("--latest", action="store_true", help="trace the most recent session (default)")
    select.add_argument("--session", default="", help="session id (or prefix, or path to a .jsonl)")
    select.add_argument("--all-active", action="store_true", help="trace all recently active sessions")
    parser.add_argument("--follow", action="store_true", help="keep tailing for new events (Ctrl-C to stop)")
    parser.add_argument("--replay", action="store_true", help="emit existing history then exit (default)")
    parser.add_argument("--claude-dir", default=str(DEFAULT_CLAUDE_DIR), help="Claude projects dir")
    parser.add_argument("--active-minutes", type=float, default=5.0, help="--all-active mtime window")
    parser.add_argument("--poll-seconds", type=float, default=2.0, help="--follow poll interval")
    parser.add_argument("--truncate", type=int, default=DEFAULT_TRUNCATE, help="max chars per content field")
    parser.add_argument("--token", default="", help="optional dashboard bearer token (ingest is unauthenticated)")
    args = parser.parse_args(argv)

    if args.token:
        _verify_token(args.server, args.token)

    selected = _select_sessions(args)
    if not selected:
        print(f"[omniscope] no Claude Code sessions found under {args.claude_dir}", file=sys.stderr)
        return 1

    tracers = []
    for session in selected:
        tracer = OmniscopeClaudeCodeTracer(session["path"], server_url=args.server, truncate=args.truncate)
        print(f"[omniscope] tracing session {tracer.trace_id} -> {args.server}", file=sys.stderr)
        tracers.append(tracer)

    # Replay existing history first in both modes.
    for tracer in tracers:
        while tracer.poll():
            pass

    if args.follow:
        try:
            while True:
                time.sleep(args.poll_seconds)
                for tracer in tracers:
                    tracer.poll()
        except KeyboardInterrupt:
            print("\n[omniscope] stopping", file=sys.stderr)

    for tracer in tracers:
        tracer.finalize()
    print(f"[omniscope] done: {len(tracers)} trace(s) sent", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
