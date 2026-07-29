"""
OMNISCOPE Example: Trace a live Claude Code session.

Programmatic equivalent of the CLI:
    python3 -m omniscope.sdk.adapters.claude_code_adapter --latest --follow

Start the server first:
    python3 -m uvicorn omniscope.server.app:app --port 8781

Then run this:
    python3 examples/claude_code_live.py

Open the dashboard (cd dashboard && npm run dev -> http://localhost:5173)
to watch the session stream in.
"""
import time

from omniscope.sdk.adapters.claude_code_adapter import OmniscopeClaudeCodeTracer, discover

# 1. Discover sessions. Claude Code keeps one append-only JSONL transcript per
#    session under ~/.claude/projects/; discover() lists them newest first.
sessions = discover()
if not sessions:
    raise SystemExit("No Claude Code sessions found under ~/.claude/projects/")
latest = sessions[0]
print(f"Tracing session {latest['session_id']} ({latest['project_dir']})")

# 2. One tracer per session; trace_id == the Claude Code session id.
#    Assistant turns become llm_call events (model + token usage), tool_use/
#    tool_result pairs become tool_call events with real durations, and
#    Agent/Workflow spawns pull the subagent transcripts into the same trace.
tracer = OmniscopeClaudeCodeTracer(latest["path"], server_url="http://localhost:8781")

# 3. Replay: each poll() processes all new transcript lines once, so looping
#    until it returns 0 drains the existing history.
while tracer.poll():
    pass
print("History replayed. Following live (Ctrl-C to stop)...")

# 4. Follow: keep polling so new events stream in while Claude Code works.
try:
    while True:
        time.sleep(2.0)
        tracer.poll()
except KeyboardInterrupt:
    pass

# 5. Flush any open turn or unresolved tools and mark the trace completed.
tracer.finalize()
print(f"Done: http://localhost:8781/api/v1/traces/{tracer.trace_id}")
