#!/usr/bin/env python3
"""
OMNISCOPE Interactive Agent Demo
================================
A multi-agent system traced live by OMNISCOPE.

Picks an LLM backend automatically:
    1. Ollama (if reachable on :11434)         - local, free
    2. Anthropic SDK (if ANTHROPIC_API_KEY set) - cloud, paid
    3. Built-in stub                            - always works, deterministic

Usage:
    python3 demo_interactive.py                # interactive REPL
    python3 demo_interactive.py "your query"   # one-shot
    python3 demo_interactive.py --backend stub # force a specific backend
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import socket
import ssl
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import webbrowser
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

from omniscope.sdk.adapters.generic_adapter import OmniscopeTracer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OMNISCOPE_HOST = "127.0.0.1"
OMNISCOPE_PORT = 8781
OMNISCOPE_SERVER = f"http://{OMNISCOPE_HOST}:{OMNISCOPE_PORT}"
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = os.environ.get("OMNISCOPE_OLLAMA_MODEL", "llama3.2")
ANTHROPIC_MODEL = os.environ.get("OMNISCOPE_ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")

# ANSI colors (auto-disabled when not a TTY)
_TTY = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _c(code: str) -> str:
    return code if _TTY else ""


C_DIM = _c("\033[2m")
C_BOLD = _c("\033[1m")
C_CYAN = _c("\033[36m")
C_GREEN = _c("\033[32m")
C_YELLOW = _c("\033[33m")
C_RED = _c("\033[31m")
C_MAGENTA = _c("\033[35m")
C_BLUE = _c("\033[34m")
C_RESET = _c("\033[0m")


def banner(text: str, color: str = C_CYAN) -> None:
    print(f"{color}{C_BOLD}{text}{C_RESET}")


def info(text: str) -> None:
    print(f"{C_DIM}  {text}{C_RESET}")


def ok(text: str) -> None:
    print(f"  {C_GREEN}[OK]{C_RESET} {text}")


def warn(text: str) -> None:
    print(f"  {C_YELLOW}[!!]{C_RESET} {text}")


def err(text: str) -> None:
    print(f"  {C_RED}[ERROR]{C_RESET} {text}")


# SSL context for Wikipedia / Open-Meteo / etc on macOS
try:
    _ssl_ctx = ssl.create_default_context()
    urllib.request.urlopen("https://en.wikipedia.org", timeout=3, context=_ssl_ctx)
except Exception:
    _ssl_ctx = ssl._create_unverified_context()


# ---------------------------------------------------------------------------
# LLM backends
# ---------------------------------------------------------------------------
@dataclass
class LLMResult:
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    model: str
    cost_usd: float = 0.0


class LLMBackend:
    name = "base"
    model = ""

    def generate(self, prompt: str, system: str = "", temperature: float = 0.3) -> LLMResult:
        raise NotImplementedError


class OllamaBackend(LLMBackend):
    name = "ollama"

    def __init__(self, url: str = OLLAMA_URL, model: str = OLLAMA_MODEL):
        self.url = url
        self.model = model

    def generate(self, prompt: str, system: str = "", temperature: float = 0.3) -> LLMResult:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {"temperature": temperature},
        }
        req = urllib.request.Request(
            f"{self.url}/api/generate",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read())
        latency = (time.time() - t0) * 1000
        text = result.get("response", "").strip()
        return LLMResult(
            text=text,
            input_tokens=result.get("prompt_eval_count", len(prompt.split())),
            output_tokens=result.get("eval_count", len(text.split())),
            latency_ms=latency,
            model=self.model,
        )


_ANTHROPIC_PRICING = {
    "claude-opus-4-7": (15.0, 75.0),
    "claude-opus-4-6": (15.0, 75.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5": (0.80, 4.0),
}


class AnthropicBackend(LLMBackend):
    name = "anthropic"

    def __init__(self, model: str = ANTHROPIC_MODEL):
        from anthropic import Anthropic  # type: ignore
        self.client = Anthropic()
        self.model = model

    def generate(self, prompt: str, system: str = "", temperature: float = 0.3) -> LLMResult:
        t0 = time.time()
        resp = self.client.messages.create(
            model=self.model,
            system=system or "You are a helpful assistant.",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=temperature,
        )
        latency = (time.time() - t0) * 1000
        text = "".join(b.text for b in resp.content if hasattr(b, "text"))
        in_tok = resp.usage.input_tokens
        out_tok = resp.usage.output_tokens
        in_rate, out_rate = next(
            (rates for key, rates in _ANTHROPIC_PRICING.items() if key in self.model),
            (3.0, 15.0),
        )
        cost = (in_tok * in_rate + out_tok * out_rate) / 1_000_000
        return LLMResult(text.strip(), in_tok, out_tok, latency, self.model, round(cost, 6))


class StubBackend(LLMBackend):
    """Deterministic, dependency-free fake LLM. Always works.

    Returns plausible router plans + summaries by inspecting the prompt for keywords.
    Lets the demo run end-to-end so OMNISCOPE traces still flow.
    """
    name = "stub"
    model = "stub-llm-v1"

    def generate(self, prompt: str, system: str = "", temperature: float = 0.3) -> LLMResult:
        t0 = time.time()
        text = self._respond(prompt, system)
        # Simulate a tiny bit of latency so traces look realistic
        time.sleep(0.05)
        latency = (time.time() - t0) * 1000
        return LLMResult(
            text=text,
            input_tokens=max(1, len(prompt) // 4),
            output_tokens=max(1, len(text) // 4),
            latency_ms=latency,
            model=self.model,
        )

    @staticmethod
    def _respond(prompt: str, system: str) -> str:
        is_router = "JSON object" in system or "router" in system.lower()
        # Extract the user query from the prompt template the agent uses
        m = re.search(r"User query:\s*(.+?)(?:\n|$)", prompt)
        query = m.group(1).strip() if m else prompt.strip().split("\n")[0]

        if is_router:
            steps = StubBackend._plan(query)
            plan = {"plan": f"Answer: {query[:60]}", "steps": steps}
            return json.dumps(plan)

        if "Briefly summarize" in prompt:
            return "Got the result above; the relevant details are captured for the final answer."

        # Synthesis prompt
        m2 = re.search(r"Research findings:\s*(.+)", prompt, re.DOTALL)
        findings = m2.group(1).strip() if m2 else ""
        if findings:
            snippet = findings[:600].replace("\n", " ")
            return f"Based on the gathered context, here is what I found: {snippet}"
        return "I don't have enough context to answer in detail, but the trace above shows the reasoning path."

    @staticmethod
    def _plan(query: str) -> list[dict]:
        q = query.lower()
        steps: list[dict] = []
        # Calculator detection
        if any(s in q for s in ["calculate", "compute", "+", "-", "*", "/", "sqrt", "^", "**"]) and re.search(r"\d", q):
            expr = re.sub(r"[^0-9+\-*/().\s a-z]", " ", q)
            expr = expr.replace("calculate", "").replace("compute", "").replace("what is", "").strip()
            steps.append({"agent": "calculator", "action": "calculator", "input": expr or query})
        # Time
        if any(s in q for s in ["time", "date", "today", "now"]):
            steps.append({"agent": "researcher", "action": "datetime", "input": ""})
        # Weather
        if "weather" in q or "temperature" in q:
            loc = re.sub(r".*(in|at|for)\s+", "", q).strip(" ?.")
            steps.append({"agent": "researcher", "action": "weather", "input": loc or "San Francisco"})
        # Otherwise, default to wiki search
        if not steps:
            steps.append({"agent": "researcher", "action": "wikipedia_search", "input": query})
        steps.append({"agent": "responder", "action": "synthesize", "input": "answer the query"})
        return steps


def pick_backend(force: str | None = None) -> LLMBackend:
    """Pick the first viable backend, in priority order."""
    if force == "stub":
        return StubBackend()
    if force in (None, "ollama"):
        try:
            urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=1)
            return OllamaBackend()
        except Exception:
            if force == "ollama":
                raise SystemExit("Ollama not reachable on :11434. Start with `ollama serve`.")
    if force in (None, "anthropic"):
        if os.environ.get("ANTHROPIC_API_KEY"):
            try:
                return AnthropicBackend()
            except Exception as ex:
                if force == "anthropic":
                    raise SystemExit(f"Anthropic backend failed: {ex}")
    return StubBackend()


# ---------------------------------------------------------------------------
# Tools (real APIs, no keys needed)
# ---------------------------------------------------------------------------
def tool_wikipedia_search(query: str) -> str:
    params = urllib.parse.urlencode({
        "action": "query", "list": "search",
        "srsearch": query, "srlimit": 3, "format": "json",
    })
    url = f"https://en.wikipedia.org/w/api.php?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": "OmniscopeDemo/1.0"})
    with urllib.request.urlopen(req, timeout=10, context=_ssl_ctx) as resp:
        data = json.loads(resp.read())
    results = data.get("query", {}).get("search", [])
    if not results:
        return "No Wikipedia results found."
    return "\n".join(
        f"- {r['title']}: {re.sub('<[^>]+>', '', r.get('snippet', ''))[:140]}"
        for r in results
    )


def tool_wikipedia_summary(title: str) -> str:
    safe = urllib.parse.quote(title)
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{safe}"
    req = urllib.request.Request(url, headers={"User-Agent": "OmniscopeDemo/1.0"})
    with urllib.request.urlopen(req, timeout=10, context=_ssl_ctx) as resp:
        data = json.loads(resp.read())
    return data.get("extract", "No summary available.")[:800]


def tool_calculator(expression: str) -> str:
    expr = expression.strip()
    if not expr:
        return "Error: empty expression"
    allowed = re.compile(r"^[\d\s+\-*/().,a-z_]+$", re.IGNORECASE)
    if not allowed.match(expr):
        return f"Error: unsafe expression '{expr}'"
    namespace = {
        "sin": math.sin, "cos": math.cos, "tan": math.tan,
        "log": math.log, "sqrt": math.sqrt, "pi": math.pi,
        "e": math.e, "abs": abs, "pow": pow,
    }
    try:
        return str(eval(expr, {"__builtins__": {}}, namespace))
    except Exception as ex:
        return f"Error: {ex}"


def tool_datetime_info(_: str = "") -> str:
    now = datetime.now()
    return (
        f"Current date: {now.strftime('%Y-%m-%d')} | "
        f"Time: {now.strftime('%H:%M:%S')} | "
        f"Day: {now.strftime('%A')} | "
        f"UTC offset: {now.astimezone().strftime('%z')}"
    )


def tool_weather(location: str) -> str:
    """Open-Meteo: free, no key. Geocodes the location, then fetches current weather."""
    location = (location or "San Francisco").strip()
    geo_url = (
        "https://geocoding-api.open-meteo.com/v1/search?"
        + urllib.parse.urlencode({"name": location, "count": 1, "format": "json"})
    )
    with urllib.request.urlopen(geo_url, timeout=10, context=_ssl_ctx) as resp:
        geo = json.loads(resp.read())
    results = geo.get("results") or []
    if not results:
        return f"No location found for '{location}'."
    g = results[0]
    lat, lon = g["latitude"], g["longitude"]
    name = f"{g.get('name', location)}, {g.get('country_code', '')}".rstrip(", ")
    wx_url = (
        "https://api.open-meteo.com/v1/forecast?"
        + urllib.parse.urlencode({
            "latitude": lat, "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code",
            "temperature_unit": "celsius",
        })
    )
    with urllib.request.urlopen(wx_url, timeout=10, context=_ssl_ctx) as resp:
        wx = json.loads(resp.read()).get("current", {})
    return (
        f"{name}: {wx.get('temperature_2m', '?')}°C, "
        f"humidity {wx.get('relative_humidity_2m', '?')}%, "
        f"wind {wx.get('wind_speed_10m', '?')} km/h"
    )


def tool_http_fetch(url: str) -> str:
    """Fetch the first ~1KB of any URL (text only)."""
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    req = urllib.request.Request(url, headers={"User-Agent": "OmniscopeDemo/1.0"})
    with urllib.request.urlopen(req, timeout=10, context=_ssl_ctx) as resp:
        ctype = resp.headers.get("Content-Type", "")
        body = resp.read(2048).decode("utf-8", errors="replace")
    body = re.sub(r"<script.*?</script>", " ", body, flags=re.DOTALL | re.I)
    body = re.sub(r"<[^>]+>", " ", body)
    body = re.sub(r"\s+", " ", body).strip()
    return f"[{ctype}] {body[:1000]}"


TOOLS: dict[str, dict] = {
    "wikipedia_search": {
        "fn": tool_wikipedia_search,
        "desc": "Search Wikipedia. Input: search query.",
    },
    "wikipedia_summary": {
        "fn": tool_wikipedia_summary,
        "desc": "Get a Wikipedia article summary. Input: exact article title.",
    },
    "calculator": {
        "fn": tool_calculator,
        "desc": "Evaluate a math expression (sqrt, sin, cos, log, pi, e). Input: expression.",
    },
    "datetime": {
        "fn": tool_datetime_info,
        "desc": "Get current local date and time. No input needed.",
    },
    "weather": {
        "fn": tool_weather,
        "desc": "Get current weather for a city. Input: city name.",
    },
    "http_fetch": {
        "fn": tool_http_fetch,
        "desc": "Fetch a URL and return cleaned text. Input: URL.",
    },
}

TOOL_LIST = "\n".join(f"  - {name}: {info['desc']}" for name, info in TOOLS.items())


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------
def _port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        try:
            return s.connect_ex((host, port)) == 0
        except Exception:
            return False


def ensure_server() -> subprocess.Popen | None:
    """Make sure the OMNISCOPE server is up. Start it in the background if not."""
    if _port_open(OMNISCOPE_HOST, OMNISCOPE_PORT):
        ok(f"OMNISCOPE server already running at {OMNISCOPE_SERVER}")
        return None

    info(f"Starting OMNISCOPE server on :{OMNISCOPE_PORT}...")
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn",
            "omniscope.server.app:app",
            "--host", OMNISCOPE_HOST,
            "--port", str(OMNISCOPE_PORT),
            "--log-level", "warning",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    # Wait up to 8s for it to come up
    for _ in range(40):
        if _port_open(OMNISCOPE_HOST, OMNISCOPE_PORT):
            ok(f"OMNISCOPE server started at {OMNISCOPE_SERVER}")
            return proc
        time.sleep(0.2)
    err("OMNISCOPE server did not come up in time. Continuing anyway.")
    return proc


# ---------------------------------------------------------------------------
# Agent pipeline
# ---------------------------------------------------------------------------
ROUTER_SYSTEM = f"""You are a task router for a multi-agent system. Given a user query, decide which agents and tools are needed.

Available tools:
{TOOL_LIST}

Respond with a JSON object:
{{
  "plan": "brief description of your plan",
  "steps": [
    {{"agent": "researcher|calculator|responder", "action": "tool_name or think", "input": "input for the tool or thought"}}
  ]
}}

Keep it to 2-5 steps. Always end with a "responder" agent step with action "synthesize".
Respond ONLY with valid JSON, no other text."""

RESPONDER_SYSTEM = (
    "You are a helpful assistant. Given research findings and context, provide a clear, "
    "informative answer to the user's question. Be concise but thorough. Cite sources "
    "when using Wikipedia information."
)

AGENT_NAMES = {
    "researcher": "Research Agent",
    "calculator": "Calculator Agent",
    "responder": "Response Agent",
}


@dataclass
class QueryStats:
    llm_calls: int = 0
    tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    duration_ms: float = 0.0
    trace_id: str = ""
    answer: str = ""
    transcript: list[str] = field(default_factory=list)


def _parse_plan(response: str, fallback_query: str) -> tuple[str, list[dict]]:
    try:
        m = re.search(r"\{.*\}", response, re.DOTALL)
        if not m:
            raise ValueError("No JSON")
        plan = json.loads(m.group())
        return plan.get("plan", "Execute query"), plan.get("steps", [])
    except (json.JSONDecodeError, ValueError):
        return (
            "Search Wikipedia and synthesize answer",
            [
                {"agent": "researcher", "action": "wikipedia_search", "input": fallback_query},
                {"agent": "responder", "action": "synthesize", "input": "answer the query"},
            ],
        )


def run_agent_pipeline(
    user_query: str,
    tracer: OmniscopeTracer,
    backend: LLMBackend,
) -> QueryStats:
    stats = QueryStats()
    t_start = time.time()
    trace_name = f"Query: {user_query[:50]}{'...' if len(user_query) > 50 else ''}"

    with tracer.trace(trace_name) as t:
        stats.trace_id = t.trace_id

        # ── Router ──
        with t.agent("router", "Task Router") as router:
            plan_prompt = f"User query: {user_query}\n\nCreate an execution plan."
            print(f"  {C_BLUE}[Router]{C_RESET} planning…")
            r = backend.generate(plan_prompt, system=ROUTER_SYSTEM)
            stats.llm_calls += 1
            stats.input_tokens += r.input_tokens
            stats.output_tokens += r.output_tokens
            stats.cost_usd += r.cost_usd
            router.llm_call(
                model=r.model, input_text=plan_prompt[:200], output_text=r.text[:300],
                input_tokens=r.input_tokens, output_tokens=r.output_tokens,
                latency_ms=int(r.latency_ms), confidence=0.9,
            )
            plan_desc, steps = _parse_plan(r.text, user_query)
            router.decision(f"Plan: {plan_desc} ({len(steps)} steps)", confidence=0.9)
            print(f"  {C_BLUE}[Router]{C_RESET} {plan_desc} {C_DIM}({len(steps)} steps){C_RESET}")

        gathered: list[str] = []
        for step_num, step in enumerate(steps, 1):
            agent_type = step.get("agent", "researcher")
            action = step.get("action", "think")
            step_input = step.get("input", "")
            agent_name = AGENT_NAMES.get(agent_type, f"Agent ({agent_type})")
            short_in = step_input if len(step_input) <= 60 else step_input[:60] + "…"
            print(f"  {C_MAGENTA}[{agent_name}]{C_RESET} {action} {C_DIM}{short_in}{C_RESET}")

            with t.agent(f"{agent_type}_{step_num}", agent_name) as agent:
                if action == "synthesize":
                    context_text = "\n\n".join(gathered) if gathered else "No additional context gathered."
                    synth_prompt = (
                        f"User question: {user_query}\n\n"
                        f"Research findings:\n{context_text}\n\n"
                        f"Provide a clear, helpful answer."
                    )
                    r = backend.generate(synth_prompt, system=RESPONDER_SYSTEM)
                    stats.llm_calls += 1
                    stats.input_tokens += r.input_tokens
                    stats.output_tokens += r.output_tokens
                    stats.cost_usd += r.cost_usd
                    stats.answer = r.text
                    agent.llm_call(
                        model=r.model, input_text=synth_prompt[:300], output_text=r.text[:500],
                        input_tokens=r.input_tokens, output_tokens=r.output_tokens,
                        latency_ms=int(r.latency_ms), confidence=0.92,
                    )
                    agent.decision("Final answer synthesized from research", confidence=0.92)

                elif action in TOOLS:
                    tool_fn: Callable = TOOLS[action]["fn"]
                    t0 = time.time()
                    try:
                        tool_result = tool_fn(step_input)
                        latency_ms = int((time.time() - t0) * 1000)
                        agent.tool_call(
                            tool_name=action,
                            tool_input={"query": step_input} if step_input else {},
                            tool_output=tool_result[:400],
                            latency_ms=latency_ms,
                        )
                        stats.tool_calls += 1
                        gathered.append(f"[{action}] {step_input}:\n{tool_result[:500]}")
                        # Auto-deepen wiki search with the top result's summary
                        if action == "wikipedia_search" and "No Wikipedia" not in tool_result:
                            first = tool_result.split("\n")[0].lstrip("- ").split(":")[0].strip()
                            if first:
                                t0 = time.time()
                                try:
                                    summary = tool_wikipedia_summary(first)
                                    agent.tool_call(
                                        tool_name="wikipedia_summary",
                                        tool_input={"title": first},
                                        tool_output=summary[:400],
                                        latency_ms=int((time.time() - t0) * 1000),
                                    )
                                    stats.tool_calls += 1
                                    gathered.append(f"[wikipedia_summary] {first}:\n{summary[:500]}")
                                except Exception:
                                    pass
                    except Exception as ex:
                        latency_ms = int((time.time() - t0) * 1000)
                        agent.tool_call(
                            tool_name=action,
                            tool_input={"query": step_input},
                            tool_output=f"Error: {ex}",
                            latency_ms=latency_ms,
                            success=False,
                            error=str(ex),
                        )
                        gathered.append(f"[{action}] Error: {ex}")
                        tool_result = f"Error: {ex}"

                    # Brief reasoning over the tool result
                    reason = backend.generate(
                        f"You used {action} with input '{step_input}'. Result:\n{tool_result[:300]}\n\nBriefly summarize.",
                        temperature=0.2,
                    )
                    stats.llm_calls += 1
                    stats.input_tokens += reason.input_tokens
                    stats.output_tokens += reason.output_tokens
                    stats.cost_usd += reason.cost_usd
                    agent.llm_call(
                        model=reason.model, input_text=action, output_text=reason.text[:300],
                        input_tokens=reason.input_tokens, output_tokens=reason.output_tokens,
                        latency_ms=int(reason.latency_ms), confidence=0.85,
                    )

                else:  # think / unknown action
                    think_prompt = (
                        f"Context so far:\n{chr(10).join(gathered[-3:])}\n\nThink about: {step_input}"
                    )
                    r = backend.generate(think_prompt)
                    stats.llm_calls += 1
                    stats.input_tokens += r.input_tokens
                    stats.output_tokens += r.output_tokens
                    stats.cost_usd += r.cost_usd
                    agent.llm_call(
                        model=r.model, input_text=think_prompt[:200], output_text=r.text[:300],
                        input_tokens=r.input_tokens, output_tokens=r.output_tokens,
                        latency_ms=int(r.latency_ms), confidence=0.8,
                    )
                    gathered.append(f"[thinking] {r.text[:300]}")

    stats.duration_ms = (time.time() - t_start) * 1000
    return stats


def print_answer(stats: QueryStats) -> None:
    print()
    banner("  ┌── Answer ─────────────────────────────────────────────", C_GREEN)
    for line in (stats.answer or "(no answer produced)").splitlines() or [""]:
        print(f"  │ {line}")
    banner("  └───────────────────────────────────────────────────────", C_GREEN)
    print(
        f"  {C_DIM}llm calls: {stats.llm_calls}  tools: {stats.tool_calls}  "
        f"tokens in/out: {stats.input_tokens}/{stats.output_tokens}  "
        f"cost: ${stats.cost_usd:.4f}  duration: {stats.duration_ms:.0f}ms{C_RESET}"
    )
    print(f"  {C_CYAN}trace:{C_RESET} {OMNISCOPE_SERVER}/?trace={stats.trace_id}")
    print()


# ---------------------------------------------------------------------------
# Interactive REPL
# ---------------------------------------------------------------------------
HELP_TEXT = """
  Commands:
    /help              show this message
    /backend [name]    show or switch backend (ollama|anthropic|stub)
    /history           list past queries this session
    /open              open the dashboard in a browser
    /quit, /exit, q    leave
"""


def run_interactive(backend: LLMBackend, tracer: OmniscopeTracer) -> None:
    history: list[QueryStats] = []
    print()
    info(f"backend: {C_BOLD}{backend.name}{C_RESET}{C_DIM} ({backend.model}){C_RESET}")
    info(f"dashboard: {OMNISCOPE_SERVER}")
    info("type /help for commands, /quit to exit")
    print()

    while True:
        try:
            query = input(f"  {C_BOLD}You ›{C_RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n  {C_DIM}bye!{C_RESET}")
            return
        if not query:
            continue

        if query.startswith("/") or query.lower() in ("quit", "exit", "q"):
            cmd, _, arg = query.lstrip("/").partition(" ")
            cmd = cmd.lower()
            if cmd in ("quit", "exit", "q"):
                print(f"  {C_DIM}bye!{C_RESET}")
                return
            if cmd == "help":
                print(HELP_TEXT)
                continue
            if cmd == "backend":
                if arg.strip():
                    try:
                        backend = pick_backend(arg.strip())
                        ok(f"switched to {backend.name} ({backend.model})")
                    except SystemExit as ex:
                        err(str(ex))
                else:
                    info(f"current backend: {backend.name} ({backend.model})")
                continue
            if cmd == "history":
                if not history:
                    info("no queries yet")
                else:
                    for i, s in enumerate(history, 1):
                        print(f"  {i}. [{s.trace_id[:8]}] llm={s.llm_calls} tools={s.tool_calls} "
                              f"${s.cost_usd:.4f} {s.duration_ms:.0f}ms")
                continue
            if cmd == "open":
                webbrowser.open(OMNISCOPE_SERVER)
                ok("opened dashboard")
                continue
            warn(f"unknown command: /{cmd}")
            continue

        try:
            stats = run_agent_pipeline(query, tracer, backend)
            history.append(stats)
            print_answer(stats)
        except Exception as ex:
            err(str(ex))
            import traceback
            traceback.print_exc()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="OMNISCOPE interactive agent demo")
    parser.add_argument("query", nargs="*", help="One-shot query (skip the REPL)")
    parser.add_argument(
        "--backend", choices=["auto", "ollama", "anthropic", "stub"], default="auto",
        help="LLM backend (default: auto-detect)",
    )
    parser.add_argument("--no-server", action="store_true", help="Don't auto-start the server")
    parser.add_argument("--open", action="store_true", help="Open the dashboard in a browser")
    args = parser.parse_args()

    print()
    banner("  OMNISCOPE — interactive multi-agent demo")
    info("=" * 56)

    server_proc = None if args.no_server else ensure_server()

    try:
        backend = pick_backend(None if args.backend == "auto" else args.backend)
        ok(f"backend: {backend.name} ({backend.model})")
    except SystemExit as ex:
        err(str(ex))
        return 1

    if args.open:
        webbrowser.open(OMNISCOPE_SERVER)

    tracer = OmniscopeTracer(server_url=OMNISCOPE_SERVER, framework=f"demo-{backend.name}")

    try:
        if args.query:
            query = " ".join(args.query)
            print()
            info(f"query: {query}")
            stats = run_agent_pipeline(query, tracer, backend)
            print_answer(stats)
        else:
            run_interactive(backend, tracer)
    finally:
        if server_proc is not None:
            info("stopping server…")
            server_proc.terminate()
            try:
                server_proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                server_proc.kill()

    return 0


if __name__ == "__main__":
    sys.exit(main())
