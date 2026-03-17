"""
OMNISCOPE Live Demo: Real multi-agent system using FREE public APIs.
No API keys required. Demonstrates actual tool calls + tracing.

Run the server first:
    python3 -m uvicorn omniscope.server.app:app --port 8781

Then run this:
    python3 demo_live.py
"""
from __future__ import annotations

import json
import ssl
import time
import urllib.request
import urllib.parse

from omniscope.sdk.adapters.generic_adapter import OmniscopeTracer

# macOS Python often lacks root certs - create unverified context as fallback
try:
    _ssl_ctx = ssl.create_default_context()
    urllib.request.urlopen("https://en.wikipedia.org", timeout=3, context=_ssl_ctx)
except ssl.SSLCertVerificationError:
    _ssl_ctx = ssl._create_unverified_context()
except Exception:
    _ssl_ctx = ssl._create_unverified_context()

SERVER = "http://localhost:8781"


def wikipedia_search(query: str, limit: int = 3) -> list[dict]:
    """Search Wikipedia API (free, no auth)."""
    params = urllib.parse.urlencode({
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": limit,
        "format": "json",
    })
    url = f"https://en.wikipedia.org/w/api.php?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": "OmniscopeDemo/1.0"})
    with urllib.request.urlopen(req, timeout=10, context=_ssl_ctx) as resp:
        data = json.loads(resp.read())
    return data.get("query", {}).get("search", [])


def wikipedia_summary(title: str) -> str:
    """Get Wikipedia page summary (free, no auth)."""
    safe_title = urllib.parse.quote(title)
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{safe_title}"
    req = urllib.request.Request(url, headers={"User-Agent": "OmniscopeDemo/1.0"})
    with urllib.request.urlopen(req, timeout=10, context=_ssl_ctx) as resp:
        data = json.loads(resp.read())
    return data.get("extract", "No summary available.")


def httpbin_post(payload: dict) -> dict:
    """POST to httpbin.org (free echo service)."""
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        "https://httpbin.org/post",
        data=data,
        headers={"Content-Type": "application/json", "User-Agent": "OmniscopeDemo/1.0"},
    )
    with urllib.request.urlopen(req, timeout=10, context=_ssl_ctx) as resp:
        return json.loads(resp.read())


def numbersapi(number: int) -> str:
    """Get a fun fact about a number (free, no auth)."""
    url = f"http://numbersapi.com/{number}?json"
    req = urllib.request.Request(url, headers={"User-Agent": "OmniscopeDemo/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=5, context=_ssl_ctx) as resp:
            data = json.loads(resp.read())
        return data.get("text", f"{number} is a number.")
    except Exception:
        return f"{number} is a number."


def run_research_agent():
    """Multi-agent research system using live Wikipedia API."""
    tracer = OmniscopeTracer(server_url=SERVER, framework="generic")

    with tracer.trace("Live Research: Quantum Computing") as t:
        # Agent 1: Research Coordinator
        with t.agent("coordinator", "Research Coordinator") as coord:
            coord.llm_call(
                model="coordinator-logic",
                input_text="Research quantum computing advances. Use Wikipedia for reliable sources.",
                output_text="Plan: 1) Search Wikipedia for quantum computing topics, 2) Get summaries, 3) Synthesize findings, 4) Generate report.",
                input_tokens=50, output_tokens=40, latency_ms=10, confidence=0.95,
            )
            coord.message_to("searcher", "Search for: quantum computing, quantum error correction, quantum supremacy", "task")

        time.sleep(0.2)

        # Agent 2: Web Searcher (REAL API CALLS)
        with t.agent("searcher", "Wikipedia Searcher") as searcher:
            # Real Wikipedia search
            t0 = time.time()
            results = wikipedia_search("quantum computing advances", limit=3)
            search_latency = int((time.time() - t0) * 1000)

            result_titles = [r["title"] for r in results]
            searcher.tool_call(
                tool_name="wikipedia_search",
                tool_input={"query": "quantum computing advances", "limit": 3},
                tool_output=json.dumps(result_titles),
                latency_ms=search_latency,
            )

            # Second search
            t0 = time.time()
            results2 = wikipedia_search("quantum error correction", limit=2)
            search_latency2 = int((time.time() - t0) * 1000)
            result_titles2 = [r["title"] for r in results2]

            searcher.tool_call(
                tool_name="wikipedia_search",
                tool_input={"query": "quantum error correction", "limit": 2},
                tool_output=json.dumps(result_titles2),
                latency_ms=search_latency2,
            )

            all_titles = result_titles + result_titles2
            searcher.llm_call(
                model="searcher-logic",
                input_text=f"Found {len(all_titles)} articles: {all_titles}",
                output_text=f"Passing {len(all_titles)} Wikipedia articles to summarizer for extraction.",
                input_tokens=len(str(all_titles)), output_tokens=30, latency_ms=5, confidence=0.9,
            )
            searcher.message_to("summarizer", f"Summarize these articles: {all_titles}", "result")

        time.sleep(0.2)

        # Agent 3: Summarizer (REAL API CALLS for summaries)
        with t.agent("summarizer", "Article Summarizer") as summarizer:
            summaries = {}
            for title in all_titles[:4]:
                t0 = time.time()
                try:
                    summary = wikipedia_summary(title)
                except Exception as e:
                    summary = f"Error fetching: {e}"
                fetch_latency = int((time.time() - t0) * 1000)

                summarizer.tool_call(
                    tool_name="wikipedia_summary",
                    tool_input={"title": title},
                    tool_output=summary[:300],
                    latency_ms=fetch_latency,
                )
                summaries[title] = summary[:300]

            combined = "\n\n".join(f"## {t}\n{s}" for t, s in summaries.items())
            summarizer.llm_call(
                model="summarizer-logic",
                input_text=f"Synthesize {len(summaries)} article summaries into key findings.",
                output_text=f"KEY FINDINGS:\n{combined[:500]}",
                input_tokens=len(combined), output_tokens=len(combined[:500]),
                latency_ms=15, confidence=0.88,
            )
            summarizer.message_to("reporter", "Synthesis complete", "result")

        time.sleep(0.2)

        # Agent 4: Report Generator
        with t.agent("reporter", "Report Generator") as reporter:
            # Use httpbin as a mock "report storage" API
            t0 = time.time()
            report_data = {
                "title": "Quantum Computing Research Report",
                "sources": list(summaries.keys()),
                "findings_count": len(summaries),
                "generated_by": "omniscope-demo",
            }
            try:
                echo = httpbin_post(report_data)
                store_latency = int((time.time() - t0) * 1000)
                reporter.tool_call(
                    tool_name="store_report",
                    tool_input=report_data,
                    tool_output=f"Report stored. Echo status: {echo.get('url', 'ok')}",
                    latency_ms=store_latency,
                )
            except Exception as e:
                store_latency = int((time.time() - t0) * 1000)
                reporter.tool_call(
                    tool_name="store_report",
                    tool_input=report_data,
                    tool_output=f"Storage skipped: {e}",
                    latency_ms=store_latency,
                    success=False,
                    error=str(e),
                )

            reporter.llm_call(
                model="reporter-logic",
                input_text="Generate final research report from synthesized findings.",
                output_text=f"# Quantum Computing Research Report\n\nBased on {len(summaries)} Wikipedia sources.\n\n{combined[:400]}\n\n## Conclusion\nQuantum computing continues to advance rapidly.",
                input_tokens=500, output_tokens=200, latency_ms=10, confidence=0.92,
            )

        time.sleep(0.1)

        # Agent 5: Quality Checker
        with t.agent("quality", "Quality Checker") as qc:
            qc.llm_call(
                model="quality-logic",
                input_text="Verify report quality: check source count, coherence, factual basis.",
                output_text=f"QUALITY REPORT:\n- Sources: {len(summaries)} (Wikipedia - reliable)\n- Coherence: Good\n- Factual basis: All claims traceable to sources\n- Grade: B+\n\nApproved for delivery.",
                input_tokens=200, output_tokens=80, latency_ms=8, confidence=0.91,
            )
            qc.decision("Report approved. All findings backed by Wikipedia sources.", confidence=0.91)


def run_fun_facts_agent():
    """Lighter demo: multi-agent fun facts using Numbers API + Wikipedia."""
    tracer = OmniscopeTracer(server_url=SERVER, framework="generic")

    with tracer.trace("Live Demo: Fun Facts Pipeline") as t:
        with t.agent("picker", "Number Picker") as picker:
            import random
            numbers = random.sample(range(1, 100), 3)
            picker.llm_call(
                model="picker-logic",
                input_text="Pick 3 interesting numbers to research.",
                output_text=f"Selected numbers: {numbers}",
                input_tokens=10, output_tokens=10, latency_ms=1, confidence=0.99,
            )

        time.sleep(0.1)

        with t.agent("fact_finder", "Fact Finder") as finder:
            facts = {}
            for n in numbers:
                t0 = time.time()
                fact = numbersapi(n)
                lat = int((time.time() - t0) * 1000)
                finder.tool_call(
                    tool_name="numbersapi",
                    tool_input={"number": n},
                    tool_output=fact,
                    latency_ms=lat,
                )
                facts[n] = fact

            finder.llm_call(
                model="finder-logic",
                input_text=f"Collected facts for {numbers}",
                output_text="\n".join(f"- {n}: {f}" for n, f in facts.items()),
                input_tokens=50, output_tokens=len(str(facts)), latency_ms=5, confidence=0.95,
            )

        time.sleep(0.1)

        with t.agent("wiki_enricher", "Wikipedia Enricher") as enricher:
            for n in numbers[:2]:
                t0 = time.time()
                try:
                    wiki_results = wikipedia_search(f"number {n} mathematics", limit=1)
                    lat = int((time.time() - t0) * 1000)
                    title = wiki_results[0]["title"] if wiki_results else f"Number {n}"
                    enricher.tool_call(
                        tool_name="wikipedia_search",
                        tool_input={"query": f"number {n} mathematics"},
                        tool_output=title,
                        latency_ms=lat,
                    )
                except Exception:
                    pass

            enricher.llm_call(
                model="enricher-logic",
                input_text="Cross-reference number facts with Wikipedia.",
                output_text="Enrichment complete. Added mathematical context to fun facts.",
                input_tokens=40, output_tokens=15, latency_ms=3, confidence=0.87,
            )

        time.sleep(0.1)

        with t.agent("compiler", "Report Compiler") as compiler:
            report_lines = [f"**{n}**: {f}" for n, f in facts.items()]
            compiler.llm_call(
                model="compiler-logic",
                input_text="Compile enriched fun facts into final report.",
                output_text="# Fun Facts Report\n\n" + "\n".join(report_lines) + "\n\nGenerated by OMNISCOPE multi-agent pipeline.",
                input_tokens=100, output_tokens=80, latency_ms=5, confidence=0.94,
            )
            compiler.decision("Report compiled and ready for delivery.", confidence=0.94)


def main():
    print("=== OMNISCOPE Live Demo (Real API Calls) ===")
    print(f"Server: {SERVER}")
    print()

    demos = [
        ("Live Research: Quantum Computing (Wikipedia API)", run_research_agent),
        ("Live Demo: Fun Facts Pipeline (Numbers API + Wikipedia)", run_fun_facts_agent),
    ]

    for name, fn in demos:
        print(f"Running: {name}...", end=" ", flush=True)
        try:
            fn()
            print("OK")
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
        time.sleep(0.5)

    print()
    print("Live demo traces generated with real API calls!")
    print("Open http://localhost:8781 to view the dashboard.")


if __name__ == "__main__":
    main()
