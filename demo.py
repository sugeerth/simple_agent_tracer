"""
OMNISCOPE Demo: Generates realistic multi-agent traces.

Run the server first:
    python -m uvicorn omniscope.server.app:app --port 8781

Then run this demo:
    python demo.py

Opens the dashboard at http://localhost:8781 with live trace data.
"""
from __future__ import annotations

import random
import time
import uuid

from omniscope.sdk.adapters.generic_adapter import OmniscopeTracer

SERVER = "http://localhost:8781"


def demo_product_listing():
    """Simulates the product photo analysis multi-agent workflow."""
    tracer = OmniscopeTracer(server_url=SERVER, framework="generic")

    with tracer.trace("Product Photo Analysis") as t:
        # Orchestrator
        with t.agent("orchestrator", "Orchestrator") as orch:
            orch.llm_call(
                model="claude-opus-4-6",
                input_text="Analyze the attached product photo and write a listing.",
                output_text="I'll decompose this into: (1) visual analysis, (2) listing draft, (3) quality review.",
                input_tokens=312, output_tokens=87, latency_ms=120, confidence=0.95,
                cost_usd=0.008,
            )
            e1 = orch.decision("Delegating to planner for task decomposition", confidence=0.95)

        time.sleep(0.1)

        # Planner
        with t.agent("planner", "Planner") as planner:
            planner.llm_call(
                model="claude-sonnet-4-6",
                input_text="Decompose: analyze product photo and write listing",
                output_text="Plan: Step 1 - Vision analysis, Step 2 - Draft listing, Step 3 - Critic review",
                input_tokens=245, output_tokens=156, latency_ms=220, confidence=0.92,
                cost_usd=0.003,
            )

        time.sleep(0.1)

        # Vision Agent
        with t.agent("vision", "Vision Agent") as vision:
            vision.tool_call(
                tool_name="image_encoder",
                tool_input={"image": "product_photo.jpg", "model": "ViT-L/14"},
                tool_output="Encoded: 768-dim embedding, patch_grid=14x14",
                latency_ms=50,
            )
            vision.llm_call(
                model="claude-opus-4-6",
                input_text="[Image embedding + photo] Describe this product in detail.",
                output_text="VISUAL ANALYSIS:\n- Product: Premium leather crossbody bag\n- Color: Cognac brown\n- Material: Full-grain leather\n- Hardware: Brushed gold clasp\n- Brand: 'PREMIUM QUALITY' embossed",
                input_tokens=1024, output_tokens=312, latency_ms=370, confidence=0.93,
                cost_usd=0.012,
            )

        time.sleep(0.1)

        # Writer Agent (draft v1 - has hallucinations)
        with t.agent("writer", "Writer Agent") as writer:
            writer.llm_call(
                model="claude-opus-4-6",
                input_text="Write an e-commerce listing based on: Premium leather crossbody bag...",
                output_text='## Premium Cognac Leather Crossbody Bag\n\nElevate your everyday carry...\n\n**Dimensions:** 10" x 7" x 3"\n\n*Free shipping on orders over $75*',
                input_tokens=1456, output_tokens=487, latency_ms=930, confidence=0.88,
                cost_usd=0.018,
            )

        time.sleep(0.1)

        # Critic Agent (finds issues)
        with t.agent("critic", "Critic Agent") as critic:
            critic.llm_call(
                model="claude-opus-4-6",
                input_text="Review this listing against the product image...",
                output_text="ISSUES FOUND:\n1. Dimensions '10x7x3' are fabricated - not verifiable from image\n2. 'Free shipping' has no basis in task context\nRECOMMENDATION: Fix and resubmit.",
                input_tokens=2100, output_tokens=234, latency_ms=650, confidence=0.85,
                cost_usd=0.014,
            )

        time.sleep(0.1)

        # Writer Agent (revision - fixes hallucinations)
        with t.agent("writer_v2", "Writer Agent (revision)") as writer2:
            writer2.llm_call(
                model="claude-opus-4-6",
                input_text="Revise listing: remove fabricated dimensions and shipping claim...",
                output_text="## Premium Cognac Leather Crossbody Bag\n\nElevate your everyday carry with this handcrafted full-grain leather crossbody bag...\n\nCrafted for those who appreciate timeless quality.",
                input_tokens=1890, output_tokens=289, latency_ms=680, confidence=0.94,
                cost_usd=0.015,
            )

        time.sleep(0.1)

        # Critic approves
        with t.agent("critic_v2", "Critic Agent (final)") as critic2:
            critic2.decision("APPROVED. All fabricated claims removed. Listing is accurate.", confidence=0.96)


def demo_research_workflow():
    """Simulates a multi-agent research workflow with tool calls."""
    tracer = OmniscopeTracer(server_url=SERVER, framework="generic")

    with tracer.trace("Research: Quantum Computing Advances 2026") as t:
        # Orchestrator
        with t.agent("coordinator", "Research Coordinator") as coord:
            coord.llm_call(
                model="claude-opus-4-6",
                input_text="Research the latest advances in quantum computing in 2026.",
                output_text="I'll coordinate: 1) Web researcher for recent papers, 2) Analyst for synthesis, 3) Writer for summary.",
                input_tokens=200, output_tokens=120, latency_ms=180, confidence=0.94,
                cost_usd=0.006,
            )
            coord.message_to("researcher", "Find recent quantum computing papers and breakthroughs from 2026", "task")

        time.sleep(0.1)

        # Researcher with tool calls
        with t.agent("researcher", "Web Researcher") as researcher:
            researcher.tool_call(
                tool_name="web_search",
                tool_input={"query": "quantum computing breakthroughs 2026", "num_results": 10},
                tool_output="Found 10 results: [1] IBM 1000-qubit processor, [2] Google quantum advantage in chemistry...",
                latency_ms=1200,
            )
            researcher.tool_call(
                tool_name="web_search",
                tool_input={"query": "quantum error correction 2026 advances"},
                tool_output="Found 8 results: [1] Microsoft topological qubits, [2] Harvard logical qubit breakthrough...",
                latency_ms=980,
            )
            researcher.tool_call(
                tool_name="arxiv_search",
                tool_input={"query": "quantum computing 2026", "max_results": 5},
                tool_output="Papers: [1] arXiv:2601.xxxxx 'Fault-tolerant quantum computing at scale', ...",
                latency_ms=450,
            )
            researcher.llm_call(
                model="claude-sonnet-4-6",
                input_text="Synthesize these search results into key findings...",
                output_text="Key findings:\n1. IBM achieved 1000 logical qubits\n2. Google demonstrated quantum advantage in drug discovery\n3. Microsoft's topological qubits reached 99.9% fidelity",
                input_tokens=3200, output_tokens=450, latency_ms=520, confidence=0.87,
                cost_usd=0.009,
            )
            researcher.message_to("analyst", "Research findings ready for analysis", "result")

        time.sleep(0.15)

        # Analyst
        with t.agent("analyst", "Research Analyst") as analyst:
            analyst.llm_call(
                model="claude-opus-4-6",
                input_text="Analyze these quantum computing findings for significance and reliability...",
                output_text="Analysis:\n- IBM claim: HIGH significance, VERIFIED by multiple sources\n- Google claim: HIGH significance, peer-reviewed in Nature\n- Microsoft claim: MEDIUM significance, preprint only",
                input_tokens=2800, output_tokens=380, latency_ms=710, confidence=0.91,
                cost_usd=0.016,
            )
            analyst.decision("Flagging Microsoft claim for additional verification - preprint only", confidence=0.78)
            analyst.message_to("writer", "Analysis complete, ready for summary draft", "result")

        time.sleep(0.1)

        # Writer
        with t.agent("summary_writer", "Summary Writer") as writer:
            writer.llm_call(
                model="claude-opus-4-6",
                input_text="Write a comprehensive research summary on quantum computing advances...",
                output_text="# Quantum Computing Advances in 2026\n\n## Executive Summary\n\n2026 marked a watershed year for quantum computing with three major milestones...\n\n## Key Breakthroughs\n\n### 1. IBM's 1000 Logical Qubit Processor...",
                input_tokens=3500, output_tokens=1200, latency_ms=1800, confidence=0.92,
                cost_usd=0.032,
            )

        time.sleep(0.1)

        # Quality check
        with t.agent("quality_checker", "Quality Checker") as qc:
            qc.llm_call(
                model="claude-sonnet-4-6",
                input_text="Verify factual accuracy of this research summary...",
                output_text="Verification: 5/6 claims verified. Note: Microsoft fidelity figure should be cited as preliminary. Overall quality: GOOD.",
                input_tokens=2200, output_tokens=180, latency_ms=420, confidence=0.89,
                cost_usd=0.005,
            )
            qc.decision("Summary approved with minor annotation on Microsoft claim", confidence=0.89)


def demo_coding_agents():
    """Simulates a coding agent workflow with tool calls and errors."""
    tracer = OmniscopeTracer(server_url=SERVER, framework="generic")

    with tracer.trace("Fix Bug: API Rate Limiting") as t:
        with t.agent("lead_dev", "Lead Developer") as lead:
            lead.llm_call(
                model="claude-opus-4-6",
                input_text="Users are reporting 429 errors on our API. Investigate and fix.",
                output_text="I'll: 1) Check error logs, 2) Review rate limiter config, 3) Implement fix, 4) Write tests.",
                input_tokens=180, output_tokens=95, latency_ms=150, confidence=0.96,
                cost_usd=0.005,
            )

        time.sleep(0.1)

        with t.agent("investigator", "Bug Investigator") as inv:
            inv.tool_call(
                tool_name="grep_logs",
                tool_input={"pattern": "429", "file": "/var/log/api/error.log", "last_n": 100},
                tool_output="Found 847 occurrences in last hour. Peak: 3:42 PM (312 in 1 minute). Source: /api/v2/search endpoint.",
                latency_ms=120,
            )
            inv.tool_call(
                tool_name="read_file",
                tool_input={"path": "src/middleware/rate_limiter.py"},
                tool_output="class RateLimiter:\n    def __init__(self):\n        self.window = 60\n        self.max_requests = 100  # per minute per IP\n...",
                latency_ms=30,
            )
            inv.llm_call(
                model="claude-opus-4-6",
                input_text="Analyze: 847 429 errors in last hour, rate limit is 100/min/IP...",
                output_text="ROOT CAUSE: Rate limiter uses IP-based bucketing, but the search endpoint is called by a CDN with shared IPs. 1000+ users share 3 CDN IPs, hitting 100/min limit immediately.\n\nFIX: Switch to API key-based rate limiting for authenticated endpoints.",
                input_tokens=1500, output_tokens=280, latency_ms=380, confidence=0.93,
                cost_usd=0.012,
            )

        time.sleep(0.1)

        with t.agent("coder", "Code Agent") as coder:
            coder.llm_call(
                model="claude-opus-4-6",
                input_text="Implement API key-based rate limiting for the search endpoint...",
                output_text="```python\nclass APIKeyRateLimiter:\n    def __init__(self, redis_client):\n        self.redis = redis_client\n        self.window = 60\n        self.max_requests = 500  # per API key\n\n    async def check(self, api_key: str) -> bool:\n        key = f'rate:{api_key}'\n        count = await self.redis.incr(key)\n        if count == 1:\n            await self.redis.expire(key, self.window)\n        return count <= self.max_requests\n```",
                input_tokens=2200, output_tokens=450, latency_ms=620, confidence=0.91,
                cost_usd=0.018,
            )
            coder.tool_call(
                tool_name="write_file",
                tool_input={"path": "src/middleware/rate_limiter_v2.py"},
                tool_output="File written: 45 lines",
                latency_ms=15,
            )
            coder.tool_call(
                tool_name="run_tests",
                tool_input={"test_path": "tests/test_rate_limiter.py"},
                tool_output="FAILED: 2 passed, 1 failed\ntest_concurrent_keys: AssertionError - expected 500, got 100",
                success=False,
                latency_ms=3200,
                error="Test failure: test_concurrent_keys",
            )
            coder.llm_call(
                model="claude-opus-4-6",
                input_text="Fix the failing test - max_requests still reading old value...",
                output_text="Found the issue: the test fixture was using the old RateLimiter class. Updated import and default value.",
                input_tokens=800, output_tokens=120, latency_ms=250, confidence=0.95,
                cost_usd=0.006,
            )
            coder.tool_call(
                tool_name="run_tests",
                tool_input={"test_path": "tests/test_rate_limiter.py"},
                tool_output="PASSED: 3 passed, 0 failed",
                latency_ms=2800,
            )

        time.sleep(0.1)

        with t.agent("reviewer", "Code Reviewer") as reviewer:
            reviewer.llm_call(
                model="claude-opus-4-6",
                input_text="Review the rate limiter changes for correctness and security...",
                output_text="LGTM. Redis-based sliding window is correct. Suggestions:\n1. Add rate limit headers (X-RateLimit-Remaining)\n2. Consider implementing token bucket for burst tolerance\n\nApproved for merge.",
                input_tokens=1800, output_tokens=200, latency_ms=340, confidence=0.94,
                cost_usd=0.012,
            )


def demo_langgraph_style():
    """Simulates a LangGraph-style stateful workflow."""
    tracer = OmniscopeTracer(server_url=SERVER, framework="langgraph")

    with tracer.trace("LangGraph: Customer Support Agent") as t:
        with t.agent("classifier", "Intent Classifier") as clf:
            clf.llm_call(
                model="gpt-4o",
                input_text="Customer: My order hasn't arrived and it's been 2 weeks",
                output_text='{"intent": "order_tracking", "sentiment": "frustrated", "urgency": "high"}',
                input_tokens=45, output_tokens=30, latency_ms=280, confidence=0.97,
            )

        time.sleep(0.05)

        with t.agent("router", "State Router") as router:
            router.decision("Routing to order_tracking node based on intent classification", confidence=0.97)

        time.sleep(0.05)

        with t.agent("order_tracker", "Order Tracker") as tracker:
            tracker.tool_call(
                tool_name="lookup_order",
                tool_input={"customer_id": "cust_12345", "status": "pending"},
                tool_output='{"order_id": "ORD-789", "status": "in_transit", "carrier": "FedEx", "tracking": "7891234", "eta": "2026-03-18"}',
                latency_ms=150,
            )
            tracker.llm_call(
                model="gpt-4o",
                input_text="Order found: ORD-789, in transit via FedEx, ETA March 18. Customer is frustrated about 2 week delay.",
                output_text="I understand your frustration. I've located your order ORD-789 - it's currently in transit with FedEx (tracking: 7891234). The expected delivery is March 18. Would you like me to escalate this to our shipping team for priority handling?",
                input_tokens=120, output_tokens=85, latency_ms=350, confidence=0.93,
            )

        time.sleep(0.05)

        with t.agent("satisfaction_checker", "Satisfaction Checker") as sat:
            sat.llm_call(
                model="gpt-4o-mini",
                input_text="Evaluate if the customer's issue was resolved adequately...",
                output_text='{"resolved": true, "satisfaction_estimate": 0.72, "follow_up_needed": true, "reason": "eta_provided_but_delay_not_explained"}',
                input_tokens=200, output_tokens=40, latency_ms=180, confidence=0.85,
            )


def main():
    print("=== OMNISCOPE Demo ===")
    print(f"Server: {SERVER}")
    print()

    demos = [
        ("Product Photo Analysis", demo_product_listing),
        ("Research: Quantum Computing", demo_research_workflow),
        ("Fix Bug: API Rate Limiting", demo_coding_agents),
        ("LangGraph: Customer Support", demo_langgraph_style),
    ]

    for name, fn in demos:
        print(f"Running: {name}...", end=" ", flush=True)
        try:
            fn()
            print("OK")
        except Exception as e:
            print(f"ERROR: {e}")
        time.sleep(0.3)

    print()
    print("Demo traces generated.")
    print(f"Open http://localhost:8781 to view the dashboard.")


if __name__ == "__main__":
    main()
