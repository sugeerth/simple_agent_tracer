"""
OMNISCOPE Quickstart — Every integration pattern in one file.

Start the server first:
    pip install fastapi uvicorn httpx pydantic
    python -m uvicorn omniscope.server.app:app --port 8781

Then run this from the repo root (as a module, so `omniscope` is importable):
    python -m examples.quickstart

Traces then land at http://localhost:8781/api/v1/traces; for the dashboard UI see the README.
"""
import time


# ============================================================
# 1. GENERIC — Works with any agent system, zero dependencies
# ============================================================
def demo_generic():
    import omniscope

    omniscope.init()  # one line to connect

    with omniscope.trace("generic-research-pipeline") as t:
        # Agent 1: Planner
        with t.agent("planner", "Research Planner") as planner:
            planner.llm_call(
                model="gpt-4",
                input_text="Plan research on quantum computing",
                output_text="Step 1: Search papers. Step 2: Summarize. Step 3: Report.",
                input_tokens=50, output_tokens=30, latency_ms=200,
                confidence=0.95,
            )
            planner.message_to("searcher", "Search for quantum computing papers")

        # Agent 2: Searcher
        with t.agent("searcher", "Paper Searcher") as searcher:
            searcher.tool_call(
                tool_name="arxiv_search",
                tool_input={"query": "quantum error correction 2024", "limit": 5},
                tool_output="Found 5 papers: [1] Quantum LDPC codes...",
                latency_ms=450,
            )
            searcher.llm_call(
                model="gpt-4",
                input_text="Rank these papers by relevance",
                output_text="Top paper: Quantum LDPC codes (relevance: 0.95)",
                input_tokens=200, output_tokens=100, latency_ms=300,
                confidence=0.88,
            )
            searcher.message_to("writer", "Top 3 papers ready for summarization")

        # Agent 3: Writer
        with t.agent("writer", "Report Writer") as writer:
            writer.llm_call(
                model="gpt-4",
                input_text="Write a research summary from these papers",
                output_text="# Quantum Computing Advances\n\nRecent work on quantum LDPC codes...",
                input_tokens=500, output_tokens=800, latency_ms=1200,
                confidence=0.91, cost_usd=0.015,
            )
            writer.decision("Report approved for delivery", confidence=0.91)


# ============================================================
# 2. DECORATOR STYLE — Minimal instrumentation for existing code
# ============================================================
def demo_decorators():
    from omniscope.sdk import trace_agent, trace_tool, trace_llm_call, omniscope

    trace_id = omniscope.start_trace("decorator-demo", framework="generic")

    @trace_tool(name="calculator")
    def calculate(expression: str, _trace_id: str = "") -> str:
        return str(eval(expression))  # noqa: S307 — demo only

    @trace_agent(name="math_agent")
    def solve_math(problem: str, _trace_id: str = "") -> str:
        result = calculate("2 + 2", _trace_id=_trace_id)
        return f"The answer is {result}"

    solve_math("What is 2+2?", _trace_id=trace_id)
    omniscope.end_trace(trace_id)


# ============================================================
# 3. LANGCHAIN / LANGGRAPH — Just pass the callback handler
# ============================================================
def demo_langchain():
    """
    import omniscope
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate

    handler = omniscope.langchain_handler(trace_name="langchain-qa")

    llm = ChatOpenAI(model="gpt-4")
    prompt = ChatPromptTemplate.from_template("Answer: {question}")
    chain = prompt | llm

    # Just add the handler to callbacks — that's it
    chain.invoke(
        {"question": "What is quantum computing?"},
        config={"callbacks": [handler]},
    )
    handler.finish()
    """
    print("  (requires langchain — see docstring for usage)")


# ============================================================
# 4. CREWAI — Wrap the crew execution
# ============================================================
def demo_crewai():
    """
    import omniscope
    from crewai import Agent, Task, Crew

    tracer = omniscope.crewai_tracer()

    researcher = Agent(role="Researcher", goal="Find info", backstory="Expert researcher")
    writer = Agent(role="Writer", goal="Write report", backstory="Expert writer")

    task1 = Task(description="Research quantum computing", agent=researcher)
    task2 = Task(description="Write a summary", agent=writer)

    crew = Crew(agents=[researcher, writer], tasks=[task1, task2])
    result = tracer.trace_crew(crew, inputs={"topic": "quantum computing"})
    """
    print("  (requires crewai — see docstring for usage)")


# ============================================================
# 5. ANTHROPIC SDK — Wrap the client
# ============================================================
def demo_anthropic():
    """
    import omniscope
    from anthropic import Anthropic

    client = omniscope.wrap_anthropic(Anthropic())

    with client.trace("anthropic-qa") as t:
        response = t.create(
            model="claude-sonnet-4-6",
            messages=[{"role": "user", "content": "What is quantum computing?"}],
            agent_id="qa_agent",
        )
        print(response.content[0].text)
    """
    print("  (requires anthropic — see docstring for usage)")


# ============================================================
# 6. OPENAI AGENTS SDK — Wrap the runner
# ============================================================
def demo_openai_agents():
    """
    import omniscope
    from agents import Agent, Runner

    tracer = omniscope.openai_agents_tracer()

    agent = Agent(name="researcher", instructions="You research topics thoroughly.")
    result = await tracer.trace_run(agent, "What is quantum computing?")
    """
    print("  (requires openai-agents — see docstring for usage)")


# ============================================================
# 7. AUTOGEN / AG2 — Wrap the chat
# ============================================================
def demo_autogen():
    """
    import omniscope
    from autogen import AssistantAgent, UserProxyAgent

    tracer = omniscope.autogen_tracer()

    assistant = AssistantAgent("assistant", llm_config={"model": "gpt-4"})
    user_proxy = UserProxyAgent("user_proxy", human_input_mode="NEVER")

    result = tracer.trace_chat(
        user_proxy, "Write a Python function to sort a list", recipient=assistant
    )
    """
    print("  (requires pyautogen — see docstring for usage)")


# ============================================================
# Run all demos
# ============================================================
if __name__ == "__main__":
    print("=== OMNISCOPE Quickstart ===\n")
    print("Server: http://localhost:8781\n")

    demos = [
        ("1. Generic (context managers)", demo_generic),
        ("2. Decorators", demo_decorators),
        ("3. LangChain / LangGraph", demo_langchain),
        ("4. CrewAI", demo_crewai),
        ("5. Anthropic SDK", demo_anthropic),
        ("6. OpenAI Agents SDK", demo_openai_agents),
        ("7. AutoGen / AG2", demo_autogen),
    ]

    for name, fn in demos:
        print(f"{name}...", end=" ", flush=True)
        try:
            fn()
            print("OK")
        except Exception as e:
            print(f"SKIP ({e})")
        time.sleep(0.3)

    print("\nDone! Open http://localhost:8781 to see traces.")
