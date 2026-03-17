/**
 * Embedded demo data for GitHub Pages standalone mode.
 * When the API is unavailable, the dashboard renders this data.
 */

import type { TraceOverview, TraceGraph, TimelineEntry, RiskScores, TraceEvent } from './useTraceData';

const DEMO_TRACES: TraceOverview[] = [
  {
    trace_id: 'demo-product-listing',
    trace_name: 'Product Photo Analysis',
    framework: 'generic',
    started_at: '2026-03-16T14:32:00.000Z',
    event_count: 12,
    agent_ids: ['orchestrator', 'planner', 'vision', 'writer', 'critic'],
    total_tokens: 8347,
    total_cost: 0.076,
    status: 'completed',
    composite_score: 0.91,
    risk_scores: {},
  },
  {
    trace_id: 'demo-research',
    trace_name: 'Research: Quantum Computing 2026',
    framework: 'generic',
    started_at: '2026-03-16T14:35:00.000Z',
    event_count: 16,
    agent_ids: ['coordinator', 'researcher', 'analyst', 'summary_writer', 'quality_checker'],
    total_tokens: 14350,
    total_cost: 0.068,
    status: 'completed',
    composite_score: null,
    risk_scores: {},
  },
  {
    trace_id: 'demo-bugfix',
    trace_name: 'Fix Bug: API Rate Limiting',
    framework: 'generic',
    started_at: '2026-03-16T14:38:00.000Z',
    event_count: 14,
    agent_ids: ['lead_dev', 'investigator', 'coder', 'reviewer'],
    total_tokens: 7625,
    total_cost: 0.053,
    status: 'completed',
    composite_score: null,
    risk_scores: {},
  },
  {
    trace_id: 'demo-langgraph',
    trace_name: 'LangGraph: Customer Support Agent',
    framework: 'langgraph',
    started_at: '2026-03-16T14:40:00.000Z',
    event_count: 8,
    agent_ids: ['classifier', 'router', 'order_tracker', 'satisfaction_checker'],
    total_tokens: 520,
    total_cost: 0.0,
    status: 'completed',
    composite_score: null,
    risk_scores: {},
  },
];

function makeNode(id: string, agentId: string, agentName: string, eventType: string, latency: number, tokens: number, confidence: number | null, output: string, model: string | null = 'claude-opus-4-6') {
  return {
    id, agent_id: agentId, agent_name: agentName, event_type: eventType,
    label: agentName, timestamp: '2026-03-16T14:32:00.000Z',
    latency_ms: latency, confidence, risk_score: confidence !== null ? Math.max(0, (1 - confidence) * 0.5) : 0,
    state: 'completed' as const, input_preview: '', output_preview: output,
    model_name: model, tokens, framework: 'generic',
  };
}

const DEMO_GRAPHS: Record<string, TraceGraph> = {
  'demo-product-listing': {
    nodes: [
      makeNode('p1', 'orchestrator', 'Orchestrator', 'agent_decision', 120, 399, 0.95, 'Decompose into: visual analysis, listing draft, quality review.'),
      makeNode('p2', 'planner', 'Planner', 'planning_step', 220, 401, 0.92, 'Plan: Step 1 - Vision, Step 2 - Draft, Step 3 - Review'),
      makeNode('p3', 'vision', 'Vision Agent', 'tool_call', 50, 0, null, 'ViT-L/14 encoding: 768-dim, patch 14x14'),
      makeNode('p4', 'vision', 'Vision Agent', 'llm_call', 370, 1336, 0.93, 'VISUAL ANALYSIS: Premium leather crossbody bag, cognac brown, full-grain leather...'),
      makeNode('p5', 'writer', 'Writer Agent', 'llm_call', 930, 1943, 0.88, '## Premium Cognac Leather Crossbody Bag\n\nDimensions: 10" x 7" x 3"\nFree shipping...'),
      makeNode('p6', 'critic', 'Critic Agent', 'llm_call', 650, 2334, 0.85, 'ISSUES: Dimensions fabricated. Free shipping has no basis. Fix and resubmit.'),
      makeNode('p7', 'writer', 'Writer (revision)', 'llm_call', 680, 2179, 0.94, '## Premium Cognac Leather Crossbody Bag\n\nCrafted for those who appreciate timeless quality.'),
      makeNode('p8', 'critic', 'Critic (final)', 'agent_decision', 220, 89, 0.96, 'APPROVED. All fabricated claims removed.'),
    ],
    edges: [
      { source: 'p1', target: 'p2', edge_type: 'causal', animated: false },
      { source: 'p2', target: 'p3', edge_type: 'causal', animated: false },
      { source: 'p3', target: 'p4', edge_type: 'causal', animated: false },
      { source: 'p4', target: 'p5', edge_type: 'data', animated: true },
      { source: 'p5', target: 'p6', edge_type: 'causal', animated: false },
      { source: 'p6', target: 'p7', edge_type: 'causal', animated: false },
      { source: 'p7', target: 'p8', edge_type: 'causal', animated: false },
    ],
  },
  'demo-research': {
    nodes: [
      makeNode('r1', 'coordinator', 'Research Coordinator', 'llm_call', 180, 320, 0.94, 'Coordinate: web researcher, analyst, writer'),
      makeNode('r2', 'researcher', 'Web Researcher', 'tool_call', 1200, 0, null, 'web_search: quantum computing breakthroughs 2026 (10 results)'),
      makeNode('r3', 'researcher', 'Web Researcher', 'tool_call', 980, 0, null, 'web_search: quantum error correction 2026 (8 results)'),
      makeNode('r4', 'researcher', 'Web Researcher', 'tool_call', 450, 0, null, 'arxiv_search: quantum computing 2026 (5 papers)'),
      makeNode('r5', 'researcher', 'Web Researcher', 'llm_call', 520, 3650, 0.87, 'Key findings: IBM 1000 qubits, Google drug discovery, Microsoft topological qubits', 'claude-sonnet-4-6'),
      makeNode('r6', 'analyst', 'Research Analyst', 'llm_call', 710, 3180, 0.91, 'IBM: HIGH significance, VERIFIED. Google: HIGH, peer-reviewed. Microsoft: MEDIUM, preprint.'),
      makeNode('r7', 'analyst', 'Research Analyst', 'agent_decision', 0, 0, 0.78, 'Flagging Microsoft claim for verification - preprint only'),
      makeNode('r8', 'summary_writer', 'Summary Writer', 'llm_call', 1800, 4700, 0.92, '# Quantum Computing Advances in 2026\n\n## Executive Summary...'),
      makeNode('r9', 'quality_checker', 'Quality Checker', 'llm_call', 420, 2380, 0.89, '5/6 claims verified. Microsoft fidelity figure should be cited as preliminary.'),
    ],
    edges: [
      { source: 'r1', target: 'r2', edge_type: 'causal', animated: false },
      { source: 'r1', target: 'r3', edge_type: 'causal', animated: false },
      { source: 'r1', target: 'r4', edge_type: 'causal', animated: false },
      { source: 'r2', target: 'r5', edge_type: 'data', animated: true },
      { source: 'r3', target: 'r5', edge_type: 'data', animated: true },
      { source: 'r4', target: 'r5', edge_type: 'data', animated: true },
      { source: 'r5', target: 'r6', edge_type: 'causal', animated: false },
      { source: 'r6', target: 'r7', edge_type: 'causal', animated: false },
      { source: 'r6', target: 'r8', edge_type: 'data', animated: true },
      { source: 'r8', target: 'r9', edge_type: 'causal', animated: false },
    ],
  },
  'demo-bugfix': {
    nodes: [
      makeNode('b1', 'lead_dev', 'Lead Developer', 'llm_call', 150, 275, 0.96, 'Plan: check logs, review rate limiter, implement fix, write tests'),
      makeNode('b2', 'investigator', 'Bug Investigator', 'tool_call', 120, 0, null, 'grep_logs: 847 occurrences of 429 in last hour'),
      makeNode('b3', 'investigator', 'Bug Investigator', 'tool_call', 30, 0, null, 'read_file: rate_limiter.py - max_requests=100/min/IP'),
      makeNode('b4', 'investigator', 'Bug Investigator', 'llm_call', 380, 1780, 0.93, 'ROOT CAUSE: CDN shares IPs. 1000+ users share 3 IPs.'),
      makeNode('b5', 'coder', 'Code Agent', 'llm_call', 620, 2650, 0.91, 'class APIKeyRateLimiter: max_requests=500 per API key'),
      makeNode('b6', 'coder', 'Code Agent', 'tool_call', 3200, 0, null, 'run_tests: FAILED - test_concurrent_keys', null),
      makeNode('b7', 'coder', 'Code Agent', 'llm_call', 250, 920, 0.95, 'Fixed: test fixture using old RateLimiter class'),
      makeNode('b8', 'coder', 'Code Agent', 'tool_call', 2800, 0, null, 'run_tests: PASSED 3/3', null),
      makeNode('b9', 'reviewer', 'Code Reviewer', 'llm_call', 340, 2000, 0.94, 'LGTM. Approved for merge.'),
    ],
    edges: [
      { source: 'b1', target: 'b2', edge_type: 'causal', animated: false },
      { source: 'b2', target: 'b3', edge_type: 'causal', animated: false },
      { source: 'b3', target: 'b4', edge_type: 'causal', animated: false },
      { source: 'b4', target: 'b5', edge_type: 'data', animated: true },
      { source: 'b5', target: 'b6', edge_type: 'causal', animated: false },
      { source: 'b6', target: 'b7', edge_type: 'causal', animated: false },
      { source: 'b7', target: 'b8', edge_type: 'causal', animated: false },
      { source: 'b8', target: 'b9', edge_type: 'causal', animated: false },
    ],
  },
  'demo-langgraph': {
    nodes: [
      makeNode('l1', 'classifier', 'Intent Classifier', 'llm_call', 280, 75, 0.97, '{"intent": "order_tracking", "sentiment": "frustrated", "urgency": "high"}', 'gpt-4o'),
      makeNode('l2', 'router', 'State Router', 'agent_decision', 0, 0, 0.97, 'Routing to order_tracking node'),
      makeNode('l3', 'order_tracker', 'Order Tracker', 'tool_call', 150, 0, null, 'lookup_order: ORD-789, in_transit, FedEx, ETA March 18', null),
      makeNode('l4', 'order_tracker', 'Order Tracker', 'llm_call', 350, 205, 0.93, 'Your order ORD-789 is in transit with FedEx. ETA March 18.', 'gpt-4o'),
      makeNode('l5', 'satisfaction_checker', 'Satisfaction Checker', 'llm_call', 180, 240, 0.85, '{"resolved": true, "satisfaction_estimate": 0.72, "follow_up_needed": true}', 'gpt-4o-mini'),
    ],
    edges: [
      { source: 'l1', target: 'l2', edge_type: 'causal', animated: false },
      { source: 'l2', target: 'l3', edge_type: 'causal', animated: false },
      { source: 'l3', target: 'l4', edge_type: 'data', animated: true },
      { source: 'l4', target: 'l5', edge_type: 'causal', animated: false },
    ],
  },
};

function buildTimeline(graph: TraceGraph): TimelineEntry[] {
  let offset = 0;
  return graph.nodes.map(n => {
    const entry: TimelineEntry = {
      event_id: n.id,
      agent_id: n.agent_id,
      agent_name: n.agent_name,
      event_type: n.event_type,
      start_ms: offset,
      duration_ms: n.latency_ms || 50,
      label: n.label,
      risk_score: n.risk_score,
      confidence: n.confidence,
    };
    offset += n.latency_ms || 50;
    return entry;
  });
}

function buildRisk(graph: TraceGraph): Record<string, RiskScores> {
  const agents = new Map<string, { scores: number[]; tools: string[] }>();
  graph.nodes.forEach(n => {
    if (!agents.has(n.agent_id)) agents.set(n.agent_id, { scores: [], tools: [] });
    const a = agents.get(n.agent_id)!;
    if (n.confidence !== null) a.scores.push(n.confidence);
    if (n.event_type === 'tool_call') a.tools.push(n.id);
  });

  const result: Record<string, RiskScores> = {};
  agents.forEach((data, agentId) => {
    const avg = data.scores.length > 0 ? data.scores.reduce((a, b) => a + b, 0) / data.scores.length : 1;
    const halluc = Math.max(0, (1 - avg) * 0.6);
    const overall = halluc * 0.4;
    result[agentId] = {
      agent_id: agentId,
      loop_probability: 0,
      hallucination_probability: Math.round(halluc * 1000) / 1000,
      context_overflow_risk: 0,
      tool_thrashing_risk: 0,
      reasoning_collapse_risk: Math.round(Math.max(0, (1 - avg) * 0.3) * 1000) / 1000,
      agent_divergence_risk: 0,
      overall_failure_risk: Math.round(overall * 1000) / 1000,
      contributing_signals: halluc > 0.1 ? [{ detector: 'hallucination', signal: `avg confidence: ${avg.toFixed(2)}`, weight: 0.8 }] : [],
      recommended_interventions: halluc > 0.3 ? ['Add retrieval step to re-ground agent'] : [],
    };
  });
  return result;
}

function buildEvent(node: ReturnType<typeof makeNode>): TraceEvent {
  return {
    event_id: node.id,
    timestamp: node.timestamp,
    trace_id: '',
    agent_id: node.agent_id,
    agent_name: node.agent_name,
    event_type: node.event_type,
    model_name: node.model_name,
    framework: node.framework,
    input_tokens: Math.floor(node.tokens * 0.4),
    output_tokens: Math.floor(node.tokens * 0.6),
    latency_ms: node.latency_ms,
    cost_usd: node.tokens > 0 ? node.tokens * 0.000005 : null,
    confidence_score: node.confidence,
    input_preview: node.input_preview,
    output_preview: node.output_preview,
    tool_name: node.event_type === 'tool_call' ? node.output_preview.split(':')[0] : null,
    tool_input: {},
    tool_output: node.event_type === 'tool_call' ? node.output_preview : '',
    tool_success: true,
    error_message: null,
    tags: {},
    metadata: {},
  };
}

export function getDemoTraces(): TraceOverview[] {
  return DEMO_TRACES;
}

export function getDemoGraph(traceId: string): TraceGraph | null {
  return DEMO_GRAPHS[traceId] || null;
}

export function getDemoTimeline(traceId: string): TimelineEntry[] {
  const graph = DEMO_GRAPHS[traceId];
  if (!graph) return [];
  return buildTimeline(graph);
}

export function getDemoRisk(traceId: string): Record<string, RiskScores> {
  const graph = DEMO_GRAPHS[traceId];
  if (!graph) return {};
  return buildRisk(graph);
}

export function getDemoEvent(eventId: string): TraceEvent | null {
  for (const graph of Object.values(DEMO_GRAPHS)) {
    const node = graph.nodes.find(n => n.id === eventId);
    if (node) return buildEvent(node);
  }
  return null;
}
