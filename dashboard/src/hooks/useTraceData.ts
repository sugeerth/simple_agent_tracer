import { useState, useEffect, useCallback } from 'react';
import { getDemoTraces, getDemoGraph, getDemoTimeline, getDemoRisk, getDemoEvent } from './demoData';

const API = '/api/v1';

export interface TraceOverview {
  trace_id: string;
  trace_name: string;
  framework: string;
  started_at: string;
  event_count: number;
  agent_ids: string[];
  total_tokens: number;
  total_cost: number;
  status: string;
  composite_score?: number | null;
  risk_scores?: Record<string, unknown>;
}

export interface GraphNode {
  id: string;
  agent_id: string;
  agent_name: string;
  event_type: string;
  label: string;
  timestamp: string;
  latency_ms: number;
  confidence: number | null;
  risk_score: number;
  state: string;
  input_preview: string;
  output_preview: string;
  model_name: string | null;
  tokens: number;
  framework: string;
}

export interface GraphEdge {
  source: string;
  target: string;
  edge_type: string;
  animated: boolean;
}

export interface TraceGraph {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

export interface TimelineEntry {
  event_id: string;
  agent_id: string;
  agent_name: string;
  event_type: string;
  start_ms: number;
  duration_ms: number;
  label: string;
  risk_score: number;
  confidence: number | null;
}

export interface RiskScores {
  agent_id: string;
  loop_probability: number;
  hallucination_probability: number;
  context_overflow_risk: number;
  tool_thrashing_risk: number;
  reasoning_collapse_risk: number;
  agent_divergence_risk: number;
  overall_failure_risk: number;
  contributing_signals: Array<{ detector: string; signal: string; weight: number }>;
  recommended_interventions: string[];
}

export interface TraceEvent {
  event_id: string;
  timestamp: string;
  trace_id: string;
  agent_id: string;
  agent_name: string;
  event_type: string;
  model_name: string | null;
  framework: string;
  input_tokens: number;
  output_tokens: number;
  latency_ms: number;
  cost_usd: number | null;
  confidence_score: number | null;
  input_preview: string;
  output_preview: string;
  tool_name: string | null;
  tool_input: Record<string, unknown>;
  tool_output: string;
  tool_success: boolean;
  error_message: string | null;
  tags: Record<string, string>;
  metadata: Record<string, unknown>;
}

async function fetchJson<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

export function useTraceList() {
  const [traces, setTraces] = useState<TraceOverview[]>([]);
  const [loading, setLoading] = useState(true);
  const [demoMode, setDemoMode] = useState(false);

  const refresh = useCallback(async () => {
    try {
      const data = await fetchJson<TraceOverview[]>(`${API}/traces?limit=50`);
      setTraces(data);
      setDemoMode(false);
    } catch {
      // API unavailable -- use demo data (GitHub Pages mode)
      if (traces.length === 0) {
        setTraces(getDemoTraces());
        setDemoMode(true);
      }
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
    const interval = setInterval(refresh, 5000);
    return () => clearInterval(interval);
  }, [refresh]);

  return { traces, loading, refresh, demoMode };
}

export function useTraceGraph(traceId: string | null) {
  const [graph, setGraph] = useState<TraceGraph | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!traceId) { setGraph(null); return; }
    setLoading(true);
    fetchJson<TraceGraph>(`${API}/traces/${traceId}/graph`)
      .then(setGraph)
      .catch(() => {
        // Fallback to demo
        setGraph(getDemoGraph(traceId));
      })
      .finally(() => setLoading(false));
  }, [traceId]);

  return { graph, loading };
}

export function useTimeline(traceId: string | null) {
  const [timeline, setTimeline] = useState<TimelineEntry[]>([]);

  useEffect(() => {
    if (!traceId) { setTimeline([]); return; }
    fetchJson<TimelineEntry[]>(`${API}/traces/${traceId}/timeline`)
      .then(setTimeline)
      .catch(() => setTimeline(getDemoTimeline(traceId)));
  }, [traceId]);

  return timeline;
}

export function useRisk(traceId: string | null) {
  const [risk, setRisk] = useState<Record<string, RiskScores>>({});

  useEffect(() => {
    if (!traceId) { setRisk({}); return; }
    fetchJson<Record<string, RiskScores>>(`${API}/traces/${traceId}/risk`)
      .then(setRisk)
      .catch(() => setRisk(getDemoRisk(traceId)));
  }, [traceId]);

  return risk;
}

export function useEvent(eventId: string | null) {
  const [event, setEvent] = useState<TraceEvent | null>(null);

  useEffect(() => {
    if (!eventId) { setEvent(null); return; }
    fetchJson<TraceEvent>(`${API}/events/${eventId}`)
      .then(setEvent)
      .catch(() => setEvent(getDemoEvent(eventId)));
  }, [eventId]);

  return event;
}
