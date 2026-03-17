import React, { useState } from 'react';
import { useTraceList, useTraceGraph, useTimeline, useRisk } from './hooks/useTraceData';
import AgentGraph from './components/AgentGraph';
import Timeline from './components/Timeline';
import RiskHeatmap from './components/RiskHeatmap';
import JudgeDashboard from './components/JudgeDashboard';
import EventDetail from './components/EventDetail';

type View = 'graph' | 'timeline' | 'risk' | 'judges';

export default function App() {
  const [selectedTrace, setSelectedTrace] = useState<string | null>(null);
  const [selectedEvent, setSelectedEvent] = useState<string | null>(null);
  const [activeView, setActiveView] = useState<View>('graph');

  const { traces, loading, demoMode } = useTraceList();
  const { graph } = useTraceGraph(selectedTrace);
  const timeline = useTimeline(selectedTrace);
  const risk = useRisk(selectedTrace);

  const currentTrace = traces.find(t => t.trace_id === selectedTrace);

  return (
    <div className="app-layout">
      {/* Sidebar: Trace List */}
      <div className="sidebar">
        <div className="sidebar-header">
          <h1>OMNISCOPE</h1>
          <div className="subtitle">Multi-Agent Observability{demoMode && <span style={{ marginLeft: 8, color: 'var(--accent-yellow)', fontSize: 10 }}>DEMO</span>}</div>
        </div>
        <div className="trace-list">
          {loading && <div style={{ padding: 20, color: 'var(--text-muted)', fontSize: 12 }}>Loading traces...</div>}
          {!loading && traces.length === 0 && (
            <div style={{ padding: 20, color: 'var(--text-muted)', fontSize: 12, lineHeight: 1.6 }}>
              No traces yet. Run the demo:<br />
              <code style={{ fontSize: 11 }}>python demo.py</code>
            </div>
          )}
          {traces.map(trace => (
            <div
              key={trace.trace_id}
              className={`trace-item ${selectedTrace === trace.trace_id ? 'active' : ''}`}
              onClick={() => { setSelectedTrace(trace.trace_id); setSelectedEvent(null); }}
            >
              <div className="name">{trace.trace_name || trace.trace_id.slice(0, 8)}</div>
              <div className="meta">
                <span className={`badge badge-${trace.status}`}>{trace.status}</span>
                <span>{trace.event_count} events</span>
                <span>{(trace.total_tokens / 1000).toFixed(1)}k tok</span>
                {trace.total_cost > 0 && <span>${trace.total_cost.toFixed(3)}</span>}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Main Content */}
      <div className="main-content">
        {currentTrace && (
          <>
            {/* Stats bar */}
            <div className="stats-bar">
              <div className="stat">
                <span className="stat-label">Agents</span>
                <span className="stat-value">{currentTrace.agent_ids.length}</span>
              </div>
              <div className="stat">
                <span className="stat-label">Events</span>
                <span className="stat-value">{currentTrace.event_count}</span>
              </div>
              <div className="stat">
                <span className="stat-label">Tokens</span>
                <span className="stat-value">{currentTrace.total_tokens.toLocaleString()}</span>
              </div>
              {currentTrace.total_cost > 0 && (
                <div className="stat">
                  <span className="stat-label">Cost</span>
                  <span className="stat-value">${currentTrace.total_cost.toFixed(4)}</span>
                </div>
              )}
              <div className="stat">
                <span className="stat-label">Status</span>
                <span className="stat-value" style={{ color: currentTrace.status === 'completed' ? 'var(--accent-green)' : currentTrace.status === 'failed' ? 'var(--accent-red)' : 'var(--accent-blue)' }}>
                  {currentTrace.status}
                </span>
              </div>
            </div>

            {/* Tab bar */}
            <div className="tab-bar">
              {(['graph', 'timeline', 'risk', 'judges'] as View[]).map(view => (
                <button
                  key={view}
                  className={`tab ${activeView === view ? 'active' : ''}`}
                  onClick={() => setActiveView(view)}
                >
                  {view === 'graph' && 'Agent Graph'}
                  {view === 'timeline' && 'Timeline'}
                  {view === 'risk' && 'Risk Heatmap'}
                  {view === 'judges' && 'Judge Scores'}
                </button>
              ))}
            </div>
          </>
        )}

        {/* View */}
        <div className="view-container">
          {!selectedTrace ? (
            <div className="empty-state">
              <h2>Select a trace to begin</h2>
              <p>
                Choose a trace from the sidebar, or generate demo data:
              </p>
              <p>
                <code>python -m uvicorn omniscope.server.app:app --port 8781</code>
              </p>
              <p>
                <code>python demo.py</code>
              </p>
            </div>
          ) : (
            <>
              {activeView === 'graph' && graph && (
                <AgentGraph graph={graph} onNodeClick={setSelectedEvent} />
              )}
              {activeView === 'timeline' && (
                <Timeline entries={timeline} onEventClick={setSelectedEvent} />
              )}
              {activeView === 'risk' && (
                <RiskHeatmap risk={risk} />
              )}
              {activeView === 'judges' && graph && (
                <JudgeDashboard nodes={graph.nodes} />
              )}
            </>
          )}
        </div>

        {/* Event detail panel */}
        {selectedEvent && (
          <EventDetail eventId={selectedEvent} onClose={() => setSelectedEvent(null)} />
        )}
      </div>
    </div>
  );
}
