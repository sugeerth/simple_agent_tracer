import React, { useMemo } from 'react';
import type { GraphNode } from '../hooks/useTraceData';

const SCORE_COLORS: Record<string, string> = {
  confidence: 'var(--accent-blue)',
  high: 'var(--accent-green)',
  medium: 'var(--accent-yellow)',
  low: 'var(--accent-red)',
};

function scoreColor(value: number): string {
  if (value >= 0.8) return 'var(--accent-green)';
  if (value >= 0.6) return 'var(--accent-yellow)';
  return 'var(--accent-red)';
}

interface Props {
  nodes: GraphNode[];
}

export default function JudgeDashboard({ nodes }: Props) {
  const { agentScores, overallStats } = useMemo(() => {
    // Group confidence scores by agent
    const byAgent = new Map<string, { scores: number[]; totalTokens: number; totalLatency: number; eventCount: number }>();

    nodes.forEach(n => {
      if (!n.agent_id || n.agent_id === 'system') return;
      const name = n.agent_name || n.agent_id;
      if (!byAgent.has(name)) {
        byAgent.set(name, { scores: [], totalTokens: 0, totalLatency: 0, eventCount: 0 });
      }
      const entry = byAgent.get(name)!;
      if (n.confidence !== null) entry.scores.push(n.confidence);
      entry.totalTokens += n.tokens;
      entry.totalLatency += n.latency_ms;
      entry.eventCount++;
    });

    const agentScores = Array.from(byAgent.entries()).map(([name, data]) => {
      const avg = data.scores.length > 0
        ? data.scores.reduce((a, b) => a + b, 0) / data.scores.length
        : null;
      return {
        name,
        avgConfidence: avg,
        minConfidence: data.scores.length > 0 ? Math.min(...data.scores) : null,
        maxConfidence: data.scores.length > 0 ? Math.max(...data.scores) : null,
        totalTokens: data.totalTokens,
        totalLatency: data.totalLatency,
        eventCount: data.eventCount,
      };
    });

    const allScores = nodes.filter(n => n.confidence !== null).map(n => n.confidence!);
    const overallStats = {
      avgConfidence: allScores.length > 0 ? allScores.reduce((a, b) => a + b, 0) / allScores.length : null,
      totalTokens: nodes.reduce((sum, n) => sum + n.tokens, 0),
      totalLatency: nodes.reduce((sum, n) => sum + n.latency_ms, 0),
      eventCount: nodes.length,
    };

    return { agentScores, overallStats };
  }, [nodes]);

  if (nodes.length === 0) {
    return (
      <div className="empty-state">
        <h2>No judge data</h2>
        <p>Quality scores will appear when trace events include confidence scores.</p>
      </div>
    );
  }

  return (
    <div className="judge-container">
      {/* Overall stats */}
      <div style={{ display: 'flex', gap: 24, marginBottom: 24 }}>
        <div className="risk-card" style={{ flex: 1 }}>
          <div className="agent-header">
            <div className="agent-name">Overall Quality</div>
            {overallStats.avgConfidence !== null && (
              <div className="risk-overall" style={{ color: scoreColor(overallStats.avgConfidence) }}>
                {(overallStats.avgConfidence * 100).toFixed(0)}%
              </div>
            )}
          </div>
          <div style={{ display: 'flex', gap: 24, fontSize: 12, color: 'var(--text-secondary)' }}>
            <div>
              <span style={{ color: 'var(--text-muted)' }}>Tokens: </span>
              <span style={{ fontWeight: 600 }}>{overallStats.totalTokens.toLocaleString()}</span>
            </div>
            <div>
              <span style={{ color: 'var(--text-muted)' }}>Latency: </span>
              <span style={{ fontWeight: 600 }}>{(overallStats.totalLatency / 1000).toFixed(1)}s</span>
            </div>
            <div>
              <span style={{ color: 'var(--text-muted)' }}>Events: </span>
              <span style={{ fontWeight: 600 }}>{overallStats.eventCount}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Per-agent scores */}
      <h3 style={{ fontSize: 13, fontWeight: 700, marginBottom: 12, color: 'var(--text-secondary)' }}>
        Per-Agent Quality Scores
      </h3>

      <div className="judge-scores">
        {agentScores.map(agent => (
          <div key={agent.name} className="judge-score-row">
            <div className="judge-score-label">{agent.name}</div>
            <div className="judge-score-bar-track">
              {agent.avgConfidence !== null && (
                <div
                  className="judge-score-bar-fill"
                  style={{
                    width: `${agent.avgConfidence * 100}%`,
                    background: scoreColor(agent.avgConfidence),
                  }}
                />
              )}
            </div>
            <div
              className="judge-score-value"
              style={{ color: agent.avgConfidence !== null ? scoreColor(agent.avgConfidence) : 'var(--text-muted)' }}
            >
              {agent.avgConfidence !== null ? `${(agent.avgConfidence * 100).toFixed(0)}%` : '--'}
            </div>
          </div>
        ))}
      </div>

      {/* Detailed table */}
      <div style={{ marginTop: 24 }}>
        <h3 style={{ fontSize: 13, fontWeight: 700, marginBottom: 12, color: 'var(--text-secondary)' }}>
          Agent Details
        </h3>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ borderBottom: '1px solid var(--border)' }}>
                <th style={{ textAlign: 'left', padding: '8px 12px', color: 'var(--text-muted)', fontWeight: 600 }}>Agent</th>
                <th style={{ textAlign: 'right', padding: '8px 12px', color: 'var(--text-muted)', fontWeight: 600 }}>Avg Confidence</th>
                <th style={{ textAlign: 'right', padding: '8px 12px', color: 'var(--text-muted)', fontWeight: 600 }}>Min</th>
                <th style={{ textAlign: 'right', padding: '8px 12px', color: 'var(--text-muted)', fontWeight: 600 }}>Max</th>
                <th style={{ textAlign: 'right', padding: '8px 12px', color: 'var(--text-muted)', fontWeight: 600 }}>Tokens</th>
                <th style={{ textAlign: 'right', padding: '8px 12px', color: 'var(--text-muted)', fontWeight: 600 }}>Latency</th>
                <th style={{ textAlign: 'right', padding: '8px 12px', color: 'var(--text-muted)', fontWeight: 600 }}>Events</th>
              </tr>
            </thead>
            <tbody>
              {agentScores.map(agent => (
                <tr key={agent.name} style={{ borderBottom: '1px solid var(--border)' }}>
                  <td style={{ padding: '8px 12px', fontWeight: 600 }}>{agent.name}</td>
                  <td style={{ padding: '8px 12px', textAlign: 'right', color: agent.avgConfidence !== null ? scoreColor(agent.avgConfidence) : 'var(--text-muted)' }}>
                    {agent.avgConfidence !== null ? `${(agent.avgConfidence * 100).toFixed(1)}%` : '--'}
                  </td>
                  <td style={{ padding: '8px 12px', textAlign: 'right', color: 'var(--text-secondary)' }}>
                    {agent.minConfidence !== null ? `${(agent.minConfidence * 100).toFixed(0)}%` : '--'}
                  </td>
                  <td style={{ padding: '8px 12px', textAlign: 'right', color: 'var(--text-secondary)' }}>
                    {agent.maxConfidence !== null ? `${(agent.maxConfidence * 100).toFixed(0)}%` : '--'}
                  </td>
                  <td style={{ padding: '8px 12px', textAlign: 'right', fontVariantNumeric: 'tabular-nums' }}>
                    {agent.totalTokens.toLocaleString()}
                  </td>
                  <td style={{ padding: '8px 12px', textAlign: 'right', fontVariantNumeric: 'tabular-nums' }}>
                    {agent.totalLatency.toFixed(0)}ms
                  </td>
                  <td style={{ padding: '8px 12px', textAlign: 'right' }}>{agent.eventCount}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
