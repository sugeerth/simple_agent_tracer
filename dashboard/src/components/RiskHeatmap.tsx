import React from 'react';
import type { RiskScores } from '../hooks/useTraceData';

function riskColor(value: number): string {
  if (value > 0.6) return 'var(--accent-red)';
  if (value > 0.3) return 'var(--accent-orange)';
  if (value > 0.1) return 'var(--accent-yellow)';
  return 'var(--accent-green)';
}

const RISK_LABELS: [keyof RiskScores, string][] = [
  ['loop_probability', 'Infinite Loop'],
  ['hallucination_probability', 'Hallucination'],
  ['context_overflow_risk', 'Context Overflow'],
  ['tool_thrashing_risk', 'Tool Thrashing'],
  ['reasoning_collapse_risk', 'Reasoning Collapse'],
  ['agent_divergence_risk', 'Agent Divergence'],
];

interface Props {
  risk: Record<string, RiskScores>;
}

export default function RiskHeatmap({ risk }: Props) {
  const agents = Object.values(risk).filter(r => r.agent_id && r.agent_id !== 'system');

  if (agents.length === 0) {
    return (
      <div className="empty-state">
        <h2>No risk data</h2>
        <p>Risk heatmap will appear when agent traces are analyzed.</p>
      </div>
    );
  }

  // Sort by overall risk descending
  const sorted = [...agents].sort((a, b) => b.overall_failure_risk - a.overall_failure_risk);

  return (
    <div className="risk-grid">
      {sorted.map(agent => (
        <div key={agent.agent_id} className="risk-card">
          <div className="agent-header">
            <div className="agent-name">{agent.agent_id}</div>
            <div
              className="risk-overall"
              style={{ color: riskColor(agent.overall_failure_risk) }}
            >
              {(agent.overall_failure_risk * 100).toFixed(0)}%
            </div>
          </div>

          {RISK_LABELS.map(([key, label]) => {
            const value = agent[key] as number;
            return (
              <div key={key} className="risk-bar-row">
                <div className="risk-bar-label">{label}</div>
                <div className="risk-bar-track">
                  <div
                    className="risk-bar-fill"
                    style={{
                      width: `${value * 100}%`,
                      background: riskColor(value),
                    }}
                  />
                </div>
                <div className="risk-bar-value" style={{ color: riskColor(value) }}>
                  {(value * 100).toFixed(0)}%
                </div>
              </div>
            );
          })}

          {agent.recommended_interventions.length > 0 && (
            <div style={{ marginTop: 12, fontSize: 11 }}>
              <div style={{ color: 'var(--accent-orange)', fontWeight: 600, marginBottom: 4 }}>
                Recommended Interventions:
              </div>
              {agent.recommended_interventions.map((int, i) => (
                <div key={i} style={{ color: 'var(--text-secondary)', marginBottom: 2 }}>
                  {int}
                </div>
              ))}
            </div>
          )}
        </div>
      ))}
    </div>
  );
}
