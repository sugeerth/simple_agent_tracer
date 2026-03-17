import React, { useMemo } from 'react';
import type { TimelineEntry } from '../hooks/useTraceData';

const AGENT_COLORS = [
  '#4a9eff', '#4ade80', '#a78bfa', '#fb923c', '#f472b6',
  '#22d3ee', '#fbbf24', '#34d399', '#c084fc', '#f87171',
];

interface Props {
  entries: TimelineEntry[];
  onEventClick: (eventId: string) => void;
}

export default function Timeline({ entries, onEventClick }: Props) {
  const { agents, maxTime, agentColorMap } = useMemo(() => {
    const agentMap = new Map<string, TimelineEntry[]>();
    let maxTime = 0;

    entries.forEach(e => {
      const key = e.agent_name || e.agent_id;
      if (!agentMap.has(key)) agentMap.set(key, []);
      agentMap.get(key)!.push(e);
      maxTime = Math.max(maxTime, e.start_ms + e.duration_ms);
    });

    const agents = Array.from(agentMap.entries());
    const agentColorMap = new Map<string, string>();
    agents.forEach(([name], i) => {
      agentColorMap.set(name, AGENT_COLORS[i % AGENT_COLORS.length]);
    });

    return { agents, maxTime: maxTime || 1, agentColorMap };
  }, [entries]);

  if (entries.length === 0) {
    return (
      <div className="empty-state">
        <h2>No timeline data</h2>
        <p>Timeline will appear when trace events are loaded.</p>
      </div>
    );
  }

  return (
    <div className="timeline-container">
      {/* Time axis */}
      <div style={{ display: 'flex', marginBottom: 12 }}>
        <div style={{ width: 160, minWidth: 160 }} />
        <div style={{ flex: 1, display: 'flex', justifyContent: 'space-between', fontSize: 10, color: 'var(--text-muted)' }}>
          <span>0ms</span>
          <span>{(maxTime / 4).toFixed(0)}ms</span>
          <span>{(maxTime / 2).toFixed(0)}ms</span>
          <span>{(maxTime * 3 / 4).toFixed(0)}ms</span>
          <span>{maxTime.toFixed(0)}ms</span>
        </div>
      </div>

      {agents.map(([agentName, events]) => {
        const color = agentColorMap.get(agentName) || '#4a9eff';
        return (
          <div key={agentName} className="timeline-agent">
            <div className="timeline-label" style={{ color }}>
              {agentName}
            </div>
            <div className="timeline-bar-container">
              {events.map(e => {
                const left = (e.start_ms / maxTime) * 100;
                const width = Math.max((e.duration_ms / maxTime) * 100, 0.5);
                const riskOpacity = e.risk_score > 0.3 ? 0.9 : 0.7;
                const bgColor = e.risk_score > 0.5
                  ? `rgba(248, 113, 113, ${riskOpacity})`
                  : `${color}`;

                return (
                  <div
                    key={e.event_id}
                    className="timeline-bar"
                    title={`${e.label}\n${e.duration_ms.toFixed(0)}ms\nRisk: ${(e.risk_score * 100).toFixed(0)}%`}
                    style={{
                      left: `${left}%`,
                      width: `${width}%`,
                      background: bgColor,
                      opacity: riskOpacity,
                    }}
                    onClick={() => onEventClick(e.event_id)}
                  >
                    {width > 5 && (
                      <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                        {e.event_type.replace(/_/g, ' ')}
                      </span>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        );
      })}

      {/* Legend */}
      <div style={{ marginTop: 24, display: 'flex', gap: 16, fontSize: 11, color: 'var(--text-muted)' }}>
        <span>Bar width = duration</span>
        <span style={{ color: 'var(--accent-red)' }}>Red = high risk</span>
        <span>Click any bar for details</span>
      </div>
    </div>
  );
}
