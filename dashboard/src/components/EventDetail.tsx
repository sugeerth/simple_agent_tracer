import React from 'react';
import { useEvent } from '../hooks/useTraceData';

interface Props {
  eventId: string;
  onClose: () => void;
}

const TYPE_LABELS: Record<string, string> = {
  llm_call: 'LLM Call',
  tool_call: 'Tool Call',
  agent_decision: 'Agent Decision',
  agent_start: 'Agent Start',
  agent_end: 'Agent End',
  chain_start: 'Chain Start',
  chain_end: 'Chain End',
  planning_step: 'Planning Step',
  inter_agent_message: 'Inter-Agent Message',
  retrieval: 'Retrieval',
  error: 'Error',
  judge_evaluation: 'Judge Evaluation',
  system_event: 'System Event',
  image_transform: 'Image Transform',
  modality_transition: 'Modality Transition',
};

export default function EventDetail({ eventId, onClose }: Props) {
  const event = useEvent(eventId);

  if (!event) {
    return (
      <div className="detail-panel">
        <div className="detail-header">
          <h3>Loading...</h3>
          <button className="detail-close" onClick={onClose}>x</button>
        </div>
      </div>
    );
  }

  return (
    <div className="detail-panel">
      <div className="detail-header">
        <h3>{TYPE_LABELS[event.event_type] || event.event_type}</h3>
        <button className="detail-close" onClick={onClose}>x</button>
      </div>

      <div className="detail-body">
        {/* Identity */}
        <div className="detail-section">
          <h4>Event</h4>
          <div className="detail-row">
            <span className="label">Event ID</span>
            <span className="value" style={{ fontSize: 10 }}>{event.event_id.slice(0, 12)}...</span>
          </div>
          <div className="detail-row">
            <span className="label">Agent</span>
            <span className="value">{event.agent_name || event.agent_id}</span>
          </div>
          <div className="detail-row">
            <span className="label">Type</span>
            <span className="value">{TYPE_LABELS[event.event_type] || event.event_type}</span>
          </div>
          <div className="detail-row">
            <span className="label">Framework</span>
            <span className="value">{event.framework}</span>
          </div>
          <div className="detail-row">
            <span className="label">Timestamp</span>
            <span className="value" style={{ fontSize: 10 }}>{event.timestamp}</span>
          </div>
        </div>

        {/* Model info */}
        {event.model_name && (
          <div className="detail-section">
            <h4>Model</h4>
            <div className="detail-row">
              <span className="label">Model</span>
              <span className="value">{event.model_name}</span>
            </div>
          </div>
        )}

        {/* Metrics */}
        <div className="detail-section">
          <h4>Metrics</h4>
          <div className="detail-row">
            <span className="label">Latency</span>
            <span className="value">{event.latency_ms.toFixed(0)}ms</span>
          </div>
          {(event.input_tokens > 0 || event.output_tokens > 0) && (
            <>
              <div className="detail-row">
                <span className="label">Input Tokens</span>
                <span className="value">{event.input_tokens.toLocaleString()}</span>
              </div>
              <div className="detail-row">
                <span className="label">Output Tokens</span>
                <span className="value">{event.output_tokens.toLocaleString()}</span>
              </div>
            </>
          )}
          {event.cost_usd !== null && event.cost_usd > 0 && (
            <div className="detail-row">
              <span className="label">Cost</span>
              <span className="value">${event.cost_usd.toFixed(4)}</span>
            </div>
          )}
          {event.confidence_score !== null && (
            <div className="detail-row">
              <span className="label">Confidence</span>
              <span className="value" style={{
                color: event.confidence_score >= 0.8 ? 'var(--accent-green)' :
                       event.confidence_score >= 0.6 ? 'var(--accent-yellow)' : 'var(--accent-red)'
              }}>
                {(event.confidence_score * 100).toFixed(0)}%
              </span>
            </div>
          )}
        </div>

        {/* Tool info */}
        {event.tool_name && (
          <div className="detail-section">
            <h4>Tool Call</h4>
            <div className="detail-row">
              <span className="label">Tool</span>
              <span className="value">{event.tool_name}</span>
            </div>
            <div className="detail-row">
              <span className="label">Success</span>
              <span className="value" style={{ color: event.tool_success ? 'var(--accent-green)' : 'var(--accent-red)' }}>
                {event.tool_success ? 'Yes' : 'No'}
              </span>
            </div>
            {Object.keys(event.tool_input).length > 0 && (
              <>
                <h4 style={{ marginTop: 8 }}>Tool Input</h4>
                <div className="detail-text">{JSON.stringify(event.tool_input, null, 2)}</div>
              </>
            )}
            {event.tool_output && (
              <>
                <h4 style={{ marginTop: 8 }}>Tool Output</h4>
                <div className="detail-text">{event.tool_output}</div>
              </>
            )}
          </div>
        )}

        {/* Error */}
        {event.error_message && (
          <div className="detail-section">
            <h4>Error</h4>
            <div className="detail-text" style={{ borderColor: 'var(--accent-red)', color: 'var(--accent-red)' }}>
              {event.error_message}
            </div>
          </div>
        )}

        {/* Input */}
        {event.input_preview && (
          <div className="detail-section">
            <h4>Input</h4>
            <div className="detail-text">{event.input_preview}</div>
          </div>
        )}

        {/* Output */}
        {event.output_preview && (
          <div className="detail-section">
            <h4>Output</h4>
            <div className="detail-text">{event.output_preview}</div>
          </div>
        )}

        {/* Tags */}
        {Object.keys(event.tags).length > 0 && (
          <div className="detail-section">
            <h4>Tags</h4>
            {Object.entries(event.tags).map(([k, v]) => (
              <div key={k} className="detail-row">
                <span className="label">{k}</span>
                <span className="value">{v}</span>
              </div>
            ))}
          </div>
        )}

        {/* Metadata */}
        {Object.keys(event.metadata).length > 0 && (
          <div className="detail-section">
            <h4>Metadata</h4>
            <div className="detail-text">{JSON.stringify(event.metadata, null, 2)}</div>
          </div>
        )}
      </div>
    </div>
  );
}
