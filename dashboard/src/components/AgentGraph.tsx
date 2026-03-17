import React, { useMemo, useCallback } from 'react';
import {
  ReactFlow,
  Node,
  Edge,
  Background,
  Controls,
  Handle,
  Position,
  useNodesState,
  useEdgesState,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import type { TraceGraph, GraphNode as GN } from '../hooks/useTraceData';

const EVENT_COLORS: Record<string, string> = {
  llm_call: '#4a9eff',
  tool_call: '#a78bfa',
  agent_decision: '#4ade80',
  agent_start: '#22d3ee',
  agent_end: '#22d3ee',
  chain_start: '#fbbf24',
  chain_end: '#fbbf24',
  planning_step: '#fb923c',
  inter_agent_message: '#f472b6',
  retrieval: '#34d399',
  error: '#f87171',
  judge_evaluation: '#c084fc',
  system_event: '#555570',
  image_transform: '#06b6d4',
  modality_transition: '#06b6d4',
};

function getColor(eventType: string): string {
  return EVENT_COLORS[eventType] || '#555570';
}

function getRiskColor(risk: number): string {
  if (risk > 0.6) return '#f87171';
  if (risk > 0.3) return '#fbbf24';
  return '#4ade80';
}

function AgentNode({ data }: { data: GN }) {
  const color = getColor(data.event_type);
  return (
    <div className="agent-node" style={{ borderColor: data.risk_score > 0.3 ? getRiskColor(data.risk_score) : undefined }}>
      <Handle type="target" position={Position.Top} style={{ background: color, width: 6, height: 6, border: 'none' }} />
      <div className="node-header">
        <div className="node-dot" style={{ background: color }} />
        <div className="node-name">{data.agent_name || data.agent_id}</div>
      </div>
      <div className="node-type">{data.event_type.replace(/_/g, ' ')}</div>
      <div className="node-metrics">
        {data.latency_ms > 0 && <span>{data.latency_ms.toFixed(0)}ms</span>}
        {data.tokens > 0 && <span>{data.tokens} tok</span>}
        {data.confidence !== null && <span>{(data.confidence * 100).toFixed(0)}%</span>}
      </div>
      {data.output_preview && (
        <div className="node-preview">{data.output_preview.slice(0, 80)}</div>
      )}
      <Handle type="source" position={Position.Bottom} style={{ background: color, width: 6, height: 6, border: 'none' }} />
    </div>
  );
}

const nodeTypes = { agentNode: AgentNode };

interface Props {
  graph: TraceGraph;
  onNodeClick: (eventId: string) => void;
}

export default function AgentGraph({ graph, onNodeClick }: Props) {
  const { flowNodes, flowEdges } = useMemo(() => {
    // Layout: simple top-down, grouping by agent
    const agentColumns: Record<string, number> = {};
    let colIdx = 0;
    const rowByNode: Record<string, number> = {};

    // Assign columns by agent, rows by order
    const agentRows: Record<string, number> = {};
    graph.nodes.forEach((n, i) => {
      if (!(n.agent_id in agentColumns)) {
        agentColumns[n.agent_id] = colIdx++;
      }
      if (!(n.agent_id in agentRows)) {
        agentRows[n.agent_id] = 0;
      }
      rowByNode[n.id] = agentRows[n.agent_id]++;
    });

    const xSpacing = 260;
    const ySpacing = 120;
    const xOffset = 40;
    const yOffset = 40;

    const flowNodes: Node[] = graph.nodes.map((n) => ({
      id: n.id,
      type: 'agentNode',
      position: {
        x: xOffset + (agentColumns[n.agent_id] || 0) * xSpacing,
        y: yOffset + (rowByNode[n.id] || 0) * ySpacing,
      },
      data: n,
    }));

    const flowEdges: Edge[] = graph.edges.map((e, i) => ({
      id: `e-${i}`,
      source: e.source,
      target: e.target,
      animated: e.animated || e.edge_type === 'data',
      style: {
        stroke: e.edge_type === 'data' ? '#a78bfa' : '#4a9eff',
        strokeWidth: 1.5,
        opacity: 0.6,
      },
      type: 'smoothstep',
    }));

    return { flowNodes, flowEdges };
  }, [graph]);

  const [nodes, , onNodesChange] = useNodesState(flowNodes);
  const [edges, , onEdgesChange] = useEdgesState(flowEdges);

  const handleNodeClick = useCallback((_: any, node: Node) => {
    onNodeClick(node.id);
  }, [onNodeClick]);

  return (
    <div style={{ width: '100%', height: '100%' }}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onNodeClick={handleNodeClick}
        nodeTypes={nodeTypes}
        fitView
        fitViewOptions={{ padding: 0.2 }}
        minZoom={0.3}
        maxZoom={2}
        defaultEdgeOptions={{ type: 'smoothstep' }}
        proOptions={{ hideAttribution: true }}
      >
        <Background color="#1a1a2e" gap={20} size={1} />
        <Controls
          style={{ background: '#1a1a2e', borderColor: '#2a2a45', borderRadius: 8 }}
        />
      </ReactFlow>
    </div>
  );
}
