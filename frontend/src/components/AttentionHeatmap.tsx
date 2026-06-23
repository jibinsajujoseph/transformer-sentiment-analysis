import React, { useState } from 'react';
import './AttentionHeatmap.css';

interface AttentionHeatmapProps {
  tokens: string[];
  attention: number[];
}

const AttentionHeatmap: React.FC<AttentionHeatmapProps> = ({ tokens, attention }) => {
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);

  if (!tokens || !attention || tokens.length !== attention.length || tokens.length === 0) {
    return null;
  }

  // Create a normalized array just in case the backend normalization isn't perfect [0, 1]
  const minVal = Math.min(...attention);
  const maxVal = Math.max(...attention);
  const range = maxVal - minVal || 1;
  const normalizedAttention = attention.map(v => (v - minVal) / range);

  // Helper to calculate color based on attention weight
  const getBackgroundColor = (weight: number) => {
    // Colors from cool slate to warm violet/cyan
    // 0 = transparent, 1 = solid accent color
    // We use rgba with the violet accent base: 129, 140, 248
    return `rgba(129, 140, 248, ${weight * 0.8})`; // Max opacity 0.8 for readability
  };

  return (
    <div className="attention-section glass-panel fade-in">
      <div className="attention-header">
        <h3 className="attention-title">Self-Attention Weights</h3>
        <p className="attention-subtitle">
          Visualizing which words the custom Transformer focused on during prediction.
        </p>
      </div>

      <div className="attention-container">
        {tokens.map((token, idx) => {
          const weight = normalizedAttention[idx];
          const isHovered = hoveredIdx === idx;
          
          return (
            <span
              key={idx}
              className={`attention-token ${isHovered ? 'hovered' : ''}`}
              style={{ backgroundColor: getBackgroundColor(weight) }}
              onMouseEnter={() => setHoveredIdx(idx)}
              onMouseLeave={() => setHoveredIdx(null)}
            >
              {token}
              {isHovered && (
                <div className="attention-tooltip">
                  Weight: {(attention[idx] * 100).toFixed(1)}%
                </div>
              )}
            </span>
          );
        })}
      </div>

      <div className="attention-legend">
        <span className="legend-label">Low Attention</span>
        <div className="legend-gradient"></div>
        <span className="legend-label">High Attention</span>
      </div>
    </div>
  );
};

export default AttentionHeatmap;
