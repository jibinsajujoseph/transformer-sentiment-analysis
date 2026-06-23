import React, { useEffect, useState } from 'react';
import './ModelCard.css';

interface ModelCardProps {
  title: string;
  architecture: string;
  params: string;
  result: {
    label: 'positive' | 'negative';
    confidence: number;
    latency_ms: number;
  } | null;
  isLoading: boolean;
  isPrimary?: boolean;
}

const ModelCard: React.FC<ModelCardProps> = ({ 
  title, 
  architecture, 
  params, 
  result, 
  isLoading,
  isPrimary = false
}) => {
  const [animatedConfidence, setAnimatedConfidence] = useState(0);

  useEffect(() => {
    if (result) {
      // Small delay for the bar animation to trigger after rendering
      const timer = setTimeout(() => {
        setAnimatedConfidence(result.confidence * 100);
      }, 100);
      return () => clearTimeout(timer);
    } else {
      setAnimatedConfidence(0);
    }
  }, [result]);

  return (
    <div className={`model-card glass-panel ${isPrimary ? 'primary-card' : ''}`}>
      <div className="card-header">
        <div className="title-group">
          <h3 className="model-title">{title}</h3>
          <span className="model-params">{params}</span>
        </div>
        <p className="model-architecture">{architecture}</p>
      </div>

      <div className="card-content">
        {isLoading ? (
          <div className="loading-skeleton">
            <div className="skeleton-line title"></div>
            <div className="skeleton-line"></div>
            <div className="skeleton-line short"></div>
          </div>
        ) : result ? (
          <div className="result-container fade-in">
            <div className="result-header">
              <span className="result-label">Prediction</span>
              <span className="latency-chip">{result.latency_ms.toFixed(1)} ms</span>
            </div>
            
            <div className={`prediction-badge ${result.label}`}>
              {result.label.toUpperCase()}
            </div>

            <div className="confidence-section">
              <div className="confidence-header">
                <span className="confidence-label">Confidence</span>
                <span className="confidence-value">{(result.confidence * 100).toFixed(1)}%</span>
              </div>
              <div className="confidence-track">
                <div 
                  className={`confidence-fill ${result.label}`} 
                  style={{ width: `${animatedConfidence}%` }}
                ></div>
              </div>
            </div>
          </div>
        ) : (
          <div className="empty-state">
            <p>Awaiting input...</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default ModelCard;
