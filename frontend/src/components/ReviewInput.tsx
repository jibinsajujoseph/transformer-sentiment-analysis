import React, { useState } from 'react';
import './ReviewInput.css';

interface ReviewInputProps {
  onAnalyze: (text: string) => void;
  isLoading: boolean;
}

const SAMPLE_REVIEWS = [
  "This movie was absolutely fantastic! The acting was superb and the plot kept me engaged throughout.",
  "Terrible film. Boring, predictable, and a complete waste of time. I want my money back.",
  "It was okay, nothing special. Some parts were good but overall pretty average.",
];

const ReviewInput: React.FC<ReviewInputProps> = ({ onAnalyze, isLoading }) => {
  const [text, setText] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (text.trim() && !isLoading) {
      onAnalyze(text.trim());
    }
  };

  const loadSample = (sample: string) => {
    setText(sample);
  };

  return (
    <section className="review-section">
      <form onSubmit={handleSubmit} className="review-form glass-panel">
        <div className="input-header">
          <h2>Enter a Movie Review</h2>
          <span className="char-count">{text.length} chars</span>
        </div>
        
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Type or paste a movie review here..."
          className="review-textarea"
          rows={6}
          disabled={isLoading}
        />
        
        <div className="form-actions">
          <div className="sample-buttons">
            <span className="sample-label">Try a sample:</span>
            {SAMPLE_REVIEWS.map((sample, idx) => (
              <button
                key={idx}
                type="button"
                className="sample-btn"
                onClick={() => loadSample(sample)}
                disabled={isLoading}
                title={sample}
              >
                Sample {idx + 1}
              </button>
            ))}
          </div>
          
          <button 
            type="submit" 
            className={`analyze-btn ${isLoading ? 'loading' : ''}`}
            disabled={!text.trim() || isLoading}
          >
            {isLoading ? (
              <>
                <span className="spinner"></span>
                Analyzing...
              </>
            ) : (
              'Analyze Sentiment'
            )}
          </button>
        </div>
      </form>
    </section>
  );
};

export default ReviewInput;
