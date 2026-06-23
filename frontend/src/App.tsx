import { useState } from 'react';
import Header from './components/Header';
import ReviewInput from './components/ReviewInput';
import ModelCard from './components/ModelCard';
import AttentionHeatmap from './components/AttentionHeatmap';
import { analyzeReview } from './api/analyze';
import type { AnalyzeResponse } from './types';
import './App.css';

function App() {
  const [results, setResults] = useState<AnalyzeResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleAnalyze = async (text: string) => {
    setIsLoading(true);
    setError(null);
    setResults(null); // Clear previous results while loading

    try {
      const data = await analyzeReview({ text });
      setResults(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An unknown error occurred');
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app-container">
      <Header />
      
      <main>
        <ReviewInput onAnalyze={handleAnalyze} isLoading={isLoading} />

        {error && (
          <div className="error-message glass-panel">
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>
            <p>{error}</p>
          </div>
        )}

        <section className="results-grid">
          <div className="model-column">
            <ModelCard 
              title="Scratch Transformer"
              architecture="Custom Encoder • 3 Blocks • 4 Heads"
              params="~6M params"
              result={results?.scratch || null}
              isLoading={isLoading}
              isPrimary={true}
            />
            
            {results?.scratch && !isLoading && (
              <AttentionHeatmap 
                tokens={results.scratch.tokens} 
                attention={results.scratch.attention} 
              />
            )}
          </div>
          
          <div className="model-column">
            <ModelCard 
              title="DistilRoBERTa"
              architecture="HuggingFace • Pre-trained + Fine-tuned"
              params="~82M params"
              result={results?.distilroberta || null}
              isLoading={isLoading}
            />
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;
