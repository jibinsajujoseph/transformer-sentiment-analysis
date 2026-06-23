export interface AnalyzeRequest {
  text: string;
}

export interface ScratchResult {
  label: 'positive' | 'negative';
  confidence: number;
  latency_ms: number;
  tokens: string[];
  attention: number[];
}

export interface DistilRoBERTaResult {
  label: 'positive' | 'negative';
  confidence: number;
  latency_ms: number;
}

export interface AnalyzeResponse {
  scratch: ScratchResult;
  distilroberta: DistilRoBERTaResult;
}
