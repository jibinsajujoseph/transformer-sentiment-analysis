import type { AnalyzeRequest, AnalyzeResponse } from '../types';

const API_BASE_URL = 'http://localhost:8000/api';

export const analyzeReview = async (request: AnalyzeRequest): Promise<AnalyzeResponse> => {
  const response = await fetch(`${API_BASE_URL}/analyze`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    let errorMessage = `HTTP error! status: ${response.status}`;
    try {
      const errorData = await response.json();
      if (errorData && errorData.detail) {
        errorMessage = typeof errorData.detail === 'string' ? errorData.detail : JSON.stringify(errorData.detail);
      }
    } catch (e) {
      // Ignore JSON parsing errors for error responses
    }
    throw new Error(errorMessage);
  }

  return response.json();
};
