import axios, { AxiosResponse } from 'axios';
import { 
  QueryRequest, 
  QueryResponse, 
  DocumentUploadResponse,
  HealthResponse,
  OpenAICompatibleRequest,
  OpenAICompatibleResponse
} from '../types/api';

// Define API base URL - uses environment variable or default
const API_URL = process.env.REACT_APP_API_URL || '/api';

// Create axios instance with base configuration
const apiClient = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

/**
 * API service functions for chat and document operations
 */
export const api = {
  /**
   * Send a query to the RAG system
   * @param query Text query to process
   */
  sendQuery: (query: string): Promise<AxiosResponse<QueryResponse>> => {
    const request: QueryRequest = { query };
    return apiClient.post<QueryResponse>('/chat', request);
  },
  
  /**
   * Upload documents to the RAG system
   * @param files List of files to upload for processing
   */
  uploadDocuments: (files: File[]): Promise<AxiosResponse<DocumentUploadResponse>> => {
    const formData = new FormData();
    files.forEach(file => {
      formData.append('files', file);
    });
    
    return apiClient.post<DocumentUploadResponse>('/documents/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
  },
  
  /**
   * Check system health
   */
  checkHealth: (): Promise<AxiosResponse<HealthResponse>> => {
    return apiClient.get<HealthResponse>('/health');
  },
  
  /**
   * OpenAI-compatible API (for migration to Azure)
   * @param messages Chat messages
   * @param options Optional parameters
   */
  openaiCompatible: (
    messages: Array<{role: 'system' | 'user' | 'assistant', content: string}>,
    options?: {temperature?: number, max_tokens?: number}
  ): Promise<AxiosResponse<OpenAICompatibleResponse>> => {
    const request: OpenAICompatibleRequest = {
      model: 'llama-2-7b-local', // Will be replaced with Azure model in azure client
      messages,
      ...options
    };
    
    return axios.post<OpenAICompatibleResponse>('/v1/chat/completions', request);
  }
};

export default apiClient;