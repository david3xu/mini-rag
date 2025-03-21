/**
 * API Type Definitions
 * Defines the request and response types for API communication
 */

import { Source, DocumentUploadResult } from './models';

/**
 * Chat query request
 * Sent to backend when user submits a question
 */
export interface QueryRequest {
  /** User's text query */
  query: string;
}

/**
 * Chat query response
 * Received from backend with answer and sources
 */
export interface QueryResponse {
  /** Generated answer text */
  answer: string;
  
  /** Source documents used for answer generation */
  sources: Source[];
}

/**
 * Document upload response
 * Received after document processing completes
 */
export interface DocumentUploadResponse {
  /** Overall status of the operation */
  status: string;
  
  /** Number of files successfully processed */
  files_processed: number;
  
  /** Detailed results for each document */
  results: DocumentUploadResult[];
}

/**
 * Health check response
 * Used to verify backend availability
 */
export interface HealthResponse {
  /** Status indicator (typically "ok") */
  status: string;
}

/**
 * OpenAI compatible chat request
 * Used for Azure migration compatibility
 */
export interface OpenAICompatibleRequest {
  /** Model identifier */
  model: string;
  
  /** Array of chat messages */
  messages: Array<{
    /** Role of the message sender */
    role: 'system' | 'user' | 'assistant';
    
    /** Message content */
    content: string;
  }>;
  
  /** Optional temperature for controlling randomness */
  temperature?: number;
  
  /** Optional token limit for response */
  max_tokens?: number;
}

/**
 * OpenAI compatible chat response
 * For Azure migration compatibility
 */
export interface OpenAICompatibleResponse {
  /** Response identifier */
  id: string;
  
  /** Object type */
  object: string;
  
  /** Timestamp of creation */
  created: number;
  
  /** Model used for generation */
  model: string;
  
  /** Array of completion choices */
  choices: Array<{
    /** Choice index */
    index: number;
    
    /** Generated message */
    message: {
      /** Role of the message */
      role: string;
      
      /** Message content */
      content: string;
    };
    
    /** Reason generation stopped */
    finish_reason: string;
  }>;
  
  /** Optional token usage statistics */
  usage?: {
    /** Prompt tokens used */
    prompt_tokens: number;
    
    /** Completion tokens generated */
    completion_tokens: number;
    
    /** Total tokens used */
    total_tokens: number;
  };
}