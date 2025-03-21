/**
 * Core domain model type definitions
 * Defines the fundamental data structures used throughout the application
 */

/**
 * Chat message representation
 * Represents a single message in the conversation between user and assistant
 */
export interface Message {
  /** Role of the message sender */
  role: 'user' | 'assistant';
  
  /** Message text content */
  content: string;
  
  /** Optional source documents used for assistant responses */
  sources?: Source[];
}

/**
 * Source document returned with assistant responses
 * Represents a document chunk used as context for generating responses
 */
export interface Source {
  /** Unique identifier for the source */
  id: string;
  
  /** Text content from the source document */
  content: string;
  
  /** Metadata associated with this source */
  metadata: DocumentMetadata;
}

/**
 * Metadata associated with source documents
 * Contains information about document origin and structure
 */
export interface DocumentMetadata {
  /** Original document source/filename */
  source: string;
  
  /** Optional page number for multi-page documents */
  page?: number;
  
  /** Optional chunk identifier within a document */
  chunk?: number;
  
  /** Optional sub-chunk identifier for further subdivision */
  sub_chunk?: number;
}

/**
 * Document upload result
 * Represents processing result for an uploaded document
 */
export interface DocumentUploadResult {
  /** Filename of the uploaded document */
  filename: string;
  
  /** Number of chunks processed from the document */
  chunks_processed?: number;
  
  /** Processing outcome status */
  status: 'success' | 'error';
  
  /** Error message if processing failed */
  error?: string;
}

/**
 * Upload status for UI display
 * Aggregates results from document upload operations
 */
export interface UploadStatus {
  /** Overall success status of the upload operation */
  success: boolean;
  
  /** Human-readable status message */
  message: string;
  
  /** Detailed results for each document */
  details: DocumentUploadResult[];
}