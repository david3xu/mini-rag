import axios, { AxiosResponse } from 'axios';
import { 
  QueryResponse, 
  DocumentUploadResponse,
  OpenAICompatibleResponse
} from '../types/api';

// Azure OpenAI endpoint
const AZURE_OPENAI_ENDPOINT = process.env.REACT_APP_AZURE_OPENAI_ENDPOINT;
// Azure Cognitive Search endpoint
const AZURE_SEARCH_ENDPOINT = process.env.REACT_APP_AZURE_SEARCH_ENDPOINT;
// Azure Storage endpoint
const AZURE_STORAGE_ENDPOINT = process.env.REACT_APP_AZURE_STORAGE_ENDPOINT;

// Azure authentication
const apiKey = process.env.REACT_APP_AZURE_API_KEY;
const sasToken = process.env.REACT_APP_AZURE_SAS_TOKEN;

// Create OpenAI client
const openaiClient = axios.create({
  baseURL: AZURE_OPENAI_ENDPOINT,
  headers: {
    'Content-Type': 'application/json',
    'api-key': apiKey
  }
});

// Create Search client
const searchClient = axios.create({
  baseURL: AZURE_SEARCH_ENDPOINT,
  headers: {
    'Content-Type': 'application/json',
    'api-key': apiKey
  }
});

// Create Storage client
const storageClient = axios.create({
  baseURL: AZURE_STORAGE_ENDPOINT,
});

/**
 * API service functions for Azure deployment
 */
export const azureApi = {
  /**
   * Send a query to Azure OpenAI with RAG context
   * @param query Text query to process
   */
  sendQuery: async (query: string): Promise<AxiosResponse<QueryResponse>> => {
    // First, get relevant documents from Azure Cognitive Search
    const searchResponse = await searchClient.post('/indexes/documents/docs/search', {
      search: query,
      select: "id,content,metadata",
      top: 3
    });
    
    const documents = searchResponse.data.value;
    
    // Then, send query with context to Azure OpenAI
    const messages = [
      { role: 'system', content: 'You are a helpful assistant that answers questions based on the provided documents.' },
      { role: 'user', content: formatRAGPrompt(query, documents) }
    ];
    
    const response = await openaiClient.post('/deployments/gpt-4-turbo/chat/completions', {
      messages,
      temperature: 0.7,
      max_tokens: 512
    });
    
    // Format response to match local API
    const answer = response.data.choices[0].message.content;
    
    // Format sources to match local API format
    const sources = documents.map((doc: any) => ({
      id: doc.id,
      content: doc.content.substring(0, 200) + (doc.content.length > 200 ? '...' : ''),
      metadata: JSON.parse(doc.metadata)
    }));
    
    return {
      data: {
        answer,
        sources
      },
      status: response.status,
      statusText: response.statusText,
      headers: response.headers,
      config: response.config
    };
  },
  
  /**
   * Upload documents to Azure Blob Storage and index in Cognitive Search
   * @param files List of files to upload for processing
   */
  uploadDocuments: async (files: File[]): Promise<AxiosResponse<DocumentUploadResponse>> => {
    const results = [];
    
    for (const file of files) {
      try {
        // Upload to Azure Blob Storage
        const formData = new FormData();
        formData.append('file', file);
        
        const uploadUrl = `${AZURE_STORAGE_ENDPOINT}/documents/${file.name}${sasToken}`;
        await axios.put(uploadUrl, file, {
          headers: {
            'x-ms-blob-type': 'BlockBlob',
            'Content-Type': file.type
          }
        });
        
        // Trigger Azure Function to process the document
        const processingResponse = await axios.post('/api/process-document', {
          filename: file.name,
          container: 'documents'
        });
        
        results.push({
          filename: file.name,
          chunks_processed: processingResponse.data.chunks_processed,
          status: 'success'
        });
      } catch (error) {
        console.error(`Error processing ${file.name}:`, error);
        results.push({
          filename: file.name,
          status: 'error',
          error: error instanceof Error ? error.message : 'Unknown error'
        });
      }
    }
    
    return {
      data: {
        status: 'success',
        files_processed: results.length,
        results
      },
      status: 200,
      statusText: 'OK',
      headers: {},
      config: {}
    };
  },
  
  /**
   * Direct access to Azure OpenAI (OpenAI-compatible API)
   */
  openaiCompatible: (
    messages: Array<{role: 'system' | 'user' | 'assistant', content: string}>,
    options?: {temperature?: number, max_tokens?: number}
  ): Promise<AxiosResponse<OpenAICompatibleResponse>> => {
    return openaiClient.post('/deployments/gpt-4-turbo/chat/completions', {
      messages,
      temperature: options?.temperature || 0.7,
      max_tokens: options?.max_tokens || 512
    });
  }
};

/**
 * Helper function to format RAG prompt with context
 * @param query User query
 * @param documents Context documents from search
 */
function formatRAGPrompt(query: string, documents: any[]): string {
  const context = documents.map((doc, i) => {
    return `Document ${i+1}:\n${doc.content}`;
  }).join('\n\n');
  
  return `
    Answer the following query based on the provided documents.
    If you don't know the answer, say you don't know.

    Documents:
    ${context}

    Query: ${query}

    Answer:
  `;
}

export default azureApi;