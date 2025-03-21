import React, { useState, useEffect } from 'react';
import { Message } from '../types/models';
import DocumentUpload from './DocumentUpload';
import MessageList from './MessageList';
import ChatInput from './ChatInput';
import EmptyState from './EmptyState';
import '../styles/components/ChatInterface.css';

/**
 * Main chat interface component that integrates message display, 
 * input handling, and document management.
 * 
 * This component serves as the primary container for the RAG application,
 * coordinating between document uploads, user queries, and response display.
 */
const ChatInterface: React.FC = () => {
  // State declarations with proper typing
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(false);

  // Load chat history from local storage on mount
  useEffect(() => {
    const savedMessages = loadFromLocalStorage();
    if (savedMessages.length > 0) {
      setMessages(savedMessages);
    }
  }, []);

  // Save chat history to local storage when messages change
  useEffect(() => {
    if (messages.length > 0) {
      saveToLocalStorage(messages);
    }
  }, [messages]);

  /**
   * Handles user message submission and API communication
   * @param query User's text query
   */
  const handleSubmit = async (query: string) => {
    // Create and add user message
    const userMessage: Message = { role: 'user', content: query };
    setMessages(prevMessages => [...prevMessages, userMessage]);
    
    // Set loading state
    setIsLoading(true);
    
    try {
      // Temporary implementation pending service layer in Segment 4
      // This will be replaced with actual API call in next segment
      await simulateApiCall(query);
      
      // Simulate assistant response with sources
      const assistantMessage: Message = { 
        role: 'assistant', 
        content: `This is a simulated response to "${query}". This placeholder will be replaced with actual API integration in the next implementation segment.`,
        sources: [
          {
            id: 'sample-1',
            content: 'Sample document content that would be retrieved from the backend.',
            metadata: {
              source: 'example-document.txt'
            }
          }
        ]
      };
      
      setMessages(prevMessages => [...prevMessages, assistantMessage]);
    } catch (error) {
      console.error('Error processing query:', error);
      
      // Handle error with user-friendly message
      const errorMessage: Message = { 
        role: 'assistant', 
        content: 'Sorry, I encountered an error processing your query. Please try again or check your document upload.'
      };
      
      setMessages(prevMessages => [...prevMessages, errorMessage]);
    } finally {
      // Reset loading state
      setIsLoading(false);
    }
  };

  /**
   * Clears chat history after confirmation
   */
  const handleClearChat = () => {
    if (messages.length > 0 && !isLoading) {
      if (window.confirm('Are you sure you want to clear the chat history?')) {
        setMessages([]);
        localStorage.removeItem('mini-rag-chat-history');
      }
    }
  };

  /**
   * Temporary function to simulate API latency
   * Will be replaced with actual API integration
   */
  const simulateApiCall = async (query: string): Promise<void> => {
    // Simulate network latency
    return new Promise(resolve => setTimeout(resolve, 1000));
  };

  /**
   * Loads message history from local storage
   */
  const loadFromLocalStorage = (): Message[] => {
    try {
      const savedData = localStorage.getItem('mini-rag-chat-history');
      if (savedData) {
        return JSON.parse(savedData) as Message[];
      }
    } catch (error) {
      console.error('Error loading chat history from localStorage:', error);
    }
    return [];
  };

  /**
   * Saves message history to local storage
   */
  const saveToLocalStorage = (messages: Message[]): void => {
    try {
      localStorage.setItem('mini-rag-chat-history', JSON.stringify(messages));
    } catch (error) {
      console.error('Error saving chat history to localStorage:', error);
    }
  };

  return (
    <div className="chat-container">
      {/* Header with title and clear button */}
      <div className="chat-header">
        <h1>Mini RAG System</h1>
        {messages.length > 0 && (
          <button 
            className="clear-button" 
            onClick={handleClearChat}
            disabled={isLoading}
            aria-label="Clear chat history"
          >
            Clear Chat
          </button>
        )}
      </div>
      
      {/* Document upload component */}
      <DocumentUpload />
      
      {/* Message display area with conditional rendering */}
      <div className="message-container">
        {messages.length === 0 ? (
          <EmptyState 
            title="No Conversations Yet"
            description="Upload documents and start asking questions about your content."
            icon={<span className="material-icons">chat</span>}
          />
        ) : (
          <MessageList 
            messages={messages} 
            isLoading={isLoading} 
          />
        )}
      </div>
      
      {/* Input component */}
      <ChatInput 
        onSubmit={handleSubmit}
        isLoading={isLoading}
        placeholder="Ask a question about your documents..."
      />
    </div>
  );
};

export default ChatInterface;