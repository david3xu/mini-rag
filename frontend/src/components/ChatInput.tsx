import React, { useState, useRef, useEffect } from 'react';
import '../styles/components/ChatInput.css';

interface ChatInputProps {
  /**
   * Callback function triggered when a message is submitted
   */
  onSubmit: (query: string) => void;
  
  /**
   * Flag indicating if a response is currently being generated
   */
  isLoading: boolean;
  
  /**
   * Optional placeholder text for the input field
   */
  placeholder?: string;
}

/**
 * Enhanced chat input component with auto-resize and keyboard shortcuts
 * 
 * Features:
 * - Auto-expanding textarea that grows with content
 * - Keyboard shortcuts (Ctrl+Enter or Cmd+Enter to submit)
 * - Accessible design with proper ARIA attributes
 * - Visual feedback during loading states
 */
const ChatInput: React.FC<ChatInputProps> = ({ 
  onSubmit, 
  isLoading,
  placeholder = "Ask a question about your documents..."
}) => {
  const [query, setQuery] = useState<string>('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  
  // Auto-resize textarea as content changes
  useEffect(() => {
    if (textareaRef.current) {
      // Reset height to auto to get the correct scrollHeight
      textareaRef.current.style.height = 'auto';
      // Set height to scrollHeight to fit content
      textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`;
    }
  }, [query]);
  
  // Handle form submission
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    
    const trimmedQuery = query.trim();
    if (!trimmedQuery || isLoading) return;
    
    onSubmit(trimmedQuery);
    setQuery('');
    
    // Reset textarea height
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }
  };
  
  // Handle keyboard shortcuts (Ctrl+Enter or Cmd+Enter to submit)
  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      handleSubmit(e);
    }
  };
  
  return (
    <form 
      onSubmit={handleSubmit} 
      className="chat-input"
      aria-label="Message input form"
    >
      <div className="chat-input__textarea-container">
        <textarea
          ref={textareaRef}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          disabled={isLoading}
          rows={1}
          className="chat-input__textarea"
          aria-label="Message input"
          aria-disabled={isLoading}
        />
      </div>
      <button 
        type="submit" 
        disabled={isLoading || !query.trim()} 
        className="chat-input__button"
        aria-label={isLoading ? "Generating response" : "Send message"}
      >
        {isLoading ? 'Processing...' : 'Send'}
      </button>
      <div className="chat-input__help">
        <small>Press Ctrl+Enter to send</small>
      </div>
    </form>
  );
};

export default ChatInput;