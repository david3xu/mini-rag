import React, { useEffect, useRef } from 'react';
import { Message } from '../types/models';
import SourceDisplay from './SourceDisplay';
import '../styles/components/MessageList.css';

interface MessageListProps {
  /**
   * Array of message objects in the conversation
   */
  messages: Message[];
  
  /**
   * Flag indicating whether a response is currently being generated
   */
  isLoading: boolean;
}

/**
 * Component to display the chat message history
 * 
 * Renders the conversation between user and assistant, including:
 * - Distinct styling for user and assistant messages
 * - Source attribution for assistant responses
 * - Loading indicator when waiting for a response
 * - Automatic scrolling to the latest message
 */
const MessageList: React.FC<MessageListProps> = ({ messages, isLoading }) => {
  // Reference for scrolling to the bottom of messages
  const messagesEndRef = useRef<HTMLDivElement>(null);
  
  // Auto-scroll to bottom when messages change or loading state changes
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  return (
    <div className="message-container">
      {messages.length === 0 ? (
        <div className="message-container__empty">
          <p>Upload documents and start asking questions about them!</p>
        </div>
      ) : (
        <div className="message-list">
          {messages.map((msg, index) => (
            <div key={index} className={`message message--${msg.role}`}>
              <div className="message__content">{msg.content}</div>
              {msg.sources && msg.sources.length > 0 && (
                <SourceDisplay sources={msg.sources} />
              )}
            </div>
          ))}
        </div>
      )}
      
      {isLoading && (
        <div className="message message--assistant message--loading">
          <div className="message__typing-indicator">
            <span></span>
            <span></span>
            <span></span>
          </div>
        </div>
      )}
      
      <div ref={messagesEndRef} className="message-list__end" />
    </div>
  );
};

export default MessageList;