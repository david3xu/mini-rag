import React, { createContext, useContext, useState, ReactNode } from 'react';
import { Message } from '../types/models';

/**
 * Interface defining the shape of our chat context
 */
interface ChatContextType {
  messages: Message[];
  addMessage: (message: Message) => void;
  clearMessages: () => void;
  isLoading: boolean;
  setIsLoading: (loading: boolean) => void;
}

// Create context with default undefined value
const ChatContext = createContext<ChatContextType | undefined>(undefined);

/**
 * Props for ChatProvider component
 */
interface ChatProviderProps {
  children: ReactNode;
}

/**
 * Provider component for chat state management
 */
export const ChatProvider: React.FC<ChatProviderProps> = ({ children }) => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(false);

  const addMessage = (message: Message) => {
    setMessages(prev => [...prev, message]);
  };

  const clearMessages = () => {
    setMessages([]);
  };

  // Context value
  const value: ChatContextType = {
    messages,
    addMessage,
    clearMessages,
    isLoading,
    setIsLoading
  };

  return (
    <ChatContext.Provider value={value}>
      {children}
    </ChatContext.Provider>
  );
};

/**
 * Custom hook for accessing chat context
 */
export const useChat = (): ChatContextType => {
  const context = useContext(ChatContext);
  if (context === undefined) {
    throw new Error('useChat must be used within a ChatProvider');
  }
  return context;
};