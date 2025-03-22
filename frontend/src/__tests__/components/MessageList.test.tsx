import React from 'react';
import { render, screen } from '@testing-library/react';
import MessageList from '../../components/MessageList';

describe('MessageList Component', () => {
  const mockMessages = [
    {
      id: '1',
      content: 'Hello, how can I help you?',
      role: 'assistant',
      timestamp: new Date().toISOString(),
    },
    {
      id: '2',
      content: 'Tell me about Mini-RAG',
      role: 'user',
      timestamp: new Date().toISOString(),
    },
    {
      id: '3',
      content: 'Mini-RAG is a lightweight RAG system...',
      role: 'assistant',
      timestamp: new Date().toISOString(),
    },
  ];
  
  test('renders empty state when no messages', () => {
    render(<MessageList messages={[]} isLoading={false} />);
    
    expect(screen.getByText(/no messages yet/i)).toBeInTheDocument();
  });
  
  test('renders loading state correctly', () => {
    render(<MessageList messages={[]} isLoading={true} />);
    
    expect(screen.getByText(/loading/i)).toBeInTheDocument();
    expect(screen.getByTestId('loading-indicator')).toBeInTheDocument();
  });
  
  test('renders all messages in correct order', () => {
    render(<MessageList messages={mockMessages} isLoading={false} />);
    
    const messageElements = screen.getAllByTestId('message-item');
    expect(messageElements).toHaveLength(3);
    
    // Check content
    expect(screen.getByText('Hello, how can I help you?')).toBeInTheDocument();
    expect(screen.getByText('Tell me about Mini-RAG')).toBeInTheDocument();
    expect(screen.getByText('Mini-RAG is a lightweight RAG system...')).toBeInTheDocument();
    
    // Check roles are displayed correctly
    const assistantLabels = screen.getAllByText(/assistant/i);
    const userLabels = screen.getAllByText(/user/i);
    
    expect(assistantLabels).toHaveLength(2);
    expect(userLabels).toHaveLength(1);
  });
  
  test('applies different styles to user and assistant messages', () => {
    render(<MessageList messages={mockMessages} isLoading={false} />);
    
    const messageElements = screen.getAllByTestId('message-item');
    
    // Check that user and assistant messages have different class names
    expect(messageElements[0]).toHaveClass('assistant-message');
    expect(messageElements[1]).toHaveClass('user-message');
    expect(messageElements[2]).toHaveClass('assistant-message');
  });
}); 