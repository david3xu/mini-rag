import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import ChatInput from '../../components/ChatInput';

describe('ChatInput Component', () => {
  const mockSubmit = jest.fn();
  
  beforeEach(() => {
    mockSubmit.mockClear();
  });
  
  test('renders with default placeholder', () => {
    render(<ChatInput onSubmit={mockSubmit} isLoading={false} />);
    
    const textarea = screen.getByRole('textbox', { name: /message input/i });
    expect(textarea).toBeInTheDocument();
    expect(textarea).toHaveAttribute('placeholder', 'Ask a question about your documents...');
  });
  
  test('handles text input', async () => {
    render(<ChatInput onSubmit={mockSubmit} isLoading={false} />);
    
    const textarea = screen.getByRole('textbox');
    await userEvent.type(textarea, 'test question');
    
    expect(textarea).toHaveValue('test question');
  });
  
  test('handles form submission', async () => {
    render(<ChatInput onSubmit={mockSubmit} isLoading={false} />);
    
    const textarea = screen.getByRole('textbox');
    await userEvent.type(textarea, 'test query');
    
    const button = screen.getByRole('button', { name: /send/i });
    await userEvent.click(button);
    
    expect(mockSubmit).toHaveBeenCalledWith('test query');
    expect(textarea).toHaveValue('');
  });
  
  test('disables input and button when loading', () => {
    render(<ChatInput onSubmit={mockSubmit} isLoading={true} />);
    
    expect(screen.getByRole('textbox')).toBeDisabled();
    expect(screen.getByRole('button')).toBeDisabled();
    expect(screen.getByText(/processing/i)).toBeInTheDocument();
  });
}); 