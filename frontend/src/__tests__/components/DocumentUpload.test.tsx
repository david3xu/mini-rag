import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import DocumentUpload from '../../components/DocumentUpload';

describe('DocumentUpload Component', () => {
  const mockOnUpload = jest.fn();
  
  beforeEach(() => {
    mockOnUpload.mockClear();
  });
  
  test('renders upload area with instructions', () => {
    render(<DocumentUpload onUpload={mockOnUpload} isUploading={false} />);
    
    expect(screen.getByText(/drag and drop/i)).toBeInTheDocument();
    expect(screen.getByText(/or click to browse/i)).toBeInTheDocument();
    expect(screen.getByTestId('upload-area')).toBeInTheDocument();
    expect(screen.getByLabelText(/browse files/i)).toBeInTheDocument();
  });
  
  test('shows loading state when uploading', () => {
    render(<DocumentUpload onUpload={mockOnUpload} isUploading={true} />);
    
    expect(screen.getByText(/uploading/i)).toBeInTheDocument();
    expect(screen.getByTestId('upload-spinner')).toBeInTheDocument();
    
    // Upload area should be disabled during upload
    const uploadArea = screen.getByTestId('upload-area');
    expect(uploadArea).toHaveClass('disabled');
  });
  
  test('handles file selection through input', async () => {
    render(<DocumentUpload onUpload={mockOnUpload} isUploading={false} />);
    
    const file = new File(['test content'], 'test.txt', { type: 'text/plain' });
    const input = screen.getByLabelText(/browse files/i);
    
    // Simulate file selection
    fireEvent.change(input, { target: { files: [file] } });
    
    // Should call onUpload with the file
    await waitFor(() => {
      expect(mockOnUpload).toHaveBeenCalledTimes(1);
      expect(mockOnUpload).toHaveBeenCalledWith([file]);
    });
  });
  
  test('handles drag and drop', async () => {
    render(<DocumentUpload onUpload={mockOnUpload} isUploading={false} />);
    
    const file = new File(['test content'], 'test.txt', { type: 'text/plain' });
    const uploadArea = screen.getByTestId('upload-area');
    
    // Simulate drag events
    fireEvent.dragEnter(uploadArea, {
      dataTransfer: {
        files: [file],
        types: ['Files'],
      },
    });
    
    expect(uploadArea).toHaveClass('drag-active');
    
    // Simulate drop
    fireEvent.drop(uploadArea, {
      dataTransfer: {
        files: [file],
        types: ['Files'],
      },
    });
    
    // Should call onUpload with the files
    await waitFor(() => {
      expect(mockOnUpload).toHaveBeenCalledTimes(1);
      expect(mockOnUpload).toHaveBeenCalledWith([file]);
    });
    
    // Drag active class should be removed
    expect(uploadArea).not.toHaveClass('drag-active');
  });
  
  test('validates file types', async () => {
    render(<DocumentUpload onUpload={mockOnUpload} isUploading={false} acceptedTypes={['.txt', '.pdf']} />);
    
    // Invalid file type
    const invalidFile = new File(['test content'], 'test.docx', { type: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' });
    const input = screen.getByLabelText(/browse files/i);
    
    // Simulate file selection with invalid file
    fireEvent.change(input, { target: { files: [invalidFile] } });
    
    // Should show error and not call onUpload
    await waitFor(() => {
      expect(screen.getByText(/unsupported file type/i)).toBeInTheDocument();
      expect(mockOnUpload).not.toHaveBeenCalled();
    });
  });
}); 