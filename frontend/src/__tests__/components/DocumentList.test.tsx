import React from 'react';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';
import DocumentList from '../../components/DocumentList';

// Mock the API client
jest.mock('../../api/documents', () => ({
  getDocuments: jest.fn().mockResolvedValue([
    { id: '1', name: 'Document 1', type: 'pdf', uploadedAt: '2023-01-01T00:00:00Z' },
    { id: '2', name: 'Document 2', type: 'docx', uploadedAt: '2023-01-02T00:00:00Z' },
  ]),
}));

describe('DocumentList Component', () => {
  test('renders loading state initially', () => {
    render(<DocumentList />);
    expect(screen.getByText(/loading documents/i)).toBeInTheDocument();
  });

  test('renders documents when loaded', async () => {
    render(<DocumentList />);
    
    // Wait for the documents to load
    expect(await screen.findByText('Document 1')).toBeInTheDocument();
    expect(screen.getByText('Document 2')).toBeInTheDocument();
    
    // Check that document types are displayed
    expect(screen.getByText('pdf')).toBeInTheDocument();
    expect(screen.getByText('docx')).toBeInTheDocument();
  });

  test('renders empty state when no documents', async () => {
    // Override the mock for this test
    const getDocumentsMock = require('../../api/documents').getDocuments;
    getDocumentsMock.mockResolvedValueOnce([]);
    
    render(<DocumentList />);
    
    // Wait for the no documents message
    expect(await screen.findByText(/no documents found/i)).toBeInTheDocument();
  });
}); 