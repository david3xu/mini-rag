import { getDocuments, uploadDocument, deleteDocument } from '../../api/documents';
import fetchMock from 'jest-fetch-mock';

// Setup fetch mock
beforeAll(() => {
  fetchMock.enableMocks();
});

afterEach(() => {
  fetchMock.resetMocks();
});

describe('Documents API', () => {
  test('getDocuments fetches documents from the API', async () => {
    const mockDocuments = [
      { id: '1', name: 'Document 1', type: 'pdf', uploadedAt: '2023-01-01T00:00:00Z' },
      { id: '2', name: 'Document 2', type: 'docx', uploadedAt: '2023-01-02T00:00:00Z' },
    ];
    
    fetchMock.mockResponseOnce(JSON.stringify(mockDocuments));
    
    const documents = await getDocuments();
    
    expect(fetchMock).toHaveBeenCalledWith('/api/documents', expect.any(Object));
    expect(documents).toEqual(mockDocuments);
  });
  
  test('uploadDocument sends file to the API', async () => {
    const mockResponse = { id: '3', name: 'Document 3', type: 'pdf', uploadedAt: '2023-01-03T00:00:00Z' };
    fetchMock.mockResponseOnce(JSON.stringify(mockResponse));
    
    const file = new File(['dummy content'], 'document.pdf', { type: 'application/pdf' });
    const result = await uploadDocument(file);
    
    expect(fetchMock).toHaveBeenCalledWith('/api/documents', expect.objectContaining({
      method: 'POST',
      body: expect.any(FormData),
    }));
    expect(result).toEqual(mockResponse);
  });
  
  test('deleteDocument removes a document', async () => {
    fetchMock.mockResponseOnce(JSON.stringify({ success: true }));
    
    const result = await deleteDocument('123');
    
    expect(fetchMock).toHaveBeenCalledWith('/api/documents/123', expect.objectContaining({
      method: 'DELETE',
    }));
    expect(result).toEqual({ success: true });
  });
  
  test('handles API errors correctly', async () => {
    fetchMock.mockRejectOnce(new Error('Network error'));
    
    await expect(getDocuments()).rejects.toThrow('Network error');
  });
}); 