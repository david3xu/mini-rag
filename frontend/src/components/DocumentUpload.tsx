import React, { useState, useRef } from 'react';
import { UploadStatus, DocumentUploadResult } from '../types/models';
import '../styles/components/DocumentUpload.css';

/**
 * Component for uploading and processing documents for the RAG system.
 * 
 * Handles file selection, validation, submission, and feedback.
 * Resource-optimized with controlled UI states and error handling.
 */
const DocumentUpload: React.FC = () => {
  // State declarations with proper typing
  const [files, setFiles] = useState<File[]>([]);
  const [uploading, setUploading] = useState<boolean>(false);
  const [uploadStatus, setUploadStatus] = useState<UploadStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  
  // Reference for file input element to programmatically control it
  const fileInputRef = useRef<HTMLInputElement>(null);

  /**
   * Handle file selection from input
   * @param e Change event from file input
   */
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFiles = Array.from(e.target.files || []);
    setFiles(selectedFiles);
    
    // Clear previous status/error when new files are selected
    setError(null);
    setUploadStatus(null);
  };

  /**
   * Process files and upload to backend
   * @param e Form submission event
   */
  const handleUpload = async (e: React.FormEvent) => {
    e.preventDefault();
    
    // Validate file selection
    if (files.length === 0) {
      setError('Please select at least one file to upload');
      return;
    }
    
    // Set uploading state
    setUploading(true);
    setError(null);
    setUploadStatus(null);
    
    try {
      // Temporary implementation pending service layer in Segment 4
      // This will be replaced with actual API call
      const results = await simulateFileUpload(files);
      
      setUploadStatus({
        success: true,
        message: `Successfully processed ${results.length} files`,
        details: results
      });
      
      // Clear file input for next upload
      setFiles([]);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    } catch (error: any) {
      console.error('Upload error:', error);
      
      // Display user-friendly error message
      setError(
        error.message || 
        'An error occurred during upload. Please try again.'
      );
    } finally {
      setUploading(false);
    }
  };

  /**
   * Reset the upload form
   */
  const handleReset = () => {
    setFiles([]);
    setError(null);
    setUploadStatus(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  /**
   * Temporary function to simulate file upload and processing
   * Will be replaced with actual API integration
   */
  const simulateFileUpload = async (filesToUpload: File[]): Promise<DocumentUploadResult[]> => {
    // Simulate network latency and processing time
    await new Promise(resolve => setTimeout(resolve, 1500));
    
    // Generate simulated results
    return filesToUpload.map(file => ({
      filename: file.name,
      chunks_processed: Math.floor(Math.random() * 10) + 1,
      status: Math.random() > 0.1 ? 'success' : 'error',
      error: Math.random() > 0.1 ? undefined : 'Simulated error processing file'
    }));
  };

  return (
    <div className="document-upload">
      <h2>Upload Documents</h2>
      
      <form onSubmit={handleUpload} className="upload-form">
        <div className="file-input-container">
          <input
            type="file"
            onChange={handleFileChange}
            multiple
            accept=".pdf,.txt,.md,.json"
            disabled={uploading}
            id="file-upload"
            ref={fileInputRef}
            aria-label="Select documents to upload"
            className="file-input"
          />
          <label htmlFor="file-upload" className={`file-input-label ${uploading ? 'disabled' : ''}`}>
            {files.length > 0 
              ? `${files.length} file(s) selected` 
              : 'Choose files'
            }
          </label>
          
          <div className="button-group">
            <button 
              type="submit" 
              className="upload-button"
              disabled={uploading || files.length === 0}
              aria-busy={uploading}
            >
              {uploading ? 'Uploading...' : 'Upload'}
            </button>
            
            {files.length > 0 && !uploading && (
              <button 
                type="button" 
                className="reset-button"
                onClick={handleReset}
                aria-label="Clear selected files"
              >
                Clear
              </button>
            )}
          </div>
        </div>
        
        {/* Error message display */}
        {error && (
          <div className="upload-error" role="alert">
            <p>{error}</p>
          </div>
        )}
        
        {/* Success status display */}
        {uploadStatus && uploadStatus.success && (
          <div className="upload-success" role="status">
            <p>{uploadStatus.message}</p>
            <ul className="upload-results">
              {uploadStatus.details.map((file, index) => (
                <li key={index} className={`upload-result upload-result--${file.status}`}>
                  <strong>{file.filename}</strong>: {file.status === 'success' 
                    ? `Processed ${file.chunks_processed} chunks` 
                    : `Error: ${file.error}`}
                </li>
              ))}
            </ul>
          </div>
        )}
      </form>
      
      <div className="supported-formats">
        <small>Supported formats: PDF, TXT, MD, JSON</small>
      </div>
    </div>
  );
};

export default DocumentUpload;