"""
Document processor service for handling file processing and chunking.

This module provides functionality for processing various document types
(text, markdown, PDF) into chunks suitable for embedding and retrieval.
"""

from typing import List, Dict, Any, Optional, Tuple
import os
import uuid
import re
import json
import logging
from pathlib import Path
import tempfile

from config import settings

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Process documents into chunks suitable for embedding.
    
    This service handles the parsing and chunking of documents, supporting
    various file formats including PDF, text, and Markdown files. It implements
    memory-efficient processing methods optimized for resource-constrained 
    environments.
    """
    
    def __init__(self):
        """Initialize the document processor with configured settings."""
        self.chunk_size = settings.CHUNK_SIZE
        self.chunk_overlap = settings.CHUNK_OVERLAP
        self.max_document_size_mb = settings.MAX_DOCUMENT_SIZE_MB
        
        logger.info(f"Document processor initialized with chunk_size={self.chunk_size}, "
                    f"chunk_overlap={self.chunk_overlap}")
    
    def process_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Process a file into chunks based on its type.
        
        Args:
            file_path: Path to the document file to process
            
        Returns:
            List of document chunks with their metadata
            
        Raises:
            ValueError: If the file type is unsupported or file is too large
            FileNotFoundError: If the file cannot be found
            RuntimeError: If processing fails for other reasons
        """
        # Validate file exists
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Check file size
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
        if file_size > self.max_document_size_mb:
            logger.error(f"File too large: {file_path} ({file_size:.2f} MB)")
            raise ValueError(f"File size exceeds maximum allowed ({self.max_document_size_mb} MB)")
        
        # Determine file type and process accordingly
        file_ext = os.path.splitext(file_path)[1].lower()
        
        logger.info(f"Processing file: {file_path} with extension {file_ext}")
        
        try:
            if file_ext == '.pdf':
                chunks = self._process_pdf(file_path)
            elif file_ext == '.txt':
                chunks = self._process_text(file_path)
            elif file_ext in ['.md', '.markdown']:
                chunks = self._process_markdown(file_path)
            elif file_ext == '.json':
                chunks = self._process_json(file_path)
            else:
                logger.error(f"Unsupported file type: {file_ext}")
                raise ValueError(f"Unsupported file type: {file_ext}")
            
            logger.info(f"Successfully processed {len(chunks)} chunks from {file_path}")
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            raise RuntimeError(f"Failed to process file {file_path}: {str(e)}")
    
    def _process_pdf(self, file_path: str) -> List[Dict[str, Any]]:
        """Process a PDF file into chunks.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            List of document chunks with metadata
            
        Raises:
            ImportError: If PDF processing dependencies are not available
            RuntimeError: If PDF processing fails
        """
        try:
            # Import here to avoid dependency if not processing PDFs
            import fitz  # PyMuPDF
        except ImportError:
            logger.error("PyMuPDF (fitz) not installed. Required for PDF processing.")
            raise ImportError("PyMuPDF (fitz) is required for PDF processing. Install with 'pip install pymupdf'")
        
        chunks = []
        try:
            logger.info(f"Opening PDF: {file_path}")
            doc = fitz.open(file_path)
            
            for page_num, page in enumerate(doc):
                logger.debug(f"Processing page {page_num+1}/{len(doc)}")
                
                # Extract text from page
                text = page.get_text()
                
                # Skip empty pages
                if not text.strip():
                    logger.debug(f"Skipping empty page {page_num+1}")
                    continue
                
                # Split by paragraphs first
                paragraphs = text.split('\n\n')
                for i, para in enumerate(paragraphs):
                    para = para.strip()
                    if len(para) < 10:  # Skip very short paragraphs
                        continue
                    
                    # Further chunk large paragraphs
                    if len(para) > self.chunk_size:
                        sub_chunks = self._split_text_into_chunks(
                            para, 
                            chunk_size=self.chunk_size, 
                            overlap=self.chunk_overlap
                        )
                        
                        for j, sub_chunk in enumerate(sub_chunks):
                            # Create unique ID for this chunk
                            chunk_id = f"{os.path.basename(file_path)}_p{page_num+1}_c{i}_s{j}"
                            
                            chunks.append({
                                "id": chunk_id,
                                "text": sub_chunk.strip(),
                                "metadata": {
                                    "source": file_path,
                                    "page": page_num + 1,
                                    "chunk": i,
                                    "sub_chunk": j
                                }
                            })
                    else:
                        chunk_id = f"{os.path.basename(file_path)}_p{page_num+1}_c{i}"
                        
                        chunks.append({
                            "id": chunk_id,
                            "text": para,
                            "metadata": {
                                "source": file_path,
                                "page": page_num + 1,
                                "chunk": i
                            }
                        })
            
            logger.info(f"PDF processing complete: {len(chunks)} chunks created")
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {str(e)}")
            raise RuntimeError(f"Failed to process PDF: {str(e)}")
    
    def _process_text(self, file_path: str) -> List[Dict[str, Any]]:
        """Process a text file into chunks.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            List of document chunks with metadata
            
        Raises:
            RuntimeError: If text processing fails
        """
        chunks = []
        
        try:
            logger.info(f"Processing text file: {file_path}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Split by paragraphs (blank lines)
            paragraphs = re.split(r'\n\s*\n', text)
            
            for i, para in enumerate(paragraphs):
                para = para.strip()
                if len(para) < 10:  # Skip very short paragraphs
                    continue
                
                # Further chunk large paragraphs
                if len(para) > self.chunk_size:
                    sub_chunks = self._split_text_into_chunks(
                        para, 
                        chunk_size=self.chunk_size, 
                        overlap=self.chunk_overlap
                    )
                    
                    for j, sub_chunk in enumerate(sub_chunks):
                        chunk_id = f"{os.path.basename(file_path)}_c{i}_s{j}"
                        
                        chunks.append({
                            "id": chunk_id,
                            "text": sub_chunk.strip(),
                            "metadata": {
                                "source": file_path,
                                "chunk": i,
                                "sub_chunk": j
                            }
                        })
                else:
                    chunk_id = f"{os.path.basename(file_path)}_c{i}"
                    
                    chunks.append({
                        "id": chunk_id,
                        "text": para,
                        "metadata": {
                            "source": file_path,
                            "chunk": i
                        }
                    })
            
            logger.info(f"Text processing complete: {len(chunks)} chunks created")
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing text file {file_path}: {str(e)}")
            raise RuntimeError(f"Failed to process text file: {str(e)}")
    
    def _process_markdown(self, file_path: str) -> List[Dict[str, Any]]:
        """Process a Markdown file into chunks.
        
        Args:
            file_path: Path to the Markdown file
            
        Returns:
            List of document chunks with metadata
            
        Raises:
            RuntimeError: If Markdown processing fails
        """
        chunks = []
        
        try:
            logger.info(f"Processing Markdown file: {file_path}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Split by headers and process each section
            header_pattern = r'^(#+)\s+(.+)'
            sections = re.split(r'(?=^#+ )', text, flags=re.MULTILINE)
            
            for i, section in enumerate(sections):
                section = section.strip()
                if not section:
                    continue
                
                # Extract heading if present
                heading = None
                match = re.match(header_pattern, section, re.MULTILINE)
                if match:
                    heading = match.group(2).strip()
                
                # Process the section content
                if len(section) > self.chunk_size:
                    sub_chunks = self._split_text_into_chunks(
                        section, 
                        chunk_size=self.chunk_size, 
                        overlap=self.chunk_overlap
                    )
                    
                    for j, sub_chunk in enumerate(sub_chunks):
                        chunk_id = f"{os.path.basename(file_path)}_s{i}_sc{j}"
                        
                        chunks.append({
                            "id": chunk_id,
                            "text": sub_chunk.strip(),
                            "metadata": {
                                "source": file_path,
                                "section": i,
                                "heading": heading,
                                "sub_chunk": j
                            }
                        })
                else:
                    chunk_id = f"{os.path.basename(file_path)}_s{i}"
                    
                    chunks.append({
                        "id": chunk_id,
                        "text": section,
                        "metadata": {
                            "source": file_path,
                            "section": i,
                            "heading": heading
                        }
                    })
            
            logger.info(f"Markdown processing complete: {len(chunks)} chunks created")
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing Markdown file {file_path}: {str(e)}")
            raise RuntimeError(f"Failed to process Markdown file: {str(e)}")
    
    def _process_json(self, file_path: str) -> List[Dict[str, Any]]:
        """Process a JSON file into chunks.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            List of document chunks with metadata
            
        Raises:
            RuntimeError: If JSON processing fails
        """
        chunks = []
        
        try:
            logger.info(f"Processing JSON file: {file_path}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Process JSON data
            # Strategy: Extract text content from keys that likely contain text data
            extracted_texts = self._extract_text_from_json(data)
            
            for i, (path, text) in enumerate(extracted_texts):
                text = text.strip()
                if len(text) < 10:  # Skip very short texts
                    continue
                
                # Chunk large text blocks
                if len(text) > self.chunk_size:
                    sub_chunks = self._split_text_into_chunks(
                        text, 
                        chunk_size=self.chunk_size, 
                        overlap=self.chunk_overlap
                    )
                    
                    for j, sub_chunk in enumerate(sub_chunks):
                        chunk_id = f"{os.path.basename(file_path)}_k{i}_sc{j}"
                        
                        chunks.append({
                            "id": chunk_id,
                            "text": sub_chunk.strip(),
                            "metadata": {
                                "source": file_path,
                                "json_path": path,
                                "chunk_index": i,
                                "sub_chunk": j
                            }
                        })
                else:
                    chunk_id = f"{os.path.basename(file_path)}_k{i}"
                    
                    chunks.append({
                        "id": chunk_id,
                        "text": text,
                        "metadata": {
                            "source": file_path,
                            "json_path": path,
                            "chunk_index": i
                        }
                    })
            
            logger.info(f"JSON processing complete: {len(chunks)} chunks created")
            return chunks
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {file_path}: {str(e)}")
            raise ValueError(f"Invalid JSON format: {str(e)}")
        except Exception as e:
            logger.error(f"Error processing JSON file {file_path}: {str(e)}")
            raise RuntimeError(f"Failed to process JSON file: {str(e)}")
    
    def _extract_text_from_json(self, data: Any, path: str = "", results: List[Tuple[str, str]] = None) -> List[Tuple[str, str]]:
        """Recursively extract text content from JSON data.
        
        Args:
            data: JSON data (dict, list, or primitive)
            path: Current JSON path (for metadata)
            results: List to accumulate results
            
        Returns:
            List of tuples (json_path, text_content)
        """
        if results is None:
            results = []
        
        # Process different data types
        if isinstance(data, dict):
            # Extract text from likely text fields
            text_keys = ["text", "content", "description", "title", "body", "summary"]
            
            # First, handle known text fields
            for key in text_keys:
                if key in data and isinstance(data[key], str) and len(data[key].strip()) > 0:
                    current_path = f"{path}.{key}" if path else key
                    results.append((current_path, data[key]))
            
            # Then recursively process all fields
            for key, value in data.items():
                current_path = f"{path}.{key}" if path else key
                self._extract_text_from_json(value, current_path, results)
                
        elif isinstance(data, list):
            # Process lists of items
            for i, item in enumerate(data):
                current_path = f"{path}[{i}]" if path else f"[{i}]"
                self._extract_text_from_json(item, current_path, results)
                
        elif isinstance(data, str) and len(data.strip()) > 50:
            # Consider standalone strings if they're reasonably long
            results.append((path, data))
        
        return results
    
    def _split_text_into_chunks(self, text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
        """Split text into overlapping chunks.
        
        Implements intelligent splitting that tries to respect paragraph and
        sentence boundaries where possible.
        
        Args:
            text: Text to split
            chunk_size: Maximum chunk size
            overlap: Overlap between chunks
            
        Returns:
            List of text chunks
        """
        if chunk_size is None:
            chunk_size = self.chunk_size
        
        if overlap is None:
            overlap = self.chunk_overlap
        
        # Ensure sensible values
        if overlap >= chunk_size:
            overlap = chunk_size // 2
        
        # If text is already small enough, return as is
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            # Find end position for this chunk
            end = min(start + chunk_size, len(text))
            
            # Adjust to respect sentence boundaries when possible
            if end < len(text):
                # Try to find sentence boundary (period, question mark, exclamation mark)
                sentence_boundary = max(
                    text.rfind('. ', start, end),
                    text.rfind('? ', start, end),
                    text.rfind('! ', start, end)
                )
                
                # If found, adjust end to include the punctuation and space
                if sentence_boundary != -1 and sentence_boundary + 2 > start + chunk_size // 2:
                    end = sentence_boundary + 2
                else:
                    # If no sentence boundary, try to find word boundary
                    space_pos = text.rfind(' ', start, end)
                    if space_pos != -1 and space_pos > start + chunk_size // 2:
                        end = space_pos + 1
            
            # Extract chunk
            chunks.append(text[start:end])
            
            # Move start position for next chunk, considering overlap
            start = end - overlap
            
            # Avoid getting stuck at the same position
            if start >= len(text) - 10:  # Near the end with little content left
                break
            if start <= chunks[-1][0]:  # No progress was made
                start = end  # Skip overlap in problematic cases
        
        return chunks
    
    def extract_metadata_from_filename(self, filename: str) -> Dict[str, Any]:
        """Extract metadata from standardized filenames.
        
        Args:
            filename: Name of the file
            
        Returns:
            Dictionary with extracted metadata
        """
        metadata = {"filename": filename}
        
        # Extract date patterns (YYYY-MM-DD)
        date_match = re.search(r'(\d{4}-\d{2}-\d{2})', filename)
        if date_match:
            metadata["date"] = date_match.group(1)
        
        # Extract category from pattern like [Category] or (Category)
        category_match = re.search(r'[\[\(]([^\]\)]+)[\]\)]', filename)
        if category_match:
            metadata["category"] = category_match.group(1).strip()
        
        return metadata

# Singleton instance for application-wide use
document_processor = DocumentProcessor()