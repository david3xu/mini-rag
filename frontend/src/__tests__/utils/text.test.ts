import { truncateText, highlightSearchTerms, getFileExtension, formatFileSize } from '../../utils/text';

describe('Text Utility Functions', () => {
  describe('truncateText', () => {
    test('returns original text when shorter than maxLength', () => {
      const text = 'Short text';
      expect(truncateText(text, 20)).toBe(text);
    });
    
    test('truncates text when longer than maxLength', () => {
      const text = 'This is a long text that should be truncated';
      expect(truncateText(text, 20)).toBe('This is a long text...');
    });
    
    test('uses default maxLength when not specified', () => {
      const text = 'A'.repeat(250);
      expect(truncateText(text)).toBe('A'.repeat(200) + '...');
    });
  });
  
  describe('formatFileSize', () => {
    test('formats bytes correctly', () => {
      expect(formatFileSize(0)).toBe('0 Bytes');
      expect(formatFileSize(500)).toBe('500 Bytes');
    });
    
    test('formats kilobytes correctly', () => {
      expect(formatFileSize(1024)).toBe('1 KB');
      expect(formatFileSize(2048)).toBe('2 KB');
    });
    
    test('formats megabytes correctly', () => {
      expect(formatFileSize(1048576)).toBe('1 MB');
      expect(formatFileSize(5242880)).toBe('5 MB');
    });
  });
  
  describe('getFileExtension', () => {
    test('returns correct extension for various file types', () => {
      expect(getFileExtension('document.pdf')).toBe('pdf');
      expect(getFileExtension('image.jpg')).toBe('jpg');
      expect(getFileExtension('data.txt')).toBe('txt');
      expect(getFileExtension('archive.tar.gz')).toBe('gz');
    });
    
    test('handles filenames without extensions', () => {
      expect(getFileExtension('README')).toBe('');
      expect(getFileExtension('no-extension')).toBe('');
    });
    
    test('handles paths correctly', () => {
      expect(getFileExtension('/path/to/file.docx')).toBe('docx');
      expect(getFileExtension('C:\\Windows\\file.xlsx')).toBe('xlsx');
    });
  });
  
  describe('highlightSearchTerms', () => {
    test('highlights single term in text', () => {
      const text = 'This is a sample text about Mini-RAG';
      const terms = ['sample'];
      const result = highlightSearchTerms(text, terms);
      
      expect(result).toContain('<mark>sample</mark>');
      expect(result).toBe('This is a <mark>sample</mark> text about Mini-RAG');
    });
    
    test('highlights multiple terms in text', () => {
      const text = 'Mini-RAG is a lightweight RAG system';
      const terms = ['mini-rag', 'system'];
      const result = highlightSearchTerms(text, terms);
      
      expect(result).toContain('<mark>Mini-RAG</mark>');
      expect(result).toContain('<mark>system</mark>');
    });
    
    test('is case insensitive for matching', () => {
      const text = 'UPPER lower MiXeD case text';
      const terms = ['upper', 'mixed'];
      const result = highlightSearchTerms(text, terms);
      
      expect(result).toContain('<mark>UPPER</mark>');
      expect(result).toContain('<mark>MiXeD</mark>');
    });
    
    test('returns original text when no terms provided', () => {
      const text = 'Original text';
      expect(highlightSearchTerms(text, [])).toBe(text);
    });
  });
}); 