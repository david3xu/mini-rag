/**
 * Utility functions for text processing
 */

/**
 * Truncate text to a specific length with ellipsis
 * @param text Text to truncate
 * @param maxLength Maximum length before truncation
 */
export function truncateText(text: string, maxLength: number = 200): string {
  if (!text || text.length <= maxLength) {
    return text;
  }
  
  return text.substring(0, maxLength) + '...';
}

/**
 * Highlight search terms in text
 * @param text Text to search in
 * @param searchTerm Term to highlight
 */
export function highlightSearchTerms(text: string, searchTerm: string): string {
  if (!text || !searchTerm) {
    return text;
  }
  
  // Simple implementation - in a real app, you'd want to use a more robust approach
  const regex = new RegExp(`(${searchTerm.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi');
  return text.replace(regex, '<mark>$1</mark>');
}

/**
 * Extract file extension from filename
 * @param filename Filename to process
 */
export function getFileExtension(filename: string): string {
  return filename.split('.').pop()?.toLowerCase() || '';
}

/**
 * Get a user-friendly file size string
 * @param bytes Size in bytes
 */
export function formatFileSize(bytes: number): string {
  if (bytes === 0) return '0 Bytes';
  
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}