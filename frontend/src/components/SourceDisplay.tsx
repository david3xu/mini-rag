import React, { useState } from 'react';
import { Source } from '../types/models';
import '../styles/components/SourceDisplay.css';

interface SourceDisplayProps {
  /**
   * Array of source documents used in RAG response
   */
  sources: Source[];
}

/**
 * Component to display source documents used in RAG responses
 * 
 * Renders a collapsible list of source documents that were used 
 * to generate the assistant's response, including:
 * - Document filenames
 * - Page numbers (if available)
 * - Relevant content snippets
 * - Toggle functionality to show/hide sources
 */
const SourceDisplay: React.FC<SourceDisplayProps> = ({ sources }) => {
  const [isExpanded, setIsExpanded] = useState<boolean>(false);

  if (!sources || sources.length === 0) {
    return null;
  }
  
  // Extract filename from source path
  const getFilename = (path: string): string => {
    return path.split('/').pop() || path;
  };

  return (
    <div className="sources">
      <button 
        className="sources__toggle"
        onClick={() => setIsExpanded(!isExpanded)}
        aria-expanded={isExpanded}
      >
        <span className="sources__toggle-text">
          {isExpanded ? 'Hide sources' : 'Show sources'} ({sources.length})
        </span>
        <span className={`sources__toggle-icon ${isExpanded ? 'sources__toggle-icon--expanded' : ''}`}>
          ▼
        </span>
      </button>
      
      {isExpanded && (
        <div className="sources__content">
          <ul className="sources__list">
            {sources.map((source, index) => (
              <li key={index} className="sources__item">
                <div className="sources__header">
                  <span className="sources__filename">
                    {getFilename(source.metadata.source)}
                  </span>
                  {source.metadata.page && 
                    <span className="sources__page">Page {source.metadata.page}</span>
                  }
                </div>
                <div className="sources__excerpt">{source.content}</div>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};

export default SourceDisplay;