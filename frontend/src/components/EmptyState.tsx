import React from 'react';
import '../styles/components/EmptyState.css';

/**
 * Interface for EmptyState component props
 */
interface EmptyStateProps {
  /** Main title text displayed in the empty state */
  title: string;
  
  /** Descriptive text providing guidance */
  description: string;
  
  /** Optional icon element to display */
  icon?: React.ReactNode;
  
  /** Optional action button */
  actionButton?: React.ReactNode;
}

/**
 * Reusable empty state component for zero-data scenarios.
 * 
 * Provides consistent styling and structure for empty views throughout the application,
 * with support for custom icons and action buttons.
 */
const EmptyState: React.FC<EmptyStateProps> = ({ 
  title, 
  description, 
  icon, 
  actionButton 
}) => {
  return (
    <div className="empty-state" role="status">
      {/* Display icon if provided */}
      {icon && (
        <div className="empty-state__icon" aria-hidden="true">
          {icon}
        </div>
      )}
      
      {/* Title with semantic heading */}
      <h3 className="empty-state__title">
        {title}
      </h3>
      
      {/* Description text */}
      <p className="empty-state__description">
        {description}
      </p>
      
      {/* Optional action button */}
      {actionButton && (
        <div className="empty-state__action">
          {actionButton}
        </div>
      )}
    </div>
  );
};

export default EmptyState;