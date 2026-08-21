import React from 'react';

export type ErrorType = 'permission' | 'stt' | 'ask' | 'empty_audio' | 'general';

export interface ErrorState {
  type: ErrorType;
  message: string;
  code?: string;
  statusCode?: number;
}

interface ErrorBannerProps {
  error: ErrorState | null;
  canRetry: boolean;
  retryLabel?: string;
  onRetry: () => void;
  onDismiss: () => void;
}

export const ErrorBanner: React.FC<ErrorBannerProps> = ({
  error,
  canRetry,
  retryLabel = 'Retry',
  onRetry,
  onDismiss,
}) => {
  if (!error) return null;

  const getErrorTitle = (type: ErrorType): string => {
    switch (type) {
      case 'permission':
        return 'Microphone Permission Required';
      case 'empty_audio':
        return 'No Speech Detected';
      case 'stt':
        return 'Transcription Failed';
      case 'ask':
        return 'Knowledge Query Failed';
      default:
        return 'An Error Occurred';
    }
  };

  const getErrorIcon = (type: ErrorType) => {
    switch (type) {
      case 'permission':
        return (
          <svg className="error-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z" />
            <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
            <line x1="1" y1="1" x2="23" y2="23" />
          </svg>
        );
      case 'empty_audio':
        return (
          <svg className="error-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="12" cy="12" r="10" />
            <line x1="12" y1="8" x2="12" y2="12" />
            <line x1="12" y1="16" x2="12.01" y2="16" />
          </svg>
        );
      default:
        return (
          <svg className="error-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
            <line x1="12" y1="9" x2="12" y2="13" />
            <line x1="12" y1="17" x2="12.01" y2="17" />
          </svg>
        );
    }
  };

  return (
    <div className={`error-banner error-type-${error.type}`} role="alert" aria-live="assertive">
      <div className="error-content">
        <div className="error-icon-wrapper">
          {getErrorIcon(error.type)}
        </div>
        <div className="error-text">
          <h4 className="error-title">{getErrorTitle(error.type)}</h4>
          <p className="error-message">{error.message}</p>
          {error.statusCode && (
            <span className="error-status-tag">Status code: {error.statusCode}</span>
          )}
        </div>
      </div>

      <div className="error-actions">
        {canRetry && (
          <button
            type="button"
            className="btn-retry"
            onClick={onRetry}
            id="error-retry-btn"
          >
            <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M21.5 2v6h-6M21.34 15.57a10 10 0 1 1-.57-8.38l5.67-5.67" />
            </svg>
            {retryLabel}
          </button>
        )}
        <button
          type="button"
          className="btn-dismiss"
          onClick={onDismiss}
          aria-label="Dismiss error"
          id="error-dismiss-btn"
        >
          &times;
        </button>
      </div>
    </div>
  );
};
