import React from 'react';

interface RecordButtonProps {
  isRecording: boolean;
  isProcessing: boolean;
  processingLabel?: string;
  durationSeconds: number;
  onStart: () => void;
  onStop: () => void;
  disabled?: boolean;
}

export const RecordButton: React.FC<RecordButtonProps> = ({
  isRecording,
  isProcessing,
  processingLabel,
  durationSeconds,
  onStart,
  onStop,
  disabled = false,
}) => {
  const formatTime = (secs: number): string => {
    const minutes = Math.floor(secs / 60);
    const remainingSecs = secs % 60;
    return `${minutes.toString().padStart(2, '0')}:${remainingSecs.toString().padStart(2, '0')}`;
  };

  const handleClick = () => {
    if (disabled || isProcessing) return;
    if (isRecording) {
      onStop();
    } else {
      onStart();
    }
  };

  return (
    <div className="record-container">
      <div className={`record-ring-wrapper ${isRecording ? 'recording-active' : ''}`}>
        {isRecording && (
          <>
            <div className="pulse-ring pulse-ring-1" />
            <div className="pulse-ring pulse-ring-2" />
          </>
        )}
        <button
          type="button"
          id="record-button"
          className={`record-button ${isRecording ? 'btn-recording' : ''} ${isProcessing ? 'btn-processing' : ''}`}
          onClick={handleClick}
          disabled={disabled || isProcessing}
          aria-label={isRecording ? 'Stop recording' : isProcessing ? processingLabel || 'Processing' : 'Start voice recording'}
        >
          {isProcessing ? (
            <div className="spinner" aria-hidden="true" />
          ) : isRecording ? (
            <div className="stop-square" aria-hidden="true" />
          ) : (
            <svg
              className="mic-icon"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
              aria-hidden="true"
            >
              <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z" />
              <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
              <line x1="12" y1="19" x2="12" y2="23" />
              <line x1="8" y1="23" x2="16" y2="23" />
            </svg>
          )}
        </button>
      </div>

      {/* Recording Indicator & Status */}
      <div className="record-status-area">
        {isRecording ? (
          <div className="live-recording-indicator">
            <span className="live-dot" />
            <span className="live-timer" id="recording-timer">{formatTime(durationSeconds)}</span>
            <div className="waveform-animation">
              <span className="wave-bar bar-1" />
              <span className="wave-bar bar-2" />
              <span className="wave-bar bar-3" />
              <span className="wave-bar bar-4" />
              <span className="wave-bar bar-5" />
            </div>
            <p className="record-hint">Click the stop button when you're done speaking</p>
          </div>
        ) : isProcessing ? (
          <div className="processing-indicator">
            <p className="processing-text">{processingLabel || 'Processing audio...'}</p>
          </div>
        ) : (
          <p className="record-hint">Click the microphone to start asking a question</p>
        )}
      </div>
    </div>
  );
};
