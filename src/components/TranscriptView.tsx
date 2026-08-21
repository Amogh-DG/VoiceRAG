import React from 'react';

interface TranscriptViewProps {
  transcript: string;
  isTranscribing: boolean;
}

export const TranscriptView: React.FC<TranscriptViewProps> = ({
  transcript,
  isTranscribing,
}) => {
  if (!transcript && !isTranscribing) return null;

  return (
    <section className="qa-card transcript-card" aria-label="Transcribed Speech">
      <div className="card-header">
        <div className="header-title-group">
          <svg className="header-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z" />
            <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
          </svg>
          <h3>Your Question</h3>
        </div>
        <span className="badge badge-stt">Transcribed</span>
      </div>

      <div className="card-body">
        {isTranscribing ? (
          <div className="transcript-loading" aria-label="Transcribing audio">
            <div className="loading-shimmer" />
            <p className="loading-subtext">Converting your speech into text with Sarvam AI...</p>
          </div>
        ) : (
          <p className="transcript-text" id="transcript-display">
            "{transcript}"
          </p>
        )}
      </div>
    </section>
  );
};
