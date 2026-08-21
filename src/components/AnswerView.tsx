import React from 'react';

interface AnswerViewProps {
  answer: string;
  latencyMs?: number;
  isAsking: boolean;
}

export const AnswerView: React.FC<AnswerViewProps> = ({
  answer,
  latencyMs,
  isAsking,
}) => {
  if (!answer && !isAsking) return null;

  return (
    <section className="qa-card answer-card" aria-label="AI Answer">
      <div className="card-header">
        <div className="header-title-group">
          <svg className="header-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z" />
          </svg>
          <h3>Answer</h3>
        </div>
        {typeof latencyMs === 'number' && (
          <span className="badge badge-latency" title={`Server latency: ${latencyMs}ms`}>
            ⚡ {latencyMs}ms
          </span>
        )}
      </div>

      <div className="card-body">
        {isAsking ? (
          <div className="answer-loading" aria-label="Fetching answer">
            <div className="skeleton-line full" />
            <div className="skeleton-line medium" />
            <div className="skeleton-line short" />
            <p className="loading-subtext">Searching knowledge base and formulating response...</p>
          </div>
        ) : (
          <div className="answer-content" id="answer-display">
            <p className="answer-text">{answer}</p>
          </div>
        )}
      </div>
    </section>
  );
};
