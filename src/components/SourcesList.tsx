import React from 'react';
import type { AskSource } from '../api';

interface SourcesListProps {
  sources: AskSource[];
}

export const SourcesList: React.FC<SourcesListProps> = ({ sources }) => {
  if (!sources || sources.length === 0) return null;

  return (
    <section className="qa-card sources-card" aria-label="Sources and Citations">
      <div className="card-header">
        <div className="header-title-group">
          <svg className="header-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20" />
            <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z" />
          </svg>
          <h3>Sources & Citations ({sources.length})</h3>
        </div>
      </div>

      <div className="sources-grid" id="sources-display">
        {sources.map((source, index) => {
          const hasMetadata = source.metadata && Object.keys(source.metadata).length > 0;

          return (
            <article key={`${source.document_id}-${index}`} className="source-item">
              <div className="source-meta-header">
                <span className="source-index">#{index + 1}</span>
                <span className="source-doc-id" title={source.document_id}>
                  📄 {source.document_id}
                </span>
              </div>

              <blockquote className="source-text">
                {source.text}
              </blockquote>

              {hasMetadata && (
                <div className="source-tags">
                  {Object.entries(source.metadata).map(([key, val]) => (
                    <span key={key} className="meta-tag">
                      <strong>{key}:</strong> {typeof val === 'object' ? JSON.stringify(val) : String(val)}
                    </span>
                  ))}
                </div>
              )}
            </article>
          );
        })}
      </div>
    </section>
  );
};
