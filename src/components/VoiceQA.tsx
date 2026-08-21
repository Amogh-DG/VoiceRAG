import React, { useState, useRef, useEffect, useCallback } from 'react';
import { RecordButton } from './RecordButton';
import { TranscriptView } from './TranscriptView';
import { AnswerView } from './AnswerView';
import { SourcesList } from './SourcesList';
import { ErrorBanner, type ErrorState, type ErrorType } from './ErrorBanner';
import { transcribe, SttError } from '../stt';
import { askQuestion, AskError, type AskSource } from '../api';

export type QAStatus = 'idle' | 'recording' | 'transcribing' | 'asking' | 'done' | 'error';

export const VoiceQA: React.FC = () => {
  const [status, setStatus] = useState<QAStatus>('idle');
  const [transcript, setTranscript] = useState<string>('');
  const [answer, setAnswer] = useState<string>('');
  const [sources, setSources] = useState<AskSource[]>([]);
  const [latencyMs, setLatencyMs] = useState<number | undefined>(undefined);
  const [error, setError] = useState<ErrorState | null>(null);
  const [recordingDuration, setRecordingDuration] = useState<number>(0);

  // Cached data to allow smart retry of individual failed steps
  const [savedAudioBlob, setSavedAudioBlob] = useState<Blob | null>(null);
  const [savedTranscript, setSavedTranscript] = useState<string>('');

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const timerIntervalRef = useRef<number | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  // Cleanup timer & streams on unmount
  useEffect(() => {
    return () => {
      if (timerIntervalRef.current) {
        clearInterval(timerIntervalRef.current);
      }
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
      }
    };
  }, []);

  const clearTimer = () => {
    if (timerIntervalRef.current) {
      clearInterval(timerIntervalRef.current);
      timerIntervalRef.current = null;
    }
  };

  const resetAll = () => {
    clearTimer();
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop();
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    setStatus('idle');
    setTranscript('');
    setAnswer('');
    setSources([]);
    setLatencyMs(undefined);
    setError(null);
    setRecordingDuration(0);
    setSavedAudioBlob(null);
    setSavedTranscript('');
  };

  /**
   * Pipeline step 2: Ask backend question with the transcribed text.
   */
  const executeAsk = useCallback(async (queryText: string) => {
    setStatus('asking');
    setError(null);

    try {
      const response = await askQuestion(queryText);
      setAnswer(response.answer);
      setSources(response.sources || []);
      setLatencyMs(response.latency_ms);
      setStatus('done');
    } catch (err: unknown) {
      setStatus('error');
      if (err instanceof AskError) {
        setError({
          type: 'ask',
          message: err.message,
          code: err.code,
          statusCode: err.statusCode,
        });
      } else {
        setError({
          type: 'ask',
          message: err instanceof Error ? err.message : 'Failed to retrieve answer from server.',
        });
      }
    }
  }, []);

  /**
   * Pipeline step 1: Transcribe audio, then automatically trigger executeAsk.
   */
  const executeTranscribe = useCallback(async (audioBlob: Blob) => {
    setStatus('transcribing');
    setError(null);

    try {
      const text = await transcribe(audioBlob);

      if (!text || text.trim() === '') {
        setStatus('error');
        setError({
          type: 'empty_audio',
          message: "We didn't catch that. Please speak clearly into your microphone and try again.",
        });
        return;
      }

      setTranscript(text);
      setSavedTranscript(text);

      // Proceed directly to asking the question
      await executeAsk(text);
    } catch (err: unknown) {
      setStatus('error');
      if (err instanceof SttError) {
        setError({
          type: 'stt',
          message: err.message,
          code: err.code,
          statusCode: err.statusCode,
        });
      } else {
        setError({
          type: 'stt',
          message: err instanceof Error ? err.message : 'Speech transcription failed.',
        });
      }
    }
  }, [executeAsk]);

  /**
   * Start audio recording with browser MediaRecorder API
   */
  const startRecording = async () => {
    resetAll();

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setStatus('error');
      setError({
        type: 'permission',
        message: 'Audio recording is not supported in this browser or environment.',
      });
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
        ? 'audio/webm;codecs=opus'
        : MediaRecorder.isTypeSupported('audio/webm')
        ? 'audio/webm'
        : MediaRecorder.isTypeSupported('audio/mp4')
        ? 'audio/mp4'
        : '';

      const mediaRecorder = mimeType
        ? new MediaRecorder(stream, { mimeType })
        : new MediaRecorder(stream);

      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event: BlobEvent) => {
        if (event.data && event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        clearTimer();
        stream.getTracks().forEach((track) => track.stop());
        streamRef.current = null;

        const mime = audioChunksRef.current[0]?.type || mimeType || 'audio/webm';
        const audioBlob = new Blob(audioChunksRef.current, { type: mime });

        if (audioBlob.size === 0) {
          setStatus('error');
          setError({
            type: 'empty_audio',
            message: 'Recorded audio was empty. Please ensure your microphone is capturing audio.',
          });
          return;
        }

        setSavedAudioBlob(audioBlob);
        await executeTranscribe(audioBlob);
      };

      mediaRecorder.start(250); // Collect data in 250ms chunks
      setStatus('recording');
      setRecordingDuration(0);

      timerIntervalRef.current = window.setInterval(() => {
        setRecordingDuration((prev) => prev + 1);
      }, 1000);
    } catch (err: unknown) {
      setStatus('error');
      const errObj = err as { name?: string; message?: string } | undefined;
      const errName = errObj?.name || '';
      const errMsg = errObj?.message || '';
      const isDenied =
        errName === 'NotAllowedError' ||
        errName === 'PermissionDeniedError' ||
        errMsg.toLowerCase().includes('permission') ||
        errMsg.toLowerCase().includes('notallowed') ||
        errMsg.toLowerCase().includes('denied');

      setError({
        type: 'permission',
        message: isDenied
          ? 'Microphone access was denied. Please allow microphone permissions in your browser.'
          : 'Could not access microphone. Please verify an input device is connected.',
      });
    }
  };

  /**
   * Stop current audio recording
   */
  const stopRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop();
    }
  };

  /**
   * Retry failed step cleanly without forcing user to re-record
   */
  const handleRetry = () => {
    if (!error) return;

    if (error.type === 'stt' && savedAudioBlob) {
      // Retry transcription with the cached audio blob
      executeTranscribe(savedAudioBlob);
    } else if (error.type === 'ask' && savedTranscript) {
      // Retry asking question with the cached transcript
      executeAsk(savedTranscript);
    } else {
      // For permission, empty audio, or general errors: reset to idle
      setStatus('idle');
      setError(null);
    }
  };

  const isRecording = status === 'recording';
  const isTranscribing = status === 'transcribing';
  const isAsking = status === 'asking';
  const isProcessing = isTranscribing || isAsking;

  const getProcessingLabel = (): string => {
    if (isTranscribing) return 'Transcribing audio with Sarvam AI...';
    if (isAsking) return 'Searching knowledge base & generating answer...';
    return '';
  };

  const canRetry = Boolean(
    (error?.type === 'stt' && savedAudioBlob) ||
    (error?.type === 'ask' && savedTranscript) ||
    error?.type === 'empty_audio' ||
    error?.type === 'permission'
  );

  const getRetryLabel = (errType: ErrorType): string => {
    if (errType === 'stt') return 'Retry Transcription';
    if (errType === 'ask') return 'Retry Question';
    return 'Try Again';
  };

  return (
    <div className="voice-qa-root">
      {/* Header section */}
      <header className="qa-header">
        <div className="qa-title-row">
          <span className="qa-badge">Voice Intelligence</span>
          <h1 className="qa-main-title">Voice-Based Q&A</h1>
        </div>
        <p className="qa-subtitle">
          Ask any question using your voice. Speak naturally into your microphone to get answers with grounded citations.
        </p>
      </header>

      {/* Main Interaction Area: Record Button & Visual Feedback */}
      <main className="qa-interactive-area">
        <RecordButton
          isRecording={isRecording}
          isProcessing={isProcessing}
          processingLabel={getProcessingLabel()}
          durationSeconds={recordingDuration}
          onStart={startRecording}
          onStop={stopRecording}
          disabled={isProcessing}
        />

        {/* Error notification & Smart Retry */}
        <ErrorBanner
          error={error}
          canRetry={canRetry}
          retryLabel={error ? getRetryLabel(error.type) : 'Retry'}
          onRetry={handleRetry}
          onDismiss={() => setError(null)}
        />

        {/* Results Container */}
        <div className="qa-results-container">
          <TranscriptView
            transcript={transcript}
            isTranscribing={isTranscribing}
          />

          <AnswerView
            answer={answer}
            latencyMs={latencyMs}
            isAsking={isAsking}
          />

          <SourcesList sources={sources} />
        </div>

        {/* Reset / New Question Button */}
        {(status === 'done' || (status === 'error' && !isProcessing)) && (
          <div className="qa-footer-actions">
            <button
              type="button"
              className="btn-new-question"
              onClick={resetAll}
              id="ask-new-question-btn"
            >
              <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M12 5v14M5 12h14" />
              </svg>
              Ask Another Question
            </button>
          </div>
        )}
      </main>
    </div>
  );
};
