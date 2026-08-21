import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import { VoiceQA } from '../VoiceQA';
import { RecordButton } from '../RecordButton';
import { TranscriptView } from '../TranscriptView';
import { AnswerView } from '../AnswerView';
import { SourcesList } from '../SourcesList';
import { ErrorBanner } from '../ErrorBanner';
import * as sttModule from '../../stt';
import * as apiModule from '../../api';

describe('VoiceQA Component and Subcomponents', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  describe('RecordButton Subcomponent', () => {
    it('renders idle state with microphone icon', () => {
      render(
        <RecordButton
          isRecording={false}
          isProcessing={false}
          durationSeconds={0}
          onStart={vi.fn()}
          onStop={vi.fn()}
        />
      );

      const button = screen.getByRole('button', { name: /start voice recording/i });
      expect(button).toBeInTheDocument();
      expect(screen.getByText(/click the microphone to start/i)).toBeInTheDocument();
    });

    it('renders active recording state with timer and waveform', () => {
      render(
        <RecordButton
          isRecording={true}
          isProcessing={false}
          durationSeconds={12}
          onStart={vi.fn()}
          onStop={vi.fn()}
        />
      );

      const button = screen.getByRole('button', { name: /stop recording/i });
      expect(button).toHaveClass('btn-recording');
      expect(screen.getByText('00:12')).toBeInTheDocument();
      expect(screen.getByText(/click the stop button when you're done/i)).toBeInTheDocument();
    });

    it('triggers onStart and onStop callbacks on click', () => {
      const handleStart = vi.fn();
      const handleStop = vi.fn();

      const { rerender } = render(
        <RecordButton
          isRecording={false}
          isProcessing={false}
          durationSeconds={0}
          onStart={handleStart}
          onStop={handleStop}
        />
      );

      fireEvent.click(screen.getByRole('button'));
      expect(handleStart).toHaveBeenCalledTimes(1);

      rerender(
        <RecordButton
          isRecording={true}
          isProcessing={false}
          durationSeconds={5}
          onStart={handleStart}
          onStop={handleStop}
        />
      );

      fireEvent.click(screen.getByRole('button'));
      expect(handleStop).toHaveBeenCalledTimes(1);
    });
  });

  describe('TranscriptView Subcomponent', () => {
    it('renders transcribed text', () => {
      render(
        <TranscriptView
          transcript="What is the refund timeline?"
          isTranscribing={false}
        />
      );

      expect(screen.getByText(/your question/i)).toBeInTheDocument();
      expect(screen.getByText(/"What is the refund timeline\?"/)).toBeInTheDocument();
      expect(screen.getByText('Transcribed')).toBeInTheDocument();
    });

    it('renders loading state when transcribing', () => {
      render(
        <TranscriptView
          transcript=""
          isTranscribing={true}
        />
      );

      expect(screen.getByText(/converting your speech into text/i)).toBeInTheDocument();
    });
  });

  describe('AnswerView Subcomponent', () => {
    it('renders answer text and latency badge', () => {
      render(
        <AnswerView
          answer="Refunds are processed in 3 to 5 business days."
          latencyMs={180}
          isAsking={false}
        />
      );

      expect(screen.getByText(/answer/i)).toBeInTheDocument();
      expect(screen.getByText('Refunds are processed in 3 to 5 business days.')).toBeInTheDocument();
      expect(screen.getByText('⚡ 180ms')).toBeInTheDocument();
    });

    it('renders cleanly when latencyMs is undefined / missing', () => {
      render(
        <AnswerView
          answer="Direct answer without latency metric."
          latencyMs={undefined}
          isAsking={false}
        />
      );

      expect(screen.getByText('Direct answer without latency metric.')).toBeInTheDocument();
      expect(screen.queryByText(/⚡/)).not.toBeInTheDocument();
    });

    it('renders skeleton loading state when asking', () => {
      render(
        <AnswerView
          answer=""
          isAsking={true}
        />
      );

      expect(screen.getByText(/searching knowledge base and formulating response/i)).toBeInTheDocument();
    });
  });

  describe('SourcesList Subcomponent', () => {
    it('renders citations with document_id and metadata', () => {
      const mockSources = [
        {
          document_id: 'policy-doc-01.pdf',
          text: 'Section 4: Refunds take 3 to 5 business days.',
          metadata: { page: 4, section: 'Refunds' },
        },
      ];

      render(<SourcesList sources={mockSources} />);

      expect(screen.getByText(/sources & citations \(1\)/i)).toBeInTheDocument();
      expect(screen.getByText(/policy-doc-01\.pdf/)).toBeInTheDocument();
      expect(screen.getByText(/Section 4: Refunds take 3 to 5 business days\./)).toBeInTheDocument();
      expect(screen.getByText('Refunds')).toBeInTheDocument();
    });
  });

  describe('ErrorBanner Subcomponent', () => {
    it('renders distinct error types and triggers retry callback', () => {
      const handleRetry = vi.fn();
      const handleDismiss = vi.fn();

      render(
        <ErrorBanner
          error={{
            type: 'stt',
            message: 'Sarvam API key is invalid',
            code: 'API_ERROR',
            statusCode: 401,
          }}
          canRetry={true}
          retryLabel="Retry Transcription"
          onRetry={handleRetry}
          onDismiss={handleDismiss}
        />
      );

      expect(screen.getByText('Transcription Failed')).toBeInTheDocument();
      expect(screen.getByText('Sarvam API key is invalid')).toBeInTheDocument();
      expect(screen.getByText('Status code: 401')).toBeInTheDocument();

      const retryBtn = screen.getByRole('button', { name: /retry transcription/i });
      fireEvent.click(retryBtn);
      expect(handleRetry).toHaveBeenCalledTimes(1);

      const dismissBtn = screen.getByRole('button', { name: /dismiss error/i });
      fireEvent.click(dismissBtn);
      expect(handleDismiss).toHaveBeenCalledTimes(1);
    });
  });

  describe('VoiceQA Integration Flow', () => {
    it('handles successful voice recording, transcription, and QA pipeline', async () => {
      vi.spyOn(sttModule, 'transcribe').mockResolvedValue('What is the return policy?');
      vi.spyOn(apiModule, 'askQuestion').mockResolvedValue({
        answer: 'Returns are accepted within 30 days of purchase.',
        sources: [
          {
            document_id: 'faq.md',
            text: 'You may return items within 30 days.',
            metadata: { category: 'returns' },
          },
        ],
        latency_ms: 220,
      });

      const mockStream = {
        getTracks: () => [{ stop: vi.fn() }],
      };

      Object.defineProperty(navigator, 'mediaDevices', {
        value: {
          getUserMedia: vi.fn().mockResolvedValue(mockStream),
        },
        writable: true,
        configurable: true,
      });

      let ondataavailableCallback: ((event: { data: Blob }) => void) | null = null;
      let onstopCallback: (() => void) | null = null;

      class MockMediaRecorder {
        state = 'inactive';
        static isTypeSupported = vi.fn().mockReturnValue(true);
        constructor() {}

        start() {
          this.state = 'recording';
        }

        stop() {
          this.state = 'inactive';
          if (ondataavailableCallback) {
            ondataavailableCallback({
              data: new Blob(['audio binary data'], { type: 'audio/webm' }),
            });
          }
          if (onstopCallback) {
            onstopCallback();
          }
        }

        set ondataavailable(cb: (event: { data: Blob }) => void) {
          ondataavailableCallback = cb;
        }

        set onstop(cb: () => void) {
          onstopCallback = cb;
        }
      }

      global.MediaRecorder = MockMediaRecorder as unknown as typeof MediaRecorder;

      render(<VoiceQA />);

      // 1. Click to start recording
      const recordBtn = screen.getByRole('button', { name: /start voice recording/i });
      await act(async () => {
        fireEvent.click(recordBtn);
      });

      // 2. Click to stop recording
      const stopBtn = screen.getByRole('button', { name: /stop recording/i });
      await act(async () => {
        fireEvent.click(stopBtn);
      });

      // 3. Wait for transcription & ask response
      await waitFor(() => {
        expect(screen.getByText(/"What is the return policy\?"/)).toBeInTheDocument();
      });

      await waitFor(() => {
        expect(screen.getByText('Returns are accepted within 30 days of purchase.')).toBeInTheDocument();
        expect(screen.getByText('⚡ 220ms')).toBeInTheDocument();
        expect(screen.getByText(/faq\.md/)).toBeInTheDocument();
      });

      // 4. "Ask Another Question" button resets state
      const newQuestionBtn = screen.getByRole('button', { name: /ask another question/i });
      fireEvent.click(newQuestionBtn);

      expect(screen.queryByText('Returns are accepted within 30 days of purchase.')).not.toBeInTheDocument();
    });

    it('denying mic permission shows a clear error state and not a silent failure', async () => {
      const permissionError = new DOMException('Permission denied', 'NotAllowedError');
      Object.defineProperty(navigator, 'mediaDevices', {
        value: {
          getUserMedia: vi.fn().mockRejectedValue(permissionError),
        },
        writable: true,
        configurable: true,
      });

      render(<VoiceQA />);

      const recordBtn = screen.getByRole('button', { name: /start voice recording/i });
      await act(async () => {
        fireEvent.click(recordBtn);
      });

      await waitFor(() => {
        expect(screen.getByText('Microphone Permission Required')).toBeInTheDocument();
        expect(screen.getByText(/microphone access was denied/i)).toBeInTheDocument();
      });
    });

    it('recording with no speech shows "we didn\'t catch that" and does NOT call askQuestion', async () => {
      vi.spyOn(sttModule, 'transcribe').mockResolvedValue('');
      const askQuestionSpy = vi.spyOn(apiModule, 'askQuestion');

      const mockStream = {
        getTracks: () => [{ stop: vi.fn() }],
      };

      Object.defineProperty(navigator, 'mediaDevices', {
        value: {
          getUserMedia: vi.fn().mockResolvedValue(mockStream),
        },
        writable: true,
        configurable: true,
      });

      let ondataavailableCallback: ((event: { data: Blob }) => void) | null = null;
      let onstopCallback: (() => void) | null = null;

      class MockMediaRecorder {
        state = 'inactive';
        static isTypeSupported = vi.fn().mockReturnValue(true);
        constructor() {}
        start() { this.state = 'recording'; }
        stop() {
          this.state = 'inactive';
          if (ondataavailableCallback) {
            ondataavailableCallback({ data: new Blob(['silent audio'], { type: 'audio/webm' }) });
          }
          if (onstopCallback) onstopCallback();
        }
        set ondataavailable(cb: (event: { data: Blob }) => void) { ondataavailableCallback = cb; }
        set onstop(cb: () => void) { onstopCallback = cb; }
      }

      global.MediaRecorder = MockMediaRecorder as unknown as typeof MediaRecorder;

      render(<VoiceQA />);

      await act(async () => {
        fireEvent.click(screen.getByRole('button', { name: /start voice recording/i }));
      });
      await act(async () => {
        fireEvent.click(screen.getByRole('button', { name: /stop recording/i }));
      });

      await waitFor(() => {
        expect(screen.getByText("No Speech Detected")).toBeInTheDocument();
        expect(screen.getByText(/we didn't catch that/i)).toBeInTheDocument();
      });

      expect(askQuestionSpy).not.toHaveBeenCalled();
    });

    it('displays clear 403 Forbidden error in UI when API key is invalid', async () => {
      vi.spyOn(sttModule, 'transcribe').mockRejectedValue(
        new sttModule.SttError(
          'Forbidden - Invalid or missing Sarvam API subscription key.',
          'FORBIDDEN',
          403
        )
      );
      const askQuestionSpy = vi.spyOn(apiModule, 'askQuestion');

      const mockStream = {
        getTracks: () => [{ stop: vi.fn() }],
      };

      Object.defineProperty(navigator, 'mediaDevices', {
        value: {
          getUserMedia: vi.fn().mockResolvedValue(mockStream),
        },
        writable: true,
        configurable: true,
      });

      let ondataavailableCallback: ((event: { data: Blob }) => void) | null = null;
      let onstopCallback: (() => void) | null = null;

      class MockMediaRecorder {
        state = 'inactive';
        static isTypeSupported = vi.fn().mockReturnValue(true);
        constructor() {}
        start() { this.state = 'recording'; }
        stop() {
          this.state = 'inactive';
          if (ondataavailableCallback) {
            ondataavailableCallback({ data: new Blob(['audio data'], { type: 'audio/webm' }) });
          }
          if (onstopCallback) onstopCallback();
        }
        set ondataavailable(cb: (event: { data: Blob }) => void) { ondataavailableCallback = cb; }
        set onstop(cb: () => void) { onstopCallback = cb; }
      }

      global.MediaRecorder = MockMediaRecorder as unknown as typeof MediaRecorder;

      render(<VoiceQA />);

      await act(async () => {
        fireEvent.click(screen.getByRole('button', { name: /start voice recording/i }));
      });
      await act(async () => {
        fireEvent.click(screen.getByRole('button', { name: /stop recording/i }));
      });

      await waitFor(() => {
        expect(screen.getByText('Transcription Failed')).toBeInTheDocument();
        expect(screen.getByText(/Forbidden - Invalid or missing Sarvam API subscription key/i)).toBeInTheDocument();
        expect(screen.getByText('Status code: 403')).toBeInTheDocument();
      });

      expect(askQuestionSpy).not.toHaveBeenCalled();
    });


    it('handles STT failure and allows retry without re-recording', async () => {
      const transcribeSpy = vi.spyOn(sttModule, 'transcribe')
        .mockRejectedValueOnce(new sttModule.SttError('Sarvam API network error', 'NETWORK_ERROR'))
        .mockResolvedValueOnce('Retried question text');

      vi.spyOn(apiModule, 'askQuestion').mockResolvedValue({
        answer: 'Answer after retry',
        sources: [],
      });

      const mockStream = {
        getTracks: () => [{ stop: vi.fn() }],
      };

      Object.defineProperty(navigator, 'mediaDevices', {
        value: {
          getUserMedia: vi.fn().mockResolvedValue(mockStream),
        },
        writable: true,
        configurable: true,
      });

      let ondataavailableCallback: ((event: { data: Blob }) => void) | null = null;
      let onstopCallback: (() => void) | null = null;

      class MockMediaRecorder {
        state = 'inactive';
        static isTypeSupported = vi.fn().mockReturnValue(true);
        constructor() {}
        start() { this.state = 'recording'; }
        stop() {
          this.state = 'inactive';
          if (ondataavailableCallback) {
            ondataavailableCallback({ data: new Blob(['audio data'], { type: 'audio/webm' }) });
          }
          if (onstopCallback) onstopCallback();
        }
        set ondataavailable(cb: (event: { data: Blob }) => void) { ondataavailableCallback = cb; }
        set onstop(cb: () => void) { onstopCallback = cb; }
      }

      global.MediaRecorder = MockMediaRecorder as unknown as typeof MediaRecorder;

      render(<VoiceQA />);

      await act(async () => {
        fireEvent.click(screen.getByRole('button', { name: /start voice recording/i }));
      });
      await act(async () => {
        fireEvent.click(screen.getByRole('button', { name: /stop recording/i }));
      });

      await waitFor(() => {
        expect(screen.getByText('Transcription Failed')).toBeInTheDocument();
        expect(screen.getByText('Sarvam API network error')).toBeInTheDocument();
      });

      const retryBtn = screen.getByRole('button', { name: /retry transcription/i });
      expect(retryBtn).toBeInTheDocument();

      await act(async () => {
        fireEvent.click(retryBtn);
      });

      await waitFor(() => {
        expect(screen.getByText('Answer after retry')).toBeInTheDocument();
      });

      expect(transcribeSpy).toHaveBeenCalledTimes(2);
    });

    it('handles /ask backend failure and allows retry without re-transcribing', async () => {
      const transcribeSpy = vi.spyOn(sttModule, 'transcribe').mockResolvedValue('What are the store hours?');
      const askQuestionSpy = vi.spyOn(apiModule, 'askQuestion')
        .mockRejectedValueOnce(new apiModule.AskError('Backend service unavailable (503)', 'API_ERROR', 503))
        .mockResolvedValueOnce({
          answer: 'We are open 9am to 9pm daily.',
          sources: [],
        });

      const mockStream = {
        getTracks: () => [{ stop: vi.fn() }],
      };

      Object.defineProperty(navigator, 'mediaDevices', {
        value: {
          getUserMedia: vi.fn().mockResolvedValue(mockStream),
        },
        writable: true,
        configurable: true,
      });

      let ondataavailableCallback: ((event: { data: Blob }) => void) | null = null;
      let onstopCallback: (() => void) | null = null;

      class MockMediaRecorder {
        state = 'inactive';
        static isTypeSupported = vi.fn().mockReturnValue(true);
        constructor() {}
        start() { this.state = 'recording'; }
        stop() {
          this.state = 'inactive';
          if (ondataavailableCallback) {
            ondataavailableCallback({ data: new Blob(['audio data'], { type: 'audio/webm' }) });
          }
          if (onstopCallback) onstopCallback();
        }
        set ondataavailable(cb: (event: { data: Blob }) => void) { ondataavailableCallback = cb; }
        set onstop(cb: () => void) { onstopCallback = cb; }
      }

      global.MediaRecorder = MockMediaRecorder as unknown as typeof MediaRecorder;

      render(<VoiceQA />);

      await act(async () => {
        fireEvent.click(screen.getByRole('button', { name: /start voice recording/i }));
      });
      await act(async () => {
        fireEvent.click(screen.getByRole('button', { name: /stop recording/i }));
      });

      // Transcript should be visible and error banner rendered
      await waitFor(() => {
        expect(screen.getByText(/"What are the store hours\?"/)).toBeInTheDocument();
        expect(screen.getByText('Knowledge Query Failed')).toBeInTheDocument();
        expect(screen.getByText(/Backend service unavailable/)).toBeInTheDocument();
      });

      const retryBtn = screen.getByRole('button', { name: /retry question/i });
      expect(retryBtn).toBeInTheDocument();

      // Click retry
      await act(async () => {
        fireEvent.click(retryBtn);
      });

      await waitFor(() => {
        expect(screen.getByText('We are open 9am to 9pm daily.')).toBeInTheDocument();
      });

      // transcribe should only have been called ONCE; askQuestion called twice
      expect(transcribeSpy).toHaveBeenCalledTimes(1);
      expect(askQuestionSpy).toHaveBeenCalledTimes(2);
    });
  });
});
