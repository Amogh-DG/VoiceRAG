import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { transcribe, SttError } from '../index';

describe('STT Module - transcribe()', () => {
  const originalEnv = process.env.VITE_SARVAM_API_KEY;

  beforeEach(() => {
    process.env.VITE_SARVAM_API_KEY = 'mock-test-sarvam-api-key';
    vi.restoreAllMocks();
  });

  afterEach(() => {
    process.env.VITE_SARVAM_API_KEY = originalEnv;
    vi.unstubAllGlobals();
  });

  it('transcribe() resolves to a plain string and passes correct contract parameters', async () => {
    const mockAudioBlob = new Blob(['dummy audio binary data'], { type: 'audio/webm' });
    let capturedUrl = '';
    let capturedOptions: RequestInit | undefined;

    vi.stubGlobal(
      'fetch',
      vi.fn().mockImplementation(async (url: string, options: RequestInit) => {
        capturedUrl = url;
        capturedOptions = options;
        return {
          ok: true,
          status: 200,
          json: async () => ({
            request_id: 'req_12345',
            transcript: 'Hello, this is a live Sarvam transcription.',
            language_code: 'hi-IN',
          }),
        };
      })
    );

    const result = await transcribe(mockAudioBlob);

    expect(typeof result).toBe('string');
    expect(result).toBe('Hello, this is a live Sarvam transcription.');
    expect(capturedUrl).toBe('https://api.sarvam.ai/speech-to-text');
    expect(capturedOptions?.method).toBe('POST');
    expect(capturedOptions?.headers).toEqual({
      'api-subscription-key': 'mock-test-sarvam-api-key',
    });

    const formData = capturedOptions?.body as FormData;
    expect(formData).toBeInstanceOf(FormData);
    expect(formData.get('file')).toBeTruthy();
    expect(formData.get('model')).toBe('saaras:v3');
    expect(formData.get('language_code')).toBe('unknown');
    expect(formData.get('with_timestamps')).toBe('false');
  });

  it('sanitizes audio/webm;codecs=opus MIME type to audio/webm to conform with Sarvam allowed types', async () => {
    const opusBlob = new Blob(['dummy opus audio'], { type: 'audio/webm;codecs=opus' });
    let capturedOptions: RequestInit | undefined;

    vi.stubGlobal(
      'fetch',
      vi.fn().mockImplementation(async (_url: string, options: RequestInit) => {
        capturedOptions = options;
        return {
          ok: true,
          status: 200,
          json: async () => ({ transcript: 'Test speech' }),
        };
      })
    );

    await transcribe(opusBlob);

    const formData = capturedOptions?.body as FormData;
    const attachedFile = formData.get('file') as Blob;
    expect(attachedFile.type).toBe('audio/webm');
  });


  it('returns empty string "" when transcript is null or missing', async () => {
    const mockAudioBlob = new Blob(['silent audio'], { type: 'audio/webm' });

    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: true,
        status: 200,
        json: async () => ({ transcript: null }),
      })
    );

    const result = await transcribe(mockAudioBlob);
    expect(result).toBe('');
  });

  it('returns empty string "" when transcript contains only whitespace', async () => {
    const mockAudioBlob = new Blob(['silent audio'], { type: 'audio/webm' });

    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: true,
        status: 200,
        json: async () => ({ transcript: '   \n\t  ' }),
      })
    );

    const result = await transcribe(mockAudioBlob);
    expect(result).toBe('');
  });

  it('maps 400 Bad Request to SttError with BAD_REQUEST code', async () => {
    const mockAudioBlob = new Blob(['corrupted audio data'], { type: 'audio/webm' });

    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: false,
        status: 400,
        json: async () => ({ detail: 'Invalid audio header format' }),
      })
    );

    try {
      await transcribe(mockAudioBlob);
      expect.unreachable('Should have thrown SttError');
    } catch (err) {
      expect(err).toBeInstanceOf(SttError);
      const sttErr = err as SttError;
      expect(sttErr.code).toBe('BAD_REQUEST');
      expect(sttErr.statusCode).toBe(400);
      expect(sttErr.message).toMatch(/Bad request/i);
    }
  });

  it('maps 403 Forbidden to SttError with FORBIDDEN code', async () => {
    const mockAudioBlob = new Blob(['audio data'], { type: 'audio/webm' });

    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: false,
        status: 403,
        json: async () => ({ error: { message: 'Invalid subscription key' } }),
      })
    );

    try {
      await transcribe(mockAudioBlob);
      expect.unreachable('Should have thrown SttError');
    } catch (err) {
      expect(err).toBeInstanceOf(SttError);
      const sttErr = err as SttError;
      expect(sttErr.code).toBe('FORBIDDEN');
      expect(sttErr.statusCode).toBe(403);
      expect(sttErr.message).toMatch(/Forbidden/i);
    }
  });

  it('maps 422 Unprocessable Entity to SttError with UNPROCESSABLE_ENTITY code', async () => {
    const mockAudioBlob = new Blob(['audio data'], { type: 'audio/webm' });

    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: false,
        status: 422,
        json: async () => ({ detail: [{ msg: 'Audio length exceeds 30s limit' }] }),
      })
    );

    try {
      await transcribe(mockAudioBlob);
      expect.unreachable('Should have thrown SttError');
    } catch (err) {
      expect(err).toBeInstanceOf(SttError);
      const sttErr = err as SttError;
      expect(sttErr.code).toBe('UNPROCESSABLE_ENTITY');
      expect(sttErr.statusCode).toBe(422);
      expect(sttErr.message).toMatch(/Unprocessable entity/i);
      expect(sttErr.message).toMatch(/Audio length exceeds 30s limit/);
    }
  });

  it('maps 429 Rate Limited to SttError with RATE_LIMITED code', async () => {
    const mockAudioBlob = new Blob(['audio data'], { type: 'audio/webm' });

    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: false,
        status: 429,
        json: async () => ({ message: 'Rate limit exceeded' }),
      })
    );

    try {
      await transcribe(mockAudioBlob);
      expect.unreachable('Should have thrown SttError');
    } catch (err) {
      expect(err).toBeInstanceOf(SttError);
      const sttErr = err as SttError;
      expect(sttErr.code).toBe('RATE_LIMITED');
      expect(sttErr.statusCode).toBe(429);
      expect(sttErr.message).toMatch(/Rate limit exceeded/i);
    }
  });

  it('maps 500 / 503 Server Error to SttError with SERVER_ERROR code', async () => {
    const mockAudioBlob = new Blob(['audio data'], { type: 'audio/webm' });

    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: false,
        status: 503,
        statusText: 'Service Unavailable',
        text: async () => 'Service Temporarily Unavailable',
      })
    );

    try {
      await transcribe(mockAudioBlob);
      expect.unreachable('Should have thrown SttError');
    } catch (err) {
      expect(err).toBeInstanceOf(SttError);
      const sttErr = err as SttError;
      expect(sttErr.code).toBe('SERVER_ERROR');
      expect(sttErr.statusCode).toBe(503);
      expect(sttErr.message).toMatch(/Sarvam service unavailable/i);
    }
  });

  it('throws SttError with NETWORK_ERROR code on network failure', async () => {
    const mockAudioBlob = new Blob(['audio data'], { type: 'audio/webm' });

    vi.stubGlobal(
      'fetch',
      vi.fn().mockRejectedValue(new Error('Failed to connect to Sarvam API host'))
    );

    try {
      await transcribe(mockAudioBlob);
      expect.unreachable('Should have thrown SttError');
    } catch (err) {
      expect(err).toBeInstanceOf(SttError);
      const sttErr = err as SttError;
      expect(sttErr.code).toBe('NETWORK_ERROR');
      expect(sttErr.message).toMatch(/Network request to Sarvam STT failed/);
    }
  });

  it('throws SttError when API key is missing', async () => {
    delete process.env.VITE_SARVAM_API_KEY;
    const mockAudioBlob = new Blob(['audio data'], { type: 'audio/webm' });

    try {
      await transcribe(mockAudioBlob);
      expect.unreachable('Should have thrown SttError');
    } catch (err) {
      expect(err).toBeInstanceOf(SttError);
      const sttErr = err as SttError;
      expect(sttErr.code).toBe('MISSING_API_KEY');
      expect(sttErr.message).toMatch(/Sarvam API key is not configured/);
    }
  });
});

