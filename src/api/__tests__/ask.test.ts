import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { askQuestion, AskError } from '../ask';

describe('API Module - askQuestion()', () => {
  const originalBaseUrl = process.env.VITE_API_BASE_URL;

  beforeEach(() => {
    process.env.VITE_API_BASE_URL = 'https://api.example.com';
    vi.restoreAllMocks();
  });

  afterEach(() => {
    process.env.VITE_API_BASE_URL = originalBaseUrl;
    vi.unstubAllGlobals();
  });

  it('sends POST /ask with query and returns parsed AskResponse', async () => {
    const mockResponseData = {
      answer: 'This is the answer from the document.',
      sources: [
        {
          document_id: 'doc-123',
          text: 'Source text snippet.',
          metadata: { page: 1, title: 'Guide' },
        },
      ],
      latency_ms: 342,
    };

    let requestedUrl = '';
    let requestedOptions: RequestInit | undefined;

    vi.stubGlobal(
      'fetch',
      vi.fn().mockImplementation((url, options) => {
        requestedUrl = String(url);
        requestedOptions = options;
        return Promise.resolve({
          ok: true,
          status: 200,
          json: async () => mockResponseData,
        });
      })
    );

    const result = await askQuestion('What is the return policy?');

    expect(requestedUrl).toBe('https://api.example.com/ask');
    expect(requestedOptions?.method).toBe('POST');
    expect(requestedOptions?.headers).toEqual({ 'Content-Type': 'application/json' });
    expect(JSON.parse(String(requestedOptions?.body))).toEqual({ query: 'What is the return policy?' });

    expect(result.answer).toBe('This is the answer from the document.');
    expect(result.sources).toHaveLength(1);
    expect(result.sources[0].document_id).toBe('doc-123');
    expect(result.latency_ms).toBe(342);
  });

  it('throws AskError on non-2xx HTTP responses', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: false,
        status: 500,
        statusText: 'Internal Server Error',
        json: async () => ({ detail: 'RAG retrieval pipeline failed' }),
      })
    );

    await expect(askQuestion('Test query')).rejects.toThrow(AskError);
    await expect(askQuestion('Test query')).rejects.toThrow(/RAG retrieval pipeline failed/);
  });

  it('throws AskError on network connection failure', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockRejectedValue(new Error('Network connection refused'))
    );

    await expect(askQuestion('Test query')).rejects.toThrow(AskError);
    await expect(askQuestion('Test query')).rejects.toThrow(/Network failure connecting to \/ask/);
  });

  it('throws AskError with TIMEOUT_ERROR when request exceeds timeout', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockImplementation((_url, options) => {
        return new Promise((_resolve, reject) => {
          // Wait for abort signal
          options?.signal?.addEventListener('abort', () => {
            reject(new DOMException('The operation was aborted', 'AbortError'));
          });
        });
      })
    );

    // Test with short 50ms timeout
    const askPromise = askQuestion('Will this timeout?', 50);

    await expect(askPromise).rejects.toThrow(AskError);
    await expect(askPromise).rejects.toThrow(/timed out/);
  });

  it('throws AskError if query is empty or whitespace', async () => {
    await expect(askQuestion('')).rejects.toThrow(AskError);
    await expect(askQuestion('   ')).rejects.toThrow(AskError);
  });
});
