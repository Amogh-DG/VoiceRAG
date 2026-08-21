/**
 * Typed error for Q&A backend API interactions.
 */
export class AskError extends Error {
  public readonly code: string;
  public readonly statusCode?: number;

  constructor(message: string, code = 'ASK_ERROR', statusCode?: number, cause?: unknown) {
    super(message);
    this.name = 'AskError';
    this.code = code;
    this.statusCode = statusCode;
    if (cause) {
      this.cause = cause;
    }

    Object.setPrototypeOf(this, AskError.prototype);
  }
}

export interface AskSource {
  document_id: string;
  text: string;
  metadata: Record<string, unknown>;
}

export interface AskResponse {
  answer: string;
  sources: AskSource[];
  latency_ms?: number;
}

const DEFAULT_TIMEOUT_MS = 30000; // 30 seconds

function getBaseUrl(): string {
  if (typeof import.meta !== 'undefined' && import.meta.env?.VITE_API_BASE_URL) {
    const envUrl = import.meta.env.VITE_API_BASE_URL.trim();
    if (envUrl) return envUrl.replace(/\/+$/, '');
  }
  if (typeof process !== 'undefined' && process.env?.VITE_API_BASE_URL) {
    const envUrl = process.env.VITE_API_BASE_URL.trim();
    if (envUrl) return envUrl.replace(/\/+$/, '');
  }
  return 'http://localhost:8000';
}

async function extractErrorDetails(response: Response): Promise<string> {
  try {
    const errorJson = await response.json();
    if (typeof errorJson.detail === 'string') {
      return errorJson.detail;
    }
    if (Array.isArray(errorJson.detail)) {
      return errorJson.detail.map((d: { msg?: string }) => d.msg || JSON.stringify(d)).join(', ');
    }
    if (errorJson.error?.message) {
      return errorJson.error.message;
    }
    if (errorJson.message) {
      return errorJson.message;
    }
    return JSON.stringify(errorJson);
  } catch {
    try {
      return await response.text();
    } catch {
      return response.statusText;
    }
  }
}

/**
 * Sends a query string to the backend /ask endpoint.
 * The frontend never communicates with vector DB, embedding model, or LLM directly.
 *
 * @param query - User question string
 * @param timeoutMs - Request timeout in milliseconds (defaults to 30,000ms)
 * @returns Promise<AskResponse> containing answer, sources, and optional latency_ms
 * @throws {AskError} on network failure, timeout, non-2xx response, or invalid payload
 */
export async function askQuestion(query: string, timeoutMs = DEFAULT_TIMEOUT_MS): Promise<AskResponse> {
  if (!query || typeof query !== 'string' || query.trim() === '') {
    throw new AskError('Query string must not be empty.', 'INVALID_QUERY');
  }

  const baseUrl = getBaseUrl();
  const endpoint = `${baseUrl}/ask`;

  const controller = new AbortController();
  const timeoutId = setTimeout(() => {
    controller.abort(new Error(`Request timed out after ${timeoutMs / 1000}s`));
  }, timeoutMs);

  let response: Response;
  try {
    response = await fetch(endpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ query: query.trim() }),
      signal: controller.signal,
    });
  } catch (err: unknown) {
    if (controller.signal.aborted) {
      throw new AskError(
        `Request to /ask timed out after ${timeoutMs / 1000} seconds.`,
        'TIMEOUT_ERROR',
        undefined,
        err
      );
    }
    throw new AskError(
      `Network failure connecting to /ask: ${err instanceof Error ? err.message : String(err)}`,
      'NETWORK_ERROR',
      undefined,
      err
    );
  } finally {
    clearTimeout(timeoutId);
  }

  if (!response.ok) {
    const errorDetails = await extractErrorDetails(response);
    const message = errorDetails
      ? `Ask request failed (${response.status}): ${errorDetails}`
      : `Ask request failed with status ${response.status}`;

    throw new AskError(message, 'API_ERROR', response.status);
  }

  let data: unknown;
  try {
    data = await response.json();
  } catch (err) {
    throw new AskError('Failed to parse JSON response from /ask.', 'INVALID_RESPONSE', response.status, err);
  }

  const res = data as AskResponse;
  if (!res || typeof res.answer !== 'string') {
    throw new AskError('Invalid response structure: missing answer string.', 'INVALID_RESPONSE_STRUCTURE', response.status);
  }

  return {
    answer: res.answer,
    sources: Array.isArray(res.sources) ? res.sources : [],
    ...(typeof res.latency_ms === 'number' ? { latency_ms: res.latency_ms } : {}),
  };
}
