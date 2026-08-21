/**
 * Custom error class for Speech-to-Text (STT) operations.
 */
export class SttError extends Error {
  public readonly code: string;
  public readonly statusCode?: number;

  constructor(message: string, code = 'STT_ERROR', statusCode?: number, cause?: unknown) {
    super(message);
    this.name = 'SttError';
    this.code = code;
    this.statusCode = statusCode;
    if (cause) {
      this.cause = cause;
    }

    // Maintain proper prototype chain
    Object.setPrototypeOf(this, SttError.prototype);
  }
}
