import { SttError } from './errors';
import type { SarvamSttSuccessResponse, SarvamSttErrorResponse } from './types';

const SARVAM_STT_ENDPOINT = 'https://api.sarvam.ai/speech-to-text';

function getApiKey(): string {
  if (typeof import.meta !== 'undefined' && import.meta.env?.VITE_SARVAM_API_KEY) {
    return import.meta.env.VITE_SARVAM_API_KEY;
  }
  if (typeof process !== 'undefined' && process.env?.VITE_SARVAM_API_KEY) {
    return process.env.VITE_SARVAM_API_KEY;
  }
  return '';
}

function getStatusErrorMessage(status: number, detail?: string): { message: string; code: string } {
  let defaultMsg: string;
  let code: string;

  switch (status) {
    case 400:
      defaultMsg = 'Bad request - Invalid audio format, encoding, or parameters.';
      code = 'BAD_REQUEST';
      break;
    case 401:
    case 403:
      defaultMsg = 'Forbidden - Invalid or missing Sarvam API subscription key.';
      code = 'FORBIDDEN';
      break;
    case 422:
      defaultMsg = 'Unprocessable entity - The audio payload could not be processed.';
      code = 'UNPROCESSABLE_ENTITY';
      break;
    case 429:
      defaultMsg = 'Rate limit exceeded - Too many requests sent to Sarvam API.';
      code = 'RATE_LIMITED';
      break;
    case 500:
      defaultMsg = 'Sarvam internal server error - Failed to process speech recognition.';
      code = 'SERVER_ERROR';
      break;
    case 503:
      defaultMsg = 'Sarvam service unavailable - Speech recognition service is temporarily overloaded or down.';
      code = 'SERVER_ERROR';
      break;
    default:
      defaultMsg = `Sarvam STT failed with HTTP status ${status}.`;
      code = 'API_ERROR';
      break;
  }

  const message = detail ? `${defaultMsg} (${detail})` : defaultMsg;
  return { message, code };
}

/**
 * Sends audio to Sarvam AI speech-to-text REST API and returns the plain transcript string.
 * Isolated internally within stt module.
 */
export async function transcribeWithSarvam(audioFile: Blob | File): Promise<string> {
  if (!audioFile || !(audioFile instanceof Blob)) {
    throw new SttError(
      'Invalid audio input: Expected a valid Blob or File instance.',
      'INVALID_AUDIO_INPUT'
    );
  }

  const apiKey = getApiKey();
  if (!apiKey || apiKey.trim() === '') {
    throw new SttError(
      'Sarvam API key is not configured. Please set VITE_SARVAM_API_KEY in your .env file.',
      'MISSING_API_KEY'
    );
  }

  const rawMime = (audioFile.type || 'audio/webm').toLowerCase();
  let cleanMime = 'audio/webm';
  let ext = 'webm';

  if (rawMime.includes('webm')) {
    cleanMime = 'audio/webm';
    ext = 'webm';
  } else if (rawMime.includes('mp4') || rawMime.includes('m4a') || rawMime.includes('aac')) {
    cleanMime = 'audio/mp4';
    ext = 'mp4';
  } else if (rawMime.includes('ogg') || rawMime.includes('opus')) {
    cleanMime = 'audio/ogg';
    ext = 'ogg';
  } else if (rawMime.includes('mp3') || rawMime.includes('mpeg')) {
    cleanMime = 'audio/mp3';
    ext = 'mp3';
  } else if (rawMime.includes('flac')) {
    cleanMime = 'audio/flac';
    ext = 'flac';
  } else if (rawMime.includes('wav')) {
    cleanMime = 'audio/wav';
    ext = 'wav';
  }

  const fileName = audioFile instanceof File && audioFile.name
    ? audioFile.name
    : `recording.${ext}`;

  // Sanitize Blob MIME type to strip codec parameters (e.g. audio/webm;codecs=opus -> audio/webm)
  // as Sarvam's REST endpoint strictly validates against a whitelist of clean MIME types.
  const sanitizedAudio = audioFile.type === cleanMime
    ? audioFile
    : (audioFile.slice ? audioFile.slice(0, audioFile.size, cleanMime) : new Blob([audioFile], { type: cleanMime }));

  const formData = new FormData();
  formData.append('file', sanitizedAudio, fileName);



  formData.append('model', 'saaras:v3');
  formData.append('language_code', 'unknown');
  formData.append('with_timestamps', 'false');

  let response: Response;
  try {
    response = await fetch(SARVAM_STT_ENDPOINT, {
      method: 'POST',
      headers: {
        'api-subscription-key': apiKey.trim(),
      },
      body: formData,
    });
  } catch (error) {
    throw new SttError(
      `Network request to Sarvam STT failed: ${error instanceof Error ? error.message : String(error)}`,
      'NETWORK_ERROR',
      undefined,
      error
    );
  }

  if (!response.ok) {
    let errorDetails = '';
    try {
      const errorJson = (await response.json()) as SarvamSttErrorResponse;
      if (typeof errorJson.detail === 'string') {
        errorDetails = errorJson.detail;
      } else if (Array.isArray(errorJson.detail)) {
        errorDetails = errorJson.detail.map((d) => d.msg || JSON.stringify(d)).join(', ');
      } else if (errorJson.error?.message) {
        errorDetails = errorJson.error.message;
      } else if (errorJson.message) {
        errorDetails = errorJson.message;
      }
    } catch {
      try {
        errorDetails = await response.text();
      } catch {
        errorDetails = response.statusText;
      }
    }

    const { message, code } = getStatusErrorMessage(response.status, errorDetails);
    throw new SttError(message, code, response.status);
  }

  let data: SarvamSttSuccessResponse;
  try {
    data = (await response.json()) as SarvamSttSuccessResponse;
  } catch (error) {
    throw new SttError(
      'Failed to parse Sarvam STT response JSON.',
      'INVALID_RESPONSE',
      response.status,
      error
    );
  }

  // Handle missing or empty transcript cleanly by returning an empty string
  if (data.transcript === null || data.transcript === undefined) {
    return '';
  }

  return String(data.transcript).trim();
}

