import { transcribeWithSarvam } from './sarvam';
export { SttError } from './errors';

/**
 * Transcribes audio from a Blob or File into a plain string transcript.
 *
 * @param audioFile - Audio data as a Blob or File
 * @returns Plain string containing ONLY the transcribed text (or "" for empty/silent audio)
 * @throws {SttError} on network failure, missing API key, or API errors
 */
export async function transcribe(audioFile: Blob | File): Promise<string> {
  return transcribeWithSarvam(audioFile);
}
