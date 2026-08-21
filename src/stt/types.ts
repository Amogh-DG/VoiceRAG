/**
 * Internal types for Sarvam STT API response handling.
 * Kept isolated inside the stt module.
 */
export interface SarvamSttSuccessResponse {
  request_id?: string;
  transcript?: string | null;
  language_code?: string;
}

export interface SarvamSttErrorResponse {
  error?: {
    message?: string;
    code?: string;
  };
  detail?: string | Array<{ msg?: string; loc?: string[] }>;
  message?: string;
}

