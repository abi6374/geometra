/** Geometra frontend type definitions */

export type ConversionDirection = "2d_to_3d" | "3d_to_2d";

export type JobStatus = "pending" | "processing" | "completed" | "failed";

export interface FileUploadResponse {
  filename: string;
  file_path: string;
  file_size_bytes: number;
  detected_format: string | null;
  detected_direction: ConversionDirection | null;
}

export interface JobResponse {
  job_id: string;
  status: JobStatus;
  direction: ConversionDirection;
  created_at: string;
}

export interface JobStatusResponse {
  job_id: string;
  status: JobStatus;
  direction: ConversionDirection;
  progress: number;
  message: string;
  result_paths: string[];
  validation_report: Record<string, unknown> | null;
  created_at: string;
  updated_at: string;
  error: string | null;
}

export interface ConversionRequest {
  direction: ConversionDirection;
  file_path: string;
  options?: Record<string, unknown>;
}

export interface HealthResponse {
  status: string;
  version: string;
}
