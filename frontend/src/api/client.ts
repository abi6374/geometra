import axios from "axios";
import type {
  ConversionRequest,
  FileUploadResponse,
  HealthResponse,
  JobResponse,
  JobStatusResponse,
} from "../types";

const api = axios.create({
  baseURL: "/api/v1",
  timeout: 30_000,
});

export const geometraApi = {
  /** Health check */
  health: async (): Promise<HealthResponse> => {
    const { data } = await api.get<HealthResponse>("/health");
    return data;
  },

  /** Upload a file */
  upload: async (file: File): Promise<FileUploadResponse> => {
    const form = new FormData();
    form.append("file", file);
    const { data } = await api.post<FileUploadResponse>("/upload", form);
    return data;
  },

  /** Start a conversion job */
  startConversion: async (req: ConversionRequest): Promise<JobResponse> => {
    const { data } = await api.post<JobResponse>("/convert", req);
    return data;
  },

  /** Get job status */
  getJobStatus: async (jobId: string): Promise<JobStatusResponse> => {
    const { data } = await api.get<JobStatusResponse>(`/jobs/${jobId}`);
    return data;
  },

  /** List all jobs */
  listJobs: async (): Promise<JobStatusResponse[]> => {
    const { data } = await api.get<JobStatusResponse[]>("/jobs");
    return data;
  },
};
