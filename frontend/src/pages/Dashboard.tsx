import { useCallback, useEffect, useRef, useState } from "react";
import { Box, RefreshCw, Activity, Clock, CheckCircle, XCircle } from "lucide-react";
import { Viewer3D } from "../components/Viewer3D";
import { FileUpload } from "../components/FileUpload";
import { geometraApi } from "../api/client";
import type { FileUploadResponse, JobStatusResponse } from "../types";

const SUPPORTED_2D = ["pdf", "png", "jpeg", "jpg", "tiff", "tif", "dxf", "dwg"];
const SUPPORTED_3D = ["step", "stp", "iges", "stl", "obj"];
const ALL_FORMATS = [...SUPPORTED_2D, ...SUPPORTED_3D];

function StatusBadge({ status }: { status: string }) {
  const colors: Record<string, string> = {
    pending: "bg-yellow-500/20 text-yellow-400 border-yellow-500/30",
    processing: "bg-blue-500/20 text-blue-400 border-blue-500/30",
    completed: "bg-emerald-500/20 text-emerald-400 border-emerald-500/30",
    failed: "bg-red-500/20 text-red-400 border-red-500/30",
  };
  return (
    <span
      className={`inline-flex items-center gap-1 rounded-full border px-2.5 py-0.5 text-xs font-medium ${colors[status] || colors.pending}`}
    >
      {status === "pending" && <Clock className="h-3 w-3" />}
      {status === "processing" && <RefreshCw className="h-3 w-3 animate-spin" />}
      {status === "completed" && <CheckCircle className="h-3 w-3" />}
      {status === "failed" && <XCircle className="h-3 w-3" />}
      {status}
    </span>
  );
}

function ProgressBar({ value }: { value: number }) {
  return (
    <div className="h-1.5 w-full overflow-hidden rounded-full bg-slate-700">
      <div
        className="h-full rounded-full bg-gradient-to-r from-indigo-500 to-violet-500 transition-all duration-500"
        style={{ width: `${Math.round(value * 100)}%` }}
      />
    </div>
  );
}

export function Dashboard() {
  const [uploadedFile, setUploadedFile] = useState<FileUploadResponse | null>(null);
  const [activeJob, setActiveJob] = useState<JobStatusResponse | null>(null);
  const [jobs, setJobs] = useState<JobStatusResponse[]>([]);
  const [health, setHealth] = useState<string>("checking...");

  useEffect(() => {
    geometraApi
      .health()
      .then((h) => setHealth(h.status))
      .catch(() => setHealth("offline"));
  }, []);

  const pollTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const pollJob = useCallback((jobId: string) => {
    const poll = async () => {
      try {
        const status = await geometraApi.getJobStatus(jobId);
        setActiveJob(status);
        if (status.status === "pending" || status.status === "processing") {
          pollTimerRef.current = setTimeout(poll, 1500);
        }
      } catch {
        pollTimerRef.current = setTimeout(poll, 3000);
      }
    };
    pollTimerRef.current = setTimeout(poll, 1000);
  }, []);

  // Cleanup polling on unmount
  useEffect(() => {
    return () => {
      if (pollTimerRef.current !== null) {
        clearTimeout(pollTimerRef.current);
      }
    };
  }, []);

  const handleUpload = useCallback(async (file: File) => {
    const result = await geometraApi.upload(file);
    setUploadedFile(result);

    const job = await geometraApi.startConversion({
      direction: result.detected_direction || "2d_to_3d",
      file_path: result.file_path,
    });
    setActiveJob({ ...job, progress: 0, message: "", result_paths: [], validation_report: null, updated_at: job.created_at, error: null });
    pollJob(job.job_id);
  }, [pollJob]);

  useEffect(() => {
    geometraApi.listJobs().then(setJobs).catch(() => {});
  }, [activeJob]);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      {/* Header */}
      <header className="border-b border-slate-800 bg-slate-900/50 backdrop-blur-sm">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-4">
          <div className="flex items-center gap-3">
            <Box className="h-7 w-7 text-indigo-400" />
            <h1 className="text-xl font-bold tracking-tight">Geometra</h1>
          </div>
          <div className="flex items-center gap-2 text-xs text-slate-400">
            <Activity className="h-3.5 w-3.5" />
            <span>API: {health}</span>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-7xl space-y-8 px-6 py-8">
        {/* Upload Section */}
        <section>
          <h2 className="mb-4 text-lg font-semibold text-slate-200">Upload Drawing or CAD File</h2>
          <FileUpload onUpload={handleUpload} acceptedExtensions={ALL_FORMATS} />
          {uploadedFile && (
            <div className="mt-3 flex items-center gap-2 text-sm text-slate-400">
              <CheckCircle className="h-4 w-4 text-emerald-400" />
              <span>
                {uploadedFile.filename} ({uploadedFile.detected_format})
              </span>
            </div>
          )}
        </section>

        {/* Active Job */}
        {activeJob && (
          <section>
            <h2 className="mb-4 text-lg font-semibold text-slate-200">Current Job</h2>
            <div className="rounded-xl border border-slate-800 bg-slate-900 p-5">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <span className="font-mono text-xs text-slate-500">{activeJob.job_id.slice(0, 8)}...</span>
                  <StatusBadge status={activeJob.status} />
                </div>
                <span className="text-xs text-slate-500">{activeJob.direction.replace("_", " → ")}</span>
              </div>
              <div className="mt-3 space-y-2">
                <ProgressBar value={activeJob.progress} />
                <p className="text-xs text-slate-400">{activeJob.message}</p>
              </div>
            </div>
          </section>
        )}

        {/* 3D Viewer + Jobs Grid */}
        <div className="grid grid-cols-1 gap-8 lg:grid-cols-3">
          {/* 3D Viewer */}
          <section className="lg:col-span-2">
            <h2 className="mb-4 text-lg font-semibold text-slate-200">3D Viewer</h2>
            <div className="h-[500px] overflow-hidden rounded-xl border border-slate-800">
              <Viewer3D
                hasModel={activeJob?.status === "completed"}
                modelPath={activeJob?.result_paths?.[0]}
              />
            </div>
          </section>

          {/* Job History */}
          <section>
            <h2 className="mb-4 text-lg font-semibold text-slate-200">Job History</h2>
            <div className="space-y-3">
              {jobs.length === 0 && (
                <p className="text-sm text-slate-500">No jobs yet. Upload a file to start.</p>
              )}
              {jobs.map((job) => (
                <div
                  key={job.job_id}
                  className="rounded-lg border border-slate-800 bg-slate-900/50 p-3 transition-colors hover:border-slate-700"
                >
                  <div className="flex items-center justify-between">
                    <span className="font-mono text-xs text-slate-500">
                      {job.job_id.slice(0, 8)}
                    </span>
                    <StatusBadge status={job.status} />
                  </div>
                  <p className="mt-1 text-xs text-slate-500">{job.direction.replace("_", " → ")}</p>
                  {job.validation_report && (
                    <p className="mt-1 text-xs text-emerald-400">
                      Score: {((job.validation_report as Record<string, number>)?.overall_score ?? 0) * 100}%
                    </p>
                  )}
                </div>
              ))}
            </div>
          </section>
        </div>
      </main>
    </div>
  );
}
