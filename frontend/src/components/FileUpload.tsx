import { useCallback, useState } from "react";
import { useDropzone } from "react-dropzone";
import { Upload, Loader2, CheckCircle, XCircle } from "lucide-react";

interface FileUploadProps {
  onUpload: (file: File) => Promise<void>;
  acceptedExtensions: string[];
}


export function FileUpload({ onUpload, acceptedExtensions }: FileUploadProps) {
  const [uploading, setUploading] = useState(false);
  const [result, setResult] = useState<"success" | "error" | null>(null);

  const onDrop = useCallback(
    async (accepted: File[]) => {
      const file = accepted[0];
      if (!file) return;
      setUploading(true);
      setResult(null);
      try {
        await onUpload(file);
        setResult("success");
      } catch {
        setResult("error");
      } finally {
        setUploading(false);
      }
    },
    [onUpload]
  );

  const { getRootProps, getInputProps, isDragActive, isDragReject } = useDropzone({
    onDrop,
    accept: acceptedExtensions.reduce<Record<string, string[]>>(
      (acc, ext) => {
        const mime =
          ext === "pdf"
            ? "application/pdf"
            : ext === "png"
              ? "image/png"
              : ext === "jpeg" || ext === "jpg"
                ? "image/jpeg"
                : ext === "tiff" || ext === "tif"
                  ? "image/tiff"
                  : "application/octet-stream";
        acc[mime] = [`.${ext}`];
        return acc;
      },
      {}
    ),
    maxFiles: 1,
    multiple: false,
  });

  return (
    <div
      {...getRootProps()}
      className={`
        relative cursor-pointer rounded-xl border-2 border-dashed p-8 text-center transition-all duration-200
        ${isDragActive && !isDragReject
          ? "border-indigo-400 bg-indigo-400/10"
          : isDragReject
            ? "border-red-400 bg-red-400/10"
            : "border-slate-600 bg-slate-800/50 hover:border-slate-500 hover:bg-slate-800/80"
        }
      `}
    >
      <input {...getInputProps()} />

      <div className="flex flex-col items-center gap-3">
        {uploading ? (
          <>
            <Loader2 className="h-10 w-10 animate-spin text-indigo-400" />
            <p className="text-sm text-slate-300">Uploading file...</p>
          </>
        ) : result === "success" ? (
          <>
            <CheckCircle className="h-10 w-10 text-emerald-400" />
            <p className="text-sm text-emerald-300">File uploaded successfully</p>
          </>
        ) : result === "error" ? (
          <>
            <XCircle className="h-10 w-10 text-red-400" />
            <p className="text-sm text-red-300">Upload failed</p>
          </>
        ) : (
          <>
            {isDragActive ? (
              <Upload className="h-10 w-10 text-indigo-400" />
            ) : (
              <Upload className="h-10 w-10 text-slate-500" />
            )}
            <div>
              <p className="text-sm font-medium text-slate-300">
                {isDragActive ? "Drop file here" : "Drag & drop or click to browse"}
              </p>
              <p className="mt-1 text-xs text-slate-500">
                Supported formats: {acceptedExtensions.join(", ")}
              </p>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

export function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}
