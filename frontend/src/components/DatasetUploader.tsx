import React, { useState } from "react";
import { DatasetProfile } from "./DatasetProfilePanel";

export interface DatasetUploaderProps {
  onUploaded?: () => void;
  onProfileReady?: (profile: DatasetProfile | null) => void;
}

export const DatasetUploader: React.FC<DatasetUploaderProps> = ({
  onUploaded,
  onProfileReady,
}) => {
  const [file, setFile] = useState<File | null>(null);
  const [status, setStatus] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const handleUpload = async () => {
    if (!file) {
      alert("Please select an Excel file first.");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    setLoading(true);
    setStatus(null);

    try {
      const res = await fetch("http://localhost:8000/upload-dataset", {
        method: "POST",
        body: formData,
      });
      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || "Upload failed");
      }
      const payload = await res.json();
      onProfileReady?.((payload.profile_preview as DatasetProfile | undefined) ?? null);
      setStatus("File uploaded successfully.");
      onUploaded?.();
    } catch (err) {
      setStatus(
        err instanceof Error
          ? `Upload failed: ${err.message}`
          : "Upload failed.",
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="card">
      <h2>Upload Dataset (Excel)</h2>
      <p className="muted">
        Upload Excel files with columns: ID, TBK1, ..., TBK8. They will be
        stored on the server under the raw data folder.
      </p>
      <div className="uploader">
        <input
          type="file"
          accept=".xlsx,.xls"
          onChange={(e) => setFile(e.target.files?.[0] ?? null)}
        />
        <button
          className="primary-btn"
          onClick={handleUpload}
          disabled={loading}
        >
          {loading ? "Uploading..." : "Upload"}
        </button>
      </div>
      {status && <p className="status">{status}</p>}
    </div>
  );
};

