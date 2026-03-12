import React from "react";
import { DatasetUploader } from "../components/DatasetUploader";

export const DatasetPage: React.FC = () => {
  return (
    <div className="page">
      <header className="page-header">
        <h2>Dataset Management</h2>
        <p className="muted">
          Upload new Excel files to expand the training dataset for research
          experiments.
        </p>
      </header>
      <DatasetUploader />
    </div>
  );
};

