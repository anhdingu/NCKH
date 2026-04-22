import React, { useEffect, useState } from "react";
import axios from "axios";
import { DatasetUploader } from "../components/DatasetUploader";
import { DatasetProfile, DatasetProfilePanel } from "../components/DatasetProfilePanel";

export const DatasetPage: React.FC = () => {
  const [profile, setProfile] = useState<DatasetProfile | null>(null);

  useEffect(() => {
    const loadProfile = async () => {
      try {
        const res = await axios.get<DatasetProfile>("http://localhost:8000/dataset/profile");
        setProfile(res.data);
      } catch {
        setProfile(null);
      }
    };
    loadProfile();
  }, []);

  return (
    <div className="page">
      <header className="page-header">
        <h2>Dataset Management</h2>
        <p className="muted">
          Upload new Excel files to expand the training dataset for research
          experiments.
        </p>
      </header>
      <DatasetUploader
        onUploaded={async () => {
          try {
            const res = await axios.get<DatasetProfile>("http://localhost:8000/dataset/profile");
            setProfile(res.data);
          } catch {
            setProfile(null);
          }
        }}
        onProfileReady={(preview) => {
          if (preview) {
            setProfile(preview);
          }
        }}
      />
      <DatasetProfilePanel profile={profile} />
    </div>
  );
};

