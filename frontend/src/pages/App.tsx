import React, { useState } from "react";
import { Dashboard } from "./Dashboard";
import { PredictionPage } from "./PredictionPage";
import { DatasetPage } from "./DatasetPage";

type PageKey = "dashboard" | "prediction" | "dataset";

export const App: React.FC = () => {
  const [page, setPage] = useState<PageKey>("dashboard");

  return (
    <div className="app-root">
      <aside className="sidebar">
        <h1 className="logo">GPA Predictor</h1>
        <nav className="nav">
          <button
            className={page === "dashboard" ? "nav-btn active" : "nav-btn"}
            onClick={() => setPage("dashboard")}
          >
            Dashboard
          </button>
          <button
            className={page === "prediction" ? "nav-btn active" : "nav-btn"}
            onClick={() => setPage("prediction")}
          >
            Prediction
          </button>
          <button
            className={page === "dataset" ? "nav-btn active" : "nav-btn"}
            onClick={() => setPage("dataset")}
          >
            Dataset
          </button>
        </nav>
      </aside>
      <main className="main">
        {page === "dashboard" && <Dashboard />}
        {page === "prediction" && <PredictionPage />}
        {page === "dataset" && <DatasetPage />}
      </main>
    </div>
  );
};

