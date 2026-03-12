import React, { useEffect, useState } from "react";
import axios from "axios";
import { MetricsChart } from "../components/MetricsChart";

interface Metrics {
  MAE: number | null;
  RMSE: number | null;
  "R²": number | null;
}

export const Dashboard: React.FC = () => {
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const fetchMetrics = async () => {
      setLoading(true);
      try {
        const res = await axios.get<Metrics>("http://localhost:8000/model-metrics");
        setMetrics(res.data);
      } catch {
        setMetrics(null);
      } finally {
        setLoading(false);
      }
    };

    fetchMetrics();
  }, []);

  return (
    <div className="page">
      <header className="page-header">
        <h2>Dashboard</h2>
        {loading && <span className="muted">Loading metrics...</span>}
      </header>
      <section className="grid-2">
        <div className="card">
          <h2>Model Summary</h2>
          {metrics ? (
            <ul className="metrics-list">
              <li>
                <span>MAE</span>
                <strong>{metrics.MAE?.toFixed(3)}</strong>
              </li>
              <li>
                <span>RMSE</span>
                <strong>{metrics.RMSE?.toFixed(3)}</strong>
              </li>
              <li>
                <span>R²</span>
                <strong>{metrics["R²"]?.toFixed(3)}</strong>
              </li>
            </ul>
          ) : (
            <p className="muted">
              Metrics are not available. Run the ML pipeline and evaluation.
            </p>
          )}
        </div>
        <MetricsChart mae={metrics?.MAE ?? null} rmse={metrics?.RMSE ?? null} />
      </section>
    </div>
  );
};

