import React, { useEffect, useState } from "react";
import axios from "axios";
import { MetricsChart } from "../components/MetricsChart";
import { RiskBandsChart } from "../components/RiskBandsChart";
import { PredictionScatterChart } from "../components/PredictionScatterChart";
import { RiskDistributionChart } from "../components/RiskDistributionChart";

interface Metrics {
  target: number;
  MAE: number | null;
  RMSE: number | null;
  R2: number | null;
}

export const Dashboard: React.FC = () => {
  const [target, setTarget] = useState<number>(8);
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [samples, setSamples] = useState<
    Array<{ actual: number; predicted: number; residual: number; risk_label: string }>
  >([]);
  const [riskDistribution, setRiskDistribution] = useState<Record<string, number>>({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchDashboardData = async () => {
      setLoading(true);
      setError(null);
      try {
        const [metricsRes, samplesRes, riskRes] = await Promise.all([
          axios.get<Metrics>(`http://localhost:8000/model-metrics?target=${target}`),
          axios.get<{ data: Array<{ actual: number; predicted: number; residual: number; risk_label: string }> }>(
            `http://localhost:8000/predictions-sample?target=${target}&limit=300`,
          ),
          axios.get<{ distribution: Record<string, number> }>(
            `http://localhost:8000/risk-distribution?target=${target}`,
          ),
        ]);
        setMetrics(metricsRes.data);
        setSamples(samplesRes.data.data ?? []);
        setRiskDistribution(riskRes.data.distribution ?? {});
      } catch {
        setMetrics(null);
        setSamples([]);
        setRiskDistribution({});
        setError("Cannot load dashboard analytics. Check backend and processed datasets.");
      } finally {
        setLoading(false);
      }
    };

    fetchDashboardData();
  }, [target]);

  return (
    <div className="page">
      <header className="page-header">
        <h2>Dashboard</h2>
        <div className="target-selector">
          <label htmlFor="targetSemester">Target TBK</label>
          <select
            id="targetSemester"
            value={target}
            onChange={(e) => setTarget(Number(e.target.value))}
          >
            <option value={4}>TBK4</option>
            <option value={5}>TBK5</option>
            <option value={6}>TBK6</option>
            <option value={7}>TBK7</option>
            <option value={8}>TBK8</option>
          </select>
        </div>
        {loading && <span className="muted">Loading metrics...</span>}
      </header>
      <section className="grid-2">
        <div className="card">
          <h2>Model Summary (TBK{target})</h2>
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
                <span>R2</span>
                <strong>{metrics.R2?.toFixed(3)}</strong>
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
      {error && <p className="muted">{error}</p>}
      <section className="grid-2">
        <PredictionScatterChart rows={samples} />
        <RiskDistributionChart distribution={riskDistribution} />
      </section>
      <RiskBandsChart />
    </div>
  );
};

