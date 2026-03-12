import React from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

export interface MetricsChartProps {
  mae: number | null;
  rmse: number | null;
}

export const MetricsChart: React.FC<MetricsChartProps> = ({ mae, rmse }) => {
  if (mae == null || rmse == null) {
    return (
      <div className="card">
        <h2>Model Error Metrics</h2>
        <p className="muted">
          Metrics are not available yet. Train and evaluate the model first.
        </p>
      </div>
    );
  }

  const data = [
    { metric: "MAE", value: mae },
    { metric: "RMSE", value: rmse },
  ];

  return (
    <div className="card">
      <h2>Model Error Metrics</h2>
      <ResponsiveContainer width="100%" height={260}>
        <BarChart data={data}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} />
          <XAxis dataKey="metric" />
          <YAxis />
          <Tooltip />
          <Legend />
          <Bar dataKey="value" name="Score" fill="#4f46e5" radius={4} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};

