import React from "react";
import {
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

interface Row {
  actual: number;
  predicted: number;
  residual: number;
  risk_label: string;
}

export const PredictionScatterChart: React.FC<{ rows: Row[] }> = ({ rows }) => {
  if (!rows.length) {
    return (
      <div className="card">
        <h2>Actual vs Predicted</h2>
        <p className="muted">No prediction sample data available yet.</p>
      </div>
    );
  }

  return (
    <div className="card">
      <h2>Actual vs Predicted</h2>
      <ResponsiveContainer width="100%" height={300}>
        <ScatterChart>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis type="number" dataKey="actual" name="Actual" domain={[0, 10]} />
          <YAxis type="number" dataKey="predicted" name="Predicted" domain={[0, 10]} />
          <Tooltip cursor={{ strokeDasharray: "3 3" }} />
          <Legend />
          <Scatter name="Predictions" data={rows} fill="#22c55e" />
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
};

