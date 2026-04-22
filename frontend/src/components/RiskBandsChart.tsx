import React from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

const palette = {
  very_poor: "#ef4444",
  poor: "#f97316",
  fair: "#f59e0b",
  good: "#84cc16",
  very_good: "#22c55e",
  excellent: "#14b8a6",
} as const;

const labelMap = {
  very_poor: "Very Poor",
  poor: "Poor",
  fair: "Fair",
  good: "Good",
  very_good: "Very Good",
  excellent: "Excellent",
} as const;

const order = [
  "very_poor",
  "poor",
  "fair",
  "good",
  "very_good",
  "excellent",
] as const;

interface RiskBandsChartProps {
  truths?: Record<string, number> | null;
}

export const RiskBandsChart: React.FC<RiskBandsChartProps> = ({ truths }) => {
  const hasTruths = Boolean(truths && Object.keys(truths).length > 0);
  const data = order.map((key, idx) => ({
    key,
    label: labelMap[key],
    color: palette[key],
    index: idx + 1,
    truth: hasTruths ? (truths?.[key] ?? 0) : 0,
  }));

  return (
    <div className="card">
      <h2>{hasTruths ? "Prediction Risk Truths" : "Early-Warning Bands"}</h2>
      <p className="muted compact">
        {hasTruths
          ? "Neutrosophic truth-membership values inferred from the predicted GPA."
          : "Reference linguistic bands used by the neutrosophic encoder."}
      </p>
      <ResponsiveContainer width="100%" height={240}>
        <BarChart data={data}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} />
          <XAxis dataKey="label" interval={0} angle={-10} textAnchor="end" height={50} />
          <YAxis domain={hasTruths ? [0, 1] : [0, 6]} ticks={hasTruths ? [0, 0.25, 0.5, 0.75, 1] : [1, 2, 3, 4, 5, 6]} />
          <Tooltip />
          <Bar dataKey={hasTruths ? "truth" : "index"} radius={4}>
            {data.map((entry) => (
              <Cell key={entry.label} fill={entry.color} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};
