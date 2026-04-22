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

const order = [
  "very_poor",
  "poor",
  "fair",
  "good",
  "very_good",
  "excellent",
] as const;

const colorMap: Record<string, string> = {
  very_poor: "#ef4444",
  poor: "#f97316",
  fair: "#f59e0b",
  good: "#84cc16",
  very_good: "#22c55e",
  excellent: "#14b8a6",
};

const humanize = (label: string) =>
  label
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");

export const RiskDistributionChart: React.FC<{
  distribution: Record<string, number>;
}> = ({ distribution }) => {
  const data = order.map((label) => ({
    label: humanize(label),
    key: label,
    count: distribution[label] ?? 0,
  }));

  const hasData = data.some((item) => item.count > 0);
  if (!hasData) {
    return (
      <div className="card">
        <h2>Risk Distribution</h2>
        <p className="muted">No risk distribution data available yet.</p>
      </div>
    );
  }

  return (
    <div className="card">
      <h2>Risk Distribution</h2>
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={data}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} />
          <XAxis dataKey="label" interval={0} angle={-10} textAnchor="end" height={55} />
          <YAxis />
          <Tooltip />
          <Bar dataKey="count" radius={4}>
            {data.map((entry) => (
              <Cell key={entry.key} fill={colorMap[entry.key]} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};

