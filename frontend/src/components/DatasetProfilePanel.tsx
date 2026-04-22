import React from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

export interface DatasetProfile {
  rows: number;
  columns: number;
  missing_by_column: Record<string, number>;
  summary_by_semester: Record<
    string,
    {
      mean: number;
      std: number;
      min: number;
      max: number;
    }
  >;
  correlation: {
    labels: string[];
    matrix: number[][];
  };
}

export const DatasetProfilePanel: React.FC<{ profile: DatasetProfile | null }> = ({
  profile,
}) => {
  if (!profile) {
    return (
      <div className="card">
        <h2>Dataset Analysis</h2>
        <p className="muted">Dataset profile is not available yet.</p>
      </div>
    );
  }

  const summaryRows = Object.entries(profile.summary_by_semester).map(([semester, stats]) => ({
    semester,
    mean: Number(stats.mean.toFixed(3)),
  }));
  const missingRows = Object.entries(profile.missing_by_column).map(([column, count]) => ({
    column,
    count,
  }));

  return (
    <div className="grid-2">
      <div className="card">
        <h2>Dataset Summary</h2>
        <ul className="metrics-list">
          <li>
            <span>Rows</span>
            <strong>{profile.rows}</strong>
          </li>
          <li>
            <span>Columns</span>
            <strong>{profile.columns}</strong>
          </li>
        </ul>
      </div>

      <div className="card">
        <h2>Mean Score by Semester</h2>
        <ResponsiveContainer width="100%" height={260}>
          <BarChart data={summaryRows}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} />
            <XAxis dataKey="semester" />
            <YAxis domain={[0, 10]} />
            <Tooltip />
            <Bar dataKey="mean" fill="#4f46e5" radius={4} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="card">
        <h2>Missing Values</h2>
        <ResponsiveContainer width="100%" height={260}>
          <BarChart data={missingRows}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} />
            <XAxis dataKey="column" />
            <YAxis />
            <Tooltip />
            <Bar dataKey="count" fill="#f97316" radius={4} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="card">
        <h2>Correlation Matrix (TBK)</h2>
        <div className="correlation-grid">
          <table>
            <thead>
              <tr>
                <th> </th>
                {profile.correlation.labels.map((label) => (
                  <th key={label}>{label}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {profile.correlation.matrix.map((row, rowIdx) => (
                <tr key={profile.correlation.labels[rowIdx]}>
                  <td>{profile.correlation.labels[rowIdx]}</td>
                  {row.map((value, colIdx) => (
                    <td key={`${rowIdx}-${colIdx}`}>{value.toFixed(2)}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

