import React from "react";

export interface PredictionResultProps {
  value: number | null;
  targetLabel?: string;
  confidence?: number | null;
  riskLabel?: string | null;
}

const prettyLabel = (raw?: string | null): string => {
  if (!raw) return "-";
  return raw
    .split("_")
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(" ");
};

export const PredictionResult: React.FC<PredictionResultProps> = ({
  value,
  targetLabel = "TBK8",
  confidence,
  riskLabel,
}) => {
  return (
    <div className="card">
      <h2>Predicted {targetLabel}</h2>
      {value == null ? (
        <p className="muted">Run a prediction to see the result.</p>
      ) : (
        <>
          <p className="prediction-value">{value.toFixed(2)}</p>
          <p className="muted">
            Risk group: <strong>{prettyLabel(riskLabel)}</strong>
          </p>
          <p className="muted">
            Confidence: <strong>{((confidence ?? 0) * 100).toFixed(1)}%</strong>
          </p>
        </>
      )}
    </div>
  );
};

