import React from "react";

export interface PredictionResultProps {
  value: number | null;
}

export const PredictionResult: React.FC<PredictionResultProps> = ({ value }) => {
  return (
    <div className="card">
      <h2>Predicted TBK8</h2>
      {value == null ? (
        <p className="muted">Run a prediction to see the result.</p>
      ) : (
        <p className="prediction-value">{value.toFixed(2)}</p>
      )}
    </div>
  );
};

