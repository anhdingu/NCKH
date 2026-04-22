import React, { useState } from "react";

export interface ScoreInputFormProps {
  labels?: [string, string, string];
  submitLabel?: string;
  onSubmit: (values: Record<string, number>) => void;
  loading?: boolean;
}

export const ScoreInputForm: React.FC<ScoreInputFormProps> = ({
  labels = ["TBK5", "TBK6", "TBK7"],
  submitLabel = "Predict TBK8",
  onSubmit,
  loading = false,
}) => {
  const [v1, setV1] = useState<string>("");
  const [v2, setV2] = useState<string>("");
  const [v3, setV3] = useState<string>("");

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const n1 = parseFloat(v1);
    const n2 = parseFloat(v2);
    const n3 = parseFloat(v3);
    if (Number.isNaN(n1) || Number.isNaN(n2) || Number.isNaN(n3)) {
      alert("Please enter valid numeric scores for all semesters.");
      return;
    }
    onSubmit({
      [labels[0]]: n1,
      [labels[1]]: n2,
      [labels[2]]: n3,
    });
  };

  return (
    <form className="card form" onSubmit={handleSubmit}>
      <h2>Input Previous Semester Scores</h2>
      <div className="form-grid">
        <label>
          {labels[0]}
          <input
            type="number"
            min={0}
            max={10}
            step={0.01}
            value={v1}
            onChange={(e) => setV1(e.target.value)}
          />
        </label>
        <label>
          {labels[1]}
          <input
            type="number"
            min={0}
            max={10}
            step={0.01}
            value={v2}
            onChange={(e) => setV2(e.target.value)}
          />
        </label>
        <label>
          {labels[2]}
          <input
            type="number"
            min={0}
            max={10}
            step={0.01}
            value={v3}
            onChange={(e) => setV3(e.target.value)}
          />
        </label>
      </div>
      <button type="submit" className="primary-btn" disabled={loading}>
        {loading ? "Predicting..." : submitLabel}
      </button>
    </form>
  );
};

