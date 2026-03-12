import React, { useState } from "react";

export interface ScoreInputFormProps {
  onSubmit: (values: { TBK5: number; TBK6: number; TBK7: number }) => void;
  loading?: boolean;
}

export const ScoreInputForm: React.FC<ScoreInputFormProps> = ({
  onSubmit,
  loading = false,
}) => {
  const [tbk5, setTbk5] = useState<string>("");
  const [tbk6, setTbk6] = useState<string>("");
  const [tbk7, setTbk7] = useState<string>("");

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const v5 = parseFloat(tbk5);
    const v6 = parseFloat(tbk6);
    const v7 = parseFloat(tbk7);
    if (Number.isNaN(v5) || Number.isNaN(v6) || Number.isNaN(v7)) {
      alert("Please enter valid numeric scores for all semesters.");
      return;
    }
    onSubmit({ TBK5: v5, TBK6: v6, TBK7: v7 });
  };

  return (
    <form className="card form" onSubmit={handleSubmit}>
      <h2>Input Previous Semester Scores</h2>
      <div className="form-grid">
        <label>
          TBK5
          <input
            type="number"
            min={0}
            max={10}
            step={0.01}
            value={tbk5}
            onChange={(e) => setTbk5(e.target.value)}
          />
        </label>
        <label>
          TBK6
          <input
            type="number"
            min={0}
            max={10}
            step={0.01}
            value={tbk6}
            onChange={(e) => setTbk6(e.target.value)}
          />
        </label>
        <label>
          TBK7
          <input
            type="number"
            min={0}
            max={10}
            step={0.01}
            value={tbk7}
            onChange={(e) => setTbk7(e.target.value)}
          />
        </label>
      </div>
      <button type="submit" className="primary-btn" disabled={loading}>
        {loading ? "Predicting..." : "Predict TBK8"}
      </button>
    </form>
  );
};

