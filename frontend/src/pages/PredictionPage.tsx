import React, { useState } from "react";
import axios from "axios";
import { ScoreInputForm } from "../components/ScoreInputForm";
import { PredictionResult } from "../components/PredictionResult";
import { RiskBandsChart } from "../components/RiskBandsChart";

interface PredictResponse {
  predicted_TBK8: number;
  confidence: number;
  risk_label: string;
  truths: Record<string, number>;
}

export const PredictionPage: React.FC = () => {
  const [targetSemester, setTargetSemester] = useState<number>(8);
  const [predicted, setPredicted] = useState<number | null>(null);
  const [confidence, setConfidence] = useState<number | null>(null);
  const [riskLabel, setRiskLabel] = useState<string | null>(null);
  const [truths, setTruths] = useState<Record<string, number> | null>(null);
  const [loading, setLoading] = useState(false);

  const labels = [
    `TBK${targetSemester - 3}`,
    `TBK${targetSemester - 2}`,
    `TBK${targetSemester - 1}`,
  ] as [string, string, string];

  const handleSubmit = async (values: Record<string, number>) => {
    setLoading(true);
    try {
      const res = await axios.post<PredictResponse>(
        "http://localhost:8000/predict",
        {
          target_semester: targetSemester,
          scores: values,
        },
      );
      setPredicted(res.data.predicted_TBK8);
      setConfidence(res.data.confidence);
      setRiskLabel(res.data.risk_label);
      setTruths(res.data.truths);
    } catch (err) {
      console.error(err);
      alert("Prediction failed. Please check backend server.");
      setConfidence(null);
      setRiskLabel(null);
      setTruths(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="page">
      <header className="page-header">
        <h2>Prediction</h2>
        <p className="muted">
          Select a target semester, then enter the previous 3 semester scores.
        </p>
        <div className="target-selector">
          <label htmlFor="predictionTarget">Target TBK</label>
          <select
            id="predictionTarget"
            value={targetSemester}
            onChange={(e) => {
              setTargetSemester(Number(e.target.value));
              setPredicted(null);
              setConfidence(null);
              setRiskLabel(null);
              setTruths(null);
            }}
          >
            <option value={4}>TBK4</option>
            <option value={5}>TBK5</option>
            <option value={6}>TBK6</option>
            <option value={7}>TBK7</option>
            <option value={8}>TBK8</option>
          </select>
        </div>
      </header>
      <section className="grid-2">
        <ScoreInputForm
          labels={labels}
          submitLabel={`Predict TBK${targetSemester}`}
          onSubmit={handleSubmit}
          loading={loading}
        />
        <PredictionResult
          value={predicted}
          targetLabel={`TBK${targetSemester}`}
          confidence={confidence}
          riskLabel={riskLabel}
        />
      </section>
      <RiskBandsChart truths={truths} />
    </div>
  );
};

