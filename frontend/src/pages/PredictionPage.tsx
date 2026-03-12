import React, { useState } from "react";
import axios from "axios";
import { ScoreInputForm } from "../components/ScoreInputForm";
import { PredictionResult } from "../components/PredictionResult";

export const PredictionPage: React.FC = () => {
  const [predicted, setPredicted] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (values: {
    TBK5: number;
    TBK6: number;
    TBK7: number;
  }) => {
    setLoading(true);
    try {
      const res = await axios.post<{ predicted_TBK8: number }>(
        "http://localhost:8000/predict",
        values,
      );
      setPredicted(res.data.predicted_TBK8);
    } catch (err) {
      console.error(err);
      alert("Prediction failed. Please check backend server.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="page">
      <header className="page-header">
        <h2>Prediction</h2>
        <p className="muted">
          Enter the GPA of semesters 5–7 to predict the next semester (TBK8).
        </p>
      </header>
      <section className="grid-2">
        <ScoreInputForm onSubmit={handleSubmit} loading={loading} />
        <PredictionResult value={predicted} />
      </section>
    </div>
  );
};

