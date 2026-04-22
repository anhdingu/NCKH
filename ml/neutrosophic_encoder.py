from typing import Dict, Iterable, List, Tuple

import numpy as np


# Linguistic partitions over GPA scale [0, 10].
LINGUISTIC_SETS: Dict[str, Tuple[float, float, float, float]] = {
    "very_poor": (0.0, 0.0, 1.5, 3.0),
    "poor": (2.0, 3.5, 4.5, 5.5),
    "fair": (4.8, 5.8, 6.6, 7.4),
    "good": (6.8, 7.4, 8.2, 8.8),
    "very_good": (8.3, 8.8, 9.3, 9.8),
    "excellent": (9.4, 9.7, 10.0, 10.0),
}


def trapezoid_membership(x: float, a: float, b: float, c: float, d: float) -> float:
    """Compute trapezoidal membership value in [0, 1]."""
    x = float(x)
    if x <= a or x >= d:
        return 0.0
    if b <= x <= c:
        return 1.0
    if a < x < b:
        return float((x - a) / (b - a + 1e-8))
    return float((d - x) / (d - c + 1e-8))


def neutrosophic_triplet(x: float, params: Tuple[float, float, float, float]) -> Tuple[float, float, float]:
    """Map one score to (Truth, Indeterminacy, Falsity)."""
    truth = trapezoid_membership(x, *params)
    indeterminacy = 1.0 - abs(2.0 * truth - 1.0)
    falsity = 1.0 - truth
    return float(truth), float(indeterminacy), float(falsity)


def encode_score(score: float) -> np.ndarray:
    """Encode one scalar score into a neutrosophic feature vector (6 sets x 3 values)."""
    encoded: List[float] = []
    for params in LINGUISTIC_SETS.values():
        t, i, f = neutrosophic_triplet(score, params)
        encoded.extend([t, i, f])
    return np.asarray(encoded, dtype=np.float32)


def encode_sequence(scores: Iterable[float]) -> np.ndarray:
    """Encode a temporal sequence of scores into shape (timesteps, 18)."""
    return np.stack([encode_score(s) for s in scores], axis=0).astype(np.float32)


def infer_risk_from_score(score: float) -> Dict[str, float | str]:
    """Infer dominant linguistic class and confidence from a scalar score."""
    set_names = list(LINGUISTIC_SETS.keys())
    truths = [trapezoid_membership(score, *LINGUISTIC_SETS[name]) for name in set_names]
    idx = int(np.argmax(truths))
    label = set_names[idx]
    confidence = float(truths[idx])
    return {
        "risk_label": label,
        "confidence": confidence,
        "truths": dict(zip(set_names, [float(v) for v in truths])),
    }
