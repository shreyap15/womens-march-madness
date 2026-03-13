from __future__ import annotations

from typing import Dict

import numpy as np


DEFAULT_WEIGHTS: Dict[str, float] = {
    "xgb": 0.50,
    "logistic": 0.25,
    "rf": 0.15,
    "elo": 0.10,
}


def blend_predictions(preds: Dict[str, np.ndarray], weights: Dict[str, float] | None = None) -> np.ndarray:
    weights = weights or DEFAULT_WEIGHTS
    total = np.zeros_like(next(iter(preds.values())))
    for key, weight in weights.items():
        if key not in preds:
            raise KeyError(f"Missing predictions for {key}")
        total += weight * preds[key]
    return total
