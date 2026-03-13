from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - optional dependency
    XGBClassifier = None


@dataclass(frozen=True)
class XGBoostConfig:
    max_depth: int = 3
    learning_rate: float = 0.02
    n_estimators: int = 900
    subsample: float = 0.7
    colsample_bytree: float = 0.7
    reg_lambda: float = 1.0
    gamma: float = 0.2
    min_child_weight: float = 5.0


def train_xgboost(
    train_df: pd.DataFrame, feature_cols: List[str], target_col: str, config: XGBoostConfig | None = None
) -> Tuple["XGBClassifier", np.ndarray]:
    if XGBClassifier is None:
        raise ImportError("xgboost is not installed. Add it to requirements.txt and install.")
    config = config or XGBoostConfig()
    X = train_df[feature_cols]
    y = train_df[target_col]
    model = XGBClassifier(
        max_depth=config.max_depth,
        learning_rate=config.learning_rate,
        n_estimators=config.n_estimators,
        subsample=config.subsample,
        colsample_bytree=config.colsample_bytree,
        reg_lambda=config.reg_lambda,
        gamma=config.gamma,
        min_child_weight=config.min_child_weight,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
    )
    model.fit(X, y)
    preds = model.predict_proba(X)[:, 1]
    return model, preds
