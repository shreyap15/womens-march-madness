from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class LogisticConfig:
    max_iter: int = 200
    C: float = 1.0


def train_logistic(
    train_df: pd.DataFrame, feature_cols: List[str], target_col: str, config: LogisticConfig | None = None
) -> Tuple[Pipeline, np.ndarray]:
    config = config or LogisticConfig()
    X = train_df[feature_cols]
    y = train_df[target_col]
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
            ("clf", LogisticRegression(max_iter=config.max_iter, C=config.C)),
        ]
    )
    model.fit(X, y)
    preds = model.predict_proba(X)[:, 1]
    return model, preds
