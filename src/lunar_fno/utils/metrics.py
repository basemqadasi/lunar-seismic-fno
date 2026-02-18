from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1-score": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def seed_metrics_to_frame(metrics_by_seed: Dict[int, Dict[str, float]]) -> pd.DataFrame:
    df = pd.DataFrame.from_dict(metrics_by_seed, orient="index")
    df.index.name = "Seed"
    return df
