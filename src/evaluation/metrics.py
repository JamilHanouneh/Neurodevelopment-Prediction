import math
from typing import Dict, Iterable

from sklearn import metrics


def rmse(y_true, y_pred) -> float:
    """Root mean squared error."""
    return math.sqrt(metrics.mean_squared_error(y_true, y_pred))


def mae(y_true, y_pred) -> float:
    """Mean absolute error."""
    return metrics.mean_absolute_error(y_true, y_pred)


def r2(y_true, y_pred) -> float:
    """Coefficient of determination."""
    return metrics.r2_score(y_true, y_pred)


def explained_variance(y_true, y_pred) -> float:
    """Explained variance score."""
    return metrics.explained_variance_score(y_true, y_pred)


METRIC_REGISTRY = {
    "rmse": rmse,
    "mae": mae,
    "r2": r2,
    "explained_variance": explained_variance,
}


def compute_metrics(y_true, y_pred, metric_names: Iterable[str]) -> Dict[str, float]:
    """Compute configured metrics."""
    results = {}
    for name in metric_names:
        func = METRIC_REGISTRY.get(name)
        if func is None:
            raise KeyError(f"Unknown metric requested: {name}")
        results[name] = float(func(y_true, y_pred))
    return results
