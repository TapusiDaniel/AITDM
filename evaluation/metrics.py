import numpy as np
from sklearn.metrics import (
    roc_auc_score, 
    average_precision_score, 
    roc_curve, 
    precision_recall_curve
)
from typing import Dict, List, Tuple, Optional, Union
import warnings


def calculate_metrics(
    labels: np.ndarray, 
    scores: np.ndarray,
    tnr_thresholds: List[float] = [0.99, 0.95, 0.90]
) -> Dict[str, Union[float, int]]:
    """
    Calculate anomaly detection metrics.
    
    Args:
        labels: Binary labels (0=normal, 1=anomaly)
        scores: Anomaly scores (higher = more anomalous)
        tnr_thresholds: TNR thresholds for TPR calculation
    
    Returns:
        Dictionary with AUROC, AP, TPR@TNR values, and sample counts
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    
    if len(labels) != len(scores):
        raise ValueError(f"Length mismatch: labels ({len(labels)}) vs scores ({len(scores)})")
    
    if len(labels) == 0:
        return _empty_metrics(tnr_thresholds)
    
    n_classes = len(np.unique(labels))
    
    if n_classes < 2:
        warnings.warn("Only one class present. Metrics cannot be computed.")
        return {
            'AUROC': np.nan,
            'AP': np.nan,
            **{f'TPR@TNR{int(t*100)}': np.nan for t in tnr_thresholds},
            'n_samples': len(labels),
            'n_anomalies': int(labels.sum()),
            'n_normal': int((labels == 0).sum())
        }
    
    auroc = roc_auc_score(labels, scores)
    ap = average_precision_score(labels, scores)
    
    fpr, tpr, _ = roc_curve(labels, scores)
    tnr = 1 - fpr
    
    tpr_at_tnr = {}
    for target_tnr in tnr_thresholds:
        idx = np.where(tnr >= target_tnr)[0]
        tpr_at_tnr[target_tnr] = float(tpr[idx[-1]]) if len(idx) > 0 else 0.0
    
    return {
        'AUROC': float(auroc),
        'AP': float(ap),
        **{f'TPR@TNR{int(t*100)}': tpr_at_tnr[t] for t in tnr_thresholds},
        'n_samples': len(labels),
        'n_anomalies': int(labels.sum()),
        'n_normal': int((labels == 0).sum())
    }


def _empty_metrics(tnr_thresholds: List[float]) -> Dict:
    """Return empty metrics dictionary."""
    return {
        'AUROC': np.nan,
        'AP': np.nan,
        **{f'TPR@TNR{int(t*100)}': np.nan for t in tnr_thresholds},
        'n_samples': 0,
        'n_anomalies': 0,
        'n_normal': 0
    }


def calculate_per_category_metrics(
    labels: List[int], 
    scores: List[float], 
    categories: List[str],
    tnr_thresholds: List[float] = [0.99, 0.95, 0.90]
) -> Dict[str, Dict]:
    """
    Calculate metrics separately for each category.
    
    Args:
        labels: List of binary labels
        scores: List of anomaly scores
        categories: List of category names
        tnr_thresholds: TNR thresholds for TPR calculation
    
    Returns:
        Dictionary mapping category names to their metrics
    """
    if not (len(labels) == len(scores) == len(categories)):
        raise ValueError("All input lists must have the same length")
    
    unique_categories = sorted(set(categories))
    results = {}
    
    for cat in unique_categories:
        cat_indices = [i for i, c in enumerate(categories) if c == cat]
        cat_labels = np.array([labels[i] for i in cat_indices])
        cat_scores = np.array([scores[i] for i in cat_indices])
        results[cat] = calculate_metrics(cat_labels, cat_scores, tnr_thresholds)
    
    return results


def calculate_optimal_threshold(
    labels: np.ndarray, 
    scores: np.ndarray,
    method: str = 'f1'
) -> Tuple[float, float]:
    """
    Calculate optimal classification threshold.
    
    Args:
        labels: Binary labels
        scores: Anomaly scores
        method: 'f1', 'youden', or 'balanced'
    
    Returns:
        Tuple of (optimal_threshold, metric_value)
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    
    if method == 'f1':
        precision, recall, thresholds = precision_recall_curve(labels, scores)
        f1_scores = np.where(
            (precision + recall) > 0,
            2 * (precision * recall) / (precision + recall),
            0
        )
        best_idx = np.argmax(f1_scores[:-1])
        return float(thresholds[best_idx]), float(f1_scores[best_idx])
    
    elif method == 'youden':
        fpr, tpr, thresholds = roc_curve(labels, scores)
        youden_j = tpr - fpr
        best_idx = np.argmax(youden_j)
        return float(thresholds[best_idx]), float(youden_j[best_idx])
    
    elif method == 'balanced':
        fpr, tpr, thresholds = roc_curve(labels, scores)
        balanced = (tpr + (1 - fpr)) / 2
        best_idx = np.argmax(balanced)
        return float(thresholds[best_idx]), float(balanced[best_idx])
    
    else:
        raise ValueError(f"Unknown method: {method}")