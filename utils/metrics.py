"""
Metrics utilities for evaluating fight detection models.
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report as sklearn_report,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    average: str = 'binary'
) -> Dict[str, float]:
    """
    Compute classification metrics.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (for ROC-AUC)
        average: Averaging method ('binary', 'micro', 'macro', 'weighted')

    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1': f1_score(y_true, y_pred, average=average, zero_division=0)
    }

    if y_prob is not None:
        try:
            if len(np.unique(y_true)) == 2:
                # Binary classification
                if y_prob.ndim == 2:
                    y_prob_binary = y_prob[:, 1]
                else:
                    y_prob_binary = y_prob
                metrics['roc_auc'] = roc_auc_score(y_true, y_prob_binary)
                metrics['avg_precision'] = average_precision_score(y_true, y_prob_binary)
            else:
                # Multi-class
                metrics['roc_auc'] = roc_auc_score(y_true, y_prob, multi_class='ovr')
        except ValueError:
            pass  # ROC-AUC not computable

    return metrics


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    normalize: Optional[str] = None
) -> np.ndarray:
    """
    Compute confusion matrix.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        normalize: Normalization method ('true', 'pred', 'all', None)

    Returns:
        Confusion matrix
    """
    return confusion_matrix(y_true, y_pred, normalize=normalize)


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    normalize: bool = True,
    save_path: Optional[str] = None,
    title: str = 'Confusion Matrix'
) -> plt.Figure:
    """
    Plot confusion matrix.

    Args:
        cm: Confusion matrix array
        class_names: List of class names
        normalize: Whether to show normalized values
        save_path: Path to save the figure
        title: Plot title

    Returns:
        Matplotlib figure
    """
    if normalize:
        cm_display = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_display = np.nan_to_num(cm_display)
        fmt = '.2f'
    else:
        cm_display = cm
        fmt = 'd'

    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(cm_display, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel='True label',
        xlabel='Predicted label'
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    # Add text annotations
    thresh = cm_display.max() / 2.
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(
                j, i, format(cm_display[i, j], fmt),
                ha='center', va='center',
                color='white' if cm_display[i, j] > thresh else 'black'
            )

    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def compute_roc_auc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    plot: bool = True,
    save_path: Optional[str] = None
) -> Tuple[float, Optional[plt.Figure]]:
    """
    Compute ROC-AUC and optionally plot ROC curve.

    Args:
        y_true: Ground truth labels
        y_prob: Predicted probabilities
        plot: Whether to plot ROC curve
        save_path: Path to save the figure

    Returns:
        Tuple of (ROC-AUC score, figure or None)
    """
    if y_prob.ndim == 2:
        y_prob = y_prob[:, 1]

    auc_score = roc_auc_score(y_true, y_prob)

    fig = None
    if plot:
        fpr, tpr, _ = roc_curve(y_true, y_prob)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {auc_score:.3f})')
        ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC) Curve')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return auc_score, fig


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot precision-recall curve.

    Args:
        y_true: Ground truth labels
        y_prob: Predicted probabilities
        save_path: Path to save the figure

    Returns:
        Matplotlib figure
    """
    if y_prob.ndim == 2:
        y_prob = y_prob[:, 1]

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AP = {ap:.3f})')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    output_dict: bool = False
) -> Union[str, Dict]:
    """
    Generate classification report.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: List of class names
        output_dict: Whether to return as dictionary

    Returns:
        Classification report (string or dict)
    """
    return sklearn_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=output_dict,
        zero_division=0
    )


def compute_segment_iou(
    pred_segments: List[Tuple[int, int]],
    gt_segments: List[Tuple[int, int]]
) -> float:
    """
    Compute segment-level IoU for temporal detection.

    Args:
        pred_segments: List of (start, end) frame tuples for predictions
        gt_segments: List of (start, end) frame tuples for ground truth

    Returns:
        Average segment IoU
    """
    if not pred_segments or not gt_segments:
        return 0.0

    ious = []

    for pred_start, pred_end in pred_segments:
        best_iou = 0.0
        for gt_start, gt_end in gt_segments:
            # Compute intersection
            inter_start = max(pred_start, gt_start)
            inter_end = min(pred_end, gt_end)
            intersection = max(0, inter_end - inter_start)

            # Compute union
            union = (pred_end - pred_start) + (gt_end - gt_start) - intersection

            if union > 0:
                iou = intersection / union
                best_iou = max(best_iou, iou)

        ious.append(best_iou)

    return np.mean(ious) if ious else 0.0


def compute_temporal_map(
    predictions: List[Dict],
    ground_truth: List[Dict],
    iou_thresholds: List[float] = [0.1, 0.25, 0.5]
) -> Dict[str, float]:
    """
    Compute mean Average Precision at different IoU thresholds.

    Args:
        predictions: List of dicts with 'start', 'end', 'score' keys
        ground_truth: List of dicts with 'start', 'end' keys
        iou_thresholds: IoU thresholds to evaluate

    Returns:
        Dictionary mapping threshold to mAP
    """
    results = {}

    for thresh in iou_thresholds:
        # Sort predictions by score
        sorted_preds = sorted(predictions, key=lambda x: x['score'], reverse=True)

        tp = np.zeros(len(sorted_preds))
        fp = np.zeros(len(sorted_preds))

        gt_matched = [False] * len(ground_truth)

        for i, pred in enumerate(sorted_preds):
            best_iou = 0.0
            best_gt_idx = -1

            for j, gt in enumerate(ground_truth):
                if gt_matched[j]:
                    continue

                # Compute IoU
                inter_start = max(pred['start'], gt['start'])
                inter_end = min(pred['end'], gt['end'])
                intersection = max(0, inter_end - inter_start)
                union = (pred['end'] - pred['start']) + (gt['end'] - gt['start']) - intersection

                if union > 0:
                    iou = intersection / union
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = j

            if best_iou >= thresh:
                tp[i] = 1
                gt_matched[best_gt_idx] = True
            else:
                fp[i] = 1

        # Compute precision and recall
        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        recall = tp_cum / len(ground_truth) if ground_truth else tp_cum
        precision = tp_cum / (tp_cum + fp_cum)

        # Compute AP
        ap = 0.0
        for t in np.arange(0, 1.1, 0.1):
            prec_at_recall = precision[recall >= t]
            if len(prec_at_recall) > 0:
                ap += prec_at_recall.max() / 11

        results[f'mAP@{thresh}'] = ap

    return results
