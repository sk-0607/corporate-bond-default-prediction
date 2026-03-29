"""
src/visualization/plots.py
----------------------------
Reusable plotting functions for the corporate bond default prediction project.
All functions optionally save figures to disk when ``save_path`` is provided.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def plot_confusion_matrix(
    y_true,
    y_pred,
    dataset_name: str = "Test",
    cmap: str = "Blues",
    title_suffix: str = "",
    save_path: str | None = None,
) -> None:
    """
    Plot and print a confusion matrix with TP/TN/FP/FN breakdown.

    Parameters
    ----------
    y_true        : array-like of int
    y_pred        : array-like of int
    dataset_name  : str   Label used in the figure title.
    cmap          : str   Matplotlib colormap (use 'Greens' for LSTM/GRU).
    title_suffix  : str   Appended to the title, e.g. ' (LSTM)'.
    save_path     : str   If given, the figure is saved to this path.
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=cmap,
        xticklabels=["No Default", "Default"],
        yticklabels=["No Default", "Default"],
    )
    plt.title(
        f"Confusion Matrix — {dataset_name} Set{title_suffix}",
        fontsize=14,
        fontweight="bold",
    )
    plt.ylabel("Actual", fontsize=12)
    plt.xlabel("Predicted", fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()

    tn, fp, fn, tp = cm.ravel()
    print(f"\n{dataset_name} Breakdown:")
    print(f"  True Negatives  : {tn:5d}  (correctly predicted no default)")
    print(f"  False Positives : {fp:5d}  (false alarms)")
    print(f"  False Negatives : {fn:5d}  (missed defaults — most costly!)")
    print(f"  True Positives  : {tp:5d}  (correctly predicted default)")


def plot_roc_curve(
    y_true,
    y_pred_proba,
    dataset_name: str = "Test",
    color: str = "steelblue",
    title_suffix: str = "",
    save_path: str | None = None,
) -> None:
    """
    Plot the ROC curve.

    Parameters
    ----------
    y_true        : array-like of int
    y_pred_proba  : array-like of float
    dataset_name  : str
    color         : str   Line colour.
    title_suffix  : str   Appended to the title, e.g. ' (LSTM)'.
    save_path     : str   Optional output path.
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    auc = roc_auc_score(y_true, y_pred_proba)

    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, linewidth=2, color=color,
             label=f"ROC Curve (AUC = {auc:.4f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Classifier")
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title(
        f"ROC Curve — {dataset_name} Set{title_suffix}",
        fontsize=14,
        fontweight="bold",
    )
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()


def plot_precision_recall_curve(
    y_true,
    y_pred_proba,
    dataset_name: str = "Test",
    color: str = "steelblue",
    title_suffix: str = "",
    save_path: str | None = None,
) -> None:
    """
    Plot the Precision-Recall curve.

    Parameters
    ----------
    y_true        : array-like of int
    y_pred_proba  : array-like of float
    dataset_name  : str
    color         : str
    title_suffix  : str
    save_path     : str   Optional output path.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)

    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, linewidth=2, color=color, label="PR Curve")
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title(
        f"Precision-Recall Curve — {dataset_name} Set{title_suffix}",
        fontsize=14,
        fontweight="bold",
    )
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()


def plot_training_history(
    history,
    title_suffix: str = "",
    save_path: str | None = None,
) -> None:
    """
    Plot training and validation loss + AUC curves from a Keras History object.

    Parameters
    ----------
    history     : keras.callbacks.History
    title_suffix: str   Appended to the suptitle, e.g. ' — LSTM'.
    save_path   : str   Optional output path.
    """
    hist = history.history

    # Detect AUC key (Keras names it 'AUC', 'auc', or 'auc_1' depending on version)
    auc_key     = next((k for k in hist if k.lower().startswith("auc") and not k.startswith("val")), None)
    val_auc_key = next((k for k in hist if k.startswith("val") and "auc" in k.lower()), None)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(hist["loss"],     label="Train")
    axes[0].plot(hist["val_loss"], label="Validation")
    axes[0].set_title("Loss per Epoch", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    if auc_key and val_auc_key:
        axes[1].plot(hist[auc_key],     label="Train")
        axes[1].plot(hist[val_auc_key], label="Validation")
        axes[1].set_title("AUC per Epoch", fontsize=13, fontweight="bold")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("AUC")
        axes[1].legend()
        axes[1].grid(alpha=0.3)

    plt.suptitle(f"Training History{title_suffix}", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()


def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 20,
    save_path: str | None = None,
) -> None:
    """
    Plot a horizontal bar chart of CatBoost feature importances.

    Parameters
    ----------
    importance_df : pd.DataFrame
        Must have columns ``feature`` and ``importance``, sorted descending.
    top_n         : int   Number of top features to plot.
    save_path     : str   Optional output path.
    """
    top = importance_df.head(top_n)

    plt.figure(figsize=(10, 8))
    plt.barh(range(len(top)), top["importance"])
    plt.yticks(range(len(top)), top["feature"])
    plt.xlabel("Importance", fontsize=12)
    plt.title(f"Top {top_n} Feature Importances", fontsize=14, fontweight="bold")
    plt.gca().invert_yaxis()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()
