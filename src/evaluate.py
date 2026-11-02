"""Metrics and figures: model comparison, ROC curves, confusion matrix, training curves."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score, precision_score, recall_score,
                             confusion_matrix, roc_curve)
import config


def compute_metrics(y, prob, thr=0.5):
    pred = (prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"Accuracy (%)": 100 * accuracy_score(y, pred), "AUC (%)": 100 * roc_auc_score(y, prob),
            "F1-Score (%)": 100 * f1_score(y, pred), "Precision (%)": 100 * precision_score(y, pred, zero_division=0),
            "Recall (%)": 100 * recall_score(y, pred), "Specificity (%)": 100 * tn / max(tn + fp, 1)}


def save_results(y, probs):
    """probs: {model_name: test probabilities}. Writes results table + all figures."""
    rows = {n: compute_metrics(y, p) for n, p in probs.items()}
    df = pd.DataFrame(rows).T.round(2)
    df.index.name = "Model"
    df.to_csv(config.RESULTS_DIR / "metrics.csv")
    print("\n================ TEST-SET RESULTS ================\n", df.to_string())
    np.savez(config.RESULTS_DIR / "test_probs.npz", y=y, **{k.replace("-", "_"): v for k, v in probs.items()})
    plot_comparison(df)
    plot_roc(y, probs)
    plot_confusion(y, probs[config.PROPOSED] if config.PROPOSED in probs else list(probs.values())[-1])
    plot_history()
    return df


def plot_comparison(df):
    ax = (df / 100).plot(kind="bar", figsize=(11, 5), width=0.8)
    ax.set_ylabel("Score"); ax.set_ylim(max(0, df.values.min() / 100 - 0.05), 1.0)
    ax.set_title("Performance comparison on the test set"); plt.xticks(rotation=0); plt.legend(ncol=3, fontsize=8)
    plt.tight_layout(); plt.savefig(config.FIGURES_DIR / "model_comparison.png", dpi=200); plt.close()


def plot_roc(y, probs):
    plt.figure(figsize=(6.5, 5.5))
    plt.plot([0, 1], [0, 1], "k--", label="Random (AUC = 0.50)")
    for n, p in probs.items():
        fpr, tpr, _ = roc_curve(y, p)
        plt.plot(fpr, tpr, lw=2 if n == config.PROPOSED else 1.2, label=f"{n} (AUC = {roc_auc_score(y, p):.4f})")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); plt.title("ROC curves")
    plt.legend(fontsize=8, loc="lower right"); plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "roc_curves.png", dpi=200); plt.close()


def plot_confusion(y, prob):
    cm = confusion_matrix(y, (prob >= 0.5).astype(int), labels=[0, 1])
    pct = cm / cm.sum(axis=1, keepdims=True) * 100
    annot = np.array([[f"{pct[i, j]:.1f}%\n({cm[i, j]} samples)" for j in range(2)] for i in range(2)])
    plt.figure(figsize=(5.5, 4.5))
    sns.heatmap(pct, annot=annot, fmt="", cmap="Blues", vmin=0, vmax=100, xticklabels=config.CLASS_NAMES,
                yticklabels=config.CLASS_NAMES, cbar_kws={"label": "Percentage (%)"})
    plt.xlabel("Predicted label"); plt.ylabel("True label"); plt.title("Confusion matrix (proposed model)")
    plt.tight_layout(); plt.savefig(config.FIGURES_DIR / "confusion_matrix.png", dpi=200); plt.close()


def plot_history():
    hists = {n: json.load(open(config.RESULTS_DIR / f"history_{n}.json")) for n in config.DEEP_MODELS
             if (config.RESULTS_DIR / f"history_{n}.json").exists()}
    for key, title, fname in [("loss", "Training vs. validation loss", "loss_curves.png"),
                              ("acc", "Training vs. validation accuracy", "accuracy_curves.png")]:
        plt.figure(figsize=(7, 4.5))
        for i, (n, h) in enumerate(hists.items()):
            c = f"C{i}"
            plt.plot(h[f"val_{key}"], color=c, lw=1.8, label=f"{n} (val)")
            plt.plot(h[f"train_{key}"], color=c, lw=1, ls="--", label=f"{n} (train)")
        plt.xlabel("Epoch"); plt.ylabel(key.capitalize()); plt.title(title); plt.legend(fontsize=7, ncol=2)
        plt.tight_layout(); plt.savefig(config.FIGURES_DIR / fname, dpi=200); plt.close()
