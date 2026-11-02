"""SHAP explainability: importance of each ECG lead for the hybrid model."""
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config


def _class1(sv):
    """Normalise the different return formats of shap across versions -> array (N, 12, T) for class 1."""
    if isinstance(sv, list):
        return np.asarray(sv[1])
    sv = np.asarray(sv)
    return sv[..., 1] if sv.ndim == 4 else sv


def lead_importance(model, background, samples):
    """Mean |SHAP| per lead (length 12) using GradientExplainer on CPU."""
    import shap
    model = model.cpu().eval()
    with torch.backends.cudnn.flags(enabled=False):          # LSTM backward needs this outside train mode
        expl = shap.GradientExplainer(model, torch.from_numpy(background).float())
        sv = _class1(expl.shap_values(torch.from_numpy(samples).float()))
    return np.abs(sv).mean(axis=(0, 2))


def run_shap(model, X_train, X_test, n_background=50, n_samples=100):
    rng = np.random.default_rng(config.SEED)
    bg = X_train[rng.choice(len(X_train), min(n_background, len(X_train)), replace=False)]
    sm = X_test[rng.choice(len(X_test), min(n_samples, len(X_test)), replace=False)]
    imp = lead_importance(model, bg, sm)
    json.dump(dict(zip(config.LEAD_NAMES, map(float, imp))), open(config.RESULTS_DIR / "shap_lead_importance.json", "w"), indent=2)
    plot_lead_importance(imp, config.FIGURES_DIR / "shap_lead_importance.png", "SHAP feature importance of ECG leads")
    order = np.argsort(-imp)
    print("\n[SHAP] lead ranking (mean |SHAP|):", ", ".join(f"{config.LEAD_NAMES[i]}={imp[i]:.4f}" for i in order))
    return imp


def plot_lead_importance(imp, path=None, title="SHAP lead importance"):
    order = np.argsort(-imp)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    bars = ax.bar([config.LEAD_NAMES[i] for i in order], imp[order], color=plt.cm.Blues(np.linspace(0.95, 0.35, len(imp))))
    for b, v in zip(bars, imp[order]):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.4f}", ha="center", va="bottom", fontsize=7)
    ax.set_ylabel("Mean |SHAP value|"); ax.set_xlabel("ECG lead"); ax.set_title(title)
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=200)
        plt.close(fig)
    return fig
