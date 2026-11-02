"""Training loop for the deep models and fitting of the classical baselines."""
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import config
from src.models import build_model
from src.features import extract_features


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_loader(X, y, batch, shuffle):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch, shuffle=shuffle)


@torch.no_grad()
def evaluate_loader(model, loader, device, criterion=None):
    model.eval()
    loss_sum, correct, n, probs = 0.0, 0, 0, []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        out = model(xb)
        if criterion is not None:
            loss_sum += criterion(out, yb).item() * len(yb)
        correct += (out.argmax(1) == yb).sum().item()
        n += len(yb)
        probs.append(torch.softmax(out, 1)[:, 1].cpu().numpy())
    return loss_sum / n, correct / n, np.concatenate(probs)


def train_deep(name, train, val, epochs=config.EPOCHS, batch=config.BATCH_SIZE, lr=config.LEARNING_RATE):
    device = get_device()
    torch.manual_seed(config.SEED)
    model = build_model(name).to(device)
    (Xtr, ytr), (Xva, yva) = train, val
    weights = torch.tensor(len(ytr) / (2.0 * np.bincount(ytr, minlength=2)), dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=config.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
    tl, vl = make_loader(Xtr, ytr, batch, True), make_loader(Xva, yva, batch, False)
    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best, path = float("inf"), config.MODELS_DIR / f"{name}.pt"
    print(f"\n=== Training {name} on {device} | params: {sum(p.numel() for p in model.parameters()):,} ===")
    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        for xb, yb in tl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        trl, tra, _ = evaluate_loader(model, tl, device, criterion)
        val_l, val_a, _ = evaluate_loader(model, vl, device, criterion)
        sched.step(val_l)
        for k, v in zip(hist, (trl, val_l, tra, val_a)):
            hist[k].append(float(v))
        if val_l < best:
            best = val_l
            torch.save(model.state_dict(), path)
        print(f"[{name}] epoch {ep:3d}/{epochs} train_loss {trl:.4f} acc {tra:.4f} | "
              f"val_loss {val_l:.4f} acc {val_a:.4f} | {time.time() - t0:.1f}s")
    json.dump(hist, open(config.RESULTS_DIR / f"history_{name}.json", "w"))
    model.load_state_dict(torch.load(path, map_location=device))      # best checkpoint by validation loss
    return model


def fit_ml(train, test):
    """SVM and Random Forest baselines on hand-crafted features. Returns {name: test probabilities}."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    Ftr, Fte = extract_features(train[0]), extract_features(test[0])
    models = {
        "SVM": make_pipeline(StandardScaler(), SVC(kernel="rbf", probability=True, class_weight="balanced",
                                                   random_state=config.SEED)),
        "RF": RandomForestClassifier(n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=config.SEED),
    }
    out = {}
    for name, m in models.items():
        print(f"\n=== Fitting {name} ===")
        m.fit(Ftr, train[1])
        out[name] = m.predict_proba(Fte)[:, 1]
    return out
