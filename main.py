"""Run the complete pipeline:  data -> preprocessing -> baselines -> deep models -> evaluation -> SHAP.

PyCharm: right-click this file > Run.  To pass options use  Run > Edit Configurations > Parameters, e.g.
    --demo --epochs 3            (quick test with synthetic data, no download needed)
    --epochs 50                  (real PTB-XL run; needs the dataset in data/ptb-xl/)
"""
import argparse
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import numpy as np
import torch
import config
from src import data, preprocess, train, evaluate, explain
from src.models import build_model


def parse():
    p = argparse.ArgumentParser(description="Hybrid CNN-BiLSTM-Transformer + SHAP for heart disease prediction from ECG")
    p.add_argument("--demo", action="store_true", help="use a small synthetic data set instead of PTB-XL")
    p.add_argument("--epochs", type=int, default=config.EPOCHS)
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--max-records", type=int, default=None, help="use only N PTB-XL records (quick tests)")
    p.add_argument("--models", nargs="+", default=config.DEEP_MODELS, choices=config.DEEP_MODELS)
    p.add_argument("--skip-ml", action="store_true", help="skip SVM / Random Forest baselines")
    p.add_argument("--demo-size", type=int, default=1200)
    return p.parse_args()


def main():
    a = parse()
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)

    # 1) data ---------------------------------------------------------------------------
    X, y, fold = data.make_demo_data(a.demo_size) if a.demo else data.load_ptbxl(max_records=a.max_records)
    print(f"[data] {len(y)} records | normal: {(y == 0).sum()} | abnormal: {(y == 1).sum()}")

    # 2) preprocessing (band-pass filter, fixed length) -> split -> normalise ---------------
    X = preprocess.clean(X)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = data.split_by_fold(X, y, fold)
    print(f"[data] train {len(ytr)} | validation {len(yva)} | test {len(yte)}")
    scaler = preprocess.Standardizer().fit(Xtr)
    scaler.save(config.MODELS_DIR / "scaler.json")

    probs = {}
    # 3) classical baselines (on filtered, un-normalised signals) -----------------------------
    if not a.skip_ml:
        probs.update(train.fit_ml((Xtr, ytr), (Xte, yte)))

    # 4) deep models ----------------------------------------------------------------------
    Ztr, Zva, Zte = scaler.transform(Xtr), scaler.transform(Xva), scaler.transform(Xte)
    trained = {}
    for name in a.models:
        model = train.train_deep(name, (Ztr, ytr), (Zva, yva), epochs=a.epochs, batch=a.batch_size)
        _, _, p = train.evaluate_loader(model, train.make_loader(Zte, yte, a.batch_size, False), train.get_device())
        probs[name] = p
        trained[name] = model

    # 5) evaluation (metrics table + figures) --------------------------------------------------
    order = [m for m in config.ML_MODELS + config.DEEP_MODELS if m in probs]
    evaluate.save_results(yte, {m: probs[m] for m in order})

    # 6) SHAP for the proposed model ---------------------------------------------------------
    if config.PROPOSED in trained:
        explain.run_shap(trained[config.PROPOSED], Ztr, Zte)
    print(f"\nDone. Tables and figures are in: {config.OUT_DIR}")
    print("Start the web app with:  python run_app.py")


if __name__ == "__main__":
    main()
