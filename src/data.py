"""Loading PTB-XL (real data) or generating a small synthetic data set (demo mode)."""
import ast
import numpy as np
import pandas as pd
import config


# --------------------------------------------------------------------------- PTB-XL
def load_ptbxl(data_dir=config.DATA_DIR, max_records=None):
    """Return X (N, 12, 1000) float32, y (N,) int64 (0 = NORM, 1 = abnormal), fold (N,) int.

    Label rule: a record is NORMAL if its only diagnostic super-class is NORM, ABNORMAL if it has at
    least one of MI / STTC / CD / HYP. Records without any diagnostic super-class are dropped.
    """
    import wfdb

    cache = config.CACHE_DIR / f"ptbxl_{config.SAMPLING_RATE}_{max_records}.npz"
    if cache.exists():
        z = np.load(cache)
        print(f"[data] loaded cached arrays from {cache}")
        return z["X"], z["y"], z["fold"]

    csv = data_dir / "ptbxl_database.csv"
    scp = data_dir / "scp_statements.csv"
    if not csv.exists() or not scp.exists():
        raise FileNotFoundError(
            f"PTB-XL files not found in {data_dir}.\n"
            "Download PTB-XL from https://physionet.org/content/ptb-xl/1.0.3/ , extract it and put the "
            "files inside data/ptb-xl/ (see README.md). Or run with --demo to test with synthetic data.")

    df = pd.read_csv(csv, index_col="ecg_id")
    df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)
    agg = pd.read_csv(scp, index_col=0)
    agg = agg[agg["diagnostic"] == 1]

    def to_superclasses(codes):
        out = set()
        for k in codes:
            if k in agg.index:
                out.add(agg.loc[k, "diagnostic_class"])
        return out

    df["superclasses"] = df["scp_codes"].apply(to_superclasses)
    df = df[df["superclasses"].apply(len) > 0]
    df["label"] = df["superclasses"].apply(lambda s: 0 if s == {"NORM"} else 1)
    if max_records:
        df = df.sample(n=min(max_records, len(df)), random_state=config.SEED)

    col = "filename_lr" if config.SAMPLING_RATE == 100 else "filename_hr"
    X = np.zeros((len(df), config.N_LEADS, config.SIGNAL_LEN), dtype=np.float32)
    keep = np.ones(len(df), dtype=bool)
    for i, f in enumerate(df[col].values):
        try:
            sig, _ = wfdb.rdsamp(str(data_dir / f))
            sig = sig.T[:, : config.SIGNAL_LEN]
            X[i, :, : sig.shape[1]] = sig
        except Exception as e:  # missing / corrupt record
            keep[i] = False
            print(f"[data] skipped {f}: {e}")
        if (i + 1) % 2000 == 0:
            print(f"[data] read {i + 1}/{len(df)} records")
    X = np.nan_to_num(X[keep])
    y = df["label"].values[keep].astype(np.int64)
    fold = df["strat_fold"].values[keep].astype(np.int64)
    np.savez_compressed(cache, X=X, y=y, fold=fold)
    return X, y, fold


def split_by_fold(X, y, fold):
    """Official PTB-XL recommendation: folds 1-8 train, 9 validation, 10 test (patient-wise split)."""
    tr, va, te = fold <= 8, fold == 9, fold == 10
    return (X[tr], y[tr]), (X[va], y[va]), (X[te], y[te])


# --------------------------------------------------------------------------- Synthetic demo data
_LEAD_GAIN = np.array([0.7, 1.0, 0.4, -0.8, 0.3, 0.7, 0.4, 0.9, 1.1, 1.3, 1.1, 0.8])
_PRECORDIAL = np.arange(6, 12)


def _g(t, mu, amp, w):
    return amp * np.exp(-0.5 * ((t - mu) / w) ** 2)


def synth_ecg(abnormal, rng):
    """One synthetic 12-lead, 10 s ECG. FOR PIPELINE TESTING ONLY - not real physiology."""
    fs, n = config.SAMPLING_RATE, config.SIGNAL_LEN
    t = np.arange(n) / fs
    hr = rng.uniform(55, 100)
    rr = 60.0 / hr
    beats = np.arange(rng.uniform(0.2, 0.8), 10.0, rr)
    beats = beats + rng.normal(0, 0.01, size=len(beats))
    kind = rng.integers(0, 3) if abnormal else -1
    sig = np.zeros((config.N_LEADS, n))
    for b in beats:
        tt = t - b
        qrs_w = 0.012 * (2.5 if kind == 1 else 1.0)
        qrs_a = 1.0 * (0.6 if kind == 1 else 1.0)
        beat = (_g(tt, -0.2, 0.15, 0.025) + _g(tt, -0.04, -0.12, 0.01) + _g(tt, 0, qrs_a, qrs_w)
                + _g(tt, 0.04, -0.25, 0.012) + _g(tt, 0.27, 0.30, 0.05))
        for l in range(config.N_LEADS):
            s = beat * _LEAD_GAIN[l]
            if l in _PRECORDIAL[1:5]:                       # V2-V5 carry the abnormality
                if kind == 0:                                # ST elevation
                    s = s + _g(tt, 0.12, 0.35, 0.06) * _LEAD_GAIN[l]
                elif kind == 2:                              # inverted T wave
                    s = s - 2.0 * _g(tt, 0.27, 0.30, 0.05) * _LEAD_GAIN[l]
            sig[l] += s
    wander = 0.1 * np.sin(2 * np.pi * rng.uniform(0.1, 0.4) * t + rng.uniform(0, 6.28))
    sig += wander + rng.normal(0, 0.03, size=sig.shape)
    return sig.astype(np.float32)


def make_demo_data(n=1200, seed=config.SEED):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n).astype(np.int64)
    X = np.stack([synth_ecg(bool(v), rng) for v in y])
    fold = rng.integers(1, 11, size=n)
    print(f"[data] DEMO MODE: generated {n} synthetic ECGs (not real data - results are meaningless clinically)")
    return X, y, fold
