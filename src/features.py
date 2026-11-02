"""Hand-crafted features for the classical ML baselines (SVM / Random Forest)."""
import numpy as np
from scipy.signal import welch
from scipy.stats import skew, kurtosis
import config


def extract_features(X):
    """X: (N, 12, T) pre-processed signals -> (N, 12 * 8) feature matrix."""
    feats = []
    for x in X:
        f = []
        for lead in x:
            freqs, psd = welch(lead, fs=config.SAMPLING_RATE, nperseg=256)
            bands = [(0.5, 5), (5, 15), (15, 40)]
            bp = [psd[(freqs >= a) & (freqs < b)].sum() for a, b in bands]
            f += [lead.mean(), lead.std(), lead.min(), lead.max(), skew(lead), kurtosis(lead)] + bp[:2]
        feats.append(f)
    return np.nan_to_num(np.asarray(feats, dtype=np.float32))
