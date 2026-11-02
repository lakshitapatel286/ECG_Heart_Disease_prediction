"""Noise reduction, fixed-length segmentation and normalisation."""
import json
import numpy as np
from scipy.signal import butter, sosfiltfilt
import config


def bandpass(X, low=0.5, high=40.0, fs=config.SAMPLING_RATE, order=3):
    """Remove baseline wander (<0.5 Hz) and high-frequency / powerline noise (>40 Hz). X: (..., T)."""
    sos = butter(order, [low, min(high, fs / 2 - 1)], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, X, axis=-1).astype(np.float32)


def fix_length(X, length=config.SIGNAL_LEN):
    """Crop or zero-pad the last axis to a fixed length (fixed-length segmentation)."""
    T = X.shape[-1]
    if T >= length:
        return X[..., :length]
    pad = [(0, 0)] * (X.ndim - 1) + [(0, length - T)]
    return np.pad(X, pad)


class Standardizer:
    """Per-lead z-score using statistics of the TRAINING set only."""

    def fit(self, X):
        self.mean = X.mean(axis=(0, 2), keepdims=True)[0].astype(np.float32)   # (12, 1)
        self.std = (X.std(axis=(0, 2), keepdims=True)[0] + 1e-6).astype(np.float32)
        return self

    def transform(self, X):
        return ((X - self.mean) / self.std).astype(np.float32)

    def save(self, path):
        json.dump({"mean": self.mean.tolist(), "std": self.std.tolist()}, open(path, "w"))

    @classmethod
    def load(cls, path):
        d = json.load(open(path))
        s = cls()
        s.mean = np.array(d["mean"], dtype=np.float32)
        s.std = np.array(d["std"], dtype=np.float32)
        return s


def clean(X):
    return bandpass(fix_length(X))
