"""Flask web app: upload an ECG (.npy / .csv), get the prediction + SHAP lead explanation. History in SQLite."""
import base64
import io
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from flask import Flask, render_template, request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from src import data, preprocess, explain
from src.models import build_model

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

app = Flask(__name__)
_state = {}


def db():
    con = sqlite3.connect(config.DB_PATH)
    con.execute("""CREATE TABLE IF NOT EXISTS history (id INTEGER PRIMARY KEY AUTOINCREMENT, time TEXT, filename TEXT,
                   prediction TEXT, probability REAL, top_leads TEXT)""")
    return con


def load_assets():
    if "model" in _state:
        return True
    mp, sp = config.MODELS_DIR / f"{config.PROPOSED}.pt", config.MODELS_DIR / "scaler.json"
    if not (mp.exists() and sp.exists()):
        return False
    m = build_model(config.PROPOSED)
    m.load_state_dict(torch.load(mp, map_location="cpu"))
    _state["model"] = m.eval()
    _state["scaler"] = preprocess.Standardizer.load(sp)
    # small background set for SHAP (synthetic noise-free baseline = zeros plus a few real-like samples)
    rng = np.random.default_rng(config.SEED)
    _state["bg"] = rng.normal(0, 0.5, size=(20, config.N_LEADS, config.SIGNAL_LEN)).astype(np.float32)
    return True


def read_ecg(file_storage):
    name = file_storage.filename.lower()
    raw = file_storage.read()
    arr = np.load(io.BytesIO(raw)) if name.endswith(".npy") else np.loadtxt(io.StringIO(raw.decode()), delimiter=",")
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("ECG must be a 2-D array (1000 samples x 12 leads, or 12 x 1000).")
    if arr.shape[0] != config.N_LEADS:
        arr = arr.T
    if arr.shape[0] != config.N_LEADS:
        raise ValueError(f"Expected 12 leads, got shape {arr.shape}.")
    return arr


def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def ecg_plot(x):
    fig, axes = plt.subplots(6, 2, figsize=(10, 7), sharex=True)
    t = np.arange(x.shape[1]) / config.SAMPLING_RATE
    for i, ax in enumerate(axes.T.ravel()):
        ax.plot(t, x[i], lw=0.7, color="#c0392b")
        ax.set_ylabel(config.LEAD_NAMES[i], rotation=0, labelpad=14, fontsize=8)
        ax.set_yticks([])
    axes[-1, 0].set_xlabel("Time (s)"); axes[-1, 1].set_xlabel("Time (s)")
    return fig_to_b64(fig)


def predict(x_raw):
    x = preprocess.clean(x_raw[None])                       # (1, 12, 1000) filtered
    z = _state["scaler"].transform(x)
    with torch.no_grad():
        p = torch.softmax(_state["model"](torch.from_numpy(z)), 1)[0, 1].item()
    imp = explain.lead_importance(_state["model"], _state["bg"], z)
    return p, imp, x[0]


@app.route("/", methods=["GET", "POST"])
def index():
    ctx = {"ready": load_assets(), "result": None, "error": None}
    if request.method == "POST" and ctx["ready"]:
        try:
            if request.form.get("demo"):
                x = data.synth_ecg(request.form["demo"] == "abnormal", np.random.default_rng())
                fname = f"synthetic_{request.form['demo']}.npy"
            else:
                f = request.files["ecg"]
                x, fname = read_ecg(f), f.filename
            p, imp, xf = predict(x)
            label = config.CLASS_NAMES[int(p >= 0.5)]
            top = [config.LEAD_NAMES[i] for i in np.argsort(-imp)[:3]]
            con = db()
            con.execute("INSERT INTO history (time, filename, prediction, probability, top_leads) VALUES (?,?,?,?,?)",
                        (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), fname, label, round(p * 100, 2), ", ".join(top)))
            con.commit(); con.close()
            ctx["result"] = {"label": label, "prob": round(p * 100, 2), "top": top, "file": fname,
                             "ecg_img": ecg_plot(xf), "shap_img": fig_to_b64(explain.plot_lead_importance(imp, None, "SHAP lead importance for this ECG"))}
        except Exception as e:
            ctx["error"] = str(e)
    con = db()
    ctx["history"] = con.execute("SELECT time, filename, prediction, probability, top_leads FROM history ORDER BY id DESC LIMIT 15").fetchall()
    con.close()
    return render_template("index.html", **ctx)


if __name__ == "__main__":
    app.run(port=5000)
