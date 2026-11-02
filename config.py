"""Central configuration. Edit the values here (or override them from the command line in main.py)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Folder that contains ptbxl_database.csv, scp_statements.csv and records100/  (see README.md)
DATA_DIR = ROOT / "data" / "ptb-xl"

SAMPLING_RATE = 100          # 100 Hz records are used (records100/)
SIGNAL_LEN = 1000            # 10 s * 100 Hz
N_LEADS = 12
LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
CLASS_NAMES = ["Normal", "Heart Disease"]

# Training
SEED = 42
EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

# Output folders
OUT_DIR = ROOT / "outputs"
MODELS_DIR = OUT_DIR / "models"
RESULTS_DIR = OUT_DIR / "results"
FIGURES_DIR = OUT_DIR / "figures"
CACHE_DIR = OUT_DIR / "cache"
DB_PATH = OUT_DIR / "inspection_history.db"

DEEP_MODELS = ["CNN", "BiLSTM", "Transformer", "CNN-BiLSTM-Transformer"]
ML_MODELS = ["SVM", "RF"]
PROPOSED = "CNN-BiLSTM-Transformer"

for _d in (MODELS_DIR, RESULTS_DIR, FIGURES_DIR, CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)
