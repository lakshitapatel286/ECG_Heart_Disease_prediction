# Hybrid CNN-BiLSTM-Transformer with SHAP for Heart Disease Prediction from ECG

Pipeline: PTB-XL data -> band-pass filtering + fixed-length segmentation + normalisation -> baselines
(SVM, Random Forest, CNN, BiLSTM, Transformer) and the proposed hybrid CNN-BiLSTM-Transformer ->
metrics, ROC, confusion matrix, training curves -> SHAP lead importance -> Flask web app (SQLite history).

## 1. Set up in PyCharm
1. `File > Open` and select this folder (`ecg_project`).
2. `File > Settings > Project > Python Interpreter > Add Interpreter > Add Local Interpreter > Virtualenv` (Python 3.9-3.12 recommended).
3. Open the terminal inside PyCharm and run: `pip install -r requirements.txt`
   (On a machine without an NVIDIA GPU you can install the smaller CPU build of PyTorch from https://pytorch.org/get-started/locally/ first.)

## 2. Quick test (no download needed)
Run `main.py` with the parameters `--demo --epochs 3`
(`Run > Edit Configurations > main.py > Parameters`) or in the terminal:

    python main.py --demo --epochs 3

This uses SYNTHETIC ECGs only to check that everything works. Do not report these numbers anywhere.

## 3. Real run on PTB-XL
1. Download PTB-XL 1.0.3 from https://physionet.org/content/ptb-xl/1.0.3/ (zip, ~1.7 GB) and extract it.
2. Copy the extracted contents so that this exists: `data/ptb-xl/ptbxl_database.csv`, `data/ptb-xl/scp_statements.csv`, `data/ptb-xl/records100/...`
3. Run:

       python main.py --epochs 50

   Use `--max-records 3000 --epochs 5` for a first small try. A GPU is strongly recommended for 50 epochs;
   on CPU a full run can take many hours. Options: `--models CNN BiLSTM` (subset), `--skip-ml`, `--batch-size 128`.

Labels: NORM only = Normal (0); any of MI / STTC / CD / HYP = Heart disease (1).
Split: official PTB-XL folds - 1-8 train, 9 validation, 10 test (patient-wise, no leakage).

## 4. Outputs (folder `outputs/`)
- `results/metrics.csv` - accuracy, AUC, F1, precision, recall, specificity for every model (your table of results)
- `figures/` - model_comparison.png, roc_curves.png, confusion_matrix.png, loss_curves.png, accuracy_curves.png, shap_lead_importance.png
- `results/shap_lead_importance.json` - mean |SHAP| per lead
- `models/` - trained weights and the scaler used by the app

All numbers come from YOUR runs - quote them from `metrics.csv`.

## 5. Web app
After `main.py` has finished: run `run_app.py` and open http://127.0.0.1:5000
Upload a 12-lead ECG (`.npy` or `.csv`, 12 x 1000 or 1000 x 12) to get the prediction, probability, ECG plot and per-ECG SHAP lead
importance. Each prediction is stored in `outputs/inspection_history.db` (SQLite).
To try a real record: `wfdb.rdsamp('data/ptb-xl/records100/00000/00001_lr')[0]` gives a 1000 x 12 array you can `np.save`.

## 6. Folder structure
    config.py          settings (paths, epochs, lead names)
    main.py            whole pipeline
    run_app.py         web app launcher
    src/data.py        PTB-XL loader, split, synthetic demo data
    src/preprocess.py  band-pass filter, segmentation, standardisation
    src/features.py    features for SVM / RF
    src/models.py      CNN, BiLSTM, Transformer, hybrid network
    src/train.py       training loop, ML baselines
    src/evaluate.py    metrics and figures
    src/explain.py     SHAP
    app/               Flask app + template

Research prototype - not a medical device.
