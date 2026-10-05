# 🌸 AI Gesture-to-Flower System

Show a hand gesture to the webcam, hold it steady for **2 seconds**, and the matching flower blooms.

```
Webcam → MediaPipe Hands → 21 landmarks → Feature engineering (101 features) → StandardScaler
       → trained FNN (MLP, TensorFlow/Keras) → gesture + confidence → 2 s stability logic → flower
```
Module: ITLPA701 Python & Fundamentals of AI · Approach: **Deep Learning – FNN** · Domain: **image/gesture recognition**.

## Gesture → flower mapping
| Gesture (class) | Flower | | Gesture (class) | Flower |
|---|---|---|---|---|
| open_palm | Rose | | thumbs_down | Orchid |
| fist | Sunflower | | pointing | Daisy |
| peace | Tulip | | rock | Lavender |
| thumbs_up | Lily | | ok_sign | Lotus |

## Setup (PyCharm / terminal)
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt                          # Python 3.10-3.13
python tests/test_core.py                                # quick self-check (no webcam needed)
```
In PyCharm: *Settings → Project → Python Interpreter → add the `.venv`*, mark the project root as **Sources Root**,
and run everything as modules (`python -m src.train`) from the project root.

## Full workflow
```bash
python -m src.collect_data            # 1. record 300 samples x 8 gestures with the webcam (SPACE to start)
python -m src.preprocess              # 2. clean -> split -> augment -> features -> scale
python -m src.train --tune            # 3. compare 4 FNN configs on validation, keep the best
python -m src.evaluate                # 4. test metrics, confusion matrix, curves, interpretation
streamlit run app.py                  # 5. run the app
```
> **Important:** this repository ships *without* `data/raw/*.csv` and `models/gesture_fnn.keras`, because the
> dataset must be recorded from your own hand (no public dataset has these 8 gestures as MediaPipe landmarks).
> Steps 1–4 take ≈ 15–20 min. Afterwards commit `models/` and `reports/` so the app can be deployed.

### Data-collection tips (they decide your accuracy)
* ≥ 300 samples per gesture, recorded in **≥ 2 separate runs** (each run = new session id).
* Vary distance, angle, lighting, background; use both hands; slowly move the hand while recording.
* Ask a classmate to record too – this is the best fix for poor generalisation.

## Dataset strategy
Custom, reproducible landmark dataset (`src/collect_data.py`). Each row = label, handedness, session, block,
and 63 raw landmark values (x,y,z × 21). Storing *landmarks* (not images) is privacy-friendly and small.
Public datasets (e.g. HaGRID, Kaggle ASL) were not used: licences/size and class sets do not match these 8 classes.

## Preprocessing (`src/preprocess.py`)
1. Validate columns, drop NaN / unknown labels / duplicates / degenerate hands.
2. **Block-wise stratified split 70/15/15**: 15 consecutive frames form a block that goes wholly to one split →
   near-identical frames cannot leak from train into test (otherwise accuracy is inflated).
3. Augment **train only** ×4 (±15° rotation, Gaussian jitter 2 % of hand size).
4. Feature engineering, then `StandardScaler` fitted on train only (saved as `models/scaler.json`).

## Feature engineering (`src/features.py`, 101 features)
| Group | # | Why useful |
|---|---|---|
| Normalised coordinates of landmarks 1–20 (x,y,z) | 60 | full hand shape; wrist-centred, size-normalised |
| Fingertip→wrist distances | 5 | how far each finger is extended |
| Fingertip↔fingertip distances | 10 | pinch (OK sign), spread (open palm), V shape |
| Joint angles (3 per finger, /π) | 15 | finger curl, independent of hand size |
| Palm direction (unit vector wrist→middle knuckle) | 3 | hand orientation |
| Thumb direction (unit vector) | 3 | **thumbs-up vs thumbs-down** |
| Extension ratio tip/PIP distance | 5 | simple open/closed cue |

Normalisation: translate wrist to origin → mirror left hands → divide by mean wrist-to-knuckle distance.
Rotation is intentionally **kept** (orientation is needed for thumbs up/down). The same function
`extract_features` + the saved scaler are used in training **and** live inference → no train/serve skew.

## FNN model (`src/train.py`)
`Input(101) → [Dense(128)+BN+Dropout(.3)] → [Dense(64)+BN+Dropout(.3)] → [Dense(32)+BN+Dropout(.3)] → Dense(8, softmax)`
| Parameter | Value | Reason |
|---|---|---|
| Activation | ReLU (He init) | fast, avoids vanishing gradients |
| Regularisation | Dropout 0.3, L2 1e-4, BatchNorm | small dataset → fight over-fitting |
| Optimizer / LR | Adam, 1e-3 (ReduceLROnPlateau ×0.5) | robust default for MLPs |
| Loss | sparse categorical cross-entropy | multi-class, integer labels |
| Batch / Epochs | 32 / max 200 | small data → small batches |
| Early stopping | patience 20 on val_loss, restore best | stops over-fitting automatically |
| Class weights | balanced | protects against imbalance |
`--tune` compares `baseline`, `wide_dropout`, `compact`, `low_lr` on the **validation** set (test set untouched).

## Evaluation (`src/evaluate.py` → `reports/`)
Accuracy, precision, recall, F1 (macro + weighted + per class), confusion matrix (counts + normalised),
training/validation accuracy and loss curves, confidence-threshold analysis (justifies 0.80), latency,
and an auto-written `interpretation.md`. **Your real numbers appear there after you train.**

## Stability logic (`src/stability.py`)
Probabilities are averaged over 5 frames; a gesture counts only if smoothed confidence ≥ 0.80; the timer
runs while the same gesture persists; another confident gesture resets it; gaps ≤ 0.35 s are tolerated;
the flower changes only when a *different* gesture is confirmed. Unit-tested in `tests/test_core.py`.

## Deploy
* **Note:** MediaPipe 0.10.15+ uses the Tasks API; the first run downloads `models/hand_landmarker.task` (~8 MB).
* **Local:** `streamlit run app.py`
* **Streamlit Community Cloud:** push to GitHub (include `models/`), choose Python 3.11 in *Advanced settings*,
  main file `app.py`. `packages.txt` installs system libs. Browser webcam uses WebRTC; on strict networks a
  TURN server may be needed (STUN is configured). Camera needs HTTPS (Cloud provides it).

git init

git add .

git status

git commit -m "AI Gesture-to-Flower"

git branch -M main

git remote add origin https://github.com/yhdjanvier/gesture_flower.git

git push -u origin main


## Project structure
```
config.py  app.py  requirements.txt  packages.txt  README.md  .gitignore  .streamlit/config.toml
src/ features.py hand_detector.py collect_data.py preprocess.py train.py evaluate.py
     inference.py stability.py flowers.py
tests/test_core.py   docs/CAT_COMPLIANCE.md
data/raw/ (CSV)  data/processed/ (npz)  models/ (gesture_fnn.keras, scaler.json, label_map.json,
model_config.json, training_history.json)  reports/ (metrics, plots)
```

## Responsible use & limitations
* **Privacy:** only 21 landmark numbers are stored, never images; video is processed in memory. Get consent before recording others.
* **Bias:** trained on few hands/skin tones/lighting → lower accuracy for others. *Mitigate:* diverse volunteers, per-group testing.
* **Misrecognition:** the 0.80 threshold + 2 s hold + temporal smoothing reduce wrong flowers.
* **Ambiguous gestures:** cultural meaning of gestures varies (thumbs-up/OK); do not use for safety-critical decisions.
* **Scope:** one hand, frontal view, similar to the training distribution.
