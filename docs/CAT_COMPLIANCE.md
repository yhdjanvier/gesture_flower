# CAT2 Practical 2026 – Compliance, marks mapping and technology summary
Module ITLPA701 · Approach: Deep Learning (FNN) · Domain: gesture (image) recognition → flower display.
> Numbers (accuracy, F1, sample counts) are produced by YOUR run and written to `reports/`; paste them here before submitting.

## 1. Requirement → evidence table
| # | CAT requirement | Marks | How the project satisfies it | Evidence |
|---|---|---|---|---|
| 1 | Environment configured / functionalities specified | 1.5+1.5 | Python venv, pinned `requirements.txt`, PyCharm instructions, `packages.txt`; functionalities listed in README + `config.py` | README, requirements.txt, config.py |
| 2 | Functionalities specification | (in 1) | Webcam detection, landmark display, prediction, confidence, 2 s timer, confirmed gesture, flower, errors | README, app.py |
| 3 | Data acquisition | 3 | Guided webcam collector → `data/raw/gesture_landmarks.csv` (8 classes × ≥300 samples, sessions, blocks) | src/collect_data.py |
| 4 | Data preprocessing | 3 | Validation/cleaning, de-duplication, block-wise stratified 70/15/15 split, train-only augmentation, scaler | src/preprocess.py, reports/class_distribution.* |
| 5 | Feature engineering (data) | 3 | 101 features: normalised coords, distances, angles, orientation, extension ratios | src/features.py |
| 6 | Model feature engineering | 3 | Standardisation (train-fit), class weights, BN/Dropout/L2, architecture candidates, feature-name registry | src/train.py, models/scaler.json |
| 7 | AI approach selection & justification | 1.5 | FNN on landmarks: tiny input, real-time (<5 ms), needs little data, privacy-friendly vs CNN on pixels | README "Dataset strategy"/"FNN model" |
| 8 | Model parameters & explanation | 1.5 | Table of layers, neurons, activation, dropout, L2, Adam, LR, loss, batch, epochs, early stopping, with reasons | README, models/model_config.json |
| 9 | Implementation & testing | 6 | Full pipeline + Streamlit app; unit tests (features, split leakage, stability logic) | src/*, app.py, tests/test_core.py |
| 10 | Evaluation metrics selected | 1 | Accuracy, precision, recall, F1 (macro/weighted/per class), confusion matrix, curves | src/evaluate.py |
| 11 | Evaluation implemented & interpreted | 2 | Test-set metrics, plots, auto `interpretation.md` (weak class, top confusion, over-fitting gap) | reports/ |
| 12 | Model improvement from evaluation | 1 | `train --tune` compares 4 configs on validation; augmentation, class weights, LR schedule; confidence threshold chosen from `confidence_threshold_analysis.csv` | reports/tuning_results.csv |
| 13 | Saved model/config for reproduction | 1 | `.keras` model, `scaler.json`, `label_map.json`, `model_config.json`, history, fixed seed 42 | models/ |
| 14 | Web app demonstration / deployment | 1 | Streamlit + WebRTC (local and Streamlit Cloud) | app.py, README "Deploy" |
| 15 | Responsible use | (task 8) | Privacy (landmarks only), bias, cultural ambiguity, no safety-critical use | README |
| 16 | Limitations & risk reduction | (task 8) | Few subjects/lighting, single hand, misclassification → threshold + 2 s hold + smoothing + more diverse data | README |

## 2. 30-mark breakdown
| Outcome | Criterion | Marks |
|---|---|---|
| Data Preprocessing (30 % = 9) | Environment/functionalities 1.5 · environment configured 1.5 · data acquired 3 · preprocessed 3 | 9 |
| Deep Learning (50 % = 15) | Data features 3 · model features 3 · approach justified 1.5 · parameters explained 1.5 · implemented & tested 6 | 15 |
| Evaluation (20 % = 6) | Metrics 1 · evaluation & interpretation 2 · improvement 1 · saved/reproducible 1 · web demo 1 | 6 |
| **Total** | | **30** |

## 3. EVERYTHING USED IN THIS PROJECT
**Dataset/source:** custom webcam dataset recorded with `collect_data.py` (your own hands; no external licence). **Samples:** ≥ 2 400 raw (8 × 300) – real count in `reports/dataset_summary.json`; ×5 after train augmentation. **Gesture classes (8):** open_palm, fist, peace, thumbs_up, thumbs_down, pointing, rock, ok_sign. **Flower classes (8):** Rose, Sunflower, Tulip, Lily, Orchid, Daisy, Lavender, Lotus.

| Technology | What it is → Why chosen → Where used |
|---|---|
| Python 3.10–3.12 | Language → required by module, rich AI ecosystem → all files |
| MediaPipe Hands 0.10.14 (legacy `solutions`) | Pretrained hand-landmark detector (21 3-D points) → fast, robust, no pixel model needed → `hand_detector.py`, collection, app |
| Feature engineering (normalisation, distances, angles, orientation) | Hand-crafted numeric descriptors → invariant to position/size/handedness → `features.py` |
| TensorFlow/Keras FNN | Fully-connected network → ideal for tabular landmark features, real-time → `train.py`, `inference.py` |
| Architecture | 101 → 128 → 64 → 32 → 8; ReLU; BatchNorm + Dropout 0.3 after each hidden layer; L2 1e-4; softmax out (~30 k parameters) |
| Training | Adam 1e-3, sparse categorical cross-entropy, batch 32, ≤200 epochs, EarlyStopping(20) + ReduceLROnPlateau, balanced class weights, seed 42 |
| Evaluation | scikit-learn metrics: accuracy, precision, recall, F1, confusion matrix, confidence-threshold analysis → `evaluate.py` |
| NumPy / pandas | Arrays & CSV handling → preprocessing |
| scikit-learn | Metrics, class weights → `train.py`, `evaluate.py` |
| Matplotlib | Curves, confusion matrix, class distribution → `reports/` |
| OpenCV | Camera capture, drawing → collection, overlay |
| Pillow | Procedural flower drawing → `flowers.py` |
| Streamlit + streamlit-webrtc + av | Web UI and browser webcam streaming → `app.py` |
| Saved formats | `.keras` (model), JSON (scaler, labels, config), CSV/NPZ (data) |

## 4. Final verification (performed)
Executed in the build sandbox: syntax check of all modules ✔ · 10 unit tests (feature shape 101, translation/scale/mirror invariance, orientation kept, block-leakage-free split, 2 s confirmation, reset on change, noise tolerance, hand-loss cancel) ✔ · preprocessing run on a synthetic CSV → shapes (8400,101)/(360,101)/(360,101) ✔ · all 8 flowers rendered ✔.
**NOT executable in the sandbox** (no TensorFlow/Streamlit/webcam/network): `train.py`, `evaluate.py`, `inference.py`, `app.py`. They were reviewed by hand for shape/path/name consistency (feature dim 101 checked in scaler, model and inference; 8 classes checked in model and label map) but you must run them once locally.
Known risks: mediapipe must stay at 0.10.14 (newer wheels removed `mp.solutions`); Streamlit Cloud may need a TURN server; accuracy depends on your recorded data.
