"""Central configuration for the AI Gesture-to-Flower system.

Single source of truth for: paths, gesture classes, gesture->flower mapping,
dataset split settings, augmentation, FNN hyper-parameters and app behaviour.
Every other module imports from here so nothing can drift out of sync.
"""
from pathlib import Path

# ----------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

RAW_CSV = RAW_DIR / "gesture_landmarks.csv"
SPLIT_PATH = PROCESSED_DIR / "dataset_splits.npz"
MODEL_PATH = MODEL_DIR / "gesture_fnn.keras"
SCALER_PATH = MODEL_DIR / "scaler.json"
LABELS_PATH = MODEL_DIR / "label_map.json"
MODEL_CONFIG_PATH = MODEL_DIR / "model_config.json"
HISTORY_PATH = MODEL_DIR / "training_history.json"

SEED = 42

# --------------------------------------------------------------- gestures
# ORDER MATTERS: the index of a gesture in this list is its class id.
GESTURES = [
    "open_palm", "fist", "peace", "thumbs_up",
    "thumbs_down", "pointing", "rock", "ok_sign",
]

GESTURE_INFO = {
    "open_palm":   {"display": "Open Palm",   "how_to": "All five fingers spread, palm facing the camera."},
    "fist":        {"display": "Fist",        "how_to": "Close all fingers into a tight fist."},
    "peace":       {"display": "Peace / V",   "how_to": "Index and middle fingers up in a V, others curled."},
    "thumbs_up":   {"display": "Thumbs Up",   "how_to": "Fist with the thumb pointing straight up."},
    "thumbs_down": {"display": "Thumbs Down", "how_to": "Fist with the thumb pointing straight down."},
    "pointing":    {"display": "Pointing",    "how_to": "Only the index finger extended, others curled."},
    "rock":        {"display": "Rock Sign",   "how_to": "Index and pinky up, middle and ring curled."},
    "ok_sign":     {"display": "OK Sign",     "how_to": "Thumb and index tips touching in a circle, other fingers up."},
}

# ------------------------------------------------------ gesture -> flower
FLOWERS = {
    "open_palm": {
        "key": "rose", "name": "Rose", "latin": "Rosa",
        "description": "Layered velvet petals - the classic flower of love and open-hearted welcome.",
        "meaning": "Love & welcome",
        "gradient": ((255, 228, 232), (255, 182, 193)),
    },
    "fist": {
        "key": "sunflower", "name": "Sunflower", "latin": "Helianthus annuus",
        "description": "A bold golden bloom that turns to follow the sun, symbolising strength and loyalty.",
        "meaning": "Strength & loyalty",
        "gradient": ((255, 247, 205), (135, 206, 235)),
    },
    "peace": {
        "key": "tulip", "name": "Tulip", "latin": "Tulipa",
        "description": "A smooth cup-shaped spring flower that stands for perfect love and renewal.",
        "meaning": "Renewal & perfect love",
        "gradient": ((255, 240, 245), (216, 191, 216)),
    },
    "thumbs_up": {
        "key": "lily", "name": "Lily", "latin": "Lilium",
        "description": "Elegant star-shaped petals with graceful stamens, a symbol of purity and renewal.",
        "meaning": "Purity & celebration",
        "gradient": ((240, 255, 245), (176, 224, 230)),
    },
    "thumbs_down": {
        "key": "orchid", "name": "Orchid", "latin": "Phalaenopsis",
        "description": "An exotic, refined bloom with a velvety lip - rare beauty and quiet strength.",
        "meaning": "Rare beauty & resilience",
        "gradient": ((245, 232, 255), (200, 170, 230)),
    },
    "pointing": {
        "key": "daisy", "name": "Daisy", "latin": "Bellis perennis",
        "description": "A cheerful meadow flower with a golden eye - innocence and simple joy.",
        "meaning": "Innocence & joy",
        "gradient": ((235, 255, 235), (173, 216, 230)),
    },
    "rock": {
        "key": "lavender", "name": "Lavender", "latin": "Lavandula",
        "description": "Fragrant violet spikes known for calm, serenity and devotion.",
        "meaning": "Calm & devotion",
        "gradient": ((240, 235, 255), (221, 204, 245)),
    },
    "ok_sign": {
        "key": "lotus", "name": "Lotus", "latin": "Nelumbo nucifera",
        "description": "Rising pure from still water, the lotus stands for enlightenment and rebirth.",
        "meaning": "Enlightenment & rebirth",
        "gradient": ((255, 240, 246), (150, 214, 220)),
    },
}

# ------------------------------------------------------ dataset / preprocessing
SPLIT_RATIOS = (0.70, 0.15, 0.15)   # train / validation / test
BLOCK_SIZE = 15                     # consecutive frames that form one split "block"
COLLECT_INTERVAL_S = 0.08           # minimum time between two recorded frames
DEFAULT_SAMPLES_PER_GESTURE = 300

AUG = {                             # applied to the TRAINING split only
    "copies": 4,                    # augmented copies per training sample
    "max_rotation_deg": 15.0,       # in-plane rotation range (+/-)
    "noise_std": 0.02,              # Gaussian landmark jitter, fraction of hand size
}

# ------------------------------------------------------------- FNN training
TRAIN = {
    "batch_size": 32,
    "epochs": 200,                  # upper bound; early stopping ends sooner
    "early_stopping_patience": 20,
    "reduce_lr_patience": 7,
    "reduce_lr_factor": 0.5,
    "min_lr": 1e-5,
}

# Candidate architectures compared on the VALIDATION set by `train.py --tune`
CANDIDATES = {
    "baseline":     {"units": [128, 64, 32],  "dropout": 0.30, "l2": 1e-4, "lr": 1e-3},
    "wide_dropout": {"units": [256, 128, 64], "dropout": 0.40, "l2": 1e-4, "lr": 1e-3},
    "compact":      {"units": [64, 32],       "dropout": 0.20, "l2": 1e-4, "lr": 1e-3},
    "low_lr":       {"units": [128, 64, 32],  "dropout": 0.30, "l2": 1e-4, "lr": 5e-4},
}
DEFAULT_CANDIDATE = "baseline"

# -------------------------------------------------------------- application
APP = {
    "hold_seconds": 2.0,            # gesture must stay stable this long
    "min_confidence": 0.80,         # smoothed softmax prob needed to count as "stable"
    "grace_seconds": 0.35,          # tolerated dropout / low-confidence gap
    "smooth_window": 5,             # frames of probability averaging
}

# ------------------------------------------------------------ MediaPipe
MP_HANDS = {
    "max_num_hands": 1,
    "model_complexity": 1,
    "min_detection_confidence": 0.6,
    "min_tracking_confidence": 0.5,
}
