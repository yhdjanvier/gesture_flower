"""Train the FNN (MLP) gesture classifier on the engineered landmark features.

Run:  python -m src.train            (baseline architecture)
      python -m src.train --tune     (compare all CANDIDATES on the VALIDATION set, keep the best)

Architecture (baseline): Input(101) -> [Dense -> BatchNorm -> Dropout] x3 (128,64,32, ReLU)
                         -> Dense(8, softmax).  Adam, sparse categorical cross-entropy.
"""
from __future__ import annotations

import argparse
import json

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

import config
from src.features import FEATURE_DIM, FEATURE_NAMES


def build_model(input_dim: int, n_classes: int, units, dropout: float, l2: float, lr: float):
    reg = tf.keras.regularizers.l2(l2)
    inputs = tf.keras.Input(shape=(input_dim,), name="landmark_features")
    x = inputs
    for i, u in enumerate(units, start=1):
        x = tf.keras.layers.Dense(u, activation="relu", kernel_initializer="he_normal",
                                  kernel_regularizer=reg, name=f"dense_{i}")(x)
        x = tf.keras.layers.BatchNormalization(name=f"bn_{i}")(x)
        x = tf.keras.layers.Dropout(dropout, name=f"dropout_{i}")(x)
    outputs = tf.keras.layers.Dense(n_classes, activation="softmax", name="gesture_probs")(x)
    model = tf.keras.Model(inputs, outputs, name="gesture_fnn")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def train_one(name: str, cfg: dict, data) -> tuple:
    tf.keras.utils.set_random_seed(config.SEED)
    X_tr, y_tr, X_va, y_va = data
    n_classes = len(config.GESTURES)
    model = build_model(X_tr.shape[1], n_classes, cfg["units"], cfg["dropout"], cfg["l2"], cfg["lr"])
    cw = compute_class_weight("balanced", classes=np.arange(n_classes), y=y_tr)
    t = config.TRAIN
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=t["early_stopping_patience"],
                                         restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=t["reduce_lr_factor"],
                                             patience=t["reduce_lr_patience"], min_lr=t["min_lr"], verbose=1),
    ]
    print(f"\n=== Training '{name}': {cfg} ===")
    hist = model.fit(X_tr, y_tr, validation_data=(X_va, y_va), epochs=t["epochs"],
                     batch_size=t["batch_size"], class_weight=dict(enumerate(cw)),
                     callbacks=callbacks, verbose=2)
    val_loss, val_acc = model.evaluate(X_va, y_va, verbose=0)
    return model, hist.history, float(val_loss), float(val_acc)


def plot_history(history: dict) -> None:
    for key, title, fname in (("accuracy", "Training vs validation accuracy", "training_accuracy.png"),
                              ("loss", "Training vs validation loss", "training_loss.png")):
        plt.figure(figsize=(7, 4.5))
        plt.plot(history[key], label=f"train {key}")
        plt.plot(history[f"val_{key}"], label=f"validation {key}")
        plt.title(title); plt.xlabel("epoch"); plt.ylabel(key); plt.grid(alpha=0.3); plt.legend()
        plt.tight_layout(); plt.savefig(config.REPORT_DIR / fname, dpi=150); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", action="store_true", help="compare all candidate architectures")
    args = ap.parse_args()

    if not config.SPLIT_PATH.exists():
        raise SystemExit("Run `python -m src.preprocess` first (no dataset_splits.npz found).")
    d = np.load(config.SPLIT_PATH)
    if d["X_train"].shape[1] != FEATURE_DIM:
        raise SystemExit("Feature dimension changed - re-run `python -m src.preprocess`.")
    data = (d["X_train"], d["y_train"], d["X_val"], d["y_val"])
    config.MODEL_DIR.mkdir(exist_ok=True); config.REPORT_DIR.mkdir(exist_ok=True)

    names = list(config.CANDIDATES) if args.tune else [config.DEFAULT_CANDIDATE]
    results, best = [], None
    for name in names:
        model, hist, vl, va = train_one(name, config.CANDIDATES[name], data)
        results.append({"config": name, **config.CANDIDATES[name], "epochs_run": len(hist["loss"]),
                        "val_loss": round(vl, 4), "val_accuracy": round(va, 4),
                        "params": int(model.count_params())})
        if best is None or (va, -vl) > (best["va"], -best["vl"]):   # accuracy first, loss breaks ties
            best = {"name": name, "model": model, "hist": hist, "vl": vl, "va": va}
    pd.DataFrame(results).to_csv(config.REPORT_DIR / "tuning_results.csv", index=False)
    print("\n", pd.DataFrame(results).to_string(index=False))
    print(f"\n[selected] {best['name']} (val_acc={best['va']:.4f}, val_loss={best['vl']:.4f})")

    best["model"].save(config.MODEL_PATH)
    config.HISTORY_PATH.write_text(json.dumps(best["hist"]))
    plot_history(best["hist"])
    (config.REPORT_DIR / "model_summary.txt").write_text(
        _summary_text(best["model"]), encoding="utf-8")
    config.MODEL_CONFIG_PATH.write_text(json.dumps({
        "model_file": config.MODEL_PATH.name, "input_dim": FEATURE_DIM, "feature_names": list(FEATURE_NAMES),
        "classes": config.GESTURES, "selected_config": best["name"],
        "hyperparameters": config.CANDIDATES[best["name"]], "training": config.TRAIN,
        "loss": "sparse_categorical_crossentropy", "optimizer": "Adam",
        "augmentation": config.AUG, "split_ratios": config.SPLIT_RATIOS, "seed": config.SEED,
        "app": config.APP, "flower_map": {g: f["name"] for g, f in config.FLOWERS.items()},
        "tensorflow_version": tf.__version__,
        "validation": {"loss": best["vl"], "accuracy": best["va"]}}, indent=2))
    print(f"[ok] saved {config.MODEL_PATH}. Next: python -m src.evaluate")


def _summary_text(model) -> str:
    lines = []
    model.summary(print_fn=lines.append)
    return "\n".join(lines)


if __name__ == "__main__":
    main()
