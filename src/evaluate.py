"""Evaluate the saved FNN on the held-out TEST (and validation) split.

Saves to reports/: metrics.json, classification_report.txt, confusion_matrix.png/.csv,
confidence_threshold_analysis.csv, training curves (re-drawn), interpretation.md
Run:  python -m src.evaluate
"""
from __future__ import annotations

import json
import time

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support)

import config


def plot_confusion(cm: np.ndarray, classes, path) -> None:
    cmn = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2))
    for ax, mat, title, fmt in ((axes[0], cm, "Confusion matrix (counts)", "d"),
                                (axes[1], cmn, "Confusion matrix (row-normalised)", ".2f")):
        ax.imshow(mat, cmap="Blues")
        ax.set_xticks(range(len(classes))); ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=45, ha="right"); ax.set_yticklabels(classes)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, format(mat[i, j], fmt), ha="center", va="center",
                        color="white" if mat[i, j] > mat.max() / 2 else "black", fontsize=8)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def main():
    if not config.MODEL_PATH.exists() or not config.SPLIT_PATH.exists():
        raise SystemExit("Need the trained model and dataset splits: run preprocess + train first.")
    model = tf.keras.models.load_model(config.MODEL_PATH, compile=False)
    d = np.load(config.SPLIT_PATH)
    classes = config.GESTURES
    metrics, report_txt = {}, []
    for split in ("val", "test"):
        X, y = d[f"X_{split}"], d[f"y_{split}"]
        proba = model.predict(X, batch_size=256, verbose=0)
        pred = proba.argmax(axis=1)
        p, r, f, _ = precision_recall_fscore_support(y, pred, average="macro", zero_division=0)
        pw, rw, fw, _ = precision_recall_fscore_support(y, pred, average="weighted", zero_division=0)
        metrics[split] = {"samples": int(len(y)), "accuracy": float(accuracy_score(y, pred)),
                          "precision_macro": float(p), "recall_macro": float(r), "f1_macro": float(f),
                          "precision_weighted": float(pw), "recall_weighted": float(rw), "f1_weighted": float(fw),
                          "per_class": classification_report(y, pred, target_names=classes,
                                                             output_dict=True, zero_division=0)}
        report_txt.append(f"===== {split.upper()} =====\n" + classification_report(
            y, pred, target_names=classes, digits=4, zero_division=0))
        if split == "test":
            y_t, pred_t, proba_t = y, pred, proba

    cm = confusion_matrix(y_t, pred_t, labels=range(len(classes)))
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(config.REPORT_DIR / "confusion_matrix.csv")
    plot_confusion(cm, classes, config.REPORT_DIR / "confusion_matrix.png")

    conf = proba_t.max(axis=1)                                    # confidence-threshold analysis
    rows = []
    for th in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        m = conf >= th
        rows.append({"threshold": th, "coverage": float(m.mean()),
                     "accuracy_on_accepted": float((pred_t[m] == y_t[m]).mean()) if m.any() else float("nan")})
    thr = pd.DataFrame(rows)
    thr.to_csv(config.REPORT_DIR / "confidence_threshold_analysis.csv", index=False)

    x1 = d["X_test"][:1]; model(x1, training=False)               # latency of one live prediction
    t0 = time.perf_counter()
    for _ in range(200):
        model(x1, training=False)
    metrics["latency_ms_per_prediction"] = (time.perf_counter() - t0) / 200 * 1000

    (config.REPORT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (config.REPORT_DIR / "classification_report.txt").write_text("\n\n".join(report_txt))

    if config.HISTORY_PATH.exists():                              # re-draw curves for the report
        from src.train import plot_history
        plot_history(json.loads(config.HISTORY_PATH.read_text()))

    # ---------------- automatic interpretation
    t = metrics["test"]; pc = t["per_class"]
    worst = min(classes, key=lambda c: pc[c]["f1-score"])
    off = cm.copy(); np.fill_diagonal(off, 0)
    i, j = np.unravel_index(off.argmax(), off.shape)
    row08 = thr[thr.threshold == 0.8].iloc[0]
    gap = metrics["val"]["accuracy"] - t["accuracy"]
    md = [
        "# Evaluation interpretation (auto-generated)", "",
        f"* Test accuracy **{t['accuracy']:.3f}**, macro precision {t['precision_macro']:.3f}, "
        f"macro recall {t['recall_macro']:.3f}, macro F1 {t['f1_macro']:.3f} on {t['samples']} unseen samples.",
        f"* Validation accuracy {metrics['val']['accuracy']:.3f} vs test {t['accuracy']:.3f} "
        f"(gap {gap:+.3f}): " + ("a small gap means no strong over-fitting to the validation set."
                                 if abs(gap) < 0.05 else "a larger gap suggests over-fitting or too little data."),
        f"* Weakest class: **{worst}** (F1 {pc[worst]['f1-score']:.3f}).",
        (f"* Most frequent confusion: **{classes[i]} -> {classes[j]}** ({off[i, j]} samples)."
         if off.max() > 0 else "* No confusions on the test set."),
        f"* With the app threshold 0.80, {row08.coverage:.1%} of test frames are accepted and "
        f"{row08.accuracy_on_accepted:.1%} of those are correct - this is why 0.80 + a 2 s hold is used "
        "to suppress wrong momentary predictions.",
        f"* Inference latency: {metrics['latency_ms_per_prediction']:.2f} ms per frame (real-time capable).", "",
        "Caveat: results come from data recorded by few people in few sessions; accuracy for new users, "
        "lighting or camera angles can be lower. Collect more varied data to improve generalisation.",
    ]
    (config.REPORT_DIR / "interpretation.md").write_text("\n".join(md), encoding="utf-8")
    print("\n\n".join(report_txt)); print("\n".join(md))


if __name__ == "__main__":
    main()
