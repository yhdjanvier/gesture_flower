"""Data preprocessing: raw landmark CSV -> cleaned, split, augmented, engineered, scaled arrays.

Pipeline
 1. load + validate the raw CSV (columns, NaN, unknown labels, duplicates, degenerate hands)
 2. block-wise stratified train/val/test split (70/15/15) - consecutive frames stay together
    so near-identical frames cannot leak between train and test
 3. augment TRAIN only (rotation + jitter on raw landmarks)
 4. feature engineering (src.features.extract_features)
 5. StandardScaler fitted on TRAIN only -> saved as JSON
 6. save arrays, label map, class-distribution table/plot, dataset summary

Run:  python -m src.preprocess
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

import config
from src.features import (FEATURE_DIM, FEATURE_NAMES, PALM_IDS, RAW_COLUMNS, augment_landmarks,
                          extract_features_batch)

SPLITS = ("train", "val", "test")


def load_and_clean(path=config.RAW_CSV) -> pd.DataFrame:
    if not path.exists():
        sys.exit(f"Raw dataset not found: {path}\nCollect it first:  python -m src.collect_data")
    df = pd.read_csv(path)
    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing columns: {missing[:5]}...")
    n0 = len(df)
    df = df.dropna(subset=list(RAW_COLUMNS)).copy()
    df = df[df["label"].isin(config.GESTURES)]
    df = df.drop_duplicates(subset=[c for c in RAW_COLUMNS if c not in ("session", "block")])
    coords = df[[c for c in RAW_COLUMNS[4:]]].to_numpy(dtype=float).reshape(len(df), 21, 3)
    wrist = coords[:, :1, :]
    palm = np.linalg.norm((coords - wrist)[:, list(PALM_IDS), :], axis=2).mean(axis=1)
    df = df[palm > 1e-3].reset_index(drop=True)
    print(f"[clean] {n0} rows -> {len(df)} rows after cleaning")
    absent = [g for g in config.GESTURES if g not in set(df["label"])]
    if absent:
        raise ValueError(f"No samples for classes: {absent}. Collect them with src.collect_data.")
    return df


def split_by_blocks(df: pd.DataFrame, ratios=config.SPLIT_RATIOS, seed=config.SEED) -> pd.DataFrame:
    """Assign each row a split. Whole blocks (label|session|block) go to ONE split."""
    rng = np.random.default_rng(seed)
    df = df.copy()
    df["group"] = df["label"] + "|" + df["session"].astype(str) + "|" + df["block"].astype(str)
    assign = {}
    for label, g in df.groupby("label"):
        groups = np.array(sorted(g["group"].unique()))
        if len(groups) < 3:
            raise ValueError(f"Class '{label}' has only {len(groups)} blocks; need >= 3 "
                             f"(collect more samples).")
        rng.shuffle(groups)
        n_test = max(1, int(round(len(groups) * ratios[2])))
        n_val = max(1, int(round(len(groups) * ratios[1])))
        for i, gname in enumerate(groups):
            assign[gname] = "test" if i < n_test else ("val" if i < n_test + n_val else "train")
    df["split"] = df["group"].map(assign)
    return df


def build_arrays(df: pd.DataFrame, seed=config.SEED):
    """Return {split: (X_features, y)} (unscaled) with augmentation on train only."""
    rng = np.random.default_rng(seed)
    label_to_id = {g: i for i, g in enumerate(config.GESTURES)}
    coord_cols = list(RAW_COLUMNS[4:])
    out = {}
    for split in SPLITS:
        part = df[df["split"] == split]
        raw = part[coord_cols].to_numpy(dtype=np.float64)
        left = (part["handedness"] == "Left").to_numpy()
        y = part["label"].map(label_to_id).to_numpy(dtype=np.int64)
        if split == "train":
            raws, lefts, ys = [raw], [left], [y]
            for _ in range(config.AUG["copies"]):
                raws.append(np.stack([augment_landmarks(r, rng, config.AUG["max_rotation_deg"],
                                                        config.AUG["noise_std"]) for r in raw]))
                lefts.append(left)
                ys.append(y)
            raw, left, y = np.concatenate(raws), np.concatenate(lefts), np.concatenate(ys)
        out[split] = (extract_features_batch(raw, left), y)
    return out


def fit_scaler(X_train: np.ndarray):
    mean = X_train.mean(axis=0)
    scale = X_train.std(axis=0)
    scale[scale < 1e-8] = 1.0
    return mean.astype(np.float32), scale.astype(np.float32)


def save_distribution(df: pd.DataFrame, arrays) -> pd.DataFrame:
    dist = pd.crosstab(df["label"], df["split"]).reindex(config.GESTURES)[list(SPLITS)]
    dist["total_raw"] = dist.sum(axis=1)
    dist.to_csv(config.REPORT_DIR / "class_distribution.csv")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ax = dist[list(SPLITS)].plot(kind="bar", figsize=(9, 4.5), edgecolor="black")
    ax.set_title("Class distribution per split (raw samples, before augmentation)")
    ax.set_ylabel("samples"); ax.set_xlabel("gesture")
    plt.xticks(rotation=30, ha="right"); plt.tight_layout()
    plt.savefig(config.REPORT_DIR / "class_distribution.png", dpi=150); plt.close()
    return dist


def main():
    for d in (config.PROCESSED_DIR, config.MODEL_DIR, config.REPORT_DIR):
        d.mkdir(parents=True, exist_ok=True)
    df = split_by_blocks(load_and_clean())
    arrays = build_arrays(df)
    mean, scale = fit_scaler(arrays["train"][0])
    scaled = {k: ((X - mean) / scale).astype(np.float32) for k, (X, _) in arrays.items()}

    np.savez_compressed(config.SPLIT_PATH,
                        X_train=scaled["train"], y_train=arrays["train"][1],
                        X_val=scaled["val"], y_val=arrays["val"][1],
                        X_test=scaled["test"], y_test=arrays["test"][1])
    config.SCALER_PATH.write_text(json.dumps({
        "type": "standard", "feature_dim": FEATURE_DIM, "feature_names": list(FEATURE_NAMES),
        "mean": mean.tolist(), "scale": scale.tolist(), "fitted_on": "train split (with augmentation)"}, indent=2))
    config.LABELS_PATH.write_text(json.dumps({str(i): g for i, g in enumerate(config.GESTURES)}, indent=2))

    dist = save_distribution(df, arrays)
    summary = {"raw_rows_after_cleaning": int(len(df)), "feature_dim": FEATURE_DIM,
               "split_ratios": config.SPLIT_RATIOS, "augmentation": config.AUG,
               "model_input_samples": {k: int(len(v[1])) for k, v in arrays.items()},
               "class_distribution": json.loads(dist.to_json())}
    (config.REPORT_DIR / "dataset_summary.json").write_text(json.dumps(summary, indent=2))
    print(dist.to_string())
    print({k: v.shape for k, v in scaled.items()})
    print(f"[ok] saved {config.SPLIT_PATH.name}, scaler.json, label_map.json")


if __name__ == "__main__":
    main()
