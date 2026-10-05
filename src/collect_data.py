"""Guided webcam data collection -> data/raw/gesture_landmarks.csv

Usage:  python -m src.collect_data [--samples 300] [--camera 0] [--gestures fist peace]
Keys :  SPACE = start recording the shown gesture | S = skip gesture | Q = quit
Tips : record every gesture in >= 2 separate runs (new session id each run), vary distance,
       angle, lighting, background, and use BOTH hands. Slowly move the hand while recording.
"""
from __future__ import annotations

import argparse
import sys
import time

import cv2
import numpy as np
import pandas as pd

import config
from src.features import RAW_COLUMNS
from src.hand_detector import HandDetector

WIN = "Gesture data collection"


def overlay(img, lines, color=(255, 255, 255), bar=None):
    cv2.rectangle(img, (0, 0), (img.shape[1], 30 + 28 * len(lines)), (20, 20, 20), -1)
    for i, t in enumerate(lines):
        cv2.putText(img, t, (12, 28 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    if bar is not None:
        h, w = img.shape[:2]
        cv2.rectangle(img, (0, h - 18), (int(w * bar), h), (80, 220, 120), -1)


def read(cap, detector):
    ok, frame = cap.read()
    if not ok:
        return None, None
    frame = cv2.flip(frame, 1)                       # mirror = selfie view (handedness is correct)
    res = detector.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if res:
        detector.draw(frame, res)
    return frame, res


def save_rows(rows):
    if not rows:
        return
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=RAW_COLUMNS)
    exists = config.RAW_CSV.exists()
    if exists:
        header = pd.read_csv(config.RAW_CSV, nrows=0).columns.tolist()
        if header != list(RAW_COLUMNS):
            sys.exit("Existing CSV has different columns; move/delete it first.")
    df.to_csv(config.RAW_CSV, mode="a", header=not exists, index=False)
    print(f"[saved] {len(rows)} rows -> {config.RAW_CSV}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=config.DEFAULT_SAMPLES_PER_GESTURE)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--gestures", nargs="*", default=config.GESTURES, choices=config.GESTURES)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        sys.exit(f"Cannot open camera index {args.camera}. Close other apps using it or try --camera 1.")
    detector = HandDetector()
    session = time.strftime("%Y%m%d_%H%M%S")
    quit_all = False
    try:
        for gesture in args.gestures:
            info = config.GESTURE_INFO[gesture]
            while not quit_all:                                   # --- waiting screen
                frame, res = read(cap, detector)
                if frame is None:
                    sys.exit("Camera returned no frame.")
                overlay(frame, [f"NEXT: {info['display']}  ({gesture})", info["how_to"],
                                "SPACE=start  S=skip  Q=quit"], (0, 255, 255))
                cv2.imshow(WIN, frame)
                k = cv2.waitKey(1) & 0xFF
                if k == ord(" "):
                    break
                if k == ord("s"):
                    gesture = None
                    break
                if k == ord("q"):
                    quit_all = True
            if quit_all:
                break
            if gesture is None:
                continue
            t0 = time.time()                                       # --- 3 s countdown
            while time.time() - t0 < 3:
                frame, _ = read(cap, detector)
                if frame is None:
                    continue
                overlay(frame, [f"Get ready: {info['display']}", f"Starting in {3 - int(time.time() - t0)}"],
                        (0, 200, 255))
                cv2.imshow(WIN, frame)
                cv2.waitKey(1)
            rows, last = [], 0.0                                   # --- recording
            while len(rows) < args.samples:
                frame, res = read(cap, detector)
                if frame is None:
                    continue
                now = time.time()
                if res is not None and now - last >= config.COLLECT_INTERVAL_S:
                    rows.append([gesture, "Left" if res.is_left else "Right", session,
                                 len(rows) // config.BLOCK_SIZE, *res.raw.tolist()])
                    last = now
                status = "REC" if res is not None else "NO HAND - show your hand"
                overlay(frame, [f"{status}: {info['display']}  {len(rows)}/{args.samples}",
                                "Slowly move / rotate / change distance"],
                        (0, 0, 255) if res is not None else (0, 165, 255), bar=len(rows) / args.samples)
                cv2.imshow(WIN, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    quit_all = True
                    break
            save_rows(rows)
    finally:
        cap.release()
        detector.close()
        cv2.destroyAllWindows()
    print("Done. Next: python -m src.preprocess")


if __name__ == "__main__":
    main()
