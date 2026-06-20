from __future__ import annotations
"""Probe a vision-based contact cue: a fingertip pad blanches (brightens, loses
saturation) when it presses the desk. Compute a per-frame "blanch" score at a
finger's tip from raw_video and check whether it peaks at the labelled onsets.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import cv2
from scipy.signal import find_peaks

S = Path("data/sessions/shuxingxing01")
FINGER = int(sys.argv[1]) if len(sys.argv) > 1 else 8  # default index
NOTE = {4: "C4", 8: "D4", 12: "E4", 16: "F4", 20: "G4"}[FINGER]
XOFF = {4: -8, 8: -8}


def main():
    rows = {r["frame_index"]: r for r in (json.loads(l) for l in open(S / "frames.jsonl", encoding="utf-8") if l.strip())}
    ann = [int(r["frame_index"]) for r in csv.DictReader(open(S / "annotations.csv", encoding="utf-8"))
           if r.get("note") == NOTE and r.get("frame_index")]
    cap = cv2.VideoCapture(str(S / "raw_video.avi"))
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    blanch = np.full(nframes + 1, np.nan)
    ydown = np.full(nframes + 1, np.nan)
    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        r = rows.get(fi)
        if r and r.get("hands"):
            lm = max(r["hands"], key=lambda h: len(h.get("landmarks", []))).get("landmarks", [])
            if len(lm) > FINGER:
                x, y = int(lm[FINGER][0]), int(lm[FINGER][1])
                rad = 6
                x1, y1 = max(0, x - rad), max(0, y - rad)
                x2, y2 = min(frame.shape[1], x + rad + 1), min(frame.shape[0], y + rad + 1)
                patch = frame[y1:y2, x1:x2]
                if patch.size:
                    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)
                    v = hsv[:, 2].mean()
                    s = hsv[:, 1].mean()
                    blanch[fi] = v - s   # pale pad => high V, low S
                    ydown[fi] = y
        fi += 1
    cap.release()

    good = np.isfinite(blanch)
    idx = np.arange(len(blanch))
    blanch[~good] = np.interp(idx[~good], idx[good], blanch[good])
    # detrend with a long moving average (lighting / skin baseline)
    k = 21
    base = np.convolve(blanch, np.ones(k) / k, mode="same")
    sig = blanch - base

    print(f"finger {FINGER} ({NOTE}) onsets: {ann}")
    print(f"blanch global mean={np.nanmean(blanch):.1f}  detrended std={np.std(sig):.2f}")
    print(f"\n{'onset':>6} | {'sig@onset':>9} {'localmax?':>9} {'peak_in_+-3':>11}")
    peaks, _ = find_peaks(sig, prominence=np.std(sig) * 0.8, distance=6)
    peakset = set(peaks.tolist())
    hit = 0
    for of in ann:
        win = range(max(0, of - 3), min(len(sig), of + 4))
        local = max(win, key=lambda i: sig[i])
        is_peak = any(abs(of - p) <= 3 for p in peaks)
        hit += is_peak
        print(f"{of:6d} | {sig[of]:9.2f} {('Y' if local==of else f'@{local}'):>9} {('YES' if is_peak else 'no'):>11}")
    print(f"\nonsets with a blanch-peak within +-3 frames: {hit}/{len(ann)}")
    print(f"total blanch peaks in clip: {len(peaks)}  (extras ~ {len(peaks)-hit})")


if __name__ == "__main__":
    main()
