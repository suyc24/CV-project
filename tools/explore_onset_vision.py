from __future__ import annotations
"""Architecture test: motion peaks (candidate taps) filtered by a vision contact
score (fingertip pad blanch). Motion gives precise timing; vision rejects passive
coupling (a finger that dips with the hand but does not actually press has no
blanch). Measures match/extra vs the 42 labelled onsets on shuxingxing01.
"""
import csv
import json
from pathlib import Path

import numpy as np
import cv2
from scipy.signal import find_peaks

S = Path("data/sessions/shuxingxing01")
FINGERS = (4, 8, 12, 16, 20)
BASE = {4: 2, 8: 5, 12: 9, 16: 13, 20: 17}
XOFF = {4: -8, 8: -8}


def build():
    rows = {r["frame_index"]: r for r in (json.loads(l) for l in open(S / "frames.jsonl", encoding="utf-8") if l.strip())}
    nf = int(cv2.VideoCapture(str(S / "raw_video.avi")).get(cv2.CAP_PROP_FRAME_COUNT))
    X = {f: np.full(nf, np.nan) for f in FINGERS}
    Y = {f: np.full(nf, np.nan) for f in FINGERS}
    RY = {f: np.full(nf, np.nan) for f in FINGERS}
    BL = {f: np.full(nf, np.nan) for f in FINGERS}   # blanch (V-S, skin region)
    zbf = {}
    cap = cv2.VideoCapture(str(S / "raw_video.avi"))
    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        r = rows.get(fi)
        if r:
            zbf[fi] = r.get("zones", [])
            hs = r.get("hands", [])
            if hs:
                lm = max(hs, key=lambda h: len(h.get("landmarks", []))).get("landmarks", [])
                if len(lm) >= 21:
                    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                    for f in FINGERS:
                        x, y = int(lm[f][0]), int(lm[f][1])
                        X[f][fi], Y[f][fi] = x, y
                        RY[f][fi] = y - lm[BASE[f]][1]
                        rad = 7
                        patch = hsv[max(0, y - rad):y + rad + 1, max(0, x - rad):x + rad + 1].reshape(-1, 3).astype(np.float32)
                        if patch.size:
                            # skin-ish: not too dark, reddish/pink hue
                            h_, s_, v_ = patch[:, 0], patch[:, 1], patch[:, 2]
                            skin = ((h_ < 25) | (h_ > 160)) & (v_ > 90)
                            sel = patch[skin] if skin.sum() >= 8 else patch
                            BL[f][fi] = sel[:, 2].mean() - sel[:, 1].mean()
        fi += 1
    cap.release()
    idx = np.arange(nf)
    for f in FINGERS:
        for arr in (X[f], Y[f], RY[f], BL[f]):
            g = np.isfinite(arr)
            if g.sum() > 2:
                arr[~g] = np.interp(idx[~g], idx[g], arr[g])
        # detrend blanch (per-finger lighting/skin baseline) -> contact spike
        base = np.convolve(BL[f], np.ones(25) / 25, mode="same")
        BL[f] = BL[f] - base
    return X, Y, RY, BL, zbf, nf


def zone_at(zones, x, y):
    for z in zones:
        if z.get("kind") != "piano":
            continue
        poly = z.get("polygon")
        if poly and cv2.pointPolygonTest(np.array(poly, np.int32), (float(x), float(y)), False) >= 0:
            return z["label"]
    return None


def detect(X, Y, RY, BL, zbf, prom, dist, blanch_thr):
    dets = []
    for f in FINGERS:
        peaks, _ = find_peaks(RY[f], prominence=prom, distance=dist)
        for p in peaks:
            if blanch_thr is not None and BL[f][p] < blanch_thr:
                continue
            note = zone_at(zbf.get(int(p), []), X[f][p] + XOFF.get(f, 0), Y[f][p])
            if note:
                dets.append((int(p), note, f))
    return dets


def score(dets, ann, tol=4):
    used = set()
    m = 0
    for af, an in ann:
        best = None
        for i, (df, dn, _) in enumerate(dets):
            if i in used or dn != an or abs(df - af) > tol:
                continue
            if best is None or abs(df - af) < abs(dets[best][0] - af):
                best = i
        if best is not None:
            used.add(best)
            m += 1
    return m, len(ann) - m, len(dets) - len(used)


def main():
    ann = [(int(r["frame_index"]), r["note"]) for r in csv.DictReader(open(S / "annotations.csv", encoding="utf-8")) if r.get("frame_index")]
    X, Y, RY, BL, zbf, nf = build()
    print(f"annotations: {len(ann)}")
    print(f"{'prom':>5} {'blanchThr':>9} | {'match':>5} {'miss':>4} {'extra':>5}")
    for prom in (12, 16, 20):
        for thr in (None, -2, 0, 2, 4, 6):
            dets = detect(X, Y, RY, BL, zbf, prom, 6, thr)
            m, mi, ex = score(dets, ann)
            print(f"{prom:5d} {str(thr):>9} | {m:5d} {mi:4d} {ex:5d}")
        print()


if __name__ == "__main__":
    main()
