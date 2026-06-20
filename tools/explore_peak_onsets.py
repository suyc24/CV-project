from __future__ import annotations
"""Prototype: detect Twinkle note onsets as prominent local maxima in each
finger's relative-y trajectory (the bottom of a tap), instead of the online
velocity-threshold state machine. Measures the ceiling against the labelled
shuxingxing01 onsets to see if a trajectory/peak approach can reach ~41/42.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import cv2
from scipy.signal import find_peaks

S = Path("data/sessions/shuxingxing01")
FINGERS = (4, 8, 12, 16, 20)
BASE = {4: 2, 8: 5, 12: 9, 16: 13, 20: 17}
XOFF = {4: -8, 8: -8}  # reversed-keyboard calibration for thumb/index


def load():
    rows = [json.loads(l) for l in open(S / "frames.jsonl", encoding="utf-8") if l.strip()]
    fmax = max(r["frame_index"] for r in rows)
    # per finger: arrays of x, y, relY over frame index (nan where missing)
    X = {f: np.full(fmax + 1, np.nan) for f in FINGERS}
    Y = {f: np.full(fmax + 1, np.nan) for f in FINGERS}
    RY = {f: np.full(fmax + 1, np.nan) for f in FINGERS}
    zones_by_frame = {}
    for r in rows:
        fi = r["frame_index"]
        zones_by_frame[fi] = r.get("zones", [])
        hs = r.get("hands", [])
        if not hs:
            continue
        lm = max(hs, key=lambda h: len(h.get("landmarks", []))).get("landmarks", [])
        if len(lm) < 21:
            continue
        for f in FINGERS:
            x, y = lm[f][0], lm[f][1]
            anchor = lm[BASE[f]][1]
            X[f][fi] = x
            Y[f][fi] = y
            RY[f][fi] = y - anchor
    # interpolate small gaps
    for f in FINGERS:
        for arr in (X[f], Y[f], RY[f]):
            idx = np.arange(len(arr))
            good = np.isfinite(arr)
            if good.sum() > 2:
                arr[~good] = np.interp(idx[~good], idx[good], arr[good])
    return X, Y, RY, zones_by_frame, fmax


def zone_at(zones, x, y):
    for z in zones:
        if z.get("kind") != "piano":
            continue
        poly = z.get("polygon")
        if poly:
            if cv2.pointPolygonTest(np.array(poly, np.int32), (float(x), float(y)), False) >= 0:
                return z["label"]
        elif z["x1"] <= x <= z["x2"] and z["y1"] <= y <= z["y2"]:
            return z["label"]
    return None


def detect(X, Y, RY, zones_by_frame, fmax, prominence, distance, min_rely):
    dets = []  # (frame, note, finger)
    for f in FINGERS:
        ry = RY[f].copy()
        ry[~np.isfinite(ry)] = np.nanmin(ry)
        peaks, props = find_peaks(ry, prominence=prominence, distance=distance)
        for p in peaks:
            if ry[p] < min_rely:
                continue
            x = X[f][p] + XOFF.get(f, 0)
            note = zone_at(zones_by_frame.get(int(p), []), x, Y[f][p])
            if note:
                dets.append((int(p), note, f))
    return dets


def score(dets, ann, tol=4):
    used = set()
    match = 0
    for (af, an) in ann:
        best = None
        for i, (df, dn, _) in enumerate(dets):
            if i in used or dn != an:
                continue
            if abs(df - af) <= tol:
                if best is None or abs(df - af) < abs(dets[best][0] - af):
                    best = i
        if best is not None:
            used.add(best)
            match += 1
    extra = len(dets) - len(used)
    return match, len(ann) - match, extra


def main():
    ann = [(int(r["frame_index"]), r["note"]) for r in csv.DictReader(open(S / "annotations.csv", encoding="utf-8")) if r.get("frame_index")]
    X, Y, RY, zbf, fmax = load()
    print(f"annotations: {len(ann)}  frames: {fmax+1}")
    print(f"{'prom':>5} {'dist':>4} {'minRY':>6} | {'match':>5} {'miss':>4} {'extra':>5}")
    best = None
    for prominence in (8, 10, 12, 15, 18, 22, 26):
        for distance in (6, 8, 10):
            for min_rely in (60, 80, 100):
                dets = detect(X, Y, RY, zbf, fmax, prominence, distance, min_rely)
                m, mi, ex = score(dets, ann)
                key = (m, -ex)
                if best is None or key > best[0]:
                    best = (key, prominence, distance, min_rely, m, mi, ex)
    _, prom, dist, mry, m, mi, ex = best
    print(f"\nBEST: prom={prom} dist={dist} minRY={mry} -> match={m}/{len(ann)} miss={mi} extra={ex}")
    # print full grid around best prominence
    for prominence in (best[1],):
        for distance in (best[2],):
            for min_rely in (60, 80, 100, 120):
                dets = detect(X, Y, RY, zbf, fmax, prominence, distance, min_rely)
                m, mi, ex = score(dets, ann)
                print(f"{prominence:5d} {distance:4d} {min_rely:6d} | {m:5d} {mi:4d} {ex:5d}")


if __name__ == "__main__":
    main()
