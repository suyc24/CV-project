from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hand_tracker import OneEuroPointFilter


def mean_abs_delta(values: list[float]) -> float:
    return sum(abs(b - a) for a, b in zip(values, values[1:])) / max(1, len(values) - 1)


def test_one_euro_filter_reduces_static_jitter():
    filt = OneEuroPointFilter(min_cutoff=1.0, beta=0.0, d_cutoff=1.0)
    raw_x = [100, 103, 98, 102, 99, 101, 100, 102, 99, 101]
    filtered_x = [filt.filter((x, 50.0, 0.0), i / 30.0)[0] for i, x in enumerate(raw_x)]

    assert mean_abs_delta(filtered_x[3:]) < mean_abs_delta([float(x) for x in raw_x[3:]])


def test_one_euro_filter_follows_fast_motion():
    filt = OneEuroPointFilter(min_cutoff=1.0, beta=0.08, d_cutoff=1.0)
    filtered = []
    for index, x in enumerate([100, 100, 100, 145, 180, 210, 230]):
        filtered.append(filt.filter((float(x), 50.0, 0.0), index / 30.0)[0])

    assert filtered[-1] > 215.0
    assert filtered[-1] <= 230.0


if __name__ == "__main__":
    test_one_euro_filter_reduces_static_jitter()
    test_one_euro_filter_follows_fast_motion()
    print("one euro filter tests passed")
