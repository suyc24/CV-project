from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_VARIANTS = {
    "baseline": [],
    "refined": ["--fingertip-refinement"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run repeatable tracker benchmarks from existing AirDesk raw_video sessions. "
            "Each variant reprocesses the same video, evaluates tracking quality, and "
            "writes an aggregate pass/fail summary."
        )
    )
    parser.add_argument("session_dirs", nargs="+", help="Recorded sessions with raw_video.avi and frames.jsonl")
    parser.add_argument("--output-root", default="data/benchmarks/tracking", help="Benchmark output directory")
    parser.add_argument("--variants", default="baseline,refined", help="Comma-separated variants: baseline,refined")
    parser.add_argument("--skip-uncalibrated-depth", action="store_true")
    parser.add_argument("--limit-frames", type=int, default=0, help="Optional smoke-test frame limit")
    parser.add_argument("--target-p95-step-px", type=float, default=16.0)
    parser.add_argument("--target-large-jumps-per-1000", type=float, default=35.0)
    parser.add_argument("--target-zone-jitter-per-1000", type=float, default=20.0)
    parser.add_argument("--target-rapid-repeats", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected_variants = parse_variants(args.variants)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    results = []
    for session in [Path(path) for path in args.session_dirs]:
        for variant_name, variant_flags in selected_variants.items():
            variant_dir = output_root / session.name / variant_name
            variant_dir.mkdir(parents=True, exist_ok=True)
            reprocess_command = [
                sys.executable,
                "tools/reprocess_session_tracking.py",
                str(session),
                "--output-dir",
                str(variant_dir),
                "--variant",
                variant_name,
                *variant_flags,
            ]
            if args.limit_frames:
                reprocess_command.extend(["--limit-frames", str(args.limit_frames)])
            run(reprocess_command)

            evaluate_command = [
                sys.executable,
                "tools/evaluate_tracking_quality.py",
                str(variant_dir),
                "--output-prefix",
                "tracking_quality",
            ]
            if args.skip_uncalibrated_depth:
                evaluate_command.append("--skip-uncalibrated-depth")
            run(evaluate_command)

            quality = json.loads((variant_dir / "tracking_quality_summary.json").read_text(encoding="utf-8"))
            item = summarize_result(session, variant_name, quality, args)
            results.append(item)

    aggregate = {
        "targets": {
            "weighted_p95_step_px": args.target_p95_step_px,
            "large_jumps_per_1000": args.target_large_jumps_per_1000,
            "zone_jitter_per_1000": args.target_zone_jitter_per_1000,
            "rapid_same_finger_repeats": args.target_rapid_repeats,
        },
        "results": results,
        "all_passed": all(item["passed"] for item in results),
    }
    summary_path = output_root / "benchmark_summary.json"
    summary_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print(json.dumps(aggregate, indent=2))
    print(f"Wrote {summary_path}")
    return 0 if aggregate["all_passed"] else 2


def parse_variants(raw: str) -> dict[str, list[str]]:
    variants = {}
    for name in [piece.strip() for piece in raw.split(",") if piece.strip()]:
        if name not in DEFAULT_VARIANTS:
            raise SystemExit(f"Unknown variant {name!r}; choose from {', '.join(DEFAULT_VARIANTS)}")
        variants[name] = DEFAULT_VARIANTS[name]
    if not variants:
        raise SystemExit("No variants selected")
    return variants


def summarize_result(session: Path, variant: str, quality: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    overall = quality.get("overall", {}) or {}
    hits = quality.get("hits", {}) or {}
    samples = max(1, int(overall.get("samples", 0) or 0))
    large_jumps = int(overall.get("large_jump_count", 0) or 0)
    zone_jitter = int(overall.get("zone_jitter_switches", 0) or 0)
    large_jumps_per_1000 = large_jumps * 1000.0 / samples
    zone_jitter_per_1000 = zone_jitter * 1000.0 / samples
    checks = {
        "weighted_p95_step_px": float(overall.get("weighted_p95_step_px", 0.0) or 0.0) <= args.target_p95_step_px,
        "large_jumps_per_1000": large_jumps_per_1000 <= args.target_large_jumps_per_1000,
        "zone_jitter_per_1000": zone_jitter_per_1000 <= args.target_zone_jitter_per_1000,
        "rapid_same_finger_repeats": int(hits.get("rapid_same_finger_repeats", 0) or 0) <= args.target_rapid_repeats,
    }
    return {
        "session": str(session),
        "variant": variant,
        "frames": quality.get("frames", 0),
        "samples": samples,
        "weighted_p95_step_px": overall.get("weighted_p95_step_px", 0.0),
        "large_jumps_per_1000": large_jumps_per_1000,
        "zone_jitter_per_1000": zone_jitter_per_1000,
        "rapid_same_finger_repeats": hits.get("rapid_same_finger_repeats", 0),
        "total_hits": hits.get("total_hits", 0),
        "checks": checks,
        "passed": all(checks.values()),
    }


def run(command: list[str]) -> None:
    print("+", " ".join(command))
    subprocess.run(command, check=True)


if __name__ == "__main__":
    raise SystemExit(main())
