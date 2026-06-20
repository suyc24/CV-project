from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark_guide import ADJACENT_KEY_NOTES, GUIDE_SEQUENCES, SINGLE_FINGER_CHECK_NOTES, TWINKLE_NOTES  # noqa: E402
from tools.run_demo_benchmark import (  # noqa: E402
    aggregate_totals,
    benchmark_config,
    evaluate_session,
    one_line_summary,
    write_hits_csv,
    write_matches_csv,
)


ROOT = Path("data/sessions/benchmarks_3d")
DEFAULT_ROOT = ROOT / "twinkle_right_hand"
DEFAULT_NOTES_PER_SECOND = 2.0
DEFAULT_GUIDE_FIRST_ONSET = 2.0


@dataclass(frozen=True)
class ClipSpec:
    split: str
    name: str
    title: str
    instruction: str
    notes: tuple[str, ...] = ()
    expected_zero_hits: bool = False
    holdout: bool = False
    guide_sequence: Optional[str] = None
    guide_notes_per_second: float = DEFAULT_NOTES_PER_SECOND

    @property
    def relative_dir(self) -> Path:
        return Path(self.split) / self.name


CLIPS = (
    ClipSpec(
        split="dev",
        name="01_rest_both_hands",
        title="双手静止防误触",
        instruction="双手自然放在键盘区域 20 秒，不弹。",
        expected_zero_hits=True,
    ),
    ClipSpec(
        split="dev",
        name="02_hover_both_hands",
        title="双手悬停防误触",
        instruction="双手五指悬停在键盘上方，左右轻微移动 20 秒，不接触。",
        expected_zero_hits=True,
    ),
    ClipSpec(
        split="dev",
        name="03_single_finger_checks",
        title="右手五指基础触发",
        instruction=(
            "右手单手：拇指 C4 6 次，食指 D4 6 次，中指 E4 6 次，"
            "无名指 F4 6 次，小指 G4 6 次。每次敲击间隔约 0.5 秒。"
        ),
        notes=(
            SINGLE_FINGER_CHECK_NOTES
        ),
        guide_sequence="single_finger_checks",
    ),
    ClipSpec(
        split="dev",
        name="04_adjacent_keys",
        title="相邻键错音检查",
        instruction="右手单手慢速弹 C4 D4 E4 F4 G4 F4 E4 D4 C4，连续 2 遍。",
        notes=ADJACENT_KEY_NOTES,
        guide_sequence="adjacent_keys",
    ),
    ClipSpec(
        split="dev",
        name="05_twinkle_slow",
        title="小星星慢速",
        instruction="右手单手弹《小星星》1 遍，略慢于正常演示速度。",
        notes=TWINKLE_NOTES,
        guide_sequence="twinkle",
        guide_notes_per_second=1.5,
    ),
    ClipSpec(
        split="dev",
        name="06_twinkle_normal_take1",
        title="小星星正常速度 take 1",
        instruction="右手单手弹《小星星》1 遍，约每秒 2 个音。",
        notes=TWINKLE_NOTES,
        guide_sequence="twinkle",
    ),
    ClipSpec(
        split="dev",
        name="07_twinkle_normal_take2",
        title="小星星正常速度 take 2",
        instruction="右手单手弹《小星星》1 遍，约每秒 2 个音。",
        notes=TWINKLE_NOTES,
        guide_sequence="twinkle",
    ),
    ClipSpec(
        split="holdout",
        name="01_rest_single_hand",
        title="单手静止防误触",
        instruction="右手自然放在键盘区域 15 秒，不弹。",
        expected_zero_hits=True,
        holdout=True,
    ),
    ClipSpec(
        split="holdout",
        name="02_twinkle_normal_take1",
        title="holdout 小星星正常速度 take 1",
        instruction="右手单手弹《小星星》1 遍，约每秒 2 个音。",
        notes=TWINKLE_NOTES,
        holdout=True,
        guide_sequence="twinkle",
    ),
    ClipSpec(
        split="holdout",
        name="03_twinkle_normal_take2",
        title="holdout 小星星正常速度 take 2",
        instruction="右手单手弹《小星星》1 遍，约每秒 2 个音。",
        notes=TWINKLE_NOTES,
        holdout=True,
        guide_sequence="twinkle",
    ),
    ClipSpec(
        split="holdout",
        name="04_twinkle_natural_variation",
        title="holdout 自然演示",
        instruction="右手单手跟随引导弹《小星星》1 遍，手型和动作保持自然。",
        notes=TWINKLE_NOTES,
        holdout=True,
        guide_sequence="twinkle",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan, annotate, and evaluate the right-hand Twinkle benchmark.")
    parser.add_argument("--root", default=str(DEFAULT_ROOT), help="Benchmark root containing dev/ and holdout/")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("plan", help="Print the agreed recording checklist")

    commands = subparsers.add_parser("record-commands", help="Print one main.py recording command per clip")
    commands.add_argument("--python", default=".venv/bin/python")
    commands.add_argument("--tracking-max-width", type=int, default=320)
    commands.add_argument("--window-width", type=int, default=640)
    commands.add_argument("--window-height", type=int, default=480)
    commands.add_argument("--guide-first-onset", type=float, default=DEFAULT_GUIDE_FIRST_ONSET)

    annotate = subparsers.add_parser("annotate", help="Generate annotations.csv for one recorded clip")
    annotate.add_argument("clip", help="Clip name, e.g. 06_twinkle_normal_take1, or split/name")
    annotate.add_argument("--first-onset", type=float, default=1.0, help="Relative time in seconds for the first expected note")
    annotate.add_argument("--notes-per-second", type=float, default=DEFAULT_NOTES_PER_SECOND)
    annotate.add_argument("--overwrite", action="store_true")

    annotate_all = subparsers.add_parser("annotate-all", help="Generate provisional annotations for all clips")
    annotate_all.add_argument("--first-onset", type=float, default=1.0)
    annotate_all.add_argument("--notes-per-second", type=float, default=DEFAULT_NOTES_PER_SECOND)
    annotate_all.add_argument("--overwrite", action="store_true")

    evaluate = subparsers.add_parser("evaluate", help="Evaluate dev or holdout clips with current hit detector")
    evaluate.add_argument("--split", choices=["dev", "holdout", "all"], default="dev")
    evaluate.add_argument("--output-root", default="data/benchmarks/twinkle_eval")
    evaluate.add_argument("--frame-tolerance", type=int, default=4)
    evaluate.add_argument("--time-tolerance", type=float, default=0.22)
    evaluate.add_argument("--annotation-padding-frames", type=int, default=45)
    evaluate.add_argument("--no-depth", action="store_true")
    evaluate.add_argument("--piano-sensitivity", choices=["current", "strict", "balanced", "sensitive"], default="current")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root)
    if args.command == "plan":
        print_plan(root)
        return 0
    if args.command == "record-commands":
        print_record_commands(root, args)
        return 0
    if args.command == "annotate":
        spec = resolve_clip(args.clip)
        write_annotations(
            root / spec.relative_dir,
            spec,
            first_onset=args.first_onset,
            notes_per_second=args.notes_per_second,
            overwrite=args.overwrite,
        )
        return 0
    if args.command == "annotate-all":
        for spec in CLIPS:
            write_annotations(
                root / spec.relative_dir,
                spec,
                first_onset=args.first_onset,
                notes_per_second=args.notes_per_second,
                overwrite=args.overwrite,
            )
        return 0
    if args.command == "evaluate":
        from tools.run_demo_benchmark import apply_piano_sensitivity

        apply_piano_sensitivity(args.piano_sensitivity)
        return evaluate_benchmark(root, args)
    raise AssertionError(args.command)


def print_plan(root: Path) -> None:
    print(f"Benchmark root: {root}")
    print("曲目: C 大调《小星星》，右手单手，正常速度约每秒 2 个音。")
    print("录制规则: 每段开始前手离开键盘，停 1 秒，清楚弹第一个音或开始动作；每段结束后停 1 秒。")
    for spec in CLIPS:
        marker = "HOLDOUT" if spec.holdout else "DEV"
        expected = "期望 0 hits" if spec.expected_zero_hits else f"期望 {len(spec.notes)} 个音"
        print(f"\n[{marker}] {spec.relative_dir}")
        print(f"  {spec.title}: {expected}")
        print(f"  {spec.instruction}")


def print_record_commands(root: Path, args: argparse.Namespace) -> None:
    for spec in CLIPS:
        session_dir = root / spec.relative_dir
        print(f"# {spec.relative_dir}: {spec.title}")
        print(
            " ".join(
                [
                    args.python,
                    "main.py",
                    "--camera-source record3d",
                    "--mode piano",
                    "--paper-keyboard",
                    "--record-session",
                    str(session_dir),
                    "--record-immediately",
                    "--instrument-roi 0.05,0.45,0.95,0.95",
                    "--piano-left-trim-keys 1.5",
                    "--piano-sensitivity balanced",
                    f"--tracking-max-width {args.tracking_max_width}",
                    "--hand-model-complexity 0",
                    f"--window-width {args.window_width}",
                    f"--window-height {args.window_height}",
                    "--no-fingertip-markers",
                ]
                + guide_command_args(spec, args.guide_first_onset)
            )
        )
        print()


def guide_command_args(spec: ClipSpec, first_onset: float) -> list[str]:
    if not spec.guide_sequence:
        return []
    if spec.guide_sequence not in GUIDE_SEQUENCES:
        raise SystemExit(f"Unknown guide sequence in clip spec: {spec.guide_sequence}")
    return [
        f"--guide-sequence {spec.guide_sequence}",
        f"--guide-first-onset {first_onset:.3f}",
        f"--guide-notes-per-second {spec.guide_notes_per_second:.3f}",
    ]


def write_annotations(
    session_dir: Path,
    spec: ClipSpec,
    first_onset: float,
    notes_per_second: float,
    overwrite: bool,
) -> None:
    session_dir.mkdir(parents=True, exist_ok=True)
    path = session_dir / "annotations.csv"
    if path.exists() and not overwrite:
        print(f"Keeping existing {path}; pass --overwrite to replace it.")
        return
    rows = annotations_for(spec, first_onset=first_onset, notes_per_second=notes_per_second)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["onset", "note", "frame_index"])
        writer.writeheader()
        writer.writerows(rows)
    write_description(session_dir, spec, first_onset, notes_per_second)
    print(f"Wrote {path} ({len(rows)} expected note(s))")


def annotations_for(spec: ClipSpec, first_onset: float, notes_per_second: float) -> list[dict[str, object]]:
    if spec.expected_zero_hits:
        return []
    interval = 1.0 / max(0.01, float(notes_per_second))
    return [
        {
            "onset": f"{first_onset + index * interval:.3f}",
            "note": note,
            "frame_index": "",
        }
        for index, note in enumerate(spec.notes)
    ]


def write_description(session_dir: Path, spec: ClipSpec, first_onset: float, notes_per_second: float) -> None:
    payload = {
        "split": spec.split,
        "name": spec.name,
        "title": spec.title,
        "instruction": spec.instruction,
        "holdout": spec.holdout,
        "expected_zero_hits": spec.expected_zero_hits,
        "note_count": len(spec.notes),
        "first_onset": first_onset,
        "notes_per_second": notes_per_second,
    }
    (session_dir / "benchmark_spec.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    (session_dir / "DESCRIPTION.md").write_text(
        "\n".join(
            [
                f"# {spec.title}",
                "",
                f"- split: `{spec.split}`",
                f"- clip: `{spec.name}`",
                f"- expected_zero_hits: `{spec.expected_zero_hits}`",
                f"- note_count: `{len(spec.notes)}`",
                "",
                spec.instruction,
                "",
            ]
        ),
        encoding="utf-8",
    )


def evaluate_benchmark(root: Path, args: argparse.Namespace) -> int:
    output_root = Path(args.output_root) / args.split
    output_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for spec in CLIPS:
        if args.split != "all" and spec.split != args.split:
            continue
        session_dir = root / spec.relative_dir
        if not (session_dir / "frames.jsonl").exists():
            print(f"Skipping missing recorded session: {session_dir}", file=sys.stderr)
            continue
        summary = evaluate_session(
            session_dir,
            frame_tolerance=args.frame_tolerance,
            time_tolerance=args.time_tolerance,
            start_frame=None,
            end_frame=None,
            annotation_padding_frames=args.annotation_padding_frames,
            use_depth=not args.no_depth,
        )
        summaries.append(summary)
        session_output = output_root / spec.split / spec.name
        session_output.mkdir(parents=True, exist_ok=True)
        (session_output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        write_hits_csv(session_output / "predicted_hits.csv", summary["predicted_hits"])
        write_matches_csv(session_output / "matches.csv", summary["matches"])
        print(one_line_summary(summary))
    aggregate = {
        "root": str(root),
        "split": args.split,
        "sessions": summaries,
        "totals": aggregate_totals(summaries),
        "config": benchmark_config(),
    }
    (output_root / "benchmark_summary.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print(json.dumps(aggregate["totals"], indent=2))
    return 0


def resolve_clip(raw: str) -> ClipSpec:
    normalized = raw.strip("/ ")
    matches = [
        spec
        for spec in CLIPS
        if normalized in {spec.name, str(spec.relative_dir)}
    ]
    if len(matches) == 1:
        return matches[0]
    known = ", ".join(str(spec.relative_dir) for spec in CLIPS)
    raise SystemExit(f"Unknown clip {raw!r}. Known clips: {known}")


if __name__ == "__main__":
    raise SystemExit(main())
