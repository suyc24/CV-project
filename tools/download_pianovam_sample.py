from __future__ import annotations

import argparse
import fnmatch
import json
import subprocess
import sys
from pathlib import Path


REPO_ID = "PianoVAM/PianoVAM_v1"
DEFAULT_PATTERNS = ("**/*.tsv", "**/*.csv", "**/*skeleton*.json", "**/*hand*.json", "**/*mediapipe*.json")
VIDEO_PATTERNS = ("**/*.mp4", "**/*.mov", "**/*.avi", "**/*.mkv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download a lightweight PianoVAM subset for AirDesk benchmarking. "
            "By default this downloads labels and hand-skeleton metadata only; "
            "pass --include-video when you have enough bandwidth/disk."
        )
    )
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument("--output", default="data/external/pianovam")
    parser.add_argument("--include-video", action="store_true")
    parser.add_argument("--max-files", type=int, default=20, help="Maximum matching files to download; use 0 for all")
    parser.add_argument(
        "--allow-pattern",
        action="append",
        default=[],
        help="Extra HuggingFace allow pattern. Can be passed multiple times.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    allow_patterns = list(DEFAULT_PATTERNS)
    if args.include_video:
        allow_patterns.extend(VIDEO_PATTERNS)
    allow_patterns.extend(args.allow_pattern)

    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError:
        return _fallback_huggingface_cli(args.repo_id, output, allow_patterns, args.max_files)

    try:
        selected_patterns = select_files(args.repo_id, allow_patterns, args.max_files, HfApi())
    except Exception as exc:
        print(
            "Could not reach HuggingFace to list PianoVAM files.\n"
            "Check network access, proxy settings, or download the dataset on a machine with internet.\n"
            f"Original error: {exc}",
            file=sys.stderr,
        )
        return 1
    manifest = {
        "repo_id": args.repo_id,
        "output": str(output),
        "include_video": args.include_video,
        "requested_patterns": allow_patterns,
        "download_patterns": selected_patterns,
    }
    (output / "aird_download_request.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    try:
        local_path = snapshot_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            local_dir=str(output),
            allow_patterns=selected_patterns,
            local_dir_use_symlinks=False,
        )
    except TypeError:
        local_path = snapshot_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            local_dir=str(output),
            allow_patterns=selected_patterns,
        )
    print(f"Downloaded PianoVAM subset to {local_path}")
    _print_next_steps(output)
    return 0


def select_files(repo_id: str, allow_patterns: list[str], max_files: int, api) -> list[str]:
    if max_files == 0:
        return allow_patterns
    files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    selected: list[str] = []
    for pattern in allow_patterns:
        for path in files:
            if path in selected:
                continue
            if fnmatch.fnmatch(path, pattern):
                selected.append(path)
                if len(selected) >= max_files:
                    return selected
    if not selected:
        raise SystemExit(f"No files matched patterns: {allow_patterns}")
    return selected


def _fallback_huggingface_cli(repo_id: str, output: Path, allow_patterns: list[str], max_files: int) -> int:
    if max_files != 0:
        print(
            "huggingface_hub is required for --max-files sampling. "
            "Install it with `pip install huggingface_hub`, or rerun with --max-files 0.",
            file=sys.stderr,
        )
        return 2
    command = [
        "huggingface-cli",
        "download",
        repo_id,
        "--repo-type",
        "dataset",
        "--local-dir",
        str(output),
    ]
    for pattern in allow_patterns:
        command.extend(["--include", pattern])

    try:
        subprocess.run(command, check=True)
    except FileNotFoundError:
        print(
            "huggingface_hub is not installed and huggingface-cli was not found.\n"
            "Install the optional downloader dependency with:\n"
            "  pip install huggingface_hub\n"
            "Then rerun this script.",
            file=sys.stderr,
        )
        return 2
    except subprocess.CalledProcessError as exc:
        print(f"huggingface-cli download failed with exit code {exc.returncode}", file=sys.stderr)
        return exc.returncode

    print(f"Downloaded PianoVAM subset to {output}")
    _print_next_steps(output)
    return 0


def _print_next_steps(output: Path) -> None:
    print("\nNext:")
    print(f"  find {output} -maxdepth 3 -type f | head")
    print("  python tools/evaluate_hit_events.py --pred <aird_hits.csv> --gt <pianovam.tsv>")


if __name__ == "__main__":
    raise SystemExit(main())
