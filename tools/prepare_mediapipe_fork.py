from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


DEFAULT_REPO_URL = "https://github.com/google-ai-edge/mediapipe.git"
DEFAULT_REF = "v0.10.21"
DEFAULT_DEST = Path("third_party/mediapipe-fork")


PATCH_NOTES = """# AirDesk MediaPipe Fork Notes

Goal: make MediaPipe's hand pipeline serve high-precision fingertip tracking,
not only whole-hand gesture recognition.

Initial patch target:

1. Keep the stock palm detector and 21-point landmark model.
2. Add an AirDesk fingertip refinement calculator after hand landmarks are
   projected back to image coordinates.
3. The calculator should mirror `fingertip_refiner.py` first: local patch,
   distal-direction edge search, conservative max shift, low-confidence no-op.
4. Expose per-fingertip confidence/disagreement so AirDesk can block hits on
   unstable frames instead of trusting a one-frame jump.

Likely graph files to inspect first:

- mediapipe/modules/hand_landmark/hand_landmark_tracking_cpu.pbtxt
- mediapipe/modules/hand_landmark/hand_landmark_tracking_gpu.pbtxt
- mediapipe/tasks/cc/vision/hand_landmarker
- mediapipe/calculators/util/landmarks_smoothing_calculator.*

Do not start by retraining the landmark model. First prove this calculator
improves AirDesk recorded-session metrics, then migrate the Python prototype.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a local MediaPipe source fork for AirDesk fingertip-tracking work. "
            "This only clones and writes patch notes; it does not build or install the wheel."
        )
    )
    parser.add_argument("--repo-url", default=DEFAULT_REPO_URL)
    parser.add_argument("--ref", default=DEFAULT_REF)
    parser.add_argument("--dest", default=str(DEFAULT_DEST))
    parser.add_argument("--skip-clone", action="store_true", help="Only write AirDesk patch notes into --dest")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dest = Path(args.dest)
    if not args.skip_clone and not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", "--depth", "1", "--branch", args.ref, args.repo_url, str(dest)])
    elif not dest.exists():
        raise SystemExit(f"{dest} does not exist; omit --skip-clone to clone it")

    notes_path = dest / "AIRDESK_FINGERTIP_PATCH_NOTES.md"
    notes_path.write_text(PATCH_NOTES, encoding="utf-8")
    print(f"Prepared MediaPipe fork workspace: {dest}")
    print(f"Wrote {notes_path}")
    print()
    print("Next manual steps:")
    print(f"  cd {dest}")
    print("  python -m pip install -U pip setuptools wheel")
    print("  python setup.py bdist_wheel")
    print("  # then install the generated wheel in this project's virtualenv")
    return 0


def run(command: list[str]) -> None:
    print("+", " ".join(command))
    subprocess.run(command, check=True)


if __name__ == "__main__":
    raise SystemExit(main())
