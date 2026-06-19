# Twinkle Right-Hand Benchmark Protocol

This protocol separates recording from algorithm debugging. The user records
all Record3D sessions once; the algorithm can then be tuned offline by replaying
`frames.jsonl` and evaluating against `annotations.csv`.

## Goal

- Song: Twinkle Twinkle Little Star in C major.
- Performer: right hand only.
- Normal speed: about 2 notes per second.
- Optimization target: demo-song correctness first, with rest/hover guardrails
  kept quiet.
- Anti-overfitting: tune on `dev`; reserve `holdout` for stage checks.

## Directory

Default root:

```text
data/sessions/benchmarks_3d/twinkle_right_hand/
  dev/
  holdout/
```

## Checklist

Print the full checklist:

```bash
.venv/bin/python tools/twinkle_benchmark.py plan
```

Print one recording command per clip:

```bash
.venv/bin/python tools/twinkle_benchmark.py record-commands
```

Each clip follows the same timing convention:

1. Move the hand away from the keyboard.
2. Pause for about 1 second.
3. Start the clip action clearly.
4. Pause for about 1 second after finishing.

## Annotation

After recording a clip, generate a provisional `annotations.csv` from the known
sequence. `--first-onset` is the relative time, in seconds, of the first note in
that session.

Example:

```bash
.venv/bin/python tools/twinkle_benchmark.py annotate dev/06_twinkle_normal_take1 --first-onset 1.20 --notes-per-second 2.0 --overwrite
```

For rest/hover clips, the generated `annotations.csv` contains only a header.
Those clips are treated as guardrails where the expected hit count is zero.

To create provisional annotations for every clip with the same initial timing:

```bash
.venv/bin/python tools/twinkle_benchmark.py annotate-all --first-onset 1.0 --notes-per-second 2.0 --overwrite
```

## Evaluation

Tune only on `dev`:

```bash
.venv/bin/python tools/twinkle_benchmark.py evaluate --split dev
```

Use `holdout` only for stage checks:

```bash
.venv/bin/python tools/twinkle_benchmark.py evaluate --split holdout
```

Outputs are written under:

```text
data/benchmarks/twinkle_eval/
```

The important summary fields are:

- `matched`: expected notes hit correctly.
- `misses`: expected notes not hit.
- `extras`: additional predicted hits.
- `wrong_near`: wrong notes near an expected onset.
- `guardrail_hits`: hits in rest/hover clips.

