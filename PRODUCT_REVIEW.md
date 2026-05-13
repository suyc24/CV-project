# AirDesk Product Review

Session analyzed: `data/sessions/test01`

## Current Findings

- Video quality is now acceptable. Average overexposure is about `0.056%`, so exposure is no longer the main blocker.
- Processing FPS is the main blocker. The 70.8 second session contains 648 processed frames, so effective runtime FPS is around 9-10 during active hand tracking.
- Two hands were detected in 524 / 648 frames, which is good enough for interaction, but low FPS makes velocity estimates and timing feel inconsistent.
- Current session produced 115 online hits. 34 hit intervals were under 80 ms, which suggests sensitivity is too high for product behavior.
- Miss reason distribution:
  - `velocity`: 3115
  - `no_zone`: 1118
  - `pressed`: 792
  - `press_line`: 182
  - `hit`: 115
  - `cooldown`: 18

## Changes Applied After This Session

- `TRACKING_ROI_Y_MIN`: `0.18 -> 0.35`
  - The session shows useful hand motion in the lower part of the frame, so the tracker should not spend time on the upper background.
- `TRACKING_MAX_WIDTH`: `640 -> 480`
  - Keeps capture at 720p while reducing MediaPipe cost.
- `PIANO_HIT_VELOCITY_THRESHOLD`: `100 -> 180`
  - The previous value was too permissive and caused toy-like accidental triggers.
- `VELOCITY_SMOOTHING_ALPHA`: `0.45 -> 0.65`
  - Makes velocity react faster after landmark smoothing.
- Exposed tracker knobs:
  - `--max-hands`
  - `--min-detection-confidence`
  - `--min-tracking-confidence`

## Second Session Update

Session analyzed: `data/sessions/test02`

- Frames: 505 over 50.9 seconds.
- Baseline replay hits with the first post-review config: 54.
- Main miss reasons:
  - `velocity`: 2321
  - `no_zone`: 1347
  - `press_line`: 53
  - `pressed`: 358
- Frame inspection shows many intended presses landing just below the visual keyboard rectangle. This means the visual overlay is useful, but the input hit target needs to be more forgiving than the rendered keybed.

Changes applied after `test02`:

- Added forgiving piano hit margins:
  - `PIANO_HIT_X_MARGIN_RATIO = 0.18`
  - `PIANO_HIT_TOP_MARGIN_RATIO = 0.20`
  - `PIANO_HIT_BOTTOM_MARGIN_RATIO = 0.90`
- Added low-speed press-line crossing trigger:
  - `PIANO_CROSSING_VELOCITY_THRESHOLD = 55.0`
- Result on `test02` replay:
  - Hits: `54 -> 87`
  - `no_zone`: `1347 -> 146`
  - This better matches user intent when fingers land slightly below the visible keyboard.

Tradeoff: `press_line` counts increase because more near-key fingertip positions are now associated with a key. That is acceptable because those frames are now diagnosable as “not yet pressed deeply enough” rather than “not on any key.”

## Third UI/Input Adjustment

User feedback: the keyboard should be larger and fill the bottom of the screen.

Changes:

- Added piano-specific ROI:
  - `PIANO_ROI_X_MIN = 0.00`
  - `PIANO_ROI_X_MAX = 1.00`
  - `PIANO_ROI_Y_MIN = 0.38`
  - `PIANO_ROI_Y_MAX = 1.00`
- Set `PIANO_AREA_HEIGHT_RATIO = 1.00`.
- Reduced hidden hit margins because the visible keybed itself is now much larger:
  - `PIANO_HIT_X_MARGIN_RATIO = 0.06`
  - `PIANO_HIT_TOP_MARGIN_RATIO = 0.08`
  - `PIANO_HIT_BOTTOM_MARGIN_RATIO = 0.18`

Geometry on a 1280x720 frame:

- Keyboard bounds: `x=0..1280`, `y=273..720`.
- White key size: about `85x447` px.

Replay results:

- `test02`: 80 hits.
- `test01`: 127 hits.

This is slightly fewer than the very forgiving invisible-margin model, but it aligns the visible UI with the actual hit target and should feel less surprising.

## Fourth Interaction/UI Adjustment

User feedback from `test03`: expected piano taps still miss too often, and the hand should remain visible above the keyboard layer.

Changes:

- Lowered piano press line:
  - `PIANO_PRESS_RATIO = 0.40`
  - The lower 60% of the visible key now counts as the playable trigger band.
- Reduced piano velocity threshold:
  - `PIANO_HIT_VELOCITY_THRESHOLD = 150`
  - This preserves velocity-sensitive triggering but accepts softer taps.
- Kept release hysteresis:
  - `PIANO_RELEASE_RATIO = 0.35`
  - A finger must still lift above the release line before another note can fire.
- Restored real hand visibility above the piano:
  - The UI keeps a copy of the camera frame before drawing the keybed.
  - A feathered mask is built from hand landmarks.
  - The hand region is blended back over the piano layer.
  - Only small fingertip markers are shown; skeleton/trail lines stay hidden.
- Set `PIANO_AREA_HEIGHT_RATIO = 1.00` so the keyboard fills the bottom ROI.

Replay results:

- `test03`: 66 -> 102 hits.
- `test02`: 54 -> 105 hits.
- `test01`: 115 -> 184 hits.

Tradeoff: this is intentionally more playable and forgiving. If later sessions show accidental hover-triggering, add a user-facing sensitivity preset rather than hard-coding a stricter default.

## Product-Level Roadmap

1. Stabilize performance to 20+ FPS.
   - Keep high-resolution capture for display.
   - Run MediaPipe on a tighter ROI.
   - Consider a worker thread for hand tracking so UI/audio does not block.

2. Replace pure velocity thresholding with a note state machine.
   - States: `released -> armed -> pressed -> held -> released`.
   - Trigger on downward crossing of the press line, not merely being below it.
   - Add per-finger adaptive baseline to tolerate different hand heights.

3. Add calibration for the interaction surface.
   - Current fixed ROI is acceptable for a demo but not product grade.
   - Four-point calibration would map camera coordinates to a stable virtual keyboard plane.

4. Add user-facing sensitivity presets.
   - `low`, `medium`, `high`, `practice`.
   - Presets should change velocity threshold, press/release ratio, and cooldown together.

5. Improve feedback.
   - Show a small hit flash on the specific key.
   - Show recording/playback state outside debug mode.
   - Add optional invisible diagnostics export for every run.

6. Build a small evaluation set.
   - Record 5-10 sessions covering slow taps, fast scales, chords, and accidental hovering.
   - Use `replay_session.py` for regression checks before each change.

## Recommended Next Run

```bash
python main.py --camera 0 --backend dshow --mode piano --debug --record-session data/sessions/test02
```

Then run:

```bash
python replay_session.py data/sessions/test02
python analysis_report.py data/sessions/test02
```
