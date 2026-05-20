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

## Fifth Interaction Model Adjustment

User feedback from the updated `test03`: the piano trigger should behave more like a real key press, and the visual keyboard should feel attached to the desk plane rather than pasted onto the screen.

Changes:

- Replaced piano hit logic with a per-finger motion state machine:
  - `idle`
  - `raised/armed`
  - `falling`
  - `pressed`
- A piano hit now requires an up/down gesture:
  - finger is armed by moving upward or staying in a lifted state;
  - finger moves downward;
  - landing point is inside a key polygon;
  - drop distance and strike velocity are large enough;
  - the finger must lift a minimum pixel distance before repeating.
- Added new tuning parameters:
  - `PIANO_ARM_RATIO = 0.28`
  - `PIANO_PRESS_RATIO = 0.35`
  - `PIANO_RELEASE_RATIO = 0.30`
  - `PIANO_FALLING_VELOCITY_THRESHOLD = 45`
  - `PIANO_STRIKE_MIN_DROP_PX = 18`
  - `PIANO_STRIKE_MIN_VELOCITY = 110`
  - `PIANO_RELEASE_LIFT_PX = 14`
- Converted piano keys from screen rectangles to projected quadrilateral zones.
  - Each key stores a `polygon`.
  - Hit testing uses point-in-polygon before the fallback margin check.
  - Session recordings now serialize polygons.
- Rendered the keybed with a perspective warp into the same quadrilateral plane used for hit testing.
  - This gives a more VR/AR-like “on the desk” feel.
  - Hands are still composited above the keyboard layer.

Replay results with the landing-anywhere state machine:

- `test03`: 93 hits.
- `test02`: 177 hits.
- `test01`: 259 hits.

The important qualitative change is that hits no longer depend on crossing a contact line. They are produced by a lifted/falling/landing sequence where the landing point can be anywhere inside a key.

## Sixth Precision/Performance Adjustment

User feedback from the latest `test03`: FPS is low, and an intended right-index tap can be reported as the thumb's key.

Root causes:

- FPS was spending avoidable CPU in per-frame perspective keybed generation, full-frame alpha blending, hand cutout blur, and debug frame-quality metrics.
- The thumb landmark is less stable in the current laptop-camera angle. In the video it can make a large apparent motion when the hand rotates, so a thumb candidate may outscore the intended long finger.

Changes:

- Cached the perspective piano keybed warp and masks instead of regenerating them every frame.
- Restricted alpha blending to the nonzero mask bounding box.
- Reduced default MediaPipe input width:
  - `TRACKING_MAX_WIDTH = 416`
- Sampled debug frame-quality metrics every `0.5s` instead of every frame.
- Reduced hand cutout blur kernel:
  - `HAND_CUTOUT_BLUR_KERNEL = 17`
- Added runtime performance switches:
  - `--no-hand-cutout`
  - `--no-fingertip-markers`
  - `--max-hands 1`
- Kept 10 fingertip visualization and later restored all 10 fingertips as default triggers:
  - `TRIGGER_FINGER_IDS = (4, 8, 12, 16, 20)`
  - Use `--no-trigger-thumb` to temporarily disable thumb triggering on unstable camera angles.
- Changed piano motion measurement to use each fingertip relative to its own base joint, not only absolute screen y motion.
- Added same-hand candidate arbitration:
  - `PIANO_MAX_HITS_PER_HAND_PER_FRAME = 1`
  - The strongest falling/landing candidate wins; weaker same-frame candidates are marked `suppressed_by_finger`.

Replay results:

- Latest `test03` before restoring thumb triggering: 6 replay hits, all from long fingers.
- `test02`: 87 replay hits, so the change does not globally kill playability on the longer sample.

Tradeoff: thumb triggering is now enabled for the intended 10-finger interaction. The `--no-trigger-thumb` switch remains useful for camera angles where the thumb landmark is unstable.

## Seventh Key-Mapping Fix

User feedback from the updated `test03`: different right-hand fingers still sounded like the same key.

Root cause:

- The perspective key polygons were being drawn correctly, but `_point_in_polygon()` used `max(1e-9, yj - yi)` as a denominator guard.
- When an edge went upward, `yj - yi` was negative, so the denominator was incorrectly replaced with `1e-9`.
- This made some polygon intersections enormous and caused points to the right of a key to be incorrectly counted inside earlier keys. In `test03`, many right-hand fingertip positions were swallowed by `C5`.

Changes:

- Fixed `_point_in_polygon()` to preserve the denominator sign and only guard true near-zero denominators.
- Added a regression test for perspective piano key mapping using points from `test03`.

Replay results on latest `test03`:

- Before fix: `C5` dominated with 10 / 17 hits.
- After fix: 18 hits spread across the landing positions:
  - `D5`, `E5`, `F5`, `G5`, `A5`
  - plus left-hand lower keys such as `C4`, `D4`, `F4`, `G4`

This directly addresses the “no matter which right-hand finger I lift, it triggers one key” failure.

## Eighth Anti-Jitter State Machine Update

User feedback from the refreshed `test01/test02`: when a hand rests on the keyboard, tiny MediaPipe landmark jitter can still trigger notes, especially when fingers are close to the camera.

Root cause:

- The previous state machine released a pressed finger from a single upward-velocity spike.
- A later one-frame downward jump could arm and trigger a new hit, even if the physical finger had barely moved.
- This is common when MediaPipe loses fingertip stability near the camera or around occlusions.

Changes:

- Added a `lifting` state before `raised`.
- A finger must accumulate a real lift before it can become armed:
  - `PIANO_ARM_MIN_LIFT_PX = 18`
- A pressed finger must satisfy release distance for consecutive frames:
  - `PIANO_RELEASE_STABLE_FRAMES = 2`
- Removed the permissive one-frame direct-hit path when the finger was not already in a falling state.
- Added synthetic regression tests:
  - pressed finger jitter does not retrigger;
  - short lift then drop does not trigger;
  - clear lift then drop still retriggers.

Replay results on refreshed sessions:

- `test01`: `182 -> 93` replay hits.
- `test02`: `138 -> 72` replay hits.
- Same-finger repeat hits under `0.35s`: `13 -> 0` on both sessions.

This is a deliberate product-level tradeoff: fewer accidental notes while a hand is resting on the virtual keys, with slightly stricter requirements for repeated taps.

## Ninth Fingertip Precision Update

User feedback: MediaPipe fingertip points can still jump when fingers are close to the camera, so the hand can be physically still while the detected fingertip bounces enough to trigger notes.

Decision:

- Training a new VLM is not the right next step for this prototype. The latency, labeling cost, and real-time deployment complexity are too high for a webcam piano interaction.
- The better product path is a dedicated hand/fingertip keypoint tracker plus temporal filtering, with recorded sessions as a regression set.

Changes:

- Added stable hand-id assignment so per-finger state is less affected by MediaPipe hand order changes.
- Added OpenCV Lucas-Kanade optical-flow stabilization for selected fingertip and finger-base landmarks.
  - MediaPipe now provides coarse pose.
  - Optical flow provides frame-to-frame continuity.
  - Large single-frame disagreement is treated as a likely landmark outlier.
- Tightened piano release:
  - release now requires upward motion evidence, or a stronger sustained lift;
  - this reduces “resting hand” retriggers caused by point jitter.
- Added piano sensitivity presets:
  - `stable` default for fewer false notes;
  - `balanced` and `sensitive` for softer taps.

Replay results after the precision guard:

- `test03`: online 22, replay 19.
- `test02`: online 138, replay 27, with 26 candidates blocked by `jitter_guard`.
- `test01`: online 182, replay 58, with 57 candidates blocked by `jitter_guard`.

The offline hit count drops on older sessions because many previous hits looked like short lift/jitter repeats. The default now intentionally favors product stability; if the next live test feels too strict, use `--piano-sensitivity balanced` before loosening the detector further.

## Tenth Record3D RGB-D Integration

User goal: move beyond toy single-webcam tracking by using Record3D/iPhone LiDAR as a more product-grade RGB-D source.

Changes:

- Added `rgbd_camera.py`.
  - Wraps the optional Record3D SDK into a blocking `read()` camera interface.
  - Supports device selection, frame timeout, rotation, mirroring, and depth-unit normalization.
- Added `depth_contact.py`.
  - Press `d` with hands away from the keys to calibrate a desk depth baseline.
  - Runtime samples local median depth around each fingertip.
  - Computes approximate height above the calibrated desk plane.
  - Returns conservative contact observations: contact, above desk, or unknown.
- Added depth-gated hit detection.
  - `--depth-contact-mode assist` blocks only when depth clearly says the finger is above the desk.
  - `--depth-contact-mode required` requires depth-confirmed contact for piano hits.
  - Default `auto` enables `assist` for Record3D and disables depth gating for webcams.
- Added session logging for depth observations and depth status.

Recommended first Record3D run:

```bash
python main.py --camera-source record3d --record3d-device 0 --mode piano --debug
```

After the window opens, remove hands from the keybed and press `d` to calibrate the desk depth.

Product note:

- This is not absolute physical contact sensing. LiDAR depth is lower resolution than RGB and noisy at finger edges.
- The robust product strategy is RGB for fingertip `x/y`, depth for contact probability, and the existing temporal state machine for tap intent.

## Product-Level Roadmap

1. Stabilize performance to 20+ FPS.
   - Keep high-resolution capture for display.
   - Run MediaPipe on a tighter ROI.
   - Consider a worker thread for hand tracking so UI/audio does not block.

2. Refine the note state machine with calibration data.
   - Add per-finger adaptive baselines to tolerate different hand heights.
   - Learn a comfortable landing/release band during a short warmup.
   - Add sensitivity presets on top of the state machine instead of changing single thresholds by hand.

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
