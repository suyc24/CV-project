# 《小星星》Record3D Benchmark 录制与标注交接文档

这份文档给负责录制 benchmark 的同学使用，不需要了解前面的项目讨论。目标是一次性录制一组可离线回放的 Record3D session，让后续算法调试尽量不再依赖人工反复测试。

## 1. 录制目标

本次 benchmark 用来优化虚拟钢琴的自由弹奏效果，目标是：

- 用右手单手弹《小星星》。
- 正常速度约每秒 2 个音。
- 录制 Record3D session，不只是普通视频。
- 程序录制时会显示音符引导，并自动生成 `annotations.csv`。
- 后续算法只用 `dev` 数据调试，`holdout` 数据用于防止过拟合。

曲目为 C 大调《小星星》，只使用白键：

```text
C4 C4 G4 G4 A4 A4 G4
F4 F4 E4 E4 D4 D4 C4
G4 G4 F4 F4 E4 E4 D4
G4 G4 F4 F4 E4 E4 D4
C4 C4 G4 G4 A4 A4 G4
F4 F4 E4 E4 D4 D4 C4
```

## 2. 数据保存位置

默认目录：

```text
data/sessions/benchmarks_3d/twinkle_right_hand/
  dev/
  holdout/
```

`dev` 用于算法开发和调参。  
`holdout` 只用于阶段性验收，不要根据 holdout 结果反复重录或调参。

## 3. 录制前准备

在项目根目录执行：

```bash
cd /home/suyc24/Python/CV-project
```

确认 Record3D 设备可见：

```bash
.venv/bin/python main.py --camera-source record3d --list-cameras
```

录制时请保持：

- iPhone/iPad 与电脑通过 USB 连接，Record3D 开启 Live RGBD Video Streaming。
- 摄像设备位置固定。
- 纸质键盘或 iPad 键盘位置固定。
- 光照稳定，避免强反光和明显阴影。
- 每段录制期间不要移动相机、键盘或桌面。

## 4. 查看录制清单

下面两个命令可以重新生成录制计划和录制命令；为了方便交接，完整内容已经直接写在第 7 部分。

打印完整录制计划：

```bash
.venv/bin/python tools/twinkle_benchmark.py plan
```

打印每段录制命令：

```bash
.venv/bin/python tools/twinkle_benchmark.py record-commands
```

建议按打印出来的顺序一段一段录制。每段命令会写入一个独立 session 目录。

## 5. Guided Recording 是怎么标注的

需要弹奏音符的片段，命令里会带类似参数：

```bash
--guide-sequence twinkle --guide-first-onset 2.000 --guide-notes-per-second 2.000
```

录制窗口会显示一个引导面板：

- 倒计时。
- 当前该弹的音，例如 `C4`。
- 下一个音。
- 当前进度。

程序在录制开始时自动写入：

```text
annotations.csv
guide_spec.json
```

`annotations.csv` 的时间和 `frames.jsonl` 使用同一个录制相对时间，所以不需要手动猜每个音的时间。

防误触片段，例如静止和悬停，不需要音符引导。它们的目标是 0 hits，评估时会作为 guardrail。

## 6. 每段如何操作

每段都遵循同一流程：

1. 运行该段命令。
2. 等窗口出现并开始录制。
3. 如果有引导面板，等待倒计时，第一个音默认在录制开始后 2 秒。
4. 跟随屏幕上显示的音符弹奏。
5. 弹完后保持手部稳定约 1 秒。
6. 按 `q` 结束当前段录制。
7. 继续下一段命令。

如果录错了某一段，例如弹错很多音、相机移动、手离开画面太久，可以直接重新运行同一条命令覆盖该 session。不要因为算法当前识别结果差而重录；只有录制行为本身明显失败时才重录。

## 7. 录制清单

请按下面顺序逐段录制。每一段下面都给出了该段的操作内容和完整命令。

### Dev 数据

1. `dev/01_rest_both_hands`
   双手自然放在键盘区域 20 秒，不弹。  
   目标：不应触发任何音。

   ```bash
   .venv/bin/python main.py --camera-source record3d --mode piano --paper-keyboard --record-session data/sessions/benchmarks_3d/twinkle_right_hand/dev/01_rest_both_hands --record-immediately --instrument-roi 0.05,0.45,0.95,0.95 --piano-left-trim-keys 1.5 --piano-sensitivity balanced --tracking-max-width 320 --hand-model-complexity 0 --window-width 640 --window-height 480 --no-fingertip-markers
   ```

2. `dev/02_hover_both_hands`
   双手五指悬停在键盘上方，左右轻微移动 20 秒，不接触。  
   目标：不应触发任何音。

   ```bash
   .venv/bin/python main.py --camera-source record3d --mode piano --paper-keyboard --record-session data/sessions/benchmarks_3d/twinkle_right_hand/dev/02_hover_both_hands --record-immediately --instrument-roi 0.05,0.45,0.95,0.95 --piano-left-trim-keys 1.5 --piano-sensitivity balanced --tracking-max-width 320 --hand-model-complexity 0 --window-width 640 --window-height 480 --no-fingertip-markers
   ```

3. `dev/03_single_finger_checks`
   右手单手跟随引导：
   - 拇指 C4 6 次
   - 食指 D4 6 次
   - 中指 E4 6 次
   - 无名指 F4 6 次
   - 小指 G4 6 次

   ```bash
   .venv/bin/python main.py --camera-source record3d --mode piano --paper-keyboard --record-session data/sessions/benchmarks_3d/twinkle_right_hand/dev/03_single_finger_checks --record-immediately --instrument-roi 0.05,0.45,0.95,0.95 --piano-left-trim-keys 1.5 --piano-sensitivity balanced --tracking-max-width 320 --hand-model-complexity 0 --window-width 640 --window-height 480 --no-fingertip-markers --guide-sequence single_finger_checks --guide-first-onset 2.000 --guide-notes-per-second 2.000
   ```

4. `dev/04_adjacent_keys`
   右手单手跟随引导，弹：

   ```text
   C4 D4 E4 F4 G4 F4 E4 D4 C4
   ```

   连续 2 遍。

   ```bash
   .venv/bin/python main.py --camera-source record3d --mode piano --paper-keyboard --record-session data/sessions/benchmarks_3d/twinkle_right_hand/dev/04_adjacent_keys --record-immediately --instrument-roi 0.05,0.45,0.95,0.95 --piano-left-trim-keys 1.5 --piano-sensitivity balanced --tracking-max-width 320 --hand-model-complexity 0 --window-width 640 --window-height 480 --no-fingertip-markers --guide-sequence adjacent_keys --guide-first-onset 2.000 --guide-notes-per-second 2.000
   ```

5. `dev/05_twinkle_slow`
   右手单手跟随引导，慢速弹《小星星》1 遍。  
   该段引导速度约每秒 1.5 个音。

   ```bash
   .venv/bin/python main.py --camera-source record3d --mode piano --paper-keyboard --record-session data/sessions/benchmarks_3d/twinkle_right_hand/dev/05_twinkle_slow --record-immediately --instrument-roi 0.05,0.45,0.95,0.95 --piano-left-trim-keys 1.5 --piano-sensitivity balanced --tracking-max-width 320 --hand-model-complexity 0 --window-width 640 --window-height 480 --no-fingertip-markers --guide-sequence twinkle --guide-first-onset 2.000 --guide-notes-per-second 1.500
   ```

6. `dev/06_twinkle_normal_take1`
   右手单手跟随引导，正常速度弹《小星星》1 遍。  
   该段引导速度约每秒 2 个音。

   ```bash
   .venv/bin/python main.py --camera-source record3d --mode piano --paper-keyboard --record-session data/sessions/benchmarks_3d/twinkle_right_hand/dev/06_twinkle_normal_take1 --record-immediately --instrument-roi 0.05,0.45,0.95,0.95 --piano-left-trim-keys 1.5 --piano-sensitivity balanced --tracking-max-width 320 --hand-model-complexity 0 --window-width 640 --window-height 480 --no-fingertip-markers --guide-sequence twinkle --guide-first-onset 2.000 --guide-notes-per-second 2.000
   ```

7. `dev/07_twinkle_normal_take2`
   右手单手跟随引导，正常速度弹《小星星》再录 1 遍。

   ```bash
   .venv/bin/python main.py --camera-source record3d --mode piano --paper-keyboard --record-session data/sessions/benchmarks_3d/twinkle_right_hand/dev/07_twinkle_normal_take2 --record-immediately --instrument-roi 0.05,0.45,0.95,0.95 --piano-left-trim-keys 1.5 --piano-sensitivity balanced --tracking-max-width 320 --hand-model-complexity 0 --window-width 640 --window-height 480 --no-fingertip-markers --guide-sequence twinkle --guide-first-onset 2.000 --guide-notes-per-second 2.000
   ```

### Holdout 数据

1. `holdout/01_rest_single_hand`
   右手自然放在键盘区域 15 秒，不弹。  
   目标：不应触发任何音。

   ```bash
   .venv/bin/python main.py --camera-source record3d --mode piano --paper-keyboard --record-session data/sessions/benchmarks_3d/twinkle_right_hand/holdout/01_rest_single_hand --record-immediately --instrument-roi 0.05,0.45,0.95,0.95 --piano-left-trim-keys 1.5 --piano-sensitivity balanced --tracking-max-width 320 --hand-model-complexity 0 --window-width 640 --window-height 480 --no-fingertip-markers
   ```

2. `holdout/02_twinkle_normal_take1`
   右手单手跟随引导，正常速度弹《小星星》1 遍。

   ```bash
   .venv/bin/python main.py --camera-source record3d --mode piano --paper-keyboard --record-session data/sessions/benchmarks_3d/twinkle_right_hand/holdout/02_twinkle_normal_take1 --record-immediately --instrument-roi 0.05,0.45,0.95,0.95 --piano-left-trim-keys 1.5 --piano-sensitivity balanced --tracking-max-width 320 --hand-model-complexity 0 --window-width 640 --window-height 480 --no-fingertip-markers --guide-sequence twinkle --guide-first-onset 2.000 --guide-notes-per-second 2.000
   ```

3. `holdout/03_twinkle_normal_take2`
   右手单手跟随引导，正常速度弹《小星星》1 遍。

   ```bash
   .venv/bin/python main.py --camera-source record3d --mode piano --paper-keyboard --record-session data/sessions/benchmarks_3d/twinkle_right_hand/holdout/03_twinkle_normal_take2 --record-immediately --instrument-roi 0.05,0.45,0.95,0.95 --piano-left-trim-keys 1.5 --piano-sensitivity balanced --tracking-max-width 320 --hand-model-complexity 0 --window-width 640 --window-height 480 --no-fingertip-markers --guide-sequence twinkle --guide-first-onset 2.000 --guide-notes-per-second 2.000
   ```

4. `holdout/04_twinkle_natural_variation`
   右手单手跟随引导，正常速度弹《小星星》1 遍。  
   手型和动作尽量自然，不要刻意配合算法。

   ```bash
   .venv/bin/python main.py --camera-source record3d --mode piano --paper-keyboard --record-session data/sessions/benchmarks_3d/twinkle_right_hand/holdout/04_twinkle_natural_variation --record-immediately --instrument-roi 0.05,0.45,0.95,0.95 --piano-left-trim-keys 1.5 --piano-sensitivity balanced --tracking-max-width 320 --hand-model-complexity 0 --window-width 640 --window-height 480 --no-fingertip-markers --guide-sequence twinkle --guide-first-onset 2.000 --guide-notes-per-second 2.000
   ```

## 8. 每段录完后检查文件

每个 session 目录至少应该包含：

```text
metadata.json
frames.jsonl
raw_video.avi
summary.json
```

带引导的弹奏片段还应该包含：

```text
annotations.csv
guide_spec.json
```

可以用下面命令快速查看已录制内容：

```bash
find data/sessions/benchmarks_3d/twinkle_right_hand -maxdepth 3 -type f | sort
```

## 9. 录完后做一次评估检查

录完 `dev` 后运行：

```bash
.venv/bin/python tools/twinkle_benchmark.py evaluate --split dev
```

录完 `holdout` 后运行：

```bash
.venv/bin/python tools/twinkle_benchmark.py evaluate --split holdout
```

评估结果输出在：

```text
data/benchmarks/twinkle_eval/
```

重要字段含义：

- `matched`：正确命中的期望音符数。
- `misses`：漏掉的期望音符。
- `extras`：多触发的音符。
- `wrong_near`：期望音符附近触发了错误音高。
- `guardrail_hits`：静止或悬停片段里的误触数量。

评估结果只是用来确认数据能正常回放，不要求录制同学根据算法结果调参。

## 10. 交付标准

录制完成后，请确认：

- `dev` 下 7 个 session 都存在。
- `holdout` 下 4 个 session 都存在。
- 所有弹奏片段都有 `annotations.csv`。
- 所有 session 都有 `frames.jsonl` 和 `raw_video.avi`。
- 相机、键盘位置在整组录制中尽量保持一致。
- 如果某段明显录坏，只重录该段。

交付时保留整个目录：

```text
data/sessions/benchmarks_3d/twinkle_right_hand/
```

不要只交付视频文件，因为算法离线调试需要 `frames.jsonl`、`annotations.csv`、metadata 和 recorded diagnostics。

## 11. 常见问题

### 看不到引导面板

确认录制命令里有 `--guide-sequence`。静止和悬停片段没有引导面板是正常的。

### 第一个音来不及弹

默认第一个音在录制开始后 2 秒。如果仍然来不及，可以打印命令时加长倒计时：

```bash
.venv/bin/python tools/twinkle_benchmark.py record-commands --guide-first-onset 3.0
```

然后使用新打印出的命令录制。

### 弹奏速度跟不上

优先跟随引导。`dev/05_twinkle_slow` 已经是较慢版本。正常速度片段尽量接近引导即可，轻微节奏偏差可以接受。

### 录错了一个音怎么办

如果只是很小的节奏偏差，可以继续录。如果音符明显弹错、漏弹很多、手离开画面、相机移动，请结束该段并重新运行同一条命令重录。

### 录制窗口里虚拟键盘没对齐

不要临时改代码。先确认命令中包含：

```bash
--instrument-roi 0.05,0.45,0.95,0.95 --piano-left-trim-keys 1.5
```

如果实际纸键盘位置明显不同，请记录问题并反馈，不要自行大量试参数。
