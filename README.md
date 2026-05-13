# AirDesk Instrument

AirDesk Instrument 是一个基于普通电脑摄像头的桌面虚拟乐器原型。它使用 OpenCV 读取实时画面，MediaPipe 追踪手部 21 个关键点，基于指尖向下速度检测“敲击”，用 Pygame 播放程序合成的鼓和钢琴音色，并提供一个无需键盘的手势 loop station。

当前版本实现两个核心能力：

- **Air Loop Station**：握拳开始/停止录制，张开手掌播放/暂停循环，拇指向上清空循环。
- **Velocity-Sensitive 视觉力度感应**：根据双手 10 个指尖的 y 方向运动速度估计相对力度，并映射为音量。

## 功能列表

- OpenCV 默认摄像头实时输入。
- MediaPipe Hands 单手/双手关键点追踪。
- Drum 模式：6 个虚拟鼓 pad：KICK、SNARE、HIHAT、TOM1、TOM2、CRASH。
- Piano 模式：两组八度白键，C4 到 C6。
- Piano 模式使用程序生成的正俯视钢琴 keybed，默认占满画面底部宽度。
- 指尖速度、触发线、release line、cooldown 联合判断有效敲击。
- 速度到音量的相对力度映射。
- 自包含音频合成，不依赖外部 wav/mp3 文件。
- Loop 录制、停止、播放/暂停、清空。
- 手势控制 loop，并带稳定帧与 cooldown 防抖。
- UI 显示虚拟区域、真实手部抠图、指尖标记、FPS、当前模式、loop 状态、最近触发音符、速度和音量。

## 安装

建议使用 Python 3.10+。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

如果启动时报 `module 'mediapipe' has no attribute 'solutions'`，通常是安装到了不兼容 legacy `solutions` API 的新版 MediaPipe。请在虚拟环境中强制重装 requirements：

```bash
pip install --upgrade --force-reinstall -r requirements.txt
```

Windows 下建议使用 `mediapipe==0.10.21`。`0.10.30` 在部分 Windows/Python 组合上会在 Tasks API 初始化时报 `function 'free' not found`。如果已经装到了 0.10.30，请执行：

```bash
pip uninstall -y mediapipe
pip install --no-cache-dir --force-reinstall -r requirements.txt
python -c "import mediapipe as mp; print(mp.__version__)"
```

确认输出是 `0.10.21` 后再运行项目。

当前代码同时兼容旧版 `solutions.hands` 和新版 MediaPipe Tasks Hand Landmarker。第一次使用新版 Tasks API 时，程序会自动下载 `models/hand_landmarker.task`。如果网络下载失败，可以手动下载：

```bash
mkdir models
curl -L https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task -o models/hand_landmarker.task
```

代码会把 `.task` 模型读成内存 buffer 再交给 MediaPipe，避免 Windows 路径中包含盘符、空格或中文目录时被 MediaPipe 拼成错误路径。

## 运行

```bash
python main.py --camera 0 --mode drum --debug
```

也可以切换到 piano：

```bash
python main.py --camera 0 --mode piano
```

参数：

- `--camera 0`：摄像头索引。
- `--mode drum|piano`：启动模式，默认 `drum`。
- `--debug`：显示每个触发指尖的 y 方向平滑速度和 pressed 状态。
- `--list-cameras`：列出当前系统可见的摄像头设备并退出。
- `--calibrate-camera`：自动测试曝光/可选对焦，按画面清晰度和过曝比例保存 `camera_profile.json`。
- `--camera-profile camera_profile.json`：读取/写入摄像头 profile。正常运行时如果该文件存在会自动加载。
- `--no-camera-profile`：忽略已保存的摄像头 profile。
- `--show-camera-profile`：打印最终会应用的摄像头参数并退出，用来确认 profile 是否生效。
- `--quality fast|balanced|high|max`：快速分辨率预设。`balanced` 是 1280x720，`high` 是 1920x1080。
- `--backend auto|dshow|msmf|v4l2`：指定 OpenCV 摄像头后端。Windows 推荐 `dshow`。
- `--width 1280 --height 720 --fps 30`：请求摄像头分辨率和帧率。默认使用 1280x720/30FPS；MediaPipe 输入仍会被缩小以保速度。
- `--manual-exposure --exposure -6`：关闭自动曝光并设置曝光值，Windows 摄像头常见可用范围约为 `-4` 到 `-8`。
- `--auto-exposure`：重新启用摄像头自动曝光。
- `--brightness/--contrast/--gain`：可选摄像头参数覆盖，不同摄像头支持程度不同。
- `--enhance auto|clahe|none`：软件亮度/对比度增强，默认 `none`。画面调参时建议先保持 `none`，避免把噪声和过曝边缘拉爆。
- `--tracking-max-width 480`：送入 MediaPipe 的最大图像宽度，越小越快但细节更少。
- `--no-tracking-roi`：关闭桌面附近 ROI 追踪，改为整帧追踪。
- `--landmark-smoothing-alpha 0.72`：landmark 时序平滑系数。越小越稳但越慢，越大越灵敏但越抖。
- `--max-hands 2`：最多追踪几只手。
- `--min-detection-confidence 0.55` / `--min-tracking-confidence 0.55`：MediaPipe 手部检测/追踪置信度。
- `--record-session data/sessions/test01`：保存可离线回放的 session 数据。
- `--no-record-video`：只保存 landmarks/diagnostics JSONL，不保存 AVI 视频。

摄像头诊断：

```bash
python main.py --list-cameras
```

摄像头自动校准：

```bash
python main.py --camera 0 --backend dshow --calibrate-camera --mode piano
```

校准时把摄像头对准实际演奏区域，桌面上最好放一张有文字或纹理的纸，并把手放到画面下半部分。程序会扫描多个分辨率和曝光值，计算亮度、过曝比例和 Laplacian 清晰度分数，保存最佳结果到 `camera_profile.json`。之后直接运行主程序会自动加载这个 profile：

```bash
python main.py --camera 0 --backend dshow --mode piano --debug
```

如果怀疑 profile 没有应用，先打印最终参数：

```bash
python main.py --camera 0 --backend dshow --show-camera-profile
```

运行主程序启动时也会打印 `Loaded camera profile`、`Profile settings` 和实际打开到的 `Camera: requested=... actual=...`。程序会在摄像头 warmup 后重新应用一次 profile，减少 DirectShow 自动改回曝光的概率。

如果摄像头支持手动对焦，也可以一起扫 focus：

```bash
python main.py --camera 0 --backend dshow --calibrate-camera --calibrate-focus --mode piano
```

如果要自定义曝光候选值，PowerShell 中建议用等号，避免负数被解析成参数：

```bash
python main.py --camera 0 --backend dshow --calibrate-camera --calibration-exposures=-4,-5,-6,-7,-8
```

如果要强制高分辨率运行：

```bash
python main.py --camera 0 --backend dshow --mode piano --quality high --tracking-max-width 640 --debug
```

如果高分辨率画面清楚但 FPS 下降，保持 `--quality high`，把 MediaPipe 输入降下来：

```bash
python main.py --camera 0 --backend dshow --mode piano --quality high --tracking-max-width 480 --enhance none
```

Windows 下如果诊断看到 `index 0: OK`，但后面继续出现其他 index 的 warning，通常只是 OpenCV 在探测不存在的摄像头编号。直接使用可用编号运行即可：

```bash
python main.py --camera 0 --mode piano
```

推荐 Windows 调试命令：

```bash
python main.py --camera 0 --backend dshow --mode piano --debug --quality balanced --manual-exposure --exposure -6 --enhance none
```

如果画面仍然过亮，优先尝试：

```bash
python main.py --camera 0 --backend dshow --mode piano --manual-exposure --exposure -8 --gain 0 --enhance none
```

如果画面太暗，尝试把曝光调回 `-4`，或者重新启用自动曝光：

```bash
python main.py --camera 0 --backend dshow --mode piano --manual-exposure --exposure -4
python main.py --camera 0 --backend dshow --mode piano --auto-exposure
```

如果追踪太慢，先关闭软件增强并进一步降低 MediaPipe 输入宽度：

```bash
python main.py --camera 0 --backend dshow --mode piano --enhance none --tracking-max-width 480
```

如果追踪抖动明显，把 smoothing alpha 调小：

```bash
python main.py --camera 0 --backend dshow --mode piano --landmark-smoothing-alpha 0.55
```

如果内置摄像头照不到桌面，可以先用空中测试模式验证软件链路：

```bash
python main.py --camera 0 --backend dshow --mode piano --air-test --debug
```

`--air-test` 会把虚拟琴键/鼓垫移动到画面中上部，并让追踪覆盖整帧。这个模式适合验证手部追踪、力度触发和 loop station，但不等价于桌面俯拍交互。

## 操作说明

键盘备用控制：

- `q`：退出。
- `m`：切换 drum/piano。
- `r`：清空 loop。
- `space`：播放/暂停 loop。
- `e`：开始/停止录制 loop，作为手势录制的备用控制。
- `[` / `]`：运行时降低 / 提高曝光，方便现场微调画面。
- `a`：切换自动曝光。
- `p`：把当前摄像头参数保存到 `camera_profile.json`。

调画面时建议打开 `--debug`。右上角会显示原始帧的 `luma`、`over` 和 `sharp`：`over` 越接近 0 越好，`sharp` 越高通常越清楚。

## 录制、回放与分析

录制可复现 session：

```bash
python main.py --camera 0 --backend dshow --mode piano --debug --record-session data/sessions/test01
```

session 目录包含：

- `metadata.json`：运行参数、摄像头设置、模式。
- `frames.jsonl`：每帧 landmarks、zones、hit events、miss reasons、亮度/清晰度指标。
- `raw_video.avi`：原始摄像头帧，不包含 UI 叠加。
- `summary.json`：帧数和时长。

如果只想保存轻量数据，不保存视频：

```bash
python main.py --camera 0 --backend dshow --mode piano --record-session data/sessions/test01 --no-record-video
```

离线回放同一段 landmarks，并测试当前 hit detector：

```bash
python replay_session.py data/sessions/test01
```

也可以临时扫参数，不改源码：

```bash
python replay_session.py data/sessions/test01 --piano-velocity-threshold 140 --piano-press-ratio 0.55
```

回放会生成 `replay_hits.csv`、`replay_miss_reasons.csv` 和 `replay_summary.json`。

生成 HTML 分析报告：

```bash
python analysis_report.py data/sessions/test01
```

报告会输出 `report.html`，并额外生成 `frame_metrics.csv`。把一个 session 目录发给协作者后，就可以不用重新连接你的摄像头，直接离线调参和验证。

钢琴触发灵敏度：

- Piano 模式使用单独的速度阈值 `PIANO_HIT_VELOCITY_THRESHOLD`，默认比 drum 更低。
- Piano 会追踪双手 10 个指尖：拇指、食指、中指、无名指、小指。
- Piano 的 press line 默认在按键高度的 40%，release line 在 35%。也就是指尖落入按键下方约 60% 高度范围时可触发。
- Piano 可视琴键默认覆盖整个底部演奏面。实际命中区域只比可视琴键略大，用来容忍边界误差。
- Piano 除了常规速度阈值，还支持“向下穿越 press line”的低速触发，降低慢按漏触发。
- 如果还是触发困难，可以继续降低 `config.py` 中的 `PIANO_HIT_VELOCITY_THRESHOLD`，例如从 `150` 调到 `130`。
- 如果误触发较多，调高 `PIANO_HIT_VELOCITY_THRESHOLD`，或把 `PIANO_PRESS_RATIO` 从 `0.40` 调到 `0.50`。

手势控制：

- `FIST` 握拳：开始录制 / 停止录制。
- `OPEN_PALM` 张开手掌：播放 / 暂停 loop。
- `THUMB_UP` 拇指向上：清空 loop。这个手势受摄像头角度影响较大，当前实现属于 experimental。

手势必须连续稳定若干帧才会触发，触发后需要离开该手势或换成其他手势才会再次触发，避免连续帧重复触发。

## 摄像头摆放建议

- 摄像头尽量俯拍桌面，画面下半部分能看到手指和桌面。
- 让桌面交互区位于画面下方 45%-90% 高度之间。
- 如果笔记本内置摄像头照不到桌面，软件无法从不可见区域恢复手指位置。推荐用外接 USB 摄像头、手机当摄像头、或小三脚架/夹臂做俯拍。
- 手机当摄像头时，把手机固定在桌面上方或侧上方，选择手机摄像头对应的 `--camera` index。画面能稳定看到手指和琴键区域，比内置摄像头可靠很多。
- 临时没有俯拍设备时，用 `--air-test` 在电脑摄像头前做空中测试，先验证声音、手势和 loop 逻辑。
- 光照尽量均匀，避免手指强反光或大片阴影。
- 如果画面像白纸一样过曝，先关自动曝光并尝试 `--exposure -6` 或 `--exposure -8`。软件增强无法恢复已经被硬件曝光打爆的细节。
- 如果手指拖影严重，降低曝光时间通常比提高分辨率更重要。
- 敲击时让任意指尖有明显的向下运动，再抬离 release line 后进行下一次敲击。
- 如果误触发较多，增大 `config.py` 中的 `HIT_VELOCITY_THRESHOLD` 或 `HIT_COOLDOWN`。

## 算法说明

### Hand Tracking

`hand_tracker.py` 使用 MediaPipe Hands 获取每只手的 21 个 landmarks，并把归一化坐标转换为像素坐标。UI 默认不绘制骨架线，而是用 landmarks 估计手部区域，把真实摄像头里的手抠回到钢琴图层上方，并用小圆点标出 10 个指尖。

为了提高实时性，当前版本默认只把画面下方桌面附近 ROI 送入 MediaPipe，并把输入宽度限制到 `TRACKING_MAX_WIDTH`。检测结果会映射回原始画面坐标。landmarks 还会经过轻量时序平滑，减少指尖抖动。

### Velocity-Sensitive Hit Detection

`hit_detector.py` 为每个 `hand_id + finger_id` 维护独立状态，包括上一帧位置、上一帧时间、y 方向平滑速度、pressed 状态、最近击打时间和轨迹。

当前 hit detector 默认追踪 MediaPipe 的 5 个指尖 landmark：

- 拇指：`4`
- 食指：`8`
- 中指：`12`
- 无名指：`16`
- 小指：`20`

双手同时出现时，每只手都会独立维护这 5 个指尖的状态。

每帧还会记录诊断原因，方便离线分析：

- `no_zone`：指尖不在任何键/pad 区域。
- `velocity`：向下速度不够。
- `press_line`：速度够但还没有落到触发线以下。
- `crossing_velocity`：穿越了触发线，但穿越速度低于低速触发阈值。
- `pressed`：还没有 release，不能重复触发。
- `cooldown`：距离上次触发太近。
- `hit`：本帧触发成功。

核心速度计算：

```python
velocity_y = (current_y - previous_y) / dt
```

图像坐标中 y 向下增大，所以向下敲击时 `velocity_y` 为正。系统使用指数平滑降低抖动。一次有效 hit 需要同时满足：

- 指尖位于某个 pad/key 区域内；
- `velocity_y > HIT_VELOCITY_THRESHOLD`；
- 指尖低于区域内 press line：`y > zone.y1 + press_ratio * zone.height`；
- 当前指尖未处于 pressed 状态；
- 距离上次 hit 超过 `HIT_COOLDOWN`。

hit 后进入 pressed 状态。只有当指尖上移到 release line 以上，或离开当前区域，才允许下一次触发。

音量映射：

```python
volume = clamp(
    (velocity_y - HIT_MIN_VELOCITY) / (HIT_MAX_VELOCITY - HIT_MIN_VELOCITY),
    0.2,
    1.0,
)
```

这里估计的是视觉上的**相对力度**，不是精确真实物理力。

### Gesture-Controlled Loop Station

`gesture_recognizer.py` 不训练模型，只用 MediaPipe landmarks 写规则：

- 食指/中指/无名指/小指：`tip_y < pip_y` 近似认为伸直。
- `FIST`：四个长手指都未伸直。
- `OPEN_PALM`：至少三个长手指伸直。
- `THUMB_UP`：拇指向上且其他长手指基本收起。

`GestureController` 要求同一手势连续稳定 `GESTURE_STABLE_FRAMES` 帧，并带 `GESTURE_COOLDOWN`。触发后必须松开或换手势才会再次触发。

`loop_station.py` 录制 `sound_id`、相对时间戳、音量和音符名。播放时按录制时的相对时间循环触发，每一轮中每个事件只播放一次，到达 loop 末尾后重置播放标记。

## 已知限制

- 单目摄像头无法得到精确真实物理力，本项目只做相对力度估计。
- MediaPipe 的 `z` 不等于真实物理深度，当前 hit detection 主要使用像素 y 方向速度。
- 光照、摄像头角度、运动模糊、遮挡都会影响 landmarks 稳定性。
- 当前版本使用固定桌面 ROI，没有做桌面平面重建或四点标定。
- `THUMB_UP` 在俯拍桌面视角下可能不如握拳和张掌稳定。

## 后续扩展

- 桌面四点标定与透视变换，把真实桌面映射到稳定坐标系。
- MIDI 输出，连接 DAW 或软件乐器。
- AR rhythm game，把节奏游戏判定和虚拟乐器结合。
- Tangible slider object tracking，用彩色物体或 ArUco marker 控制滤波器、音量、效果器。

## 调参建议

- 敲击不触发：降低 `HIT_VELOCITY_THRESHOLD`，或把摄像头放低一点，让指尖运动在画面中更明显。
- 太容易误触发：提高 `HIT_VELOCITY_THRESHOLD`，增大 `HIT_COOLDOWN`，或提高 `PRESS_RATIO`。
- 必须抬很高才能再次触发：增大 `RELEASE_RATIO`；想更严格则减小它。
- 音量变化不明显：调整 `HIT_MIN_VELOCITY` 和 `HIT_MAX_VELOCITY`。
- 手势太敏感：提高 `GESTURE_STABLE_FRAMES` 或 `GESTURE_COOLDOWN`。

## 自动测试

不依赖摄像头的合成 hit detector 测试：

```bash
python tests/test_hit_detector_synthetic.py
```
