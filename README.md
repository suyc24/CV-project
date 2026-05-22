# AirDesk Instrument

AirDesk Instrument 是一个基于普通电脑摄像头的桌面虚拟乐器原型。它使用 OpenCV 读取实时画面，MediaPipe 追踪手部 21 个关键点，基于指尖向下速度检测“敲击”，用 Pygame 播放程序合成的鼓和钢琴音色，并提供一个无需键盘的手势 loop station。

当前版本实现两个核心能力：

- **Air Loop Station**：握拳开始/停止录制，张开手掌播放/暂停循环，拇指向上清空循环。
- **Velocity-Sensitive 视觉力度感应**：追踪双手 10 个指尖，并根据指尖相对下落速度估计音量。

## 功能列表

- OpenCV 默认摄像头实时输入。
- MediaPipe Hands 单手/双手关键点追踪。
- Drum 模式：6 个虚拟鼓 pad：KICK、SNARE、HIHAT、TOM1、TOM2、CRASH。
- Piano 模式：7 个大白键，C4 到 B4。
- Piano 模式使用程序生成的透视钢琴平面，默认贴在画面下方桌面区域。
- 指尖抬起/下落状态机、落点区域、下落幅度、速度和 cooldown 联合判断有效敲击。
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

Record3D / iPhone LiDAR 是可选输入源，不放进默认依赖里，避免没有 iPhone 的环境安装失败。需要 RGB-D 输入时额外安装：

```bash
pip install -r requirements-record3d.txt
```

## 运行

```bash
python main.py --camera 0 --mode drum --debug
```

也可以切换到 piano：

```bash
python main.py --camera 0 --mode piano
```

参数：

- `--camera-source webcam|record3d`：输入源，默认 `webcam`。`record3d` 使用 iPhone/iPad 的 RGB-D 流。
- `--camera 0`：摄像头索引。
- `--mode drum|piano`：启动模式，默认 `drum`。
- `--debug`：显示每个触发指尖的 y 方向平滑速度和 pressed 状态。
- `--display-scale 1.25`：放大 OpenCV 显示窗口，不改变检测坐标。
- `--window-width/--window-height`：指定显示窗口尺寸。画面会等比例缩放并补黑边，不会强行拉伸。
- `--fullscreen`：全屏显示。
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
- `--tracking-max-width 416`：送入 MediaPipe 的最大图像宽度，越小越快但细节更少。检测不到手指时可以试 `480` 或 `640`。
- `--no-tracking-roi`：关闭桌面附近 ROI 追踪，改为整帧追踪。默认模式会先查 ROI，失败时自动全帧重捕获。
- `--landmark-smoothing-alpha 0.72`：landmark 时序平滑系数。越小越稳但越慢，越大越灵敏但越抖。
- `--no-optical-stabilization`：关闭指尖光流稳定层和短时丢手桥接。默认开启，用来抑制 MediaPipe 单帧跳点。
- `--piano-sensitivity stable|balanced|sensitive`：钢琴触发预设。默认 `stable` 更抗抖，`sensitive` 更容易触发轻敲。
- `--max-hands 2`：最多追踪几只手。
- `--no-hand-cutout`：不把真实手部抠回到钢琴图层上方，可换取更高 FPS。
- `--no-fingertip-markers`：隐藏指尖圆点。
- `--trigger-thumb`：允许拇指触发音符；当前默认已启用，保留这个参数用于兼容旧命令。
- `--no-trigger-thumb`：关闭拇指触发。如果某个摄像头角度下拇指 landmark 抖动误触，可以临时使用。
- `--min-detection-confidence 0.45` / `--min-tracking-confidence 0.45`：MediaPipe 手部检测/追踪置信度。数值越低越容易找回手，但误检风险更高。
- `--record-session data/sessions/test01`：保存可离线回放的 session 数据。
- `--no-record-video`：只保存 landmarks/diagnostics JSONL，不保存 AVI 视频。
- `--record3d-device 0`：Record3D 设备索引。
- `--record3d-rotate 0|90|180|270`：旋转 Record3D 画面，用来适配手机横竖屏安装方向。
- `--record3d-mirror`：水平镜像 Record3D 画面。
- `--record3d-depth-unit auto|m|cm|mm`：深度单位，默认自动判断。
- `--depth-contact-mode auto|off|assist|required`：RGB-D 接触判定模式。`assist` 只在深度明确显示手指离桌面很高时拦截；`required` 要求明确接触才触发。
- `--depth-contact-threshold 0.075`：指尖高于桌面多少米以内算接触。
- `--depth-release-threshold 0.14`：指尖高于桌面多少米以上算明确离桌面。

摄像头诊断：

```bash
python main.py --list-cameras
```

Record3D 设备诊断：

```bash
python main.py --camera-source record3d --list-cameras
```

## Record3D RGB-D 输入

Record3D 适合有 LiDAR 的 iPhone/iPad。推荐使用 USB streaming，不建议用 Wi-Fi 做触键主输入。

手机端：

- 安装并打开 Record3D。
- 用 USB 连接电脑，并在手机上信任电脑。
- 在 Record3D 里开启 Live RGBD Video Streaming over USB。
- 把手机固定在桌面侧上方或斜上方，能同时看到手指和虚拟键盘区域。

电脑端运行：

```bash
python main.py --camera-source record3d --record3d-device 0 --mode piano --debug
```

如果手机画面方向不对：

```bash
python main.py --camera-source record3d --mode piano --record3d-rotate 90 --debug
```

启动后先把手移开虚拟琴键区域，按 `d` 校准桌面深度。校准期间琴键会暂时隐藏，校准成功后才会在画面下方生成一块贴合桌面角度的钢琴平面。之后系统仍然用 RGB/MediaPipe 追踪指尖 `x/y`，但会用 Record3D depth 判断指尖是否接近桌面平面。

默认 `--depth-contact-mode auto` 在 Record3D 下等价于 `assist`：深度信息只作为防误触辅助。如果你想让触发必须满足深度接触，使用：

```bash
python main.py --camera-source record3d --mode piano --depth-contact-mode required --debug
```

`required` 更产品化但也更挑摄像头摆位和深度质量。如果漏触多，先用 `assist`。

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

如果追踪太慢，先关闭软件增强、降低 MediaPipe 输入宽度，并在只测单手时把 `--max-hands` 设为 1：

```bash
python main.py --camera 0 --backend dshow --mode piano --enhance none --tracking-max-width 360 --max-hands 1
```

如果只是想优先验证触键逻辑，也可以临时关掉手部抠图：

```bash
python main.py --camera 0 --backend dshow --mode piano --no-hand-cutout
```

如果追踪抖动明显，把 smoothing alpha 调小：

```bash
python main.py --camera 0 --backend dshow --mode piano --landmark-smoothing-alpha 0.55
```

如果手停在键盘上不动仍然误触发，保持默认 `--piano-sensitivity stable`，并确认没有加 `--no-optical-stabilization`。如果稳定后轻敲变得不够灵敏，再试：

```bash
python main.py --camera 0 --backend dshow --mode piano --piano-sensitivity balanced
python main.py --camera 0 --backend dshow --mode piano --piano-sensitivity sensitive
```

Record3D 已经校准深度后，如果手停在桌面上仍然因为 landmark 抖动误触，优先试更严格的深度模式。默认 `assist` 不会强制用 depth 判断 release，因为部分录像里 depth 会把真实抬指也看成 contact；`required` 会更稳但更挑桌面校准质量：

```bash
python main.py --camera-source record3d --mode piano --depth-contact-mode required --debug
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
- `d`：采集多帧 Record3D depth，校准桌面深度并生成桌面贴合钢琴平面。校准时手需要离开琴键区域。
- `o`：运行时旋转画面 90 度。窗口会自动适配新的横竖比例；旋转后会清空旧 depth 校准，需要重新按 `d`。
- `+` / `-`：放大 / 缩小显示窗口。
- `f`：切换全屏。
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
- `raw_video.avi`：原始摄像头帧，不包含 UI 叠加。如果录制期间切换横竖屏，视频会等比例补边，避免被 OpenCV 裁切或拉伸。
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
python replay_session.py data/sessions/test01 --piano-strike-velocity 100 --piano-strike-drop 16
python replay_session.py data/sessions/test01 --depth-contact-mode required --output-prefix replay_required
```

回放会生成 `replay_hits.csv`、`replay_miss_reasons.csv` 和 `replay_summary.json`。

## 视频标注流程

推荐先用 AirDesk 自己录 session，这样视频、每帧 hand landmarks、zones 和诊断信息都会在同一个目录里，后续我可以直接离线优化：

```bash
python main.py --camera-source record3d --mode piano --debug --record-session data/sessions/ipad_test01
```

如果你用别的软件录视频，也可以把视频放进 `data/sessions/ipad_test01/raw_video.avi`，但最好仍然用同一摄像机视角、同一 iPad 摆位。

最省事的标注方式是准备一个只有音符顺序的 score 文件，不需要写时间戳：

```csv
note
C4
D4
E4
F4
G4
A4
B4
```

项目里有一个示例：[data/annotation_score_example.csv](/home/suyc24/Python/CV-project/data/annotation_score_example.csv)。

打开标注器：

```bash
python tools/annotate_piano_video.py data/sessions/ipad_test01 --score data/sessions/ipad_test01/score.csv
```

标注器会播放 `raw_video.avi`，输出 `data/sessions/ipad_test01/annotations.csv`。操作很少：

- `space`：播放 / 暂停。
- `Enter`：把当前帧标成 score 里的下一个音。
- `1-7`：不用 score 时，直接标 C4、D4、E4、F4、G4、A4、B4。
- `,` / `.`：前进 / 后退一帧。
- `[` / `]`：后退 / 前进 1 秒。
- `Backspace`、`u` 或 `z`：撤销上一个标注。
- `s`：保存。
- `q`：保存并退出。

生成标注后，先跑当前检测器：

```bash
python replay_session.py data/sessions/ipad_test01 --output-prefix baseline
```

再和人工标注对齐评测：

```bash
python tools/evaluate_hit_events.py \
  --pred data/sessions/ipad_test01/baseline_hits.csv \
  --gt data/sessions/ipad_test01/annotations.csv \
  --gt-time-column onset \
  --match-notes \
  --tolerance 0.08 \
  --output data/sessions/ipad_test01/baseline_eval.json
```

这会得到 precision、recall、F1、误触、漏触和平均时间误差。后续优化时我会用这个 `annotations.csv` 当标准答案，而不是凭肉眼猜。

外部数据集评测入口：

```bash
python tools/download_pianovam_sample.py --output data/external/pianovam
python tools/evaluate_hit_events.py --pred data/sessions/test01/replay_hits.csv --gt data/external/pianovam/<labels>.tsv --gt-time-column onset --tolerance 0.08
```

当前优先参考 PianoVAM，因为它比普通手势数据集更接近“钢琴演奏视频 + 按键 onset 真值”。下载脚本默认只拉标签和手部骨架元数据；如果要拉原始视频，加 `--include-video`，但会明显占用带宽和磁盘。评测脚本做的是事件级 precision/recall/F1，不会假装用没有人工标注的测试视频算 99% 准确率。

生成 HTML 分析报告：

```bash
python analysis_report.py data/sessions/test01
```

报告会输出 `report.html`，并额外生成 `frame_metrics.csv`。把一个 session 目录发给协作者后，就可以不用重新连接你的摄像头，直接离线调参和验证。

重新导出当前 session 的截图：

```bash
python extract_session_frames.py data/sessions/test01 --count 5 --include-hit-frames
python extract_session_frames.py data/sessions/test02 --count 5 --include-hit-frames
```

这会重建 `extracted_frames/`。默认会在截图上叠加当前 `frames.jsonl` 里的琴键 polygon、指尖和 hit 标记。如果只想看原始摄像头画面，加 `--no-overlay`。

钢琴触发灵敏度：

- Piano 模式使用单独的触键状态机参数，主要包括 `PIANO_STRIKE_MIN_VELOCITY`、`PIANO_STRIKE_MIN_DROP_PX` 和 `PIANO_RELEASE_LIFT_PX`。
- Piano 会使用双手 10 个指尖：拇指、食指、中指、无名指、小指。
- 拇指默认也能触发琴键。如果某个摄像头角度下拇指 landmark 抖动误触，可以加 `--no-trigger-thumb` 临时关闭。
- Piano 不再要求指尖落到某条 contact line 以下；只要一次有效下落的落点位于琴键区域内，就可以触发。
- Piano 可视琴键默认覆盖整个底部演奏面。实际命中区域只比可视琴键略大，用来容忍边界误差。
- Piano 使用 `lifting -> raised -> falling -> pressed` 状态机，更接近真实钢琴的“抬起、下落、触键”动作。
- `lifting` 必须累计至少 `PIANO_ARM_MIN_LIFT_PX` 的抬起幅度才会进入可触发状态；`pressed` 后也要连续 `PIANO_RELEASE_STABLE_FRAMES` 帧满足抬起距离才会 release。这样可以过滤 MediaPipe 在手指贴近镜头或停在键盘上时的轻微抖动。
- 如果某个指尖点明显偏离光流预测，或刚从短暂丢手中恢复，系统会把该指尖标记为 `unstable_tracking`。这类帧不会更新下落状态，也不会触发琴键，避免把 MediaPipe 跳点当成敲击。
- 如果还是触发困难，可以继续降低 `config.py` 中的 `PIANO_STRIKE_MIN_VELOCITY`，例如从当前默认 `80` 调到 `70`，或把 `PIANO_STRIKE_MIN_DROP_PX` 从当前默认 `12` 调到 `10`。
- 如果误触发较多，优先增大 `PIANO_ARM_MIN_LIFT_PX` 或 `PIANO_RELEASE_STABLE_FRAMES`；其次再调高 `PIANO_STRIKE_MIN_VELOCITY`，或增大 `PIANO_STRIKE_MIN_DROP_PX` / `PIANO_RELEASE_LIFT_PX`。

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
- 敲击时让任意指尖先抬起一点再向下落到琴键区域内，下一次敲击前也需要有明显抬起。
- 如果误触发较多，增大 `config.py` 中的 `HIT_VELOCITY_THRESHOLD` 或 `HIT_COOLDOWN`。

## 算法说明

### Hand Tracking

`hand_tracker.py` 使用 MediaPipe Hands 获取每只手的 21 个 landmarks，并把归一化坐标转换为像素坐标。UI 默认不绘制骨架线，而是用 landmarks 估计手部区域，把真实摄像头里的手抠回到钢琴图层上方，并用小圆点标出 10 个指尖。

为了减少指尖靠近镜头时的跳点，当前版本会先做稳定 hand id 分配，再对指尖和指根等关键点使用 OpenCV Lucas-Kanade 光流进行时序稳定：MediaPipe 给出粗位置，光流提供相邻帧连续运动估计；光流还会做 forward-backward 一致性检查，过滤“前向能跟上、反向回不去”的不可靠点。当 MediaPipe 与光流差异过大时，系统会降低单帧 MediaPipe 点的权重，并把对应 landmark 标记为不稳定，禁止这一帧触发音符。

为了提高实时性，当前版本默认只把画面下方桌面附近 ROI 送入 MediaPipe，并把输入宽度限制到 `TRACKING_MAX_WIDTH`。如果 ROI 内没检测到手，会自动用整帧再跑一次重捕获。若 MediaPipe 连续少量帧丢手，系统会用上一帧 landmarks 的光流预测短暂桥接，最多 `TRACKING_BRIDGE_MAX_FRAMES` 帧；桥接帧只用于视觉连续性，不允许触发琴键或手势控制。

### Velocity-Sensitive Hit Detection

`hit_detector.py` 为每个 `hand_id + finger_id` 维护独立状态，包括上一帧位置、上一帧时间、y 方向平滑速度、pressed 状态、最近击打时间和轨迹。

当前 UI 默认显示 MediaPipe 的 5 个指尖 landmark：

- 拇指：`4`
- 食指：`8`
- 中指：`12`
- 无名指：`16`
- 小指：`20`

双手同时出现时，每只手都会显示并检测这 5 个指尖：

- 拇指：`4`
- 食指：`8`
- 中指：`12`
- 无名指：`16`
- 小指：`20`

如果需要临时降低拇指误触风险，运行时加 `--no-trigger-thumb`，或把 `config.py` 的 `TRIGGER_FINGER_IDS` 改成 `(8, 12, 16, 20)`。

每帧还会记录诊断原因，方便离线分析：

- `no_zone`：指尖不在任何键/pad 区域。
- `velocity`：向下速度不够。
- `armed`：指尖已抬起，等待下落。
- `lifted`：下落过程中又上抬，本次候选敲击被取消。
- `short_drop`：下落幅度不够。
- `strike_velocity`：下落距离够，但敲击速度不够。
- `pressed`：还没有明显抬起，不能重复触发。
- `cooldown`：距离上次触发太近。
- `unstable_tracking`：该指尖刚丢失、刚重捕获，或与光流预测严重冲突。
- `jitter_guard`：单帧跳点或净下落证据不足，本次候选触发被防抖层拦截。
- `hit`：本帧触发成功。

核心速度计算：

```python
velocity_y = (current_y - previous_y) / dt
```

图像坐标中 y 向下增大，所以向下敲击时 `velocity_y` 为正。Piano 模式还会计算“指尖相对本手指根部关节”的 y 位移和速度，避免整只手或拇指 landmark 抖动被误判成食指敲击。系统使用指数平滑降低抖动，并为每个触发指尖维护一个小状态机：

- `idle`：指尖不在当前键附近。
- `lifting`：指尖正在上抬，但累计抬起幅度还不够，暂时不能触发。
- `raised/armed`：指尖在键面上方，或已经完成足够明确的抬起动作。
- `falling`：指尖开始向下运动，并记录这次下落的最高点和最大下落速度。
- `pressed`：指尖完成有效下落并触发音符，随后需要明显抬起才会再次触发。

一次有效 piano hit 需要同时满足：

- 指尖位于某个 pad/key 区域内；
- 指尖先出现抬起/armed 或正在向下落；
- 下落位移超过 `PIANO_STRIKE_MIN_DROP_PX`；
- 下落速度超过 `PIANO_STRIKE_MIN_VELOCITY`；
- 当前落点位于某个琴键 polygon 内；
- 当前指尖未处于 pressed 状态；
- 距离上次 hit 超过 `HIT_COOLDOWN`。

同一只手在同一帧里如果多个指尖都像是命中，会只保留分数最高的那个候选。这个仲裁可以减少“抬食指却触发到拇指对应键”的错配，但也会牺牲一点单手同帧和弦能力。

hit 后进入 pressed 状态。只有当指尖明显抬起，或离开当前区域，才允许下一次触发。Record3D 使用 `--depth-contact-mode required` 时，release 还会检查指尖是否仍然贴近桌面；如果 depth 认为还在接触，就不会因为 2D landmark 抖动而解除 pressed 状态。

FPS 低时通常不是摄像头本身慢，而是实时管线里有几项很吃 CPU：MediaPipe 手部追踪、透视钢琴图层合成、手部抠图、debug 清晰度指标和 session 录制。当前版本已经缓存钢琴透视贴图、降低默认 MediaPipe 输入宽度、降低手部 mask 模糊半径，并让 debug 画质指标按间隔采样。如果仍然低于 20 FPS，优先尝试 `--max-hands 1 --tracking-max-width 360 --no-hand-cutout`。

### RGB-D Contact Gating

`rgbd_camera.py` 把 Record3D 的 callback 流包装成 `read()` 接口，输出 RGB、depth、confidence 和 timestamp。`depth_contact.py` 负责桌面接触估计：

- 按 `d` 时记录当前虚拟琴键区域的桌面 depth baseline；
- 每帧把 depth resize 到 RGB 画面尺寸；
- 对每个指尖采样局部 median depth；
- 计算 `height_above_desk = desk_depth - finger_depth`；
- 如果高度小于 `DEPTH_CONTACT_THRESHOLD_M`，认为指尖接近桌面；
- 如果高度大于 `DEPTH_RELEASE_THRESHOLD_M`，在 `assist` 模式下会拦截这次触发。

这不是精确物理深度重建，而是用 RGB-D 作为“接触概率”的额外证据。产品上它比单目 `y` 速度更稳，尤其能减少手停在键盘上时的误触。

音量映射：

```python
volume = clamp(
    (velocity_y - HIT_MIN_VELOCITY) / (HIT_MAX_VELOCITY - HIT_MIN_VELOCITY),
    0.2,
    1.0,
)
```

Piano 模式单独使用更低的视觉速度范围：`PIANO_HIT_MIN_VELOCITY`、`PIANO_HIT_MAX_VELOCITY` 和 `PIANO_MIN_VOLUME`。这是因为钢琴状态机使用的是相对指尖运动，数值通常显著低于 drum 的屏幕 y 方向速度。

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
- Record3D/LiDAR depth 分辨率低于 RGB，指尖边缘会有噪声和空洞，所以默认只作为辅助证据。
- Record3D 的 RGB/depth 对齐、画面旋转和镜像依赖手机安装方向；必要时用 `--record3d-rotate` 和 `--record3d-mirror` 调整。
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
- 音量变化不明显：drum 调整 `HIT_MIN_VELOCITY` / `HIT_MAX_VELOCITY`；piano 调整 `PIANO_HIT_MIN_VELOCITY` / `PIANO_HIT_MAX_VELOCITY`。
- 手势太敏感：提高 `GESTURE_STABLE_FRAMES` 或 `GESTURE_COOLDOWN`。

## 自动测试

不依赖摄像头的合成 hit detector 测试：

```bash
python tests/test_hit_detector_synthetic.py
```
