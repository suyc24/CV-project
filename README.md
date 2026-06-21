# AirDesk Virtual Piano

基于 RGB-D 摄像头（Record3D + iPhone LiDAR）和 MediaPipe 手部追踪的实时虚拟钢琴系统。通过多级指尖稳定管线和速度敏感六态状态机实现弹奏检测，无需物理键盘。

## 安装

Python 3.10+ 推荐。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-record3d.txt  # Record3D RGB-D 支持
```

## 运行

**推荐命令（Record3D + iPhone LiDAR）：**

```bash
python main.py --camera-source record3d --mode piano \
  --instrument-roi 0.05,0.45,0.95,0.95 --depth-contact-mode assist \
  --piano-sensitivity stable --no-fingertip-markers
```

启动后将手移开琴键区域，按 `d` 校准桌面深度。校准成功后即可开始演奏。

**重要提示：请确保演奏环境有充足、均匀的光照。** MediaPipe 在弱光下检测率和关键点稳定性会显著下降，导致追踪抖动增大和误触发。建议使用台灯或环形灯补光。

## 运行时按键

| 按键 | 功能 |
|------|------|
| `q` | 退出 |
| `d` | 校准桌面深度 |
| `s` | 延音踏板开/关 |
| `c` | 和弦模式开/关 |
| `t` | 节拍器开/关 |
| `v` | 切换力度曲线 (linear/natural/dramatic) |
| `x` | 导出 MIDI 文件 |
| `1`-`6` | 切换音阶 (major/minor/pentatonic/blues/dorian/mixolydian) |
| `9`/`0` | 节拍器 BPM -10 / +10 |
| `f` | 全屏 |
| `o` | 旋转画面 90° |

## 系统架构

```
RGB-D Camera → ROI Crop → MediaPipe Hands (21 landmarks)
    → EMA Smoothing → Optical Flow Stabilization
    → Six-State Hit Detector (+ Depth Assist Gate)
    → Audio Engine (Procedural Synthesis, Velocity Layers)
    → Visual Feedback
```

## 核心特性

- **多级指尖稳定**：EMA 平滑 + Lucas-Kanade 光流锚定，抖动降低 60-70%
- **六态状态机**：idle → lifting → raised → falling → hit → pressed，速度敏感力度映射
- **深度辅助架构**：iPhone LiDAR 作为负向门控（阻止明显悬空的误触发）
- **程序化音频合成**：3 层力度音色（pp/mf/ff），零外部音频文件依赖
- **可选扩展功能**：音阶切换、和弦模式、延音踏板、节拍器、MIDI 导出、手势控制、Loop Station

## 项目结构

```
main.py                 # 主程序入口
config.py               # 全局配置参数
hand_tracker.py         # MediaPipe 封装 + 多级滤波
hit_detector.py         # 六态弹奏检测状态机
audio_engine.py         # 程序化音频合成引擎
scales.py               # 音阶/调式/和弦工具
midi_writer.py          # MIDI Type 0 文件导出
ui.py                   # 可视化界面渲染
rgbd_camera.py          # Record3D RGB-D 输入封装
depth_contact.py        # 深度接触估计
instrument.py           # 键盘几何布局
gesture_recognizer.py   # 手势识别
loop_station.py         # Loop Station 录制/回放
session_recorder.py     # Session 录制（离线分析用）
report/                 # 项目报告 (LaTeX + PDF)
demo/                   # 演示截图和视频
```
