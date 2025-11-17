# UnitySentis-YOLOv8 Pose 示例

一个使用 Unity Sentis 在本地进行 YOLOv8 姿态估计（17 关键点）的小型演示工程：从摄像头读取画面，经 Letterbox 适配后送入 ONNX 模型推理，解析关键点并在 UI 上渲染，同时提供简单的手势检测与性能 HUD。

## 功能特性
- 摄像头采集 → 纹理转张量 → 推理（GPU/CPU/Burst 等后端可选）
- 兼容 YOLOv8 Pose 常见输出形状（[1,56,N] / [1,N,56] 及部分 4D 变体）
- 关键点渲染与基础骨架绘制（`YoloPoseRenderer`）
- 简单的手势检测接口（示例 `HandWaveDetector`）
- 性能监控 HUD（推理耗时、帧间调度）
- Git LFS 管理大模型与二进制资源

## 目录结构（节选）
- `Assets/Scripts/Inference/`
  - `YoloPoseRunner.cs`：主流程（摄像头、Letterbox、推理、解码、分发、渲染）
  - `YoloPoseRenderer.cs`：关键点与骨架 UI 渲染
  - `PerfHUD.cs`：性能显示
  - `GestureDetector.cs` / `HandWaveDetector.cs`：手势接口与示例实现
- `Assets/Scenes/SampleScene.unity`：演示场景
- `Assets/StreamingAssets/`：演示所需的 `.sentis` 资源
- `Assets/*.onnx`：YOLOv8 Pose 模型（n/s/m 尺寸）
- `Packages/`、`ProjectSettings/`：Unity 工程配置（含 URP）

## 环境要求
- Unity 6.0 LTS（URP 项目；Sentis 2.4.1 示例文件已包含）
- macOS 或 Windows（需可用摄像头）
- Git LFS 已安装并启用（用于拉取 `*.onnx`、`*.sentis` 等大文件）

安装 Git LFS（macOS）：
```zsh
brew install git-lfs
git lfs install
```

安装 Git LFS（Windows）：
```powershell
# 方式一：winget（推荐）
winget install GitHub.GitLFS
git lfs install

# 方式二：Chocolatey
choco install git-lfs -y
git lfs install

# 方式三：官方安装包（交互式）
# https://git-lfs.com/ 下载并安装，随后在命令行执行：
git lfs install
```

## 获取与运行
1. 克隆并确保 LFS 生效：
   ```zsh
   git clone <repo-url>
   cd UnitySentis
   git lfs pull
   ```
2. 用 Unity 打开工程（Unity 6.0 LTS）。
3. 打开 `Assets/Scenes/SampleScene.unity` 并点击运行。
4. 若在 macOS 无法访问摄像头，请在「系统设置 → 隐私与安全性 → 相机」允许 Unity/Editor 使用相机。

## 关键脚本与参数
- `YoloPoseRunner`
  - `modelAsset`：Sentis 的 `ModelAsset`（绑定 YOLOv8 Pose ONNX 或已转换资源）。
  - `backend`：选择后端（例如 `GPUCompute`/`CPU`）。
  - `modelInputSize`：模型输入分辨率（默认 `640x640`），内部采用 Letterbox。
  - `confThreshold`：检测置信度阈值（默认 `0.25`）。
  - `keypointThreshold`：关键点置信度阈值（默认 `0.2`）。
  - `inferenceInterval`：推理间隔帧（`1` 表示每帧推理，>1 可减轻压力）。
  - `poseRenderer` / `pointPrefab`：关键点渲染配置。
  - `detectors`：手势检测器列表，示例包含 `HandWaveDetector`。

- `YoloPoseRenderer`
  - 负责将 `YoloPoseRunner` 解码出的 17 点绘制到叠加 UI。

- `PerfHUD`
  - 显示推理耗时、节流状况等。

## 模型与预处理
- 本示例对摄像头纹理进行 **Letterbox** 以匹配模型输入尺寸：
  - 等比缩放，空白区域填充黑色。
  - 使用 `TextureConverter.ToTensor()` 将 `RenderTexture` 转为 NCHW 张量。
- 解码支持 YOLOv8 Pose 常见布局（通道维度 `>=56` 且包含 `4+1+17*3=56` 项）。

## 手势检测扩展
实现自定义手势：继承 `GestureDetector` 并实现以下方法：
- `Setup(Vector2Int modelInputSize)`：初始化一次。
- `OnPose(in PoseFrame frame)`：收到单人最佳姿态（17 点 + 置信度）。
- `OnNoPose()`：未检测到人体时回调。
- `ResetState()`：切换场景或重新开始时重置。
将自定义脚本挂到场景中，并添加到 `YoloPoseRunner.detectors` 列表即可。

## 性能建议
- 首选 `GPUCompute` 后端；在弱机型上可以调大 `inferenceInterval`。
- 模型选择：`yolov8n-pose`（更快）/`yolov8s`/`yolov8m`（更准）。
- 降低输入尺寸（如 512/416）可换取更高帧率，但需同时调整阈值与渲染尺度。

## Git 与 LFS
本仓库已启用 Git LFS，追踪以下类型：
- `*.onnx`、`*.sentis`（模型及转换资源）
- `*.fbx`、`*.psd`、`*.mp4`（常见大体积二进制）

拉取时请确保：
```zsh
git lfs install
git lfs pull
```

## 常见问题
- 相机黑屏：
  - 等待摄像头初始化（宽高 > 16）
  - 检查系统相机权限
  - 在 `Game` 视图选择正确显示分辨率
- 模型不匹配或输出异常：
  - 确认使用 YOLOv8 Pose（含 17 关键点）的权重
  - `modelInputSize` 与 Letterbox 预处理逻辑保持一致
- 帧率低：
  - 切换更小模型（`n`/`s`）或增大 `inferenceInterval`
  - 选择更合适的后端（GPU 优先）

## 免责声明
仅用于学习与演示，模型与资源版权归各自作者所有。
