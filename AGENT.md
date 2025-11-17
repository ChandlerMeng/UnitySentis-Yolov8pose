# UnitySentis Agent Plan (YOLOv8 Pose)

本项目目标：使用 Unity 的 AI Inference（原 Sentis）包，在运行时拉起摄像头，执行 YOLOv8 Pose 模型进行人体关键点识别，并在画面中渲染骨架。

## 环境与依赖
- Unity 编辑器：`6000.0.62f1`
- 推理包：`com.unity.ai.inference@2.4.1`（自 Sentis 更名而来）
- 操作系统：macOS
- 主要资源：
  - 模型：`Assets/yolov8m-pose.onnx`
  - 官方样例：`Assets/Samples/Sentis/2.4.1/*`（已安装示例，供 API/用法参考）

备注：Unity 包名已由 `com.unity.sentis` 更名为 `com.unity.ai.inference`，但示例目录与文档中仍可能以“Sentis”称呼组件/功能；API 命名空间以项目内样例为准。

## 目标功能
1. 打开摄像头并显示实时画面。
2. 将摄像头纹理转换为张量，按模型输入尺寸/格式进行预处理（缩放、letterbox、归一化等）。
3. 载入 ONNX 模型，在合适的后端（优先 GPU）上进行推理。
4. 解析 YOLOv8 Pose 输出，执行后处理（阈值过滤、NMS、关键点反变换）。
5. 在屏幕空间绘制关键点与骨架连线（叠加到摄像头画面）。
6. 优化与资源复用（Tensor/Worker 复用、最小化分配、批量绘制）。

## 分阶段计划
- 阶段 A：摄像头与显示
  - 使用 `WebCamTexture` 拉起摄像头。
  - 将帧显示到 `RawImage` 或 `MeshRenderer` 材质。
- 阶段 B：纹理->张量
  - 参考示例：`Assets/Samples/Sentis/2.4.1/Convert textures to tensors/TextureToTensor.cs`。
  - 在 GPU 上进行尺寸变换/颜色变换以减少 CPU 拖拽。
- 阶段 C：模型与推理
  - `ModelLoader.Load("Assets/yolov8m-pose.onnx")` 加载模型。
  - 使用 `WorkerFactory` 创建 `GPU`/`Compute` 后端 `IWorker`，全局复用。
- 阶段 D：后处理与解码
  - 根据模型实际输出名称与张量形状进行解析（通过日志打印确认）。
  - 置信度阈值、NMS、关键点坐标映射回原图尺寸。
- 阶段 E：可视化
  - 采用 `LineRenderer` 或 UI 绘制点与边；COCO 典型骨架边集作为默认连线。
- 阶段 F：优化与稳定性
  - 预分配 `Tensor` 与 `ComputeBuffer`，避免每帧 GC。
  - 降低纹理拷贝次数，优先保持在 GPU 流水线内。

## 目录与文件
- 场景：`Assets/Scenes/SampleScene.unity`（可用于初期验证）。
- 模型：`Assets/yolov8m-pose.onnx`（已放置）。
- 示例：`Assets/Samples/Sentis/2.4.1/*`（官方 API 使用范例）。
- 建议新增：
  - `Assets/Scripts/Inference/YoloPoseRunner.cs`：摄像头、推理、后处理主控。
  - `Assets/Scripts/Inference/YoloPoseRenderer.cs`：关键点/骨架绘制与对象池。
  - `Assets/Scripts/Inference/PerfHUD.cs`：帧率、内存与推理耗时显示。
  - `Assets/Materials/*`：用于摄像头画面与关键点渲染的材质/着色器（可选）。

## 运行步骤（当前）
1. 打开 `SampleScene` 或新建场景，创建空物体 `YoloPoseAgent`。
2. 添加 `YoloPoseRunner` 脚本（创建后），在 Inspector 中引用：
   - 模型资产：`Assets/yolov8m-pose.onnx`
  - `Camera View`：一个 `RawImage` 用于显示摄像头画面（建议添加 `AspectRatioFitter(FitInParent)` 以保持比例全屏）。
  - `Overlay`：一个与 `Camera View` 同尺寸的 `RectTransform`，Stretch + pivot(0.5,0.5)。
  - `Pose Renderer`：场景中的 `YoloPoseRenderer` 组件实例。
  - `Point Prefab`：用于关键点/线段的 `UI/Image` 预制（方块/条形）。
  - `PerfHUD`（可选）：`PerfHUD` 组件实例与其 `Text` 目标。
 3. 允许摄像头权限（首次运行会弹出系统权限对话框）。
 4. 运行场景，默认使用 GPUCompute 后端，并以 letterbox 预处理喂入模型；关键点与骨架将叠加在画面上。
3. 允许摄像头权限（首次运行会弹出系统权限对话框）。
4. 运行场景，检查控制台日志中的模型 IO 名称/形状，并微调预处理/后处理参数。

## 模型 IO 与解码提示
- 不同导出版本的 YOLOv8 Pose 输出名称/形状可能不同。
- 当前工程已适配常见 3D 输出形状 `[1,56,N] / [1,N,56]`；同时保留 4D 兜底。
- 关键点解码恒定输出 17 个点/目标，置信度低的点以 `(-1,-1)` 占位，便于稳定连线。
- 依据实际输出实现：
  - 置信度阈值与类别过滤（人体类别）。
  - NMS 抑制重叠框。
  - 关键点数量/顺序与坐标反变换（从模型输入空间映射回源图）。

## 性能与稳定性
- 后端优先：GPU（Compute 或 Metal）> CPU。
- 复用 `IWorker` 与输入/输出 `Tensor`，避免每帧 new。
- 预处理采用 GPU 侧的 letterbox 到 `RenderTexture`，再转张量，避免拉伸并与 UI 显示一致。
- 分离推理与渲染职责：渲染仅做必要的数据拷贝与绘制。
- 可选“1 帧延迟流水线 + 隔帧推理”：
  - 第 1 帧 Schedule（非阻塞），第 2 帧 Readback（可能阻塞），主线程更平滑。
  - 通过 `inferenceInterval` 控制推理频率（1=每帧，2/3=隔帧）。
- `PerfHUD` 显示 FPS、推理耗时（最近/平均）与内存占用，便于快速评估性能。

## 参数说明（关键）
- `confThreshold`：目标置信度阈值（0~1），过滤低置信度人体，建议 0.25~0.4。
- `keypointThreshold`：关键点置信度阈值（0~1），过滤低置信度关键点（低于阈值以 `(-1,-1)` 忽略），建议 0.2~0.5。
- `maxDetections`：每帧最大渲染人数，单人设 1，多人设 3~5。
- `inferenceInterval`：推理频率（帧间隔），1=每帧，2/3=隔帧；值越大卡顿越小、实时性下降。

## 已实现清单
- 摄像头采集与画面显示（保持比例全屏）。
- 纹理→张量：GPU letterbox 预处理，模型输入尺寸对齐（默认 640x640）。
- 推理：GPUCompute 后端，Worker/Tensor 复用。
- 解码：适配 3D `[1,56,N]`/`[1,N,56]` 输出；17 关键点稳定索引。
- 可视化：
  - 关键点 UI 渲染（点池）。
  - 骨架连线（线段池，COCO-17 拓扑）。
  - 与 UI 显示 letterbox 映射完全一致（避免位移/缩放误差）。
- 性能 HUD：FPS、内存与推理耗时(last/avg)。
- 可选降载：1 帧延迟流水线 + 隔帧推理。

## 待办/可选增强
- NMS 与多人目标框提取（当前以关键点为主，未渲染框）。
- 多目标配色与置信度可视化（颜色/透明度编码）。
- 异步读回/ComputeBuffer 路线以进一步降低主线程阻塞。
- 截图/录制与参数热调（UI Slider）。

## 权限与平台注意
- macOS 编辑器/Standalone 首次使用摄像头需要系统授权。
- 如未显示画面，检查“系统设置 -> 隐私与安全性 -> 相机”中是否允许 Unity 编辑器/应用。

## 近期里程碑（可勾选）
- [ ] 摄像头取流与画面显示
- [ ] 模型加载与单帧推理成功
- [ ] 打印并确认模型 IO 形状
- [ ] 实现最小可行后处理（阈值 + NMS + 关键点）
- [ ] 屏幕空间渲染关键点与骨架
- [ ] 基本性能优化（GPU 后端、复用张量）

## 参考
- 项目内样例：`Assets/Samples/Sentis/2.4.1`
- Unity AI Inference（Sentis 2.x）文档与发行说明（与 2.4.1 对应）
