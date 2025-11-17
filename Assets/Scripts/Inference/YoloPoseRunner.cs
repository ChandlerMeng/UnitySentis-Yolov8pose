using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using Unity.InferenceEngine;
using Stopwatch = System.Diagnostics.Stopwatch;

namespace Inference.Demo
{
    // 最小可运行：摄像头 -> 纹理转张量 -> YOLOv8 Pose 推理 -> 解析关键点 -> UI 绘制
    public class YoloPoseRunner : MonoBehaviour
    {
        [Header("Model & Backend")]
        [SerializeField] ModelAsset modelAsset;
        [SerializeField] BackendType backend = BackendType.GPUCompute;

        [Header("Camera & Display")]
        [SerializeField] RawImage cameraView;       // 显示摄像头画面
        [SerializeField] AspectRatioFitter cameraAspect; // 用于保持摄像头比例全屏适配
        [SerializeField] RectTransform overlay;     // 关键点叠加层（与 cameraView 同尺寸）
        [SerializeField] Vector2Int modelInputSize = new Vector2Int(640, 640);

        [Header("Thresholds")]
        [SerializeField] float confThreshold = 0.25f;
        [SerializeField] float keypointThreshold = 0.2f;
        [SerializeField] int maxDetections = 5;

        [Header("Renderer")]
        [SerializeField] YoloPoseRenderer poseRenderer;
        [SerializeField] GameObject pointPrefab; // 供渲染器使用
        [SerializeField] PerfHUD perfHUD;

        [Header("Perf")]
        [SerializeField] int inferenceInterval = 1; // 每隔多少帧推理一次，1=每帧

        [Header("Gesture Detectors")] 
        [SerializeField] List<GestureDetector> detectors = new List<GestureDetector>();

        Worker _worker;
        Tensor<float> _inputTensor; // [1,3,H,W]
        WebCamTexture _webcam;
        readonly Tensor[] _inputs = new Tensor[1];
        RenderTexture _letterboxRT;

        // 缓存：关键点坐标（模型输入空间，像素）
        readonly List<Vector2> _points = new List<Vector2>(17 * 5);
        readonly List<Vector2> _lastPoints = new List<Vector2>(17 * 5);
        bool _loggedShape;
        bool _inflight = false; // 本帧是否已有在跑的推理（等待下一帧读回）

        // 手势检测逻辑由外部 Detector 组件管理

        void OnEnable()
        {
            if (modelAsset == null)
            {
                Debug.LogError("ModelAsset 未设置。请在 Inspector 绑定 yolov8 pose onnx 资源。");
                enabled = false;
                return;
            }

            // 1) 初始化摄像头
            if (_webcam == null)
            {
                var devices = WebCamTexture.devices;
                if (devices == null || devices.Length == 0)
                {
                    Debug.LogError("未检测到摄像头设备。");
                    enabled = false;
                    return;
                }
                _webcam = new WebCamTexture(devices[0].name, 1280, 720, 30);
                _webcam.Play();
            }
            if (cameraView != null)
                cameraView.texture = _webcam;
            if (cameraAspect == null && cameraView != null)
            {
                cameraAspect = cameraView.GetComponent<AspectRatioFitter>();
                if (cameraAspect == null) cameraAspect = cameraView.gameObject.AddComponent<AspectRatioFitter>();
                cameraAspect.aspectMode = AspectRatioFitter.AspectMode.FitInParent;
            }

            // 2) 初始化模型与 Worker
            var model = ModelLoader.Load(modelAsset);
            _worker = new Worker(model, backend);

            // 3) 预分配输入张量（NCHW）
            _inputTensor = new Tensor<float>(new TensorShape(1, 3, modelInputSize.y, modelInputSize.x));
            _inputs[0] = _inputTensor;

            // 确保渲染器准备就绪
            if (poseRenderer != null && poseRenderer.gameObject.activeInHierarchy)
            {
                // 将 pointPrefab 传递/校验
                var field = typeof(YoloPoseRenderer).GetField("pointPrefab", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
                if (field != null && pointPrefab != null)
                    field.SetValue(poseRenderer, pointPrefab);
            }

            // 通知所有 Detector 进行设置/复位
            if (detectors != null)
            {
                foreach (var d in detectors)
                {
                    if (d == null) continue;
                    d.Setup(modelInputSize);
                    d.ResetState();
                }
            }
        }

        void OnDisable()
        {
            try { _worker?.Dispose(); } catch { /* ignore */ }
            try { _inputTensor?.Dispose(); } catch { /* ignore */ }
            if (_webcam != null)
            {
                if (_webcam.isPlaying) _webcam.Stop();
                Destroy(_webcam);
                _webcam = null;
            }
        }

        void Update()
        {
            if (_webcam == null || !_webcam.isPlaying) return;
            if (_webcam.width <= 16 || _webcam.height <= 16) return; // 等待摄像头初始化

            int srcW = _webcam.width;
            int srcH = _webcam.height;

            // 情况A：有在飞推理 → 尝试读回并解码（造成的阻塞集中在这一步，变为隔一帧发生）
            if (_inflight)
            {
                var output = _worker.PeekOutput() as Tensor<float>;
                if (output != null)
                {
                    var sw = Stopwatch.StartNew();
                    using var cpu = output.ReadbackAndClone(); // 可能阻塞，但只在有推理完成的帧发生
                    sw.Stop();
                    if (perfHUD != null) perfHUD.ReportInferenceMillis(sw.Elapsed.TotalMilliseconds);

                    if (!_loggedShape)
                    {
                        var sh = cpu.shape;
                        // 形状只打印一次（避免刷屏）
                        // Debug.Log($"YOLOv8 Pose 输出形状: [{sh[0]},{sh[1]},{sh[2]}]");
                        _loggedShape = true;
                    }

                    _points.Clear();
                    DecodeYoloV8Pose(cpu, _points, confThreshold, keypointThreshold, maxDetections);

                    // 更新可复用的上次结果
                    _lastPoints.Clear();
                    _lastPoints.AddRange(_points);

                    // 手势检测：挑选单人最佳候选，分发给所有 Detector
                    if (detectors != null && detectors.Count > 0)
                    {
                        if (TryDecodeBestPose(cpu, confThreshold, keypointThreshold, out var pts, out var confs))
                        {
                            // 归一化到[0,1]（左上原点）
                            for (int i = 0; i < pts.Length; i++)
                            {
                                if (pts[i].x >= 0f && pts[i].y >= 0f)
                                {
                                    pts[i].x /= modelInputSize.x;
                                    pts[i].y /= modelInputSize.y;
                                }
                            }
                            var frame = new PoseFrame { points = pts, confidences = confs, frameCount = Time.frameCount };
                            foreach (var d in detectors) d?.OnPose(in frame);
                        }
                        else
                        {
                            foreach (var d in detectors) d?.OnNoPose();
                        }
                    }
                }
                _inflight = false;
            }
            // 情况B：根据频率决定是否发起新一轮推理。非推理帧只渲染上一次结果。
            else if (Time.frameCount % Mathf.Max(1, inferenceInterval) == 0)
            {
                EnsureLetterboxRT();
                LetterboxCopy(_webcam, _letterboxRT, modelInputSize.x, modelInputSize.y);
                TextureConverter.ToTensor(_letterboxRT, _inputTensor);
                _worker.Schedule(_inputs);   // 非阻塞
                _inflight = true;            // 下一帧再读回
            }

            // 渲染：始终渲染上一次结果，避免非推理帧卡顿
            if (poseRenderer != null && _lastPoints.Count > 0)
            {
                float scale = Mathf.Min((float)modelInputSize.x / srcW, (float)modelInputSize.y / srcH);
                poseRenderer.RenderPoints(_lastPoints, modelInputSize.x, modelInputSize.y, srcW, srcH, scale);
            }

        }

        void LateUpdate()
        {
            // 驱动相机图像按实际比例自适应全屏
            if (cameraAspect != null && _webcam != null && _webcam.width > 16 && _webcam.height > 16)
            {
                cameraAspect.aspectRatio = (float)_webcam.width / _webcam.height;
            }
        }

        void EnsureLetterboxRT()
        {
            if (_letterboxRT != null && (_letterboxRT.width != modelInputSize.x || _letterboxRT.height != modelInputSize.y))
            {
                _letterboxRT.Release();
                Destroy(_letterboxRT);
                _letterboxRT = null;
            }
            if (_letterboxRT == null)
            {
                _letterboxRT = new RenderTexture(modelInputSize.x, modelInputSize.y, 0, RenderTextureFormat.ARGB32)
                {
                    wrapMode = TextureWrapMode.Clamp,
                    filterMode = FilterMode.Bilinear
                };
                _letterboxRT.Create();
            }
        }

        static void LetterboxCopy(Texture src, RenderTexture dst, int dstW, int dstH)
        {
            var srcW = src.width;
            var srcH = src.height;
            if (srcW <= 0 || srcH <= 0) return;

            float scale = Mathf.Min((float)dstW / srcW, (float)dstH / srcH);
            float drawW = srcW * scale;
            float drawH = srcH * scale;
            float offX = (dstW - drawW) * 0.5f;
            float offY = (dstH - drawH) * 0.5f;

            var prev = RenderTexture.active;
            RenderTexture.active = dst;
            GL.PushMatrix();
            GL.LoadPixelMatrix(0, dstW, dstH, 0); // 像素坐标，原点左上
            GL.Clear(true, true, Color.black);
            Graphics.DrawTexture(new Rect(offX, offY, drawW, drawH), src);
            GL.PopMatrix();
            RenderTexture.active = prev;
        }

        // 解析 YOLOv8 Pose：支持 [1,56,N] / [1,N,56]（常见 YOLOv8 Pose 输出），以及部分 4D 变体
        // 56 = 4(xywh)+1(obj)+17*3(kpts)
        static void DecodeYoloV8Pose(Tensor<float> t, List<Vector2> outPoints, float confThr, float kptThr, int maxDet)
        {
            var s = t.shape;
            int d0 = s[0];
            int d1 = s[1];
            int d2 = s[2];

            // 优先处理 3D 输出 [1,56,N] 或 [1,N,56]
            if (d0 == 1 && (d1 >= 56 || d2 >= 56))
            {
                int C, N, cAxis, nAxis;
                if (d1 >= 56) { cAxis = 1; nAxis = 2; C = d1; N = d2; }
                else { cAxis = 2; nAxis = 1; C = d2; N = d1; }

                if (C < 56 || N < 1) return;

                int picked = 0;
                for (int i = 0; i < N && picked < maxDet; i++)
                {
                    float conf = Read3D(t, cAxis, 4, nAxis, i);
                    if (conf < confThr) continue;

                    for (int k = 0; k < 17; k++)
                    {
                        float kx = Read3D(t, cAxis, 5 + k * 3 + 0, nAxis, i);
                        float ky = Read3D(t, cAxis, 5 + k * 3 + 1, nAxis, i);
                        float ks = Read3D(t, cAxis, 5 + k * 3 + 2, nAxis, i);
                        if (ks >= kptThr)
                            outPoints.Add(new Vector2(kx, ky));
                        else
                            outPoints.Add(new Vector2(-1f, -1f));
                    }
                    picked++;
                }
                return;
            }

            // 兼容部分 4D 变体（如 [1,56,1,N] 等）
            try
            {
                int[] dims = { s[0], s[1], s[2], s[3] };
                int cAxis = -1, nAxis = -1, C = -1, N = -1;
                for (int i = 0; i < 4; i++)
                {
                    if (dims[i] >= 56) { cAxis = i; C = dims[i]; break; }
                }
                for (int i = 0; i < 4; i++)
                {
                    if (i == cAxis) continue;
                    if (dims[i] >= 1000) { nAxis = i; N = dims[i]; break; }
                }
                if (cAxis == -1 || nAxis == -1 || C < 56 || N < 1)
                {
                    Debug.LogWarning($"无法识别 YOLOv8 Pose 输出形状 4D: [{dims[0]},{dims[1]},{dims[2]},{dims[3]}]");
                    return;
                }

                int picked = 0;
                for (int i = 0; i < N && picked < maxDet; i++)
                {
                    float conf = Read4DByAxes(t, cAxis, 4, nAxis, i);
                    if (conf < confThr) continue;
                    for (int k = 0; k < 17; k++)
                    {
                        float kx = Read4DByAxes(t, cAxis, 5 + k * 3 + 0, nAxis, i);
                        float ky = Read4DByAxes(t, cAxis, 5 + k * 3 + 1, nAxis, i);
                        float ks = Read4DByAxes(t, cAxis, 5 + k * 3 + 2, nAxis, i);
                        if (ks >= kptThr)
                            outPoints.Add(new Vector2(kx, ky));
                        else
                            outPoints.Add(new Vector2(-1f, -1f));
                    }
                    picked++;
                }
            }
            catch (Exception e)
            {
                Debug.LogWarning($"解码 YOLOv8 Pose 输出失败: {e.Message}");
            }
        }

        // 在 3D Tensor [1,C,N] 或 [1,N,C] 上读取
        static float Read3D(Tensor<float> t, int cAxis, int cIndex, int nAxis, int nIndex)
        {
            if (cAxis == 1 && nAxis == 2) return t[0, cIndex, nIndex];
            if (cAxis == 2 && nAxis == 1) return t[0, nIndex, cIndex];
            // 兜底
            return 0f;
        }

        // 在 4D Tensor 上按指定轴读取：将某轴作为通道索引、某轴作为候选索引，其余取 0
        static float Read4DByAxes(Tensor<float> t, int cAxis, int cIndex, int nAxis, int nIndex)
        {
            // 轴顺序：[0,1,2,3] -> (n, c, h, w) 中的任意映射，这里暴力匹配
            // 我们以 (b,c,h,w) 访问器读取，未知轴一律给 0
            int a0 = 0, a1 = 0, a2 = 0, a3 = 0;
            // 先将 nIndex 映射
            if (nAxis == 0) a0 = nIndex; else if (nAxis == 1) a1 = nIndex; else if (nAxis == 2) a2 = nIndex; else a3 = nIndex;
            // 再将 cIndex 映射
            if (cAxis == 0) a0 = cIndex; else if (cAxis == 1) a1 = cIndex; else if (cAxis == 2) a2 = cIndex; else a3 = cIndex;

            // 其余轴取 0，注意边界
            a0 = Mathf.Clamp(a0, 0, t.shape[0] - 1);
            a1 = Mathf.Clamp(a1, 0, t.shape[1] - 1);
            a2 = Mathf.Clamp(a2, 0, t.shape[2] - 1);
            a3 = Mathf.Clamp(a3, 0, t.shape[3] - 1);
            return t[a0, a1, a2, a3];
        }

        // 选择单人最佳候选，输出一组17点与其关键点置信度
        bool TryDecodeBestPose(Tensor<float> t, float confThr, float kptThr, out Vector2[] points, out float[] kptScores)
        {
            points = null; kptScores = null;
            var s = t.shape;
            int d0 = s[0];
            int d1 = s[1];
            int d2 = s[2];

            int bestIndex = -1;
            float bestConf = -1f;
            int cAxis = -1, nAxis = -1, C = -1, N = -1;

            // 3D优先 [1,56,N] 或 [1,N,56]
            if (d0 == 1 && (d1 >= 56 || d2 >= 56))
            {
                if (d1 >= 56) { cAxis = 1; nAxis = 2; C = d1; N = d2; }
                else { cAxis = 2; nAxis = 1; C = d2; N = d1; }
                if (C < 56 || N < 1) return false;

                for (int i = 0; i < N; i++)
                {
                    float conf = Read3D(t, cAxis, 4, nAxis, i);
                    if (conf < confThr) continue;
                    if (conf > bestConf) { bestConf = conf; bestIndex = i; }
                }

                if (bestIndex < 0) return false;
                points = new Vector2[17];
                kptScores = new float[17];
                for (int k = 0; k < 17; k++)
                {
                    float kx = Read3D(t, cAxis, 5 + k * 3 + 0, nAxis, bestIndex);
                    float ky = Read3D(t, cAxis, 5 + k * 3 + 1, nAxis, bestIndex);
                    float ks = Read3D(t, cAxis, 5 + k * 3 + 2, nAxis, bestIndex);
                    if (ks >= kptThr)
                    {
                        points[k] = new Vector2(kx, ky);
                        kptScores[k] = ks;
                    }
                    else
                    {
                        points[k] = new Vector2(-1f, -1f);
                        kptScores[k] = 0f;
                    }
                }
                return true;
            }

            // 4D兼容
            try
            {
                int[] dims = { s[0], s[1], s[2], s[3] };
                for (int i = 0; i < 4; i++)
                {
                    if (dims[i] >= 56) { cAxis = i; C = dims[i]; break; }
                }
                for (int i = 0; i < 4; i++)
                {
                    if (i == cAxis) continue;
                    if (dims[i] >= 1000) { nAxis = i; N = dims[i]; break; }
                }
                if (cAxis == -1 || nAxis == -1 || C < 56 || N < 1) return false;

                for (int i = 0; i < N; i++)
                {
                    float conf = Read4DByAxes(t, cAxis, 4, nAxis, i);
                    if (conf < confThr) continue;
                    if (conf > bestConf) { bestConf = conf; bestIndex = i; }
                }

                if (bestIndex < 0) return false;
                points = new Vector2[17];
                kptScores = new float[17];
                for (int k = 0; k < 17; k++)
                {
                    float kx = Read4DByAxes(t, cAxis, 5 + k * 3 + 0, nAxis, bestIndex);
                    float ky = Read4DByAxes(t, cAxis, 5 + k * 3 + 1, nAxis, bestIndex);
                    float ks = Read4DByAxes(t, cAxis, 5 + k * 3 + 2, nAxis, bestIndex);
                    if (ks >= kptThr)
                    {
                        points[k] = new Vector2(kx, ky);
                        kptScores[k] = ks;
                    }
                    else
                    {
                        points[k] = new Vector2(-1f, -1f);
                        kptScores[k] = 0f;
                    }
                }
                return true;
            }
            catch { return false; }
        }
    }
}
