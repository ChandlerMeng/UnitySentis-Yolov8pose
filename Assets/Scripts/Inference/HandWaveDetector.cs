using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Events;

namespace Inference.Demo
{
    // 挥手动作识别器（单人）：继承通用 GestureDetector
    public class HandWaveDetector : GestureDetector
    {
        [Header("Wave Parameters")]
        [SerializeField] int historyLength = 30;               // 采样帧数
        [SerializeField] float minYAboveShoulder = 0.05f;      // 手腕高于肩的最小距离（归一化）
        [SerializeField] float minXDelta = 0.15f;              // 左右摆动的最小幅度（归一化）
        [SerializeField] float minConfidence = 0.5f;           // 关键点置信度阈值
        [SerializeField] int cooldownFrames = 30;              // 触发冷却（帧）

        [System.Serializable]
        public class SimpleEvent : UnityEvent { }
        [Header("Events")]
        [SerializeField] SimpleEvent OnRightHandWave;
        [SerializeField] SimpleEvent OnLeftHandWave;

        readonly List<Vector2[]> _histPoints = new List<Vector2[]>();
        readonly List<float[]> _histConfs = new List<float[]>();
        int _cooldownR = 0, _cooldownL = 0;

        public override void ResetState()
        {
            _histPoints.Clear();
            _histConfs.Clear();
            _cooldownR = _cooldownL = 0;
        }

        public override void OnNoPose()
        {
            // 冷却递减
            if (_cooldownR > 0) _cooldownR--;
            if (_cooldownL > 0) _cooldownL--;
        }

        public override void OnPose(in PoseFrame frame)
        {
            // 拷贝入历史，避免外部数组被复用
            var ptsCopy = new Vector2[frame.points.Length];
            frame.points.CopyTo(ptsCopy, 0);
            var confCopy = new float[frame.confidences.Length];
            frame.confidences.CopyTo(confCopy, 0);

            _histPoints.Add(ptsCopy);
            _histConfs.Add(confCopy);

            while (_histPoints.Count > Mathf.Max(5, historyLength))
            {
                _histPoints.RemoveAt(0);
                _histConfs.RemoveAt(0);
            }

            if (_histPoints.Count < Mathf.Max(5, historyLength))
            {
                OnNoPose();
                return;
            }

            if (_cooldownR <= 0 && DetectWave(true))
            {
                try { OnRightHandWave?.Invoke(); } catch { }
                _cooldownR = Mathf.Max(0, cooldownFrames);
                Debug.Log("Right hand wave detected.");
            }
            if (_cooldownL <= 0 && DetectWave(false))
            {
                try { OnLeftHandWave?.Invoke(); } catch { }
                _cooldownL = Mathf.Max(0, cooldownFrames);
                Debug.Log("Left hand wave detected.");
            }

            OnNoPose();
        }

        bool DetectWave(bool isRightHand)
        {
            int len = Mathf.Min(historyLength, _histPoints.Count);
            int start = _histPoints.Count - len;
            int wristIdx = isRightHand ? 10 : 9;
            int shoulderIdx = isRightHand ? 6 : 5;

            // 置信度过滤与数据收集
            float minX = float.PositiveInfinity, maxX = float.NegativeInfinity, sumYDiff = 0f;
            for (int i = 0; i < len; i++)
            {
                var pts = _histPoints[start + i];
                var conf = _histConfs[start + i];
                if (wristIdx >= pts.Length || shoulderIdx >= pts.Length) return false;
                if (conf[wristIdx] < minConfidence || conf[shoulderIdx] < minConfidence) return false;

                var w = pts[wristIdx];
                var s = pts[shoulderIdx];
                if (w.x < 0f || w.y < 0f || s.x < 0f || s.y < 0f) return false;

                if (w.x < minX) minX = w.x;
                if (w.x > maxX) maxX = w.x;
                sumYDiff += (s.y - w.y); // 手腕高于肩 => 正值
            }

            float avgYDiff = sumYDiff / len;
            if (avgYDiff < minYAboveShoulder) return false;
            if ((maxX - minX) < minXDelta) return false;

            return true;
        }
    }
}
