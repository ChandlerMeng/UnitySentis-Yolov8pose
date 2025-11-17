using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Profiling;
using UnityEngine.UI;

namespace Inference.Demo
{
    public class PerfHUD : MonoBehaviour
    {
        [SerializeField] Text targetText;
        [SerializeField] float updateInterval = 0.5f;
        [SerializeField] int inferenceWindow = 60;

        float _timeAcc;
        int _frameCount;
        float _fps;

        readonly Queue<double> _inferHist = new Queue<double>();
        double _inferLastMs;
        double _inferAvgMs;

        public void ReportInferenceMillis(double ms)
        {
            _inferLastMs = ms;
            _inferHist.Enqueue(ms);
            while (_inferHist.Count > inferenceWindow) _inferHist.Dequeue();
            double sum = 0;
            foreach (var v in _inferHist) sum += v;
            _inferAvgMs = _inferHist.Count > 0 ? sum / _inferHist.Count : 0;
        }

        void Update()
        {
            _timeAcc += Time.unscaledDeltaTime;
            _frameCount++;
            if (_timeAcc >= updateInterval)
            {
                _fps = _frameCount / _timeAcc;
                _timeAcc = 0f;
                _frameCount = 0;

                if (targetText != null)
                {
                    long monoUsed = Profiler.GetMonoUsedSizeLong();
                    long alloc = Profiler.GetTotalAllocatedMemoryLong();
                    long reserved = Profiler.GetTotalReservedMemoryLong();

                    targetText.text =
                        $"FPS: {_fps:F1}\n" +
                        $"Infer(ms): last={_inferLastMs:F2}, avg={_inferAvgMs:F2}\n" +
                        $"Mono: {BytesToMB(monoUsed):F1} MB\n" +
                        $"Alloc: {BytesToMB(alloc):F1} MB  Reserved: {BytesToMB(reserved):F1} MB";
                }
            }
        }

        static float BytesToMB(long bytes) => bytes / (1024f * 1024f);
    }
}
