using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

namespace Inference.Demo
{
    // 简单的 UI 点渲染器：将模型空间坐标映射到 Overlay UI 上，以小方块显示关键点
    public class YoloPoseRenderer : MonoBehaviour
    {
        [SerializeField] RectTransform overlay; // 与摄像头画面同尺寸的 UI 容器
        [SerializeField] GameObject pointPrefab; // 建议为一个带 Image 组件的方块，无需 Sprite
        [SerializeField] Vector2 pointSize = new Vector2(8, 8);
        [SerializeField] float lineThickness = 3f;

        readonly List<RectTransform> _pool = new List<RectTransform>();
        int _active;
        readonly List<RectTransform> _linePool = new List<RectTransform>();
        int _activeLines;

        public void Clear()
        {
            for (int i = 0; i < _active; i++)
                _pool[i].gameObject.SetActive(false);
            _active = 0;
            for (int i = 0; i < _activeLines; i++)
                _linePool[i].gameObject.SetActive(false);
            _activeLines = 0;
        }

        RectTransform Get()
        {
            if (_active < _pool.Count)
            {
                var r = _pool[_active++];
                r.gameObject.SetActive(true);
                return r;
            }
            var go = Instantiate(pointPrefab, overlay);
            var rt = go.GetComponent<RectTransform>();
            if (rt == null) rt = go.AddComponent<RectTransform>();
            
            // 确保子物体使用中心锚点和 pivot，避免 stretch 干扰
            rt.anchorMin = new Vector2(0.5f, 0.5f);
            rt.anchorMax = new Vector2(0.5f, 0.5f);
            rt.pivot = new Vector2(0.5f, 0.5f);
            rt.sizeDelta = pointSize;
            
            _pool.Add(rt);
            _active++;
            return rt;
        }

        RectTransform GetLine()
        {
            if (_activeLines < _linePool.Count)
            {
                var r = _linePool[_activeLines++];
                r.gameObject.SetActive(true);
                return r;
            }
            var go = Instantiate(pointPrefab, overlay);
            var rt = go.GetComponent<RectTransform>();
            if (rt == null) rt = go.AddComponent<RectTransform>();
            rt.anchorMin = new Vector2(0.5f, 0.5f);
            rt.anchorMax = new Vector2(0.5f, 0.5f);
            rt.pivot = new Vector2(0.5f, 0.5f);
            _linePool.Add(rt);
            _activeLines++;
            return rt;
        }

        // 将模型输入坐标系中的点（单位：像素，原点左上）绘制到 overlay
        // letterboxScale: 源图到模型输入的缩放比例（来自预处理）
        public void RenderPoints(IReadOnlyList<Vector2> pointsModelSpace, int modelW, int modelH,
                                 int sourceW, int sourceH, float letterboxScale)
        {
            if (overlay == null || pointPrefab == null) return;
            Clear();

            var rect = overlay.rect;
            float ow = rect.width;
            float oh = rect.height;

            // 摄像头画面在 overlay 内的 letterbox 拟合（保持比例居中，与 UI AspectRatioFitter 一致）
            float uiScale = Mathf.Min(ow / sourceW, oh / sourceH);
            float dispW = sourceW * uiScale;
            float dispH = sourceH * uiScale;
            float offX = (ow - dispW) * 0.5f;
            float offY = (oh - dispH) * 0.5f;

            // 模型坐标是 letterbox 后的像素坐标，需要先映射回源图坐标，再映射到 UI
            // 模型输入中 letterbox 的有效区域尺寸
            float letterboxW = sourceW * letterboxScale;
            float letterboxH = sourceH * letterboxScale;
            float letterboxOffX = (modelW - letterboxW) * 0.5f;
            float letterboxOffY = (modelH - letterboxH) * 0.5f;
            for (int i = 0; i < pointsModelSpace.Count; i++)
            {
                var p = pointsModelSpace[i];
                if (p.x < 0f || p.y < 0f) continue; // 跳过无效点
                // 1) 从模型空间坐标 -> 源图坐标（去除 letterbox 偏移和缩放）
                float srcX = (p.x - letterboxOffX) / letterboxScale;
                float srcY = (p.y - letterboxOffY) / letterboxScale;
                
                // 2) 从源图坐标 -> UI 像素坐标（应用 UI letterbox）
                float uiX = offX + srcX * uiScale;
                float uiY = offY + srcY * uiScale;
                
                var rt = Get();
                // 3) 从 UI 像素坐标（左上原点）-> anchoredPosition（中心原点，y向上）
                rt.anchoredPosition = new Vector2(uiX - ow * 0.5f, oh * 0.5f - uiY);
            }

            // 绘制骨架连线：假设 points 以 17 个为一组
            // YOLOv8 COCO17 关键点索引：
            // 0:鼻, 1:左眼, 2:右眼, 3:左耳, 4:右耳, 5:左肩, 6:右肩, 7:左肘, 8:右肘,
            // 9:左腕,10:右腕,11:左髋,12:右髋,13:左膝,14:右膝,15:左踝,16:右踝
            int[][] edges = new int[][]
            {
                new[]{5,6},    // 双肩
                new[]{5,7}, new[]{7,9},   // 左臂
                new[]{6,8}, new[]{8,10},  // 右臂
                new[]{11,12},             // 髋
                new[]{5,11}, new[]{6,12}, // 躯干
                new[]{11,13}, new[]{13,15}, // 左腿
                new[]{12,14}, new[]{14,16},  // 右腿
                new[]{1,0}, new[]{2,0}, new[]{1,3}, new[]{2,4}
                // 可选: 头部 new[]{1,0}, new[]{2,0}, new[]{1,3}, new[]{2,4}
            };

            int personCount = pointsModelSpace.Count / 17;
            for (int pIdx = 0; pIdx < personCount; pIdx++)
            {
                int baseIdx = pIdx * 17;
                for (int e = 0; e < edges.Length; e++)
                {
                    int a = baseIdx + edges[e][0];
                    int b = baseIdx + edges[e][1];
                    var pa = pointsModelSpace[a];
                    var pb = pointsModelSpace[b];
                    if (pa.x < 0f || pa.y < 0f || pb.x < 0f || pb.y < 0f) continue;

                    // A: 模型->源图
                    float a_srcX = (pa.x - letterboxOffX) / letterboxScale;
                    float a_srcY = (pa.y - letterboxOffY) / letterboxScale;
                    float b_srcX = (pb.x - letterboxOffX) / letterboxScale;
                    float b_srcY = (pb.y - letterboxOffY) / letterboxScale;

                    // B: 源图->UI 像素
                    float a_uiX = offX + a_srcX * uiScale;
                    float a_uiY = offY + a_srcY * uiScale;
                    float b_uiX = offX + b_srcX * uiScale;
                    float b_uiY = offY + b_srcY * uiScale;

                    // C: UI 像素->anchoredPosition（中心原点，y向上）
                    Vector2 A = new Vector2(a_uiX - ow * 0.5f, oh * 0.5f - a_uiY);
                    Vector2 B = new Vector2(b_uiX - ow * 0.5f, oh * 0.5f - b_uiY);

                    var lr = GetLine();
                    Vector2 mid = (A + B) * 0.5f;
                    Vector2 dir = (B - A);
                    float len = dir.magnitude;
                    float ang = Mathf.Atan2(dir.y, dir.x) * Mathf.Rad2Deg;
                    lr.anchoredPosition = mid;
                    lr.sizeDelta = new Vector2(len, lineThickness);
                    lr.localRotation = Quaternion.Euler(0, 0, ang);
                }
            }
        }
    }
}
