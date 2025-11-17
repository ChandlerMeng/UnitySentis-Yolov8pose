using UnityEngine;

namespace Inference.Demo
{
    // 单人姿态帧（关键点已归一化到[0,1]，左上原点）
    public struct PoseFrame
    {
        public Vector2[] points;   // 长度=17，可能包含(-1,-1)表示无效
        public float[] confidences; // 长度=17，对应关键点置信度
        public int frameCount;     // Unity Time.frameCount
    }

    // 手势检测基类（MonoBehaviour）
    public abstract class GestureDetector : MonoBehaviour
    {
        // 供 Runner 调用，提供模型输入尺寸等上下文（可选）
        public virtual void Setup(Vector2Int modelInputSize) { }

        // 收到单人姿态时调用
        public abstract void OnPose(in PoseFrame frame);

        // 当前帧无有效姿态时调用
        public virtual void OnNoPose() { }

        // 当系统启停或需要清空状态时调用
        public virtual void ResetState() { }

        // 手势名称（可用于调试/HUD）
        public virtual string GestureName => GetType().Name;
    }
}
