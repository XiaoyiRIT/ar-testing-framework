📁 AR Action Discovery Framework – Current Progress Summary

（可直接用于新对话的上下文）

1. 项目目标简述

本项目的最终目标是构建一个 自动化 AR Interaction Event Generation + Action Discovery 系统，完成：

AR 对象检测（YOLO）

手势采样（tap / drag / rotate / pinch）

多指手势注入（Appium dispatchGesture）

操作前后场景分析（FoELS + optical flow + geometry + SSIM）

N/M 策略统计操作支持情况

输出 Action Support Matrix（JSONL）

当前工程已完成基础骨架，但正式版本代码在 src/ 下尚未补齐。
已有代码大部分还在 common/、cv/、experiments/ 中。

2. 📦 当前项目目录结构（已改造成正式工程）

已包含：

src/：正式版模块的目标目录（仍是空骨架，需要逐个补齐代码）

common/、cv/、experiments/：旧版/实验代码（用于迁移参考）

configs/ad.yaml：Action Discovery 的配置文件

scripts/run_discovery.sh：运行入口脚本

3. 📌 模块状态与待办事项
3.1 Detector（YOLO Detector）
已有

YOLO 训练、推理代码在 cv/strategy_yolo.py

YOLO 模型、数据集等资源完整

待完成（src/detector/yolo_detector.py）

创建 YOLODetector 类

初始化模型（from ultralytics import YOLO）

detect(frame_bgr) → 返回统一格式：

{
  "objects": [
    {"id": 0, "cls": "AR_Object", "bbox": [x, y, w, h], "center_xy": [cx, cy], "score": 0.92}
  ]
}

3.2 Executor（Appium 多指手势执行器）
已有

完整的手势执行逻辑在 experiments/v3_ar_monkey_appium.py

输入事件相关工具：common/device.py、common/actions.py

待完成（src/executor/appium_executor.py）

封装 Appium driver 建立逻辑

snapshot_screen() → BGR numpy array

perform(op, region, params) → 调用多指手势注入执行一次操作

所有截图/执行都由该类统一管理

3.3 Sampler（采样模块）
已有

随机策略在 common/policy_random.py

待完成（src/sampler/default_sampler.py）

增加 sample(op, region) 方法

drag：方向/距离

rotate：角度/半径

pinch：scale_sign（in/out）

tap：抖动半径

3.4 Verifier（多证据验证器）
已有

src/verifier/backends/motion_similarity.py：几何+光流验证的完整后端

common/verify_motion.py：旧版（参考）

待完成（src/verifier/verifier.py）

将 YOLO bbox/center 与前后帧交给 motion_similarity

构造 extra 参数（像素阈值等）

返回 success, evidence, metrics

未来可加入 FoELS / SSIM / optical flow 多通道融合

3.5 Policy（N/M 判定）
已有

src/policy/policy.py 已有基本骨架

待完成

确保 decide_support(op, trial_results) 正常返回布尔值即可（简单部分）

3.6 Discovery（总控流程）
已有

src/discovery/run_discovery.py 框架已创建

待完成

整合 executor.snapshot_screen()

实现 select_targets(det_result)

对每个 region + op 做 N 次尝试，记录 JSONL

写入 trial 记录与 final support 结果

3.7 Main Entry（程序入口）
已有

src/__main__.py 已提供完整入口骨架（但未绑定实际模块）

待完成

在 make_components(...) 中实例化：

YOLODetector

DefaultSampler

AppiumExecutor

4. 🚀 开发路线（按优先级）

完成 Detector（yolo_detector.py）
→ 这是所有步骤的输入，优先级最高

完成 Executor（appium_executor.py）
→ 指令执行 + 截图入口

完成 Sampler
→ 可以先只做 drag/tap，后续再扩展 rotate/pinch

把 Verifier 连接上 motion_similarity 后端

把 run_discovery.py 接上 detector/sampler/executor/verifier

测试 dry-run + offline mock → online real app
