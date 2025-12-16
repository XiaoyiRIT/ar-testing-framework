
---

# 🚀 **Action Discovery Framework – 全局工程进度总结（截至当前对话）**

本文件总结了当前 AR 自动化测试框架（Action Discovery Framework）的全部代码结构、完成状态与待办事项，用于新的 ChatGPT 对话作为上下文输入。

---

# 1. 🎯 项目目标概述

本工程旨在构建一个 **用于 AR 应用的自动化 Action Discovery 工具**，通过视觉检测、操作采样、手势执行、变化验证与策略判定，最终输出：

* **Action Support Matrix（支持矩阵）**
* 可用于后续 LLM 驱动智能脚本生成的基础能力图谱

系统核心流程如下：

1. **YOLODetector**：检测 AR 物体位置（bbox / center）
2. **DefaultSampler**：根据物体与操作生成执行参数
3. **AppiumExecutor**：对 Android 设备注入手势
4. **Verifier**：通过视觉证据（motion similarity 等）判断操作是否成功
5. **NMPolicy**：N/M 判定，决定该目标是否“支持”某个动作
6. **run_discovery**：主循环，将上述模块整合为完整 Action Discovery 流程

---

# 2. 📁 代码目录结构（当前有效结构）

```
program/
├── src/
│   ├── __init__.py
│   ├── __main__.py
│   ├── common/
│   │   ├── __init__.py
│   │   ├── actions.py
│   │   ├── device.py
│   │   ├── locator_iface.py
│   │   ├── policy_random.py
│   │   ├── timing.py
│   │   └── verify_motion.py
│   │
│   ├── detector/
│   │   ├── __init__.py
│   │   └── yolo_detector.py
│   │
│   ├── sampler/
│   │   ├── __init__.py
│   │   └── default_sampler.py
│   │
│   ├── executor/
│   │   ├── __init__.py
│   │   └── appium_executor.py
│   │
│   ├── verifier/
│   │   ├── __init__.py
│   │   ├── backends/
│   │   │   └── motion_similarity.py
│   │   └── verifier.py
│   │
│   ├── policy/
│   │   ├── __init__.py
│   │   └── policy.py
│   │
│   └── discovery/
│       ├── __init__.py
│       └── run_discovery.py
│
├── configs/
│   └── ad.yaml
│
├── scripts/
│   └── run_discovery.sh
│
├── experiments/
│   └── v0_v1_v2_v3_archive/
│       └── v2_ar_monkey_appium.py  (参考用)
│
├── README.md
└── requirements.txt
```

---

# 3. 🧩 各模块完成度与当前状态

## ✔️ **3.1 Detector (YOLODetector)**

* 已完成：`yolo_detector.py`
* 功能：加载 YOLO 模型、输出 bbox + center + score
* 输出格式标准化完毕

## ✔️ **3.2 Sampler (DefaultSampler)**

* 已完成：`default_sampler.py`
* 支持动作：

  * tap/single_tap
  * drag/drag_short/drag_long
  * rotate/rotate_cw/rotate_ccw
  * pinch/pinch_in/pinch_out/zoom_in/zoom_out
* 所有参数都规范化并可通过 cfg 调整

## ✔️ **3.3 Executor (AppiumExecutor)**

* 已完成：`appium_executor.py`
* 与 `common/actions.py` 与 `common/device.py` 完全对齐
* 支持动作家族：tap / drag / rotate / pinch（自动处理别名）

## ✔️ **3.4 Verifier**

* 已完成：

  * `verifier/backends/motion_similarity.py`（已有）
  * `verifier.py`（新写）
* 功能：将操作映射到 motion-based 几何验证
* 支持：

  * drag：方向/幅度
  * rotate：旋转幅度
  * pinch：缩放幅度
  * tap：暂时 always False（未来可接 FoELS/SSIM）

## ✔️ **3.5 NM Policy**

* 已完成：`policy.py`
* 简单 N/M 判定模块（统计成功次数 ≥ M）

## ✔️ **3.6 Discovery Pipeline**

* 已完成：`run_discovery.py`
* 功能：

  1. 初次截图 → YOLO 检测 → 选 target
  2. 对 target × op 执行 N 次试验
  3. 调用 sampler → executor → verifier
  4. 写 trial.jsonl
  5. N/M 判定 → 写 support.jsonl

## ✔️ **3.7 Main entry**

* `__main__.py` 需要对齐 sampler/executor/detector，但结构已经存在，可直接运行。

---

# 4. 🛠️ 最近关键修复点（已处理）

* `common/` 从顶层移到 `src/common/` → 统一成包内导入
* 修改所有 `from common import xxx` → `from ..common import xxx`
* `src/__init__.py` 不再强制修改 sys.path
* discovery 全流程第一次集成成功

---

# 5. 📌 待办事项（Next Steps）

以下是下一阶段可继续的开发点：

### 🟦 5.1（可选）增强 tap verifier

* 接入 FoELS / SSIM 判断是否“有轻微变化”
* 目前 tap = always False

### 🟩 5.2 增加 AR app 内 UI 导航（未来）

* 自动点击 UI 元素进入 AR 页面
* 适合作为后续 LLM agent 部分的基础

### 🟧 5.3 日志可视化（实验阶段）

* 绘制 bbox+flow+rotation 检测的可视化调试图

### 🟨 5.4 系统性 benchmark（论文部分）

* Sceneform sample app 上做验证
* 多 app cross-validation

---

# 6. 🚦 当前系统可运行的命令

在项目根目录运行：

```bash
scripts/run_discovery.sh
```

或等价：

```bash
python -m src --cfg configs/ad.yaml
```

如果模块路径无误、Appium 连接正常、YOLO 权重可用，则系统将执行完整一轮 Action Discovery。

---

# 7. 📄 本文件的用途

你可以将本文件作为**新对话输入**，它包含：

* 完整的项目结构
* 全部模块完成情况
* 下一步开发重点
* 你之前的工程决策记录
* ChatGPT 继续开发所需的全部上下文

在新对话中上传此文件并说：

> “这是我的项目状态，我们继续开发某某模块。”

即可无缝继续。

---

如需我帮你生成一个 **clean 版初始框架模板** 或一个 **自动生成 docs 的脚本**，也可以告诉我。
