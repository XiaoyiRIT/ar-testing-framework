# CV 算法评估脚本使用说明

## 📋 概述

`v2_evaluation.py` 是用于评估 CV (Computer Vision) 算法准确性的脚本。它通过对比 **CV 验证结果** 与 **Ground Truth（代码层面的真实结果）** 来计算准确率指标。

## 🎯 评估目标

验证 CV 算法能否准确识别 AR 物体的操作响应，证明 CV 方法与代码层面验证的等效性。

## 📊 评估指标

### Ground Truth 定义

| 类型 | CV 结果 | Ground Truth | 说明 |
|------|---------|--------------|------|
| **True Positive (TP)** | ✓ 有运动 | ✓ 有记录 | CV 正确识别了成功的操作 |
| **True Negative (TN)** | ✗ 无运动 | ✗ 无记录 | CV 正确识别了失败的操作 |
| **False Positive (FP)** | ✓ 有运动 | ✗ 无记录 | CV 误判（app不支持或失败）|
| **False Negative (FN)** | ✗ 无运动 | ✓ 有记录 | CV 漏检（未检测到运动）|

### 计算公式

- **准确率 (Accuracy)** = (TP + TN) / (TP + TN + FP + FN)
- **精确率 (Precision)** = TP / (TP + FP)
- **召回率 (Recall)** = TP / (TP + FN)
- **F1 分数 (F1-Score)** = 2 × Precision × Recall / (Precision + Recall)

## 🔧 新增功能

相比 v2_ar_monkey_appium.py，增加了：

1. **更多操作类型** (共10种)：

   **支持的操作** (app实现了这些)：
   - `tap` - 单击
   - `double_tap` - 双击
   - `drag` - 拖拽
   - `long_press` - 长按操作（800-1200ms）
   - `pinch_in` - 捏合缩放
   - `rotate` - 旋转

   **不支持的操作** (用于测试False Positive)：
   - `triple_tap` - 三连击
   - `swipe` - 快速滑动
   - `two_finger_tap` - 双指点击
   - `flick` - 轻弹

2. **均匀操作分配**：
   - 操作次数在各类型之间均匀分配
   - 使用random seed确保可重现性
   - 操作顺序随机但可重现

3. **Negative Samples**：
   - 50%的操作故意在AR物体外执行
   - 用于测试CV算法的True Negative识别能力
   - 增加评估的全面性

4. **Ground Truth 检测**：
   - 每次操作后自动从 logcat 读取真实结果
   - 与 CV 验证结果进行对比

5. **详细的评估报告**：
   - TP/TN/FP/FN 统计
   - Accuracy, Precision, Recall, F1-Score
   - CSV 日志包含 GT 结果、是否为negative sample、是否为支持的操作等信息

## 🚀 使用方法

### 重要提示

**必须从项目根目录运行此脚本**，因为它需要访问 `src/` 和 `cv/` 目录下的模块。

```bash
# 切换到项目根目录
cd /path/to/ar-testing-framework

# 或者如果你在其他目录
cd /home/user/ar-testing-framework
```

### 基础用法

```bash
# 基础运行（使用默认参数）
python experiments/v0_v1_v2_v3_archive/v2_evaluation.py \
    --pkg com.google.ar.sceneform.samples.hellosceneform \
    --rounds 100
```

### 完整参数示例

```bash
python experiments/v0_v1_v2_v3_archive/v2_evaluation.py \
    --pkg com.google.ar.sceneform.samples.hellosceneform \
    --activity auto \
    --serial emulator-5554 \
    --rounds 200 \
    --seed 42 \
    --supported_ops tap,double_tap,drag,long_press,pinch_in,rotate \
    --unsupported_ops triple_tap,swipe,two_finger_tap,flick \
    --negative_sample_ratio 0.5 \
    --verify_wait_ms 200 \
    --log_csv results/evaluation_$(date +%Y%m%d_%H%M%S).csv \
    --print-interval 20
```

### 关键参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--pkg` | hellosceneform | 目标 Android app 包名 |
| `--activity` | auto | Activity 名称（auto 自动检测）|
| `--serial` | - | ADB 设备序列号（单设备可不填）|
| `--rounds` | 100 | 测试轮数 |
| `--supported_ops` | tap,double_tap,... | app支持的操作类型（逗号分隔）|
| `--unsupported_ops` | triple_tap,swipe,... | app不支持的操作类型（逗号分隔）|
| `--negative_sample_ratio` | 0.5 | Negative sample比例（0.0-1.0）|
| `--verify_wait_ms` | 200 | 操作后等待时间（ms）|
| `--log_csv` | - | CSV 日志输出路径 |
| `--seed` | - | 随机种子（用于可重复性）|
| `--prime_tap` | 1 | 操作前是否先轻触（1=是，0=否）|

### 操作类型选项

**支持的操作** (默认值)：
- `tap` - 单击
- `double_tap` - 双击
- `drag` - 拖拽
- `long_press` - 长按
- `pinch_in` - 捏合（缩小）
- `rotate` - 旋转

**不支持的操作** (默认值，用于测试FP)：
- `triple_tap` - 三连击
- `swipe` - 快速滑动
- `two_finger_tap` - 双指点击
- `flick` - 轻弹

**自定义示例**：
```bash
# 只测试部分支持的操作
--supported_ops tap,drag,rotate

# 添加更多不支持的操作
--unsupported_ops triple_tap,swipe,two_finger_tap,flick,long_drag

# 调整negative sample比例
--negative_sample_ratio 0.3  # 30%为negative samples
```

## 📝 输出说明

### 控制台输出

#### 启动时输出（操作分布）
```
[v2_eval] Operation distribution:
  double_tap        :  20 times  (✓ supported)
  drag              :  20 times  (✓ supported)
  flick             :  20 times  (✗ unsupported)
  long_press        :  20 times  (✓ supported)
  pinch_in          :  20 times  (✓ supported)
  rotate            :  20 times  (✓ supported)
  swipe             :  20 times  (✗ unsupported)
  tap               :  20 times  (✓ supported)
  triple_tap        :  20 times  (✗ unsupported)
  two_finger_tap    :  20 times  (✗ unsupported)
[v2_eval] Negative samples: 100/200 (50.0%)
[v2_eval] Random seed: 42
```

#### 实时输出
```
[v2_eval r001] cap=45.2ms  cv=123.4ms  action=856.3ms  verify&wait=245.8ms  TOTAL=1270.7ms  CV=1 GT=1 ✓:drag
[001/100] ✓ tap+drag from (512,384) to (612,384) bbox=(450,320,124,128)

[v2_eval r002] cap=42.1ms  cv=118.2ms  action=723.5ms  verify&wait=220.3ms  TOTAL=1104.1ms  CV=0 GT=0 ✓:triple_tap
[002/100] NEGATIVE(triple_tap): triple_tap at (250,180) interval=95ms

[v2_eval r003] cap=43.8ms  cv=121.7ms  action=890.2ms  verify&wait=235.1ms  TOTAL=1290.8ms  CV=1 GT=0 ✗:swipe
[003/100] ✗ swipe from (520,390) to (620,390)
```

**说明**：
- `CV=1` 表示 CV 验证通过，`CV=0` 表示未通过
- `GT=1` 表示 Ground Truth 确认成功，`GT=0` 表示失败
- `✓` 表示 CV 结果正确，`✗` 表示不正确
- `✓ 操作名` 表示支持的操作，`✗ 操作名` 表示不支持的操作
- `NEGATIVE(操作名)` 表示这是 negative sample（在AR物体外操作）

#### 最终评估报告
```
============================================================
[EVALUATION RESULTS]
============================================================
Total operations: 95
CV verified: 87/100 (87.0%)
GT verified: 91/100 (91.0%)
------------------------------------------------------------
True Positive (TP):     82  (CV=1, GT=1) ✓
True Negative (TN):      8  (CV=0, GT=0) ✓
False Positive (FP):     5  (CV=1, GT=0) ✗ CV误判
False Negative (FN):     0  (CV=0, GT=1) ✗ CV漏检
------------------------------------------------------------
Accuracy:  0.9474 (94.74%)
Precision: 0.9425
Recall:    1.0000
F1-Score:  0.9705
============================================================
```

### CSV 日志格式

生成的 CSV 文件包含以下列：

| 列名 | 说明 |
|------|------|
| step | 步骤序号 |
| detected | 是否检测到 AR 物体（1/0）|
| cv_verified | CV 验证结果（1/0）|
| gt_verified | Ground Truth 结果（1/0）|
| cv_correct | CV 是否正确（1/0）|
| operation | 操作类型 |
| is_negative | 是否为 negative sample（1/0）|
| is_supported | 操作是否被app支持（1/0）|
| cx_img, cy_img | AR 物体中心坐标（图像空间）|
| bbox_x, bbox_y, bbox_w, bbox_h | 边界框（图像空间）|
| message | 操作描述 |
| cap_ms, cv_ms, action_ms, verify&wait_ms, total_ms | 各阶段耗时 |

**CSV数据分析示例**：

```python
import pandas as pd

df = pd.read_csv('results/eval.csv')

# 按操作类型分析准确率
accuracy_by_op = df.groupby('operation').agg({
    'cv_correct': 'mean',
    'is_supported': 'first',
    'step': 'count'
}).rename(columns={'step': 'count', 'cv_correct': 'accuracy'})
print(accuracy_by_op)

# 分析 supported vs unsupported 的准确率
print("\nSupported operations accuracy:")
print(df[df['is_supported'] == 1]['cv_correct'].mean())

print("\nUnsupported operations accuracy:")
print(df[df['is_supported'] == 0]['cv_correct'].mean())

# 分析 positive vs negative samples 的准确率
print("\nPositive samples accuracy:")
print(df[df['is_negative'] == 0]['cv_correct'].mean())

print("\nNegative samples accuracy:")
print(df[df['is_negative'] == 1]['cv_correct'].mean())
```

## ⚠️ 重要注意事项

### 1. Ground Truth 来源

确保你的 Sample app 正确输出 logcat 日志：

```logcat
AR_OP: {"kind":"drag","ok":true,"ts_wall":1767649464274,"dTrans_m":0.06711415}
AR_OP: {"kind":"rotate","ok":true,"ts_wall":1767649467975,"dYaw_deg":124.665764}
AR_OP: {"kind":"long_press_end","ok":true,"ts_wall":1767649462287,"tap_id":4}
AR_OP: {"kind":"double_tap","ok":true,"ts_wall":1767649457869,"tap_id":3}
```

**必需格式**：
- TAG: `AR_OP`
- Level: `D` (Debug)
- 内容: JSON 格式，包含 `kind` 和 `ok` 字段

### 2. 操作与 logcat kind 映射

| 脚本操作名 | logcat kind |
|------------|-------------|
| drag | drag |
| rotate | rotate |
| pinch_in | pinch |
| long_press | long_press_end |
| double_tap | double_tap |

### 3. 避免 long_press_hold 误触

脚本已优化 drag 操作的按压时间（仅 30ms），避免触发 `long_press_hold`。

### 4. 时序匹配

脚本在操作后：
1. 等待 `verify_wait_ms` (默认 200ms)
2. 捕获后置图像进行 CV 验证
3. 额外等待 100ms
4. 读取最近的 logcat 记录检测 GT

确保 `verify_wait_ms` 足够长，让 AR app 完成响应并输出日志。

## 🔍 常见问题

### Q1: 为什么 GT verified 总是 0？

**可能原因**：
- Sample app 没有正确输出 logcat
- logcat 被清空或没有权限读取
- `verify_wait_ms` 太短，日志还没输出

**解决方法**：
```bash
# 检查 logcat 输出
adb logcat -s AR_OP:D

# 增加等待时间
--verify_wait_ms 300
```

### Q2: 准确率很低怎么办？

**可能原因**：
- CV 阈值参数不合适
- AR app 对某些操作不响应
- 时间窗口设置不当

**解决方法**：
- 调整 CV 参数（`--verify_min_frac`, `--rotate_min_deg` 等）
- 只测试 app 支持的操作类型
- 增加 `verify_wait_ms`

### Q3: 如何只测试特定操作？

使用 `--operations` 参数：
```bash
# 只测试 drag 和 rotate
--operations drag,rotate
```

## 📈 结果分析建议

1. **准确率 > 90%**：CV 算法可靠，可以用于实际测试
2. **高 FP（误判）**：CV 过于敏感，考虑提高阈值
3. **高 FN（漏检）**：CV 过于保守，考虑降低阈值
4. **TN 很少**：说明大部分操作都成功了（正常现象）

## 🛠️ 调试技巧

### 启用详细日志

```bash
# 同时查看实时 logcat
adb logcat -s AR_OP:D &

# 运行评估
python experiments/v0_v1_v2_v3_archive/v2_evaluation.py --rounds 10
```

### 保存完整日志

```bash
python experiments/v0_v1_v2_v3_archive/v2_evaluation.py \
    --rounds 100 \
    --log_csv results/eval.csv \
    2>&1 | tee results/eval.log
```

### 分析 CSV 数据

```python
import pandas as pd

df = pd.read_csv('results/eval.csv')

# 查看 CV 错误的案例
errors = df[df['cv_correct'] == 0]
print(errors[['operation', 'cv_verified', 'gt_verified', 'message']])

# 按操作类型统计准确率
accuracy_by_op = df.groupby('operation').agg({
    'cv_correct': 'mean',
    'cv_verified': 'sum',
    'gt_verified': 'sum'
})
print(accuracy_by_op)
```

## 📚 相关文件

- `src/common/actions.py` - 新增了 `double_tap` 函数
- `src/common/verify_motion.py` - 扩展支持 `long_press` 和 `double_tap`
- `cv/verify_motion.py` - CV 验证算法
- `cv/strategy_yolo.py` - YOLO 目标检测

## 🔄 更新记录

- **2026-01-05**: 初始版本
  - 添加 long_press 和 double_tap 支持
  - 实现 Ground Truth 检测
  - 计算 Accuracy, Precision, Recall, F1-Score
  - 生成详细的评估报告和 CSV 日志
