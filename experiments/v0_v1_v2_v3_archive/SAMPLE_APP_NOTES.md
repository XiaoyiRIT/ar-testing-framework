# HelloSceneform Sample App - 实现细节

## 📱 App信息

- **包名**: `com.google.ar.sceneform.samples.hellosceneform`
- **Activity**: `HelloSceneformActivity`
- **Logcat TAG**: `AR_OP`
- **源码路径**: `experiments/hellosceneform/app/src/main/java/.../HelloSceneformActivity.java`

## 🎯 支持的操作

### 单指操作

| 操作 | logcat kind | 触发条件 | 输出字段 |
|------|-------------|----------|----------|
| **单击** | `tap` | 快速轻触（< 双击超时） | tap_id, target (plane/node/empty), selected |
| **双击** | `double_tap` | 两次快速点击 | tap_id |
| **长按** | `long_press_hold`<br>`long_press_end` | 按住 > 长按超时 | tap_id |
| **拖拽** | `drag` | 单指移动AR物体 | dTrans_m (移动距离，单位米) |

### 双指操作

| 操作 | logcat kind | 触发条件 | 输出字段 |
|------|-------------|----------|----------|
| **捏合/缩放** | `pinch` | 双指距离变化 | scale_factor, dScale_abs |
| **旋转** | `rotate` | 双指角度变化 | dYaw_deg (旋转角度) |

### 放置操作

| 操作 | logcat kind | 说明 |
|------|-------------|------|
| **开始放置** | `place_start` | 点击平面开始放置AR物体 |
| **放置成功** | `place_ok` | AR物体成功锚定（3帧TRACKING确认）|
| **放置失败** | `place_fail` | 锚点丢失或停止追踪 |

## 🔍 关键实现细节

### 1. Tap vs Place
- 每次单击**总是**触发`tap`事件
- 如果点击在AR平面上，**额外**触发`place_*`事件
- 两者共享同一个`tap_id`

### 2. Pinch操作（重要！）
- App中的`pinch`是**双向的**：
  - `scale_factor > 1.0` → 放大
  - `scale_factor < 1.0` → 缩小
- 测试工具中`pinch_in`会映射到app的`pinch`（正确）
- 如果需要区分放大/缩小，需要检查`scale_factor`字段

### 3. Long Press两阶段
- **`long_press_hold`**: 长按触发时立即输出（中间状态）
- **`long_press_end`**: 长按结束时输出（最终确认）
- 测试工具映射`long_press` → `long_press_end`（正确）

### 4. Drag vs Rotate vs Pinch 优先级
App使用"优势规则"（dominance rule）决定输出哪个操作：
```
如果 pinch明显 且 rotate不明显 → 输出 pinch
如果 rotate明显 且 pinch不明显 → 输出 rotate
如果 两者都明显:
    - 比较归一化分数（相对阈值的倍数）
    - 使用1.6倍优势比例（R=1.6）
    - 输出优势更大的操作
```

### 5. 成功判定阈值

| 操作 | 阈值常量 | 值 | 说明 |
|------|---------|-----|------|
| Drag | `EPS_T_M` | 0.002m | 2mm 最小移动距离 |
| Pinch | `EPS_S` | 0.02 | 2% 最小缩放比例 |
| Rotate | `EPS_R_DEG` | 3° | 3度 最小旋转角度 |

这些阈值决定了什么时候操作会被记录为成功（`ok: true`）。

## 📊 Logcat输出示例

### 正常操作序列
```logcat
AR_OP: {"kind":"tap","ok":true,"ts_wall":1767649456790,"tap_id":1,"target":"plane","selected":true}
AR_OP: {"kind":"place_start","ok":true,"ts_wall":1767649456489,"tap_id":1,"anchor_pose":"-0.114,-0.909,-1.334"}
AR_OP: {"kind":"place_ok","ok":true,"ts_wall":1767649456545,"tap_id":1,"anchor_pose":"-0.114,-0.909,-1.334"}
AR_OP: {"kind":"double_tap","ok":true,"ts_wall":1767649457869,"tap_id":3}
AR_OP: {"kind":"drag","ok":true,"ts_wall":1767649464274,"dTrans_m":0.06711415}
AR_OP: {"kind":"rotate","ok":true,"ts_wall":1767649467975,"dYaw_deg":124.665764}
AR_OP: {"kind":"pinch","ok":true,"ts_wall":1767649470123,"scale_factor":1.45,"dScale_abs":0.15}
```

### Long Press序列
```logcat
AR_OP: {"kind":"long_press_hold","ok":true,"ts_wall":1767649461576,"tap_id":4}
AR_OP: {"kind":"long_press_end","ok":true,"ts_wall":1767649462287,"tap_id":4}
```

### 误操作（hold太久触发long_press_hold，然后drag）
```logcat
AR_OP: {"kind":"long_press_hold","ok":true,"ts_wall":1767649477037,"tap_id":7}
AR_OP: {"kind":"drag","ok":true,"ts_wall":1767649478170,"dTrans_m":0.021772636}
```

## ⚠️ 测试工具注意事项

### 1. 避免误触long_press
- **问题**: drag操作如果press时间过长，会先触发`long_press_hold`
- **解决**: `drag_line()` 中的pause已设为30ms（✓ 已实现）
- **验证**: 确保`verify_wait_ms`足够长，让所有事件输出

### 2. 操作名映射（已正确）
```python
op_map = {
    "pinch_in": "pinch",      # ✓ 测试工具的pinch_in映射到app的pinch
    "long_press": "long_press_end",  # ✓ 使用最终确认事件
    "tap": "tap",
    "double_tap": "double_tap",
    "drag": "drag",
    "rotate": "rotate",
}
```

### 3. 不支持的操作（期望无输出）
- `triple_tap` - App不实现，不会有logcat输出 ✓
- `swipe` - App不区分swipe和drag，可能被识别为drag
- `two_finger_tap` - App不实现，不会有输出 ✓
- `flick` - App不区分flick和drag，可能被识别为drag

### 4. Double Tap的副作用
**重要**：App中双击会**删除**被点击的AR节点！
- 这是正常行为
- 测试时需要注意节点可能被删除
- 可能影响后续操作（节点不存在了）

### 5. Ground Truth检测时间窗口
建议设置：
- `verify_wait_ms`: 200-300ms（等待AR响应和logcat输出）
- `time_window_sec`: 2.0s（在logcat中搜索最近的匹配记录）

## 🔬 CV验证阈值建议

基于app的成功阈值，CV验证阈值应设置为相近值：

```python
# 推荐的CV阈值（experiments/v0_v1_v2_v3_archive/v2_evaluation.py）
--drag_min_px 8.0          # ~2mm in typical phone screen
--rotate_min_deg 15.0      # app用3度，CV可以宽松些
--pinch_scale_thr 0.10     # app用2%，CV用10%（宽松）
--verify_min_frac 0.5      # 50%特征点一致
```

## 📝 建议的测试配置

### 基础测试（验证app是否正常）
```bash
python experiments/v0_v1_v2_v3_archive/v2_evaluation.py \
    --pkg com.google.ar.sceneform.samples.hellosceneform \
    --rounds 60 \
    --seed 42 \
    --supported_ops tap,double_tap,drag,long_press,pinch_in,rotate \
    --unsupported_ops triple_tap,two_finger_tap \
    --negative_sample_ratio 0.3 \
    --verify_wait_ms 250 \
    --log_csv results/baseline.csv
```

### 完整评估
```bash
python experiments/v0_v1_v2_v3_archive/v2_evaluation.py \
    --pkg com.google.ar.sceneform.samples.hellosceneform \
    --rounds 200 \
    --seed 42 \
    --supported_ops tap,double_tap,drag,long_press,pinch_in,rotate \
    --unsupported_ops triple_tap,swipe,two_finger_tap,flick \
    --negative_sample_ratio 0.5 \
    --verify_wait_ms 250 \
    --log_csv results/full_eval_$(date +%Y%m%d_%H%M%S).csv
```

## 🎯 预期结果

### Supported Operations
- **Tap/Double-tap/Long-press**: 准确率应该很高（> 95%）
- **Drag**: 中等准确率（80-90%），取决于CV检测灵敏度
- **Rotate**: 中等准确率（75-85%），旋转检测较复杂
- **Pinch**: 中等准确率（75-85%），缩放检测较复杂

### Unsupported Operations
- **Triple-tap**: 应该100%识别为不支持（GT=0）
- **Two-finger-tap**: 应该100%识别为不支持（GT=0）
- **Swipe/Flick**: 可能被app识别为drag（需要注意！）

### Negative Samples
- CV应该正确识别大部分negative samples（在AR物体外操作）
- 预期准确率 > 90%

## 🐛 已知问题和限制

1. **Swipe/Flick可能被识别为Drag**
   - App不区分这些快速手势
   - 如果移动距离 > 2mm，会被记录为drag
   - 这**不是bug**，而是app设计决策

2. **Place操作需要特殊处理**
   - Place会同时触发tap和place_*事件
   - 测试工具中place不在默认操作列表中
   - 如果需要测试place，需要单独配置

3. **双击删除节点**
   - 多次双击可能导致所有节点被删除
   - 需要控制double_tap的测试次数

## 📚 相关文件

- **App源码**: `experiments/hellosceneform/app/src/main/java/.../HelloSceneformActivity.java`
- **测试工具**: `experiments/v0_v1_v2_v3_archive/v2_evaluation.py`
- **MotionVerifier**: `src/common/verify_motion.py`
- **Actions**: `src/common/actions.py`
