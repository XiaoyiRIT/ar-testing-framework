# Project Context

- **Path**: `/Users/yangxiaoyi/Desktop/phd/project/newproject/program`
- **Time (UTC)**: 2025-10-31 17:06:01Z
- **Git Branch**: main
- **Git Commit**: d18c655

## Directory Tree (depth=3)

```
program/
├── common/
│   ├── actions.py
│   ├── device.py
│   ├── locator_iface.py
│   ├── policy_random.py
│   ├── timing.py
│   └── verify_motion.py
├── cv/
│   ├── yolo/
│   │   ├── testimages/
│   │   ├── coco.yaml
│   │   ├── coco8.yaml
│   │   ├── myar2.yaml
│   │   ├── myar_simp.yaml
│   │   └── test_yolo_train.py
│   ├── strategy_bgmedian.py
│   ├── strategy_flow_cluster.py
│   ├── strategy_FoELS.py
│   ├── strategy_grabcut_refine.py
│   ├── strategy_riri_invariant.py
│   ├── strategy_templatematch.py
│   ├── strategy_twodiff_orb.py
│   ├── strategy_yolo.py
│   └── verify_motion.py
├── val/
├── .gitignore
├── config.yaml
├── gen_project_context.py
├── main.py
├── README.md
├── requirements.txt
├── smoke.py
├── temp.java
├── v0_ar_monkey_adb.py
├── v0_ar_monkey_appium.py
├── v1_ar_monkey_appium.py
├── v2_ar_monkey_appium.py
├── v3_ar_monkey_appium.py
├── yolo_adb_screencap.py
├── yolo_scratch.py
└── yolo_val.py
```

## File Stats

- Files counted (excluded patterns applied): **9995**
- Total size: **1.7 MB**

### Top 15 Largest Files (excluded patterns applied)

| Size | Path |
|---:|---|
| 17.2 KB | `cv/strategy_FoELS.py` |
| 14.8 KB | `temp.java` |
| 14.7 KB | `v1_ar_monkey_appium.py` |
| 14.7 KB | `v2_ar_monkey_appium.py` |
| 13.9 KB | `v3_ar_monkey_appium.py` |
| 12.3 KB | `cv/yolo/runs/train/exp3/results.csv` |
| 11.7 KB | `cv/yolo/runs/train/exp2/results.csv` |
| 10.0 KB | `cv/verify_motion.py` |
| 9.9 KB | `cv/yolo/runs/train/exp4/results.csv` |
| 9.4 KB | `gen_project_context.py` |
| 8.6 KB | `v0_ar_monkey_adb.py` |
| 8.2 KB | `common/actions.py` |
| 8.1 KB | `common/verify_motion.py` |
| 6.7 KB | `yolo_adb_screencap.py` |
| 6.4 KB | `cv/yolo/test_yolo_train.py` |

## Recent Commits

```
d18c655 Create README.md
7143e3e init: import existing project
```

## File Snippets (first 120 lines)

### `common/actions.py`

```text
# common/actions.py
# ------------------------------------------------------------
# Description:
#   Coordinate-only gesture primitives implemented with
#   Selenium ActionBuilder (touch pointers). This mirrors
#   your working v0 semantics to avoid API differences across
#   Selenium/Appium versions.
# ------------------------------------------------------------

import math
from selenium.webdriver.common.actions.action_builder import ActionBuilder
from selenium.webdriver.common.actions.pointer_input import PointerInput
from selenium.webdriver.common.actions.interaction import Interaction

def tap(driver, x, y, press_ms=60):
    a  = ActionBuilder(driver)
    f1 = a.add_pointer_input(kind="touch", name="finger1")
    f1.create_pointer_move(duration=0, x=int(x), y=int(y))
    f1.create_pointer_down(button=0)
    f1.create_pause(press_ms/1000.0)
    f1.create_pointer_up(button=0)
    a.perform()

def long_press(driver, x, y, hold_ms=900):
    a  = ActionBuilder(driver)
    f1 = a.add_pointer_input(kind="touch", name="finger1")
    f1.create_pointer_move(duration=0, x=int(x), y=int(y))
    f1.create_pointer_down(button=0)
    f1.create_pause(hold_ms/1000.0)
    f1.create_pointer_up(button=0)
    a.perform()

def drag_line(driver, x1, y1, x2, y2, duration_ms=600):
    a  = ActionBuilder(driver)
    f1 = a.add_pointer_input(kind="touch", name="finger1")
    f1.create_pointer_move(duration=0, x=int(x1), y=int(y1))
    f1.create_pointer_down(button=0)
    f1.create_pause(0.03)
    f1.create_pointer_move(duration=duration_ms, x=int(x2), y=int(y2))
    f1.create_pointer_up(button=0)
    a.perform()

def pinch_or_zoom(driver, cx, cy, start_dist=60, end_dist=220, duration_ms=600):
    """
    If end_dist < start_dist => pinch; else => zoom.
    Uses diagonal symmetry (↙↗).
    """
    dx0 = dy0 = start_dist // 2
    dx1 = dy1 = end_dist   // 2
    s1 = (int(cx - dx0), int(cy - dy0)); e1 = (int(cx - dx1), int(cy - dy1))
    s2 = (int(cx + dx0), int(cy + dy0)); e2 = (int(cx + dx1), int(cy + dy1))

    a  = ActionBuilder(driver)
    f1 = a.add_pointer_input(kind="touch", name="finger1")
    f2 = a.add_pointer_input(kind="touch", name="finger2")

    f1.create_pointer_move(0, x=s1[0], y=s1[1])
    f2.create_pointer_move(0, x=s2[0], y=s2[1])
    f1.create_pointer_down(button=0)
    f2.create_pointer_down(button=0)
    f1.create_pointer_move(duration_ms, x=e1[0], y=e1[1])
    f2.create_pointer_move(duration_ms, x=e2[0], y=e2[1])
    f1.create_pointer_up(button=0)
    f2.create_pointer_up(button=0)
    a.perform()

def rotate(
    driver,
    cx, cy,
    radius=200,
    angle_deg=60,
    duration_ms=800,
    steps=8,
    direction="ccw",   # "ccw" 逆时针 | "cw" 顺时针
    start_deg=0        # 可选：两指起始基准角（第二指为 start_deg+180）
):
    """
    Two-finger rotation around (cx, cy) with both fingers sweeping the SAME angular direction.
    - direction: "ccw" (counter-clockwise) or "cw" (clockwise)
    - start_deg: starting angle for finger1; finger2 starts at start_deg + 180
    Screen coords notice: y 轴向下增长，若方向与预期相反，切换 direction 即可。
    """
    sgn = +1 if str(direction).lower() == "ccw" else -1

    def pos(deg):
        r = math.radians(deg)
        return int(cx + radius * math.cos(r)), int(cy + radius * math.sin(r))

    # 起点：两指相差 180°
    f1_start = float(start_deg)
    f2_start = float(start_deg + 180.0)

    a  = ActionBuilder(driver)
    f1 = a.add_pointer_input(kind="touch", name="finger1")
    f2 = a.add_pointer_input(kind="touch", name="finger2")

    x1, y1 = pos(f1_start)
    x2, y2 = pos(f2_start)
    f1.create_pointer_move(0, x=x1, y=y1)
    f2.create_pointer_move(0, x=x2, y=y2)
    f1.create_pointer_down(button=0)
    f2.create_pointer_down(button=0)

    # 分段同步走弧线
    seg = max(1, int(duration_ms / max(1, int(steps))))
    for i in range(1, steps + 1):
        delta = sgn * (angle_deg / steps) * i  # 两指同向，同幅度
        nx1, ny1 = pos(f1_start + delta)
        nx2, ny2 = pos(f2_start + delta)
        f1.create_pointer_move(seg, x=nx1, y=ny1)
        f2.create_pointer_move(seg, x=nx2, y=ny2)

    f1.create_pointer_up(button=0)
    f2.create_pointer_up(button=0)
    a.perform()

    
    # common/actions.py

def smart_correct_rule_based(
... (truncated)
```

### `common/device.py`

```text
# common/device.py
# ------------------------------------------------------------
# Description:
#   Appium driver bootstrap (UiAutomator2Options) + adb helpers
#   + simple utilities (window size, screenshot if需要扩展).
#   Mirrors your working connection semantics.
# ------------------------------------------------------------

import subprocess, cv2, numpy as np, time
from typing import Optional
from appium import webdriver
from appium.options.android import UiAutomator2Options
from selenium.common.exceptions import WebDriverException


def _try_connect(url, options):
    try:
        drv = webdriver.Remote(url, options=options)
        return drv
    except WebDriverException as e:
        # 典型 404/UnknownCommandError：base-path 不对
        msg = str(e)
        if "UnknownCommandError" in msg or "requested resource could not be found" in msg or "404" in msg:
            return None
        # 其他异常直接抛
        raise

def sh(cmd: str) -> str:
    return subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT).decode("utf-8", "ignore")

def adb(cmd: str, serial: Optional[str] = None) -> str:
    base = f"adb -s {serial} " if serial else "adb "
    return sh(base + cmd)

def one_device_or_none() -> Optional[str]:
    out = sh("adb devices")
    lines = [l for l in out.strip().splitlines() if "\tdevice" in l and not l.startswith("List")]
    return lines[0].split("\t")[0] if len(lines) == 1 else None

def resolve_main_activity(pkg: str, serial: Optional[str] = None) -> Optional[str]:
    intent = 'intent:#Intent;action=android.intent.action.MAIN;category=android.intent.category.LAUNCHER;end'
    try:
        out = adb(f'shell cmd package query-activities --brief "{intent}"', serial)
        for line in out.splitlines():
            line = line.strip()
            if line.startswith(pkg + "/"):
                return line.split("/", 1)[1]
    except Exception:
        pass
    return None

def make_driver(pkg: str,
                activity: Optional[str],
                serial: Optional[str],
                warmup_wait: float = 3.0):
    """Create driver with your v0-style capabilities."""
    caps = {
        "platformName": "Android",
        "appium:automationName": "UiAutomator2",
        "appium:deviceName": serial or (one_device_or_none() or "Android"),
        "appium:udid": serial if serial else one_device_or_none(),
        "appium:noReset": True,
        "appium:autoGrantPermissions": True,
        "appium:newCommandTimeout": 300,
        "appium:appPackage": pkg,
        "appium:disableWindowAnimation": True,
    }

    if activity and activity not in ("auto", None):
        caps["appium:appActivity"] = activity
        caps["appium:appWaitActivity"] = "*"
    else:
        caps["appium:intentAction"]   = "android.intent.action.MAIN"
        caps["appium:intentCategory"] = "android.intent.category.LAUNCHER"
        caps["appium:intentFlags"]    = "0x10200000"
        caps["appium:appWaitActivity"] = "*"

    options = UiAutomator2Options().load_capabilities(caps)
    
    # 允许从环境或 config 里传 server_url；没有就用默认
    server_url = "http://127.0.0.1:4723"  # 你也可以从 config 传进来
    # 组装两条候选：根路径 与 /wd/hub（去重）
    candidates = []
    if server_url.rstrip("/").endswith("/wd/hub"):
        root = server_url.rstrip("/").rsplit("/wd/hub", 1)[0]
        candidates = [server_url, root]
    else:
        hub = server_url.rstrip("/") + "/wd/hub"
        candidates = [server_url, hub]

    last_err = None
    for url in candidates:
        drv = _try_connect(url, options)
        if drv is not None:
            print(f"[device] connected via {url}")
            time.sleep(warmup_wait)
            return drv
    # 两条都失败，抛出更友好的错误
    raise RuntimeError(
        f"Failed to create session. Tried: {candidates}. "
        "Check Appium base path (root vs /wd/hub) and that uiautomator2 driver is installed "
        "(appium driver install uiautomator2)."
    )

def get_window_size(driver):
    s = driver.get_window_size()
    return int(s["width"]), int(s["height"])

def capture_bgr(driver):
    png = driver.get_screenshot_as_png()
    arr = np.frombuffer(png, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)
```

### `common/locator_iface.py`

```text
# common/locator_iface.py
# ------------------------------------------------------------
# Description:
#   Target locator interface. Implementations return an object
#   with "center" and "bbox" for the detected AR object.
# ------------------------------------------------------------

from typing import Optional, Tuple, Dict, Any
import numpy as np

class ITargetLocator:
    def reset(self) -> None:
        """Reset internal state if any."""
        ...

    def locate(self, curr_bgr: np.ndarray, prev_bgr: Optional[np.ndarray] = None
              ) -> Optional[Dict[str, Any]]:
        """
        Return:
          {"center": (cx, cy), "bbox": (x, y, w, h)}  or  None if not found.
        """
        ...
```

### `common/policy_random.py`

```text
# common/policy_random.py
# ------------------------------------------------------------
# Description:
#   Random gesture policy used by v0 and as fallback in v1.
#   Generates coordinates + calls coordinate-only actions.
# ------------------------------------------------------------

import random
from typing import Tuple
from common.actions import tap, drag_line, long_press, pinch_or_zoom, rotate

def _rand_point(w: int, h: int, margin_ratio: float = 0.05) -> Tuple[int,int]:
    L = int(w*margin_ratio); T = int(h*margin_ratio)
    R = w - L - 1;          B = h - T - 1
    return random.randint(L, R), random.randint(T, B)

def step(driver,
         w: int, h: int,
         safe_box=(0.12, 0.18, 0.88, 0.88),
         drag_ms=(300, 900), long_ms=(700, 1500),
         pinch_start=80, pinch_end=220,
         rotate_radius=(160, 260), rotate_angle=(30, 90), rotate_steps=8):
    """执行一次随机动作，返回字符串描述便于日志记录。"""
    L, T, R, B = int(w*safe_box[0]), int(h*safe_box[1]), int(w*safe_box[2]), int(h*safe_box[3])
    op = random.choice(["tap","drag","long","pinch","zoom","rotate"])
    #op = random.choice(["drag","pinch","zoom","rotate"])

    if op == "tap":
        x = random.randint(L, R); y = random.randint(T, B)
        tap(driver, x, y, press_ms=random.randint(40, 120))
        return f"tap({x},{y})"

    if op == "drag":
        x1 = random.randint(L, R); y1 = random.randint(T, B)
        x2 = random.randint(L, R); y2 = random.randint(T, B)
        dur = random.randint(*drag_ms)
        drag_line(driver, x1, y1, x2, y2, duration_ms=dur)
        return f"drag({x1},{y1}->{x2},{y2},{dur}ms)"

    if op == "long":
        x = random.randint(L, R); y = random.randint(T, B)
        hold = random.randint(*long_ms)
        long_press(driver, x, y, hold_ms=hold)
        return f"long({x},{y},{hold}ms)"

    if op in ("pinch", "zoom"):
        cx = random.randint(L + 40, R - 40)
        cy = random.randint(T + 40, B - 40)
        s, e = (pinch_end, pinch_start) if op == "pinch" else (pinch_start, pinch_end)
        pinch_or_zoom(driver, cx, cy, start_dist=s, end_dist=e,
                      duration_ms=random.randint(450, 900))
        return f"{op}({cx},{cy},{s}->{e})"

    # rotate
    cx = random.randint(L + 60, R - 60)
    cy = random.randint(T + 60, B - 60)
    radius = random.randint(*rotate_radius)
    angle  = random.randint(*rotate_angle)
    rotate(driver, cx, cy, radius=radius, angle_deg=angle,
           duration_ms=random.randint(600, 1100), steps=rotate_steps)
    return f"rotate({cx},{cy},r={radius},θ={angle}°)"
```

### `common/timing.py`

```text
# -*- coding: utf-8 -*-
"""
common/timing.py
--------------------------------
通用计时工具，用于 v0/v1/v3 统计：
- 程序总时长（含准备阶段）
- 模拟操作时长（你控制开始/结束）
- 平均每个操作耗时（传入总操作数；可选分手势统计）
- v0 场景可从 monkey 的标准输出解析 "Events injected: N"

用法要点：
- 创建 Timing() 即启动“程序总时长”计时
- 在模拟操作开始处调用 .start_sim()
- 在模拟操作结束处调用 .stop_sim()
- 结束前调用 .set_ops_total(n)（或 .set_ops_by_kind({...})），然后打印 .summary_str()
"""

import time
import re
from typing import Optional, Dict, IO

_MONKEY_INJECTED_RE = re.compile(r"Events injected:\s*(\d+)", re.IGNORECASE)

class Timing:
    def __init__(self) -> None:
        self._t_prog_start = time.time()
        self._t_sim_start: Optional[float] = None
        self._t_sim_end: Optional[float] = None
        self._ops_total: Optional[int] = None
        self._ops_by_kind: Dict[str, int] = {}   # 可选：{"place":N, "drag":M, ...}

    # --- 时钟控制 ---
    def start_sim(self) -> None:
        if self._t_sim_start is None:
            self._t_sim_start = time.time()

    def stop_sim(self) -> None:
        if self._t_sim_start is not None and self._t_sim_end is None:
            self._t_sim_end = time.time()

    # --- 事件数设置 ---
    def set_ops_total(self, n: int) -> None:
        self._ops_total = max(0, int(n))

    def set_ops_by_kind(self, d: Dict[str, int]) -> None:
        # 允许只传有意义的 key，例如 {"drag": 120, "pinch": 0}
        clean = {}
        for k, v in (d or {}).items():
            try:
                clean[str(k)] = max(0, int(v))
            except Exception:
                pass
        self._ops_by_kind = clean
        # 若未设置总数，自动合计
        if self._ops_total is None:
            self._ops_total = sum(clean.values())

    # --- 从 monkey 输出解析注入事件数（v0 专用，可选） ---
    @staticmethod
    def parse_monkey_injected_from_stream(stream: IO[str]) -> Optional[int]:
        """
        读取 monkey 的 stdout（文本流），找到 'Events injected: N'。
        若未找到返回 None。你可以一边读取一边把每行丢给本函数解析。
        """
        for line in stream:
            m = _MONKEY_INJECTED_RE.search(line)
            if m:
                try:
                    return int(m.group(1))
                except Exception:
                    pass
        return None

    # --- 统计取值 ---
    def total_runtime(self) -> float:
        return time.time() - self._t_prog_start

    def sim_runtime(self) -> float:
        if self._t_sim_start is None:
            return 0.0
        end = self._t_sim_end if self._t_sim_end is not None else time.time()
        return max(0.0, end - self._t_sim_start)

    def avg_time_per_op(self) -> float:
        ops = (self._ops_total or 0)
        return 0.0 if ops <= 0 else self.sim_runtime() / float(ops)

    def avg_time_per_kind(self) -> Dict[str, float]:
        if not self._ops_by_kind:
            return {}
        rt = self.sim_runtime()
        return {k: (0.0 if v <= 0 else rt / float(v)) for k, v in self._ops_by_kind.items()}

    # --- 输出 ---
    def summary_dict(self) -> Dict[str, object]:
        out = {
            "total_runtime_sec": round(self.total_runtime(), 6),
            "sim_runtime_sec": round(self.sim_runtime(), 6),
            "ops_total": int(self._ops_total or 0),
            "avg_time_per_op_sec": round(self.avg_time_per_op(), 6),
        }
        if self._ops_by_kind:
            out["ops_by_kind"] = {k: int(v) for k, v in self._ops_by_kind.items()}
            out["avg_time_per_kind_sec"] = {k: round(v, 6) for k, v in self.avg_time_per_kind().items()}
        return out

    def summary_str(self) -> str:
        d = self.summary_dict()
        lines = []
        lines.append("[timing]")
        lines.append("  total_runtime_sec       = {:.3f}".format(d["total_runtime_sec"]))
        lines.append("  sim_runtime_sec         = {:.3f}".format(d["sim_runtime_sec"]))
        lines.append("  ops_total               = {}".format(d["ops_total"]))
        lines.append("  avg_time_per_op_sec     = {:.6f}".format(d["avg_time_per_op_sec"]))
        if "ops_by_kind" in d:
            lines.append("  ops_by_kind             = {}".format(d["ops_by_kind"]))
            lines.append("  avg_time_per_kind_sec   = {}".format(d["avg_time_per_kind_sec"]))
        return "\n".join(lines)
```

### `common/verify_motion.py`

```text
# -*- coding: utf-8 -*-
"""
common/verify_motion.py
--------------------------------
- 解析 TAG=AR_OP 的单行 JSON（place/drag/pinch/rotate/tap）
- 计算 per-kind 与 overall 成功率
- overall 的口径可配置：overall_basis ∈ {"place","tap"}（默认 "place"）
  * "place": overall = place + drag + pinch + rotate（tap 不计，避免 place+tap 双计）
  * "tap":   overall = tap   + drag + pinch + rotate（place 完全当作 tap 的特例，不计入 overall）

注：place 在 app 里发生时通常也会有一条 tap；若选 "tap" 作为口径，place 仍单独打印，但不会进 overall。
"""

import json
import re
import time
import threading
import subprocess
from collections import Counter
from typing import Optional, Dict, Any, List

TAG = "AR_OP"
JSON_RE = re.compile(r'\bAR_OP\b[^{]*({.*})')  # 兼容 "AR_OP   :" / "AR_OP(1234):" 等

KINDS_ALL = ("place_start", "place_ok", "place_fail", "drag", "pinch", "rotate", "tap")

def _safe_rate(ok: int, total: int) -> float:
    return 0.0 if total <= 0 else ok / float(total)

def parse_line(line: str) -> Optional[Dict[str, Any]]:
    if "AR_OP" not in line:
        return None
    m = JSON_RE.search(line)
    if not m:
        return None
    try:
        evt = json.loads(m.group(1))
        k = evt.get("kind")
        if isinstance(k, str):
            evt["kind"] = k.strip().lower()
        return evt
    except Exception:
        return None

class MotionStats:
    """线程安全计数与成功率。"""
    def __init__(self, overall_basis: str = "place") -> None:
        self._cnt = Counter()
        self._t0 = time.time()
        self._lock = threading.Lock()
        self._pending_place = 0
        self._overall_basis = overall_basis if overall_basis in ("place","tap") else "place"

    def set_overall_basis(self, basis: str) -> None:
        if basis not in ("place","tap"):
            basis = "place"
        with self._lock:
            self._overall_basis = basis

    def reset(self) -> None:
        with self._lock:
            self._cnt.clear()
            self._t0 = time.time()
            self._pending_place = 0

    def feed_event(self, evt: Dict[str, Any]) -> None:
        kind = str(evt.get("kind", "")).strip().lower()
        ok = bool(evt.get("ok", False))
        if kind not in KINDS_ALL:
            return
        with self._lock:
            if kind == "place_start":
                self._cnt["place_start_total"] += 1
                self._pending_place += 1
                return
            if kind == "place_ok":
                if self._pending_place > 0:
                    self._cnt["place_ok_total"] += 1
                    self._pending_place -= 1
                return
            if kind == "place_fail":
                if self._pending_place > 0:
                    self._pending_place -= 1
                self._cnt["place_fail_total"] += 1
                return
            # 其余：drag/pinch/rotate/tap
            self._cnt[f"{kind}_total"] += 1
            if ok:
                self._cnt[f"{kind}_ok"] += 1

    def feed_line(self, line: str) -> None:
        evt = parse_line(line)
        if evt:
            self.feed_event(evt)

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            c = dict(self._cnt)
            elapsed = time.time() - self._t0
            basis = self._overall_basis

        place_ok  = c.get("place_ok_total", 0)
        place_try = c.get("place_start_total", 0)
        drag_ok, drag_total = c.get("drag_ok", 0), c.get("drag_total", 0)
        pin_ok,  pin_total  = c.get("pinch_ok", 0), c.get("pinch_total", 0)
        rot_ok,  rot_total  = c.get("rotate_ok", 0), c.get("rotate_total", 0)
        tap_ok,  tap_total  = c.get("tap_ok", 0),  c.get("tap_total", 0)

        if basis == "tap":
            ok_sum  = tap_ok + drag_ok + pin_ok + rot_ok
            try_sum = tap_total + drag_total + pin_total + rot_total
        else:  # "place"
            ok_sum  = place_ok + drag_ok + pin_ok + rot_ok
            try_sum = place_try + drag_total + pin_total + rot_total

        return {
            "elapsed_sec": elapsed,
            "counts": c,
            "place":   {"ok": place_ok, "total": place_try, "rate": _safe_rate(place_ok, place_try)},
            "drag":    {"ok": drag_ok,  "total": drag_total, "rate": _safe_rate(drag_ok, drag_total)},
... (truncated)
```

### `cv/strategy_bgmedian.py`

```text
# cv/strategy_bgmedian.py
# ------------------------------------------------------------
# Background subtraction (MOG2/median-ish) + contour selection.
# Exports: locate(curr_bgr, prev_bgr=None, min_area_ratio=0.002, max_w=640)
# Notes:
#   - Stateless wrapper: builds a pseudo background by lightly blurring
#     and subtracting; if prev_bgr is given, we align roughly by ECC; otherwise
#     we just threshold high-gradient regions as foreground proxy.
# ------------------------------------------------------------
import cv2, numpy as np
from typing import Optional, Dict, Any

def _resize_keep(src, max_w=640):
    h, w = src.shape[:2]
    if w <= max_w: return src, 1.0
    s = max_w / w
    return cv2.resize(src, (int(w*s), int(h*s)), cv2.INTER_AREA), s

def _align_ecc(ref_gray, mov_gray):
    # ECC homography alignment (robust for small camera shakes)
    warp = np.eye(2,3, dtype=np.float32)
    try:
        cc, warp = cv2.findTransformECC(ref_gray, mov_gray, warp, cv2.MOTION_EUCLIDEAN,
                                        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 1e-4))
    except cv2.error:
        pass
    return warp

def locate(curr_bgr, prev_bgr=None, min_area_ratio:float=0.002, max_w:int=640) -> Optional[Dict[str, Any]]:
    img, scale = _resize_keep(curr_bgr, max_w)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.bilateralFilter(g, 5, 35, 35)

    if prev_bgr is not None:
        ref, _ = _resize_keep(prev_bgr, max_w)
        refg = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
        warp = _align_ecc(refg, g)
        refg = cv2.warpAffine(refg, warp, (g.shape[1], g.shape[0]), flags=cv2.INTER_LINEAR+cv2.WARP_INVERSE_MAP)
        fg = cv2.absdiff(g, refg)
    else:
        # fall back: high-frequency as proxy for foreground
        fg = cv2.GaussianBlur(g, (0,0), 1.5)
        fg = cv2.Laplacian(fg, cv2.CV_16S, ksize=3)
        fg = cv2.convertScaleAbs(fg)

    # Post-process
    th = cv2.threshold(fg, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), 1)
    th = cv2.dilate(th, np.ones((5,5), np.uint8), 1)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None

    h, w = th.shape[:2]
    min_area = (w*h) * min_area_ratio
    cand = [(cv2.contourArea(c), cv2.boundingRect(c)) for c in cnts if cv2.contourArea(c) >= min_area]
    if not cand: return None

    _, (x, y, ww, hh) = sorted(cand, key=lambda z: z[0], reverse=True)[0]
    inv = 1.0/scale
    x, y, ww, hh = int(x*inv), int(y*inv), int(ww*inv), int(hh*inv)
    cx, cy = x + ww//2, y + hh//2
    return {"center": (cx, cy), "bbox": (x, y, ww, hh)}
```

### `cv/strategy_flow_cluster.py`

```text
# cv/strategy_flow_cluster.py
# ------------------------------------------------------------
# Sparse optical flow (KLT) + motion vector clustering.
# Exports: locate(curr_bgr, prev_bgr, min_area_ratio=0.002, max_w=640)
# Notes:
#   - Requires prev_bgr; if absent, returns None.
# ------------------------------------------------------------
import cv2, numpy as np
from typing import Optional, Dict, Any

def _resize_keep(src, max_w=640):
    h, w = src.shape[:2]
    if w <= max_w: return src, 1.0
    s = max_w / w
    return cv2.resize(src, (int(w*s), int(h*s)), cv2.INTER_AREA), s

def locate(curr_bgr, prev_bgr=None, min_area_ratio:float=0.002, max_w:int=640) -> Optional[Dict[str, Any]]:
    if prev_bgr is None:
        return None
    post, scale = _resize_keep(curr_bgr, max_w)
    pre,  _    = _resize_keep(prev_bgr, max_w)
    pre_g = cv2.cvtColor(pre,  cv2.COLOR_BGR2GRAY)
    post_g= cv2.cvtColor(post, cv2.COLOR_BGR2GRAY)

    # good features to track
    pts = cv2.goodFeaturesToTrack(pre_g, maxCorners=600, qualityLevel=0.01, minDistance=7, blockSize=7)
    if pts is None:
        return None
    p1, st, err = cv2.calcOpticalFlowPyrLK(pre_g, post_g, pts, None, winSize=(21,21), maxLevel=3,
                                           criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    good_new = p1[st==1]; good_old = pts[st==1].reshape(-1,2)
    v = good_new - good_old
    mag = np.linalg.norm(v, axis=1)
    if len(mag) < 10:
        return None
    # filter small motions
    thr = np.percentile(mag, 70)
    sel = v[np.where(mag >= thr)]
    loc = good_new[np.where(mag >= thr)]
    if len(sel) < 6:
        return None

    # k-means (k=2) on end-point positions to pick dominant cluster
    Z = np.float32(loc)
    K = 2 if len(Z) >= 8 else 1
    if K > 1:
        criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
        ret, labels, centers = cv2.kmeans(Z, K, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
        # pick cluster with more members
        k = np.argmax(np.bincount(labels.flatten()))
        pts_k = Z[labels.flatten()==k]
    else:
        pts_k = Z

    x,y,w,h = cv2.boundingRect(pts_k.astype(np.float32))
    inv = 1.0/scale
    X,Y,W,H = int(x*inv), int(y*inv), int(w*inv), int(h*inv)
    cx, cy = X + W//2, Y + H//2
    return {"center": (cx, cy), "bbox": (X, Y, W, H)}
```

### `cv/strategy_FoELS.py`

```text
# cv/strategy_FoELS.py  (clean & robust, relaxed-by-default)
import cv2, time, numpy as np
from typing import Optional, Dict, Any, Tuple

# -------------------- small utils --------------------
def _ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000.0

def _resize_keep(src: np.ndarray, max_w: int) -> Tuple[np.ndarray, float]:
    h, w = src.shape[:2]
    if w <= max_w: return src, 1.0
    s = max_w / w
    dst = cv2.resize(src, (int(w*s), int(h*s)), cv2.INTER_AREA)
    return dst, s

def _to_gray(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr

def _unit(v: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True) + eps
    return v / n

# -------------------- flows --------------------
def _flow_farneback(prev_g: np.ndarray, curr_g: np.ndarray) -> np.ndarray:
    # 轻微平滑以稳住纹理（对低纹理/噪声有帮助）
    prev_g = cv2.GaussianBlur(prev_g, (3,3), 0)
    curr_g = cv2.GaussianBlur(curr_g, (3,3), 0)
    flow = cv2.calcOpticalFlowFarneback(prev_g, curr_g, None,
                                        pyr_scale=0.5, levels=4, winsize=21,
                                        iterations=5, poly_n=5, poly_sigma=1.2,
                                        flags=0)
    return flow.astype(np.float32)

def _tvl1_available() -> bool:
    return hasattr(cv2, "optflow") and (
        hasattr(cv2.optflow, "DualTVL1OpticalFlow_create") or
        hasattr(cv2.optflow, "createOptFlow_DualTVL1")
    )

def _flow_tvl1(prev_g: np.ndarray, curr_g: np.ndarray) -> np.ndarray:
    if not _tvl1_available():
        raise RuntimeError("cv2.optflow TV-L1 not available")
    try:
        tvl1 = cv2.optflow.DualTVL1OpticalFlow_create()
    except AttributeError:
        tvl1 = cv2.optflow.createOptFlow_DualTVL1()
    return tvl1.calc(prev_g, curr_g, None)

# -------------------- geometry --------------------
def _intersect(p: np.ndarray, d: np.ndarray, q: np.ndarray, e: np.ndarray, eps: float = 1e-6):
    A = np.stack([d, -e], axis=1)
    b = (q - p)
    det = A[0,0]*A[1,1] - A[0,1]*A[1,0]
    if abs(det) < eps: return None
    invA = np.array([[ A[1,1], -A[0,1]],
                     [-A[1,0],  A[0,0]]], dtype=np.float32) / det
    t = (invA @ b)[0]
    return p + t * d

# -------------------- FoE estimation --------------------
def _foe_ransac(flow: np.ndarray, mask: np.ndarray,
                trials: int, ang_tol_deg: float,
                min_mag: float, rng=np.random.default_rng(0)) -> Optional[Dict[str, Any]]:
    H, W, _ = flow.shape
    ys, xs = np.where(mask)
    if len(xs) < 150: return None

    vec_all = flow[ys, xs]; mag_all = np.linalg.norm(vec_all, axis=1)
    keep = mag_all >= min_mag
    if keep.sum() < 120: return None

    xs = xs[keep]; ys = ys[keep]; vec = vec_all[keep]; mag = mag_all[keep]
    # 幅值分位裁剪 + 子采样
    ql, qh = np.quantile(mag, [0.20, 0.98])
    sel = (mag >= ql) & (mag <= qh)
    xs = xs[sel]; ys = ys[sel]; vec = vec[sel]
    MMAX = 4000
    if xs.size > MMAX:
        idx = rng.choice(xs.size, size=MMAX, replace=False)
        xs = xs[idx]; ys = ys[idx]; vec = vec[idx]
    if xs.size < 120: return None

    pts  = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
    dirs = _unit(vec)
    M = xs.size
    cos_thr = np.cos(np.deg2rad(ang_tol_deg))
    best = None
    stop_hits = int(0.65 * M)
    xm_lo, xm_hi = -0.4*W, 1.4*W
    ym_lo, ym_hi = -0.4*H, 1.4*H

    for _ in range(trials):
        i, j = rng.choice(M, size=2, replace=False)
        p, q = pts[i], pts[j]; d, e = dirs[i], dirs[j]
        foe = _intersect(p, d, q, e)
        if foe is None: continue
        if not (xm_lo <= foe[0] <= xm_hi and ym_lo <= foe[1] <= ym_hi): continue
        vF = pts - foe; vFn = _unit(vF)
        cosang = np.sum(vFn * dirs, axis=1)
        inliers = (cosang > cos_thr)
        score = int(inliers.sum())
        if (best is None) or (score > best["score"]):
            sign_votes = np.sign(np.sum(np.sum(vF[inliers] * vec[inliers], axis=1)))
            sign = 1.0 if sign_votes >= 0 else -1.0
            best = dict(foe=foe, inliers=inliers, score=score, sign=sign)
            if score >= stop_hits: break

    if best is None: return None
    if best["score"] / float(M) < 0.18: return None
    return best

def _foe_lsq(flow: np.ndarray, mask: np.ndarray, min_mag: float) -> Optional[Dict[str, Any]]:
    """
    线性最小二乘 FoE 备选：最小化 Σ|| (I - dd^T)(e - p) ||^2
    推导： (ΣP) e = ΣP p, 其中 P = I - dd^T
    """
    ys, xs = np.where(mask)
    if xs.size < 80: return None
    vec = flow[ys, xs]; mag = np.linalg.norm(vec, axis=1)
    keep = mag >= min_mag
... (truncated)
```

### `cv/strategy_grabcut_refine.py`

```text
# cv/strategy_grabcut_refine.py
# ------------------------------------------------------------
# Saliency pre-detection + GrabCut refinement.
# Exports: locate(curr_bgr, prev_bgr=None, min_area_ratio=0.002, max_w=640)
# Requires: OpenCV with ximgproc/saliency (falls back to gradient if absent).
# ------------------------------------------------------------
import cv2, numpy as np
from typing import Optional, Dict, Any

def _resize_keep(src, max_w=640):
    h, w = src.shape[:2]
    if w <= max_w: return src, 1.0
    s = max_w / w
    return cv2.resize(src, (int(w*s), int(h*s)), cv2.INTER_AREA), s

def _saliency(gray):
    # Try OpenCV saliency; fallback to spectral residual approximation
    try:
        sal = cv2.saliency.StaticSaliencyFineGrained_create()
        ok, m = sal.computeSaliency(gray)
        if ok and m is not None: 
            m = (m*255).astype("uint8")
            return m
    except Exception:
        pass
    # Fallback: spectral residual (simple)
    g = cv2.GaussianBlur(gray, (0,0), 2.0)
    spec = np.fft.fft2(g.astype(np.float32))
    logA = np.log(np.abs(spec)+1e-6)
    phase = np.angle(spec)
    avg = cv2.boxFilter(logA, ddepth=-1, ksize=(9,9))
    resid = logA - avg
    recon = np.exp(resid + 1j*phase)
    m = np.fft.ifft2(recon)
    m = np.abs(m)
    m = cv2.GaussianBlur(m, (0,0), 2.0)
    m = cv2.normalize(m, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
    return m

def _largest_bbox(bin_img, min_area):
    cnts, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    cand = [(cv2.contourArea(c), cv2.boundingRect(c)) for c in cnts if cv2.contourArea(c) >= min_area]
    if not cand: return None
    return sorted(cand, key=lambda z: z[0], reverse=True)[0][1]

def locate(curr_bgr, prev_bgr=None, min_area_ratio:float=0.002, max_w:int=640) -> Optional[Dict[str, Any]]:
    img, scale = _resize_keep(curr_bgr, max_w)
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sal = _saliency(gray)
    sal = cv2.GaussianBlur(sal, (7,7), 0)
    th = cv2.threshold(sal, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), 1)
    min_area = int(h*w*min_area_ratio)
    rect = _largest_bbox(th, min_area)
    if rect is None:
        return None
    x,y,ww,hh = rect
    # Expand rect slightly as GrabCut initialization
    pad = int(0.08 * max(ww, hh))
    x0 = max(0, x - pad); y0 = max(0, y - pad)
    x1 = min(w-1, x + ww + pad); y1 = min(h-1, y + hh + pad)
    rect = (x0, y0, x1-x0, y1-y0)

    # GrabCut refinement
    mask = np.zeros((h,w), np.uint8)
    bgdModel = np.zeros((1,65), np.float64)
    fgdModel = np.zeros((1,65), np.float64)
    try:
        cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        # If grabcut fails, fall back to rectangle
        fin = (x0,y0,x1-x0,y1-y0)
        inv = 1.0/scale
        X,Y,W,H = [int(v*inv) for v in fin]
        cx, cy = X + W//2, Y + H//2
        return {"center": (cx, cy), "bbox": (X, Y, W, H)}

    fgMask = np.where((mask==cv2.GC_FGD) | (mask==cv2.GC_PR_FGD), 255, 0).astype('uint8')
    fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8), 1)
    rect2 = _largest_bbox(fgMask, min_area)
    if rect2 is None:
        rect2 = rect
    X,Y,W,H = rect2
    inv = 1.0/scale
    X,Y,W,H = int(X*inv), int(Y*inv), int(W*inv), int(H*inv)
    cx, cy = X + W//2, Y + H//2
    return {"center": (cx, cy), "bbox": (X, Y, W, H)}
```

### `cv/strategy_riri_invariant.py`

```text
# cv/strategy_riri_invariant.py
import cv2, time, numpy as np
from typing import Optional, Dict, Any, Tuple

# —— 小工具：与 strategy_FoELS 风格一致 —— #
def _ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000.0

def _resize_keep(src: np.ndarray, max_w: int) -> Tuple[np.ndarray, float]:
    h, w = src.shape[:2]
    if w <= max_w: return src, 1.0
    s = max_w / float(w)
    dst = cv2.resize(src, (int(w*s), int(h*s)), cv2.INTER_AREA)
    return dst, s

def _to_gray(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr

def _flow_farneback(prev_g: np.ndarray, curr_g: np.ndarray) -> np.ndarray:
    prev_g = cv2.GaussianBlur(prev_g, (3,3), 0)
    curr_g = cv2.GaussianBlur(curr_g, (3,3), 0)
    flow = cv2.calcOpticalFlowFarneback(prev_g, curr_g, None,
                                        pyr_scale=0.5, levels=4, winsize=21,
                                        iterations=5, poly_n=5, poly_sigma=1.2,
                                        flags=0)
    return flow.astype(np.float32)

def _unit(v: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True) + eps
    return v / n

# —— 估计 FoE（复用简化 LSQ，细节参考 strategy_FoELS） —— #
def _foe_lsq(flow: np.ndarray, mask: np.ndarray, min_mag: float = 0.10):
    ys, xs = np.where(mask)
    if xs.size < 80: return None
    vec = flow[ys, xs]; mag = np.linalg.norm(vec, axis=1)
    keep = mag >= min_mag
    if keep.sum() < 60: return None
    xs = xs[keep]; ys = ys[keep]; vec = vec[keep]
    pts  = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
    dirs = _unit(vec)

    S = np.zeros((2,2), np.float32)
    b = np.zeros((2,), np.float32)
    I = np.eye(2, dtype=np.float32)
    for (p, d) in zip(pts, dirs):
        P = I - np.outer(d, d)
        S += P; b += P @ p
    S += 1e-6 * I
    try:
        foe = np.linalg.solve(S, b)
    except np.linalg.LinAlgError:
        return None
    # 符号判定：径向是否发散（与 strategy_FoELS 类似）
    vF = pts - foe
    sign_votes = np.sign(np.sum(np.sum(vF * vec, axis=1)))
    sign = 1.0 if sign_votes >= 0 else -1.0
    return dict(foe=foe, sign=sign)

# —— 主入口：与 strategy_FoELS.locate 同名同参 —— #
def locate(curr_bgr: np.ndarray,
           prev_bgr: Optional[np.ndarray] = None,
           min_area_ratio: float = 0.0015,
           max_w: int = 720,
           angle_thr_deg: float = 20.0,
           min_flow_mag: float = 0.08,
           camera_move_thr: float = 0.01,
           debug_profile: bool = False) -> Optional[Dict[str, Any]]:

    if prev_bgr is None:
        if debug_profile: print("[RIRI:None] no prev frame")
        return None

    Tall = time.perf_counter()

    # 1) resize / gray
    T = time.perf_counter()
    c_small, s  = _resize_keep(curr_bgr, max_w)
    p_small, s2 = _resize_keep(prev_bgr, max_w)
    t_resize = _ms(T)
    if abs(s - s2) > 1e-6:
        if debug_profile: print(f"[RIRI:None] resize scale mismatch s={s:.4f}, s2={s2:.4f}")
        return None

    T = time.perf_counter()
    c_g = _to_gray(c_small); p_g = _to_gray(p_small)
    t_gray = _ms(T)

    # 2) flow
    T = time.perf_counter()
    flow = _flow_farneback(p_g, c_g)
    t_flow = _ms(T)

    # 3) 简单“相机在动”检查：统计全图/低先验区域的光流比例
    T = time.perf_counter()
    mag = np.linalg.norm(flow, axis=-1)
    ratio_move = float((mag > min_flow_mag).mean())
    t_cam = _ms(T)
    if ratio_move < camera_move_thr:
        if debug_profile: print(f"[RIRI:None] camera static ratio={ratio_move:.3f} < thr={camera_move_thr}")
        return None

    # 4) 估计 FoE（不固定 FoE，逐帧估计以构造“查找图”）
    T = time.perf_counter()
    mask = mag > min_flow_mag
    foe_est = _foe_lsq(flow, mask, min_mag=min_flow_mag)
    if foe_est is None:
        if debug_profile: print("[RIRI:None] LSQ FoE failed")
        return None
    foe, sign = foe_est["foe"], foe_est["sign"]
    t_foe = _ms(T)

    # 5) 比值不变量残差（角度差在 -90~90 度内等价于 arctan 比值差）
    T = time.perf_counter()
    Hs, Ws = c_g.shape[:2]
    X, Y = np.meshgrid(np.arange(Ws, dtype=np.float32),
                       np.arange(Hs, dtype=np.float32))
    vF = np.stack([(X - foe[0]) * sign, (Y - foe[1]) * sign], axis=-1)  # 期望径向（带正负）
    F  = flow

... (truncated)
```

### `cv/strategy_templatematch.py`

```text
# cv/strategy_templatematch.py
# ------------------------------------------------------------
# Self-template from prev frame + NCC matching (with ECC alignment).
# Exports: locate(curr_bgr, prev_bgr=None, min_area_ratio=0.002, max_w=640)
# Behavior:
#   - If prev_bgr is provided, we find the most salient/textured patch in prev
#     (Harris corner density) and treat it as a template, then locate it in curr.
#   - If prev_bgr is None, falls back to edge-density contour selection.
# ------------------------------------------------------------
import cv2, numpy as np
from typing import Optional, Dict, Any

def _resize_keep(src, max_w=640):
    h, w = src.shape[:2]
    if w <= max_w: return src, 1.0
    s = max_w / w
    return cv2.resize(src, (int(w*s), int(h*s)), cv2.INTER_AREA), s

def _best_patch(gray, ksize=48, stride=24):
    h, w = gray.shape[:2]
    resp = cv2.cornerHarris(gray, 2, 3, 0.04)
    resp = cv2.normalize(resp, None, 0, 1.0, cv2.NORM_MINMAX)
    best = None
    for y in range(0, h-ksize, stride):
        for x in range(0, w-ksize, stride):
            s = resp[y:y+ksize, x:x+ksize].mean()
            if best is None or s > best[0]:
                best = (s, (x,y,ksize,ksize))
    return best[1] if best else None

def _align_ecc(ref_gray, mov_gray):
    warp = np.eye(2,3, dtype=np.float32)
    try:
        _, warp = cv2.findTransformECC(ref_gray, mov_gray, warp, cv2.MOTION_EUCLIDEAN,
                                       criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT, 30, 1e-4))
    except cv2.error:
        pass
    return warp

def locate(curr_bgr, prev_bgr=None, min_area_ratio:float=0.002, max_w:int=640) -> Optional[Dict[str, Any]]:
    img, scale = _resize_keep(curr_bgr, max_w)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = g.shape[:2]

    if prev_bgr is not None:
        pre, _ = _resize_keep(prev_bgr, max_w)
        pre_g = cv2.cvtColor(pre, cv2.COLOR_BGR2GRAY)
        warp = _align_ecc(pre_g, g)
        pre_g = cv2.warpAffine(pre_g, warp, (w, h), flags=cv2.INTER_LINEAR+cv2.WARP_INVERSE_MAP)
        # choose a textured patch in prev
        rect = _best_patch(pre_g, ksize=max(24, min(h,w)//10), stride=max(12, min(h,w)//20))
        if rect is None:
            rect = (w//3, h//3, w//3, h//3)
        x,y,ww,hh = rect
        tmpl = pre_g[y:y+hh, x:x+ww]
        res = cv2.matchTemplate(g, tmpl, cv2.TM_CCOEFF_NORMED)
        _, maxv, _, maxloc = cv2.minMaxLoc(res)
        X, Y = maxloc
        W, H = ww, hh
    else:
        # Fallback: edge density region
        edges = cv2.Canny(g, 60, 160)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), 1)
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: return None
        min_area = int(h*w*min_area_ratio)
        cand = [(cv2.contourArea(c), cv2.boundingRect(c)) for c in cnts if cv2.contourArea(c) >= min_area]
        if not cand: return None
        X,Y,W,H = sorted(cand, key=lambda z: z[0], reverse=True)[0][1]

    inv = 1.0/scale
    X,Y,W,H = int(X*inv), int(Y*inv), int(W*inv), int(H*inv)
    cx, cy = X + W//2, Y + H//2
    return {"center": (cx, cy), "bbox": (X, Y, W, H)}
```

### `cv/strategy_twodiff_orb.py`

```text
# cv/strategy_twodiff_orb.py
# ------------------------------------------------------------
# Two-frame differencing + ORB alignment.
# Exports: locate(curr_bgr, prev_bgr=None, min_area_ratio=0.002, max_w=640)
# ------------------------------------------------------------
import cv2, numpy as np
from typing import Optional, Dict, Any

def _resize_keep(src, max_w=640):
    h, w = src.shape[:2]
    if w <= max_w: return src, 1.0
    s = max_w / w
    return cv2.resize(src, (int(w*s), int(h*s)), cv2.INTER_AREA), s

def _align_orb(prev_bgr, curr_bgr, max_kp=800):
    prev_g = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
    curr_g = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=max_kp)
    k1, d1 = orb.detectAndCompute(prev_g, None)
    k2, d2 = orb.detectAndCompute(curr_g, None)
    if d1 is None or d2 is None: return prev_bgr
    m = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(d1, d2)
    m = sorted(m, key=lambda x: x.distance)[:200]
    if len(m) < 10: return prev_bgr
    src = np.float32([k1[a.queryIdx].pt for a in m]).reshape(-1,1,2)
    dst = np.float32([k2[a.trainIdx].pt for a in m]).reshape(-1,1,2)
    H, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if H is None: return prev_bgr
    h, w = curr_bgr.shape[:2]
    return cv2.warpAffine(prev_bgr, H, (w, h))

def locate(curr_bgr, prev_bgr=None, min_area_ratio=0.002, max_w=640) -> Optional[Dict[str, Any]]:
    if prev_bgr is None: return None
    p_small, s1 = _resize_keep(prev_bgr, max_w)
    c_small, s2 = _resize_keep(curr_bgr, max_w)
    if abs(s1 - s2) > 1e-6: return None
    scale = s1
    p_aln = _align_orb(p_small, c_small)
    pb = cv2.GaussianBlur(p_aln, (5,5), 0)
    cb = cv2.GaussianBlur(c_small, (5,5), 0)
    diff = cv2.absdiff(cb, pb)
    g = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), 1)
    th = cv2.dilate(th, np.ones((5,5), np.uint8), 1)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    h, w = th.shape[:2]
    min_area = (w*h) * min_area_ratio
    cand = [(cv2.contourArea(c), cv2.boundingRect(c)) for c in cnts if cv2.contourArea(c) >= min_area]
    if not cand: return None
    _, (x, y, ww, hh) = sorted(cand, key=lambda z: z[0], reverse=True)[0]
    inv = 1.0 / scale
    x, y, ww, hh = int(x*inv), int(y*inv), int(ww*inv), int(hh*inv)
    cx, cy = x + ww//2, y + hh//2
    return {"center": (cx, cy), "bbox": (x, y, ww, hh)}
```

### `cv/strategy_yolo.py`

```text
# cv/strategy_yolo.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Any, Dict, Optional
import time
import os
import tempfile

import numpy as np
import cv2
from ultralytics import YOLO

_MODEL = None
_MODEL_PATH = None

def _lazy_load(model_path: str):
    global _MODEL, _MODEL_PATH
    if _MODEL is None or _MODEL_PATH != model_path:
        _MODEL = YOLO(model_path)
        _MODEL_PATH = model_path

def _pick_index(res) -> Optional[int]:
    boxes = res.boxes
    probs = getattr(res, "probs", None)
    if boxes is None or boxes.xywh is None or len(boxes) == 0:
        return None
    # 优先你指定的策略：probs.top1 -> 否则按最高 conf
    if probs is not None and hasattr(probs, "top1") and probs.top1 is not None:
        idx = int(probs.top1)
        if 0 <= idx < len(boxes):
            return idx
    confs = boxes.conf.cpu().numpy()
    return int(confs.argmax())

def locate(
    image_bgr: np.ndarray,
    *,
    model_path: str = "cv/yolo/best.pt",
    save_format: str = ".png",     # ".png" 或 ".jpg" 都可；默认 .png
    debug_profile: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    最接近 Ultralytics sample 的用法：
    1) 把 ndarray 截图原样写成临时 PNG/JPG
    2) 直接把文件路径传给 YOLO(...)，不设 conf/imgsz/classes 等额外参数
    """
    assert image_bgr is not None and image_bgr.ndim == 3 and image_bgr.shape[2] == 3, "expect HxWx3 BGR"
    if image_bgr.dtype != np.uint8:
        image_bgr = np.clip(image_bgr, 0, 255).astype(np.uint8)

    _lazy_load(model_path)
    t0 = time.perf_counter()

    # 1) 写临时文件（不做任何 resize/预处理）
    #    使用 cv2.imwrite 避免二次编码/解码差异；suffix 控制 png/jpg
    tmp = tempfile.NamedTemporaryFile(prefix="yolo_frame_", suffix=save_format, delete=False)
    tmp_path = tmp.name
    tmp.close()
    ok = cv2.imwrite(tmp_path, image_bgr)
    if not ok:
        if debug_profile:
            print("[yolo] cv2.imwrite failed")
        try:
            os.remove(tmp_path)
        finally:
            return None

    # 2) 以“路径”方式推理（不传 conf/imgsz/classes）
    try:
        t1 = time.perf_counter()
        results = _MODEL(tmp_path, conf=0.05, iou=0.45, imgsz=960, verbose=False)
        infer_ms = (time.perf_counter() - t1) * 1000.0
    finally:
        # 清理临时图片
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    if not results or len(results) == 0:
        if debug_profile:
            print(f"[yolo] empty results  io={(t1 - t0)*1000:.1f}ms  infer={infer_ms:.1f}ms")
        return None

    res = results[0]
    boxes = getattr(res, "boxes", None)
    if boxes is None or boxes.xywh is None or len(boxes) == 0:
        if debug_profile:
            print(f"[yolo] no boxes  infer={infer_ms:.1f}ms")
        return None

    # 选择一个框
    idx = _pick_index(res)
    if idx is None:
        if debug_profile:
            print("[yolo] selection failed (no valid index)")
        return None

    # 直接用 Ultralytics 的 xywh（中心点坐标 + 宽高）
    xywh = boxes.xywh.cpu().numpy()
    cx_f, cy_f, w_f, h_f = map(float, xywh[idx])

    cx, cy = int(round(cx_f)), int(round(cy_f))
    x = int(round(cx_f - 0.5 * w_f))
    y = int(round(cy_f - 0.5 * h_f))
    w = int(round(w_f))
    h = int(round(h_f))

    out = {
        "center": (cx, cy),
        "bbox": (x, y, w, h),
        "meta": {
            "model_path": _MODEL_PATH,
            "conf": float(boxes.conf[idx]),
            "cls": int(boxes.cls[idx]) if boxes.cls is not None else None,
            "infer_ms": infer_ms,
            "input": "path",      # 标记这次是“路径输入”
            "format": save_format
        },
    }
... (truncated)
```

### `cv/verify_motion.py`

```text
# cv/verify_motion.py
# ------------------------------------------------------------
# （暂时弃用）
# Description:
#   Post-action verification utilities for AR interactions.
#   Given pre/post frames + (center, bbox) + op metadata,
#   verify whether the intended gesture truly occurred.
#   v1+ version with:
#     (1) background motion compensation (camera shake removal)
#     (2) similarity transform fitting (scale/rotation/translation)
# ------------------------------------------------------------
from typing import Tuple, Optional, Dict, Any
import numpy as np
import cv2

Point = Tuple[int, int]
BBox  = Tuple[int, int, int, int]

# ------------------------------
# Utility helpers
# ------------------------------
def _to_gray(img_bgr: np.ndarray) -> np.ndarray:
    if img_bgr.ndim == 3:
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return img_bgr

def _clip_bbox(bbox: BBox, w: int, h: int) -> BBox:
    x, y, bw, bh = bbox
    x = int(max(0, min(x, w-1)))
    y = int(max(0, min(y, h-1)))
    bw = int(max(1, min(bw, w - x)))
    bh = int(max(1, min(bh, h - y)))
    return (x, y, bw, bh)

def _good_points(gray: np.ndarray, bbox: BBox, max_pts=120):
    """GoodFeatures within bbox area."""
    x, y, w, h = _clip_bbox(bbox, gray.shape[1], gray.shape[0])
    mask = np.zeros_like(gray, np.uint8)
    mask[y:y+h, x:x+w] = 255
    pts = cv2.goodFeaturesToTrack(
        gray, maxCorners=max_pts, qualityLevel=0.01, minDistance=6, mask=mask
    )
    if pts is None:
        return None
    return pts.reshape(-1, 2).astype(np.float32)

def _track(pre_g: np.ndarray, post_g: np.ndarray, pts: np.ndarray):
    """PyrLK optical flow, return matched p0, p1 arrays."""
    nxt, st, err = cv2.calcOpticalFlowPyrLK(
        pre_g, post_g, pts, None,
        winSize=(21, 21), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)
    )
    if nxt is None or st is None:
        return None, None
    ok = (st.reshape(-1) == 1)
    p0 = pts[ok]
    p1 = nxt.reshape(-1, 2)[ok]
    if len(p0) == 0:
        return None, None
    return p0, p1

def _bg_compensate(pre_g: np.ndarray, post_g: np.ndarray, bbox: BBox) -> np.ndarray:
    """Estimate background affine (camera motion) using points OUTSIDE bbox, then warp post_g."""
    H, W = pre_g.shape
    x, y, w, h = _clip_bbox(bbox, W, H)

    mask = np.ones_like(pre_g, np.uint8) * 255
    mask[y:y+h, x:x+w] = 0  # exclude target region

    pts_bg = cv2.goodFeaturesToTrack(
        pre_g, maxCorners=300, qualityLevel=0.01, minDistance=6, mask=mask
    )
    if pts_bg is None or len(pts_bg) < 20:
        return post_g  # fallback

    p0, p1 = _track(pre_g, post_g, pts_bg.reshape(-1, 2).astype(np.float32))
    if p0 is None or len(p0) < 12:
        return post_g

    # Fit partial affine (rotation+scale+translation)
    M, inliers = cv2.estimateAffinePartial2D(
        p1, p0, method=cv2.RANSAC, ransacReprojThreshold=3.0, maxIters=2000, confidence=0.99
    )
    if M is None:
        return post_g
    return cv2.warpAffine(post_g, M, (W, H), flags=cv2.INTER_LINEAR)

def _fit_similarity(p0: np.ndarray, p1: np.ndarray, center: Optional[Tuple[float, float]]=None):
    """
    Robustly fit similarity transform from p0 -> p1 using partial affine, then decompose.
    Return dict: {'scale': s, 'theta': rad, 'tx': tx, 'ty': ty, 'inlier_ratio': r, 'M': 2x3}
    """
    if len(p0) < 6:
        return None

    a0 = p0.copy().astype(np.float32)
    a1 = p1.copy().astype(np.float32)

    if center is not None:
        cx, cy = center
        a0 -= np.array([[cx, cy]], dtype=np.float32)
        a1 -= np.array([[cx, cy]], dtype=np.float32)

    M, inliers = cv2.estimateAffinePartial2D(
        a0, a1, method=cv2.RANSAC, ransacReprojThreshold=2.5, maxIters=2000, confidence=0.99
    )
    if M is None:
        return None

    # Decompose similarity parameters from 2x3 affine: [[a, b, tx],[c, d, ty]]
    a, b, tx = M[0, 0], M[0, 1], M[0, 2]
    c, d, ty = M[1, 0], M[1, 1], M[1, 2]
    # For similarity, c ≈ -b, d ≈ a. Use average.
    # Scale and rotation from the rotation-scale matrix [[a, b],[-b, a]]
    s = float(np.sqrt(max(1e-12, ((a + d) * 0.5)**2 + ((b - c) * 0.5)**2)))
    # Use atan2 of average (b, a) form
    theta = float(np.arctan2((b - c) * 0.5, (a + d) * 0.5))

    r = float(inliers.sum() / len(a0)) if inliers is not None else 1.0
... (truncated)
```

### `cv/yolo/test_yolo_train.py`

```text
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
python test_yolo_train.py train --data myar2.yaml --epochs 100 --imgsz 640

python test_yolo_train.py val --data myar2.yaml --weights runs/train/exp/weights/best.pt

python test_yolo_train.py test --test-dir testimages --weights runs/train/exp4/weights/best.pt --conf 0.05
"""

import argparse
import sys
import platform
from tqdm import tqdm
from pathlib import Path
from ultralytics import YOLO

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

def pick_device(prefer="auto"):
    # 简单设备选择：优先 MPS（Apple Silicon），其次 CUDA，最后 CPU
    if prefer != "auto":
        return prefer
    try:
        import torch
        if platform.system() == "Darwin" and torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"

def ensure_yaml(data_yaml: str):
    p = Path(data_yaml)
    if not p.exists():
        print(f"[WARN] data yaml 不存在：{p.resolve()}，请确认路径。")
    return str(p)

def cmd_train(args):
    device = pick_device(args.device)
    print(f"[INFO] device = {device}")

    # 1) 选择基模型：默认 yolo11s.pt（更稳），可用 --weights 指定
    model = YOLO(args.weights)

    # 2) 开始训练：只使用 data yaml 的 train/val
    results = model.train(
        data=ensure_yaml(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        lr0=args.lr0,
        patience=args.patience,
        seed=args.seed,
        workers=args.workers,
        project=args.project,
        name=args.name,
        pretrained=True
    )
    print(results)

def cmd_val(args):
    device = pick_device(args.device)
    print(f"[INFO] device = {device}")

    model = YOLO(args.weights)
    metrics = model.val(
        data=ensure_yaml(args.data),
        split="val",              # 依赖 data.yaml 的 val
        imgsz=args.imgsz,
        device=device,
        iou=0.7,                  # 可按需调整
        conf=0.001,               # 评测用低阈值，做 PR 曲线更稳
        project=args.project,
        name=args.name
    )
    print(metrics)                # dict-like，含 mAP50、mAP50-95 等

def cmd_test(args):
    device = pick_device(args.device)
    print(f"[INFO] device = {device}")

    model = YOLO(args.weights)

    test_dir = Path(args.test_dir)
    # 统计测试集图片数量（递归）
    total_imgs = sum(1 for p in test_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS)

    if total_imgs == 0:
        print(f"[WARN] 在 {test_dir} 未找到图片（支持扩展名：{sorted(IMAGE_EXTS)}）")
        return

    pbar = tqdm(total=total_imgs, desc="Inferencing (stream)", unit="img")

    # 关键：stream=True，并通过 for 循环消费生成器来触发推理与保存
    for r in model.predict(
            source=str(test_dir),
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            device=device,
            project=args.project,
            name=args.name,
            stream=True,        # ✅ 流式
            save=True,          # 保存渲染图到 runs/predict-*
            save_txt=True,      # 保存 YOLO txt
            save_conf=True,
            verbose=False       # 由 tqdm 负责展示进度
        ):
        # r.path 是当前图片路径字符串
        pbar.set_postfix_str(Path(r.path).name[:40])
        pbar.update(1)

    pbar.close()
    print("[INFO] 推理完成，查看 runs 目录输出。")

def main():
... (truncated)
```

### `gen_project_context.py`

```text
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成工程结构的 Markdown 文件（项目“快照说明”），便于在对话中共享上下文。
- 输出：默认写入 project_context.md
- 目录树：可设最大深度
- 排除：默认忽略 .git、虚拟环境、缓存与大文件等，可自定义
- 可选摘录：按通配符挑选关键文件，截取前 N 行作为上下文片段

python gen_project_context.py --out chat_context.md --tree-depth 4

"""

import argparse
import datetime as dt
import os
import sys
import subprocess
from pathlib import Path
from typing import Iterable, List, Tuple

# 目录名级别的硬排除（出现即跳过整个子树）
EXCLUDE_DIR_NAMES = {".git", ".svn", ".hg", ".venv", ".venv311", "venv", "env", "ENV", ".conda", "conda"}
# 文件后缀级别的硬排除（可选）
EXCLUDE_FILE_SUFFIXES = {".pt", ".onnx", ".ckpt", ".mp4", ".mov", ".avi", ".zip", ".tar", ".gz", ".7z"}

DEFAULT_EXCLUDES = [
    ".git", ".svn", ".hg",
    ".DS_Store",
    "__pycache__", "*.pyc", "*.pyo", "*.pyd",
    ".idea", ".vscode",
    ".venv", ".venv311", "venv", "env", "ENV", ".conda", "conda", ".env",
    "build", "dist", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "node_modules",
    "runs", "outputs", "logs",
    # 大文件常见后缀（仅用于目录树与统计时忽略；不影响真实代码）
    "*.pt", "*.onnx", "*.ckpt",
    "*.mp4", "*.mov", "*.avi",
    "*.png", "*.jpg", "*.jpeg", "*.webp",
    "*.zip", "*.tar", "*.gz", "*.7z",
]

DEFAULT_SNIPPETS = [
    "README.md",
    "*.md",
    "*.py",
    "requirements.txt",
    "pyproject.toml",
    "setup.cfg",
]

def parse_args():
    p = argparse.ArgumentParser(description="生成工程结构 Markdown 文档")
    p.add_argument("--root", type=str, default=".", help="工程根目录（默认当前目录）")
    p.add_argument("--out", type=str, default="project_context.md", help="输出文件名")
    p.add_argument("--tree-depth", type=int, default=3, help="目录树最大深度（默认 3）")
    p.add_argument("--exclude", action="append", default=[], help="追加排除模式（可多次）")
    p.add_argument("--no-default-excludes", action="store_true", help="不使用默认排除列表")
    p.add_argument("--snippet-glob", action="append", default=[], help="需要摘录片段的通配符（可多次）")
    p.add_argument("--snippet-lines", type=int, default=120, help="每个文件摘录的最大行数")
    p.add_argument("--topn", type=int, default=15, help="体积 Top-N 文件列表条数")
    return p.parse_args()

def is_git_repo(root: Path) -> bool:
    return (root / ".git").exists()

def run_cmd(cmd: List[str], cwd: Path) -> str:
    try:
        out = subprocess.check_output(cmd, cwd=str(cwd), stderr=subprocess.DEVNULL, text=True, timeout=5)
        return out.strip()
    except Exception:
        return ""

def gather_git_info(root: Path) -> Tuple[str, str, str]:
    if not is_git_repo(root):
        return ("N/A", "N/A", "N/A")
    branch = run_cmd(["git", "rev-parse", "--abbrev-ref", "HEAD"], root) or "N/A"
    commit = run_cmd(["git", "rev-parse", "--short", "HEAD"], root) or "N/A"
    latest = run_cmd(["git", "log", "--oneline", "-n", "5"], root)
    return (branch, commit, latest or "N/A")

def match_any(path: Path, patterns: Iterable[str]) -> bool:
    from fnmatch import fnmatch
    rel = str(path).replace("\\", "/")
    name = path.name
    # 1) 通配匹配：支持 **/dir/** 形式
    if any(fnmatch(rel, pat) or fnmatch(name, pat) for pat in patterns):
        return True
    # 2) 目录名硬排除：只要任何一段路径命中就排除
    for part in path.parts:
        if part in EXCLUDE_DIR_NAMES:
            return True
    # 3) 文件后缀硬排除
    if path.is_file() and path.suffix.lower() in EXCLUDE_FILE_SUFFIXES:
        return True
    return False

def iter_tree(root: Path, max_depth: int, excludes: List[str]) -> List[str]:
    lines = []
    root = root.resolve()

    def walk(d: Path, depth: int, prefix: str = ""):
        if depth > max_depth:
            return
        try:
            entries = sorted(d.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        except PermissionError:
            return

        # 先分组，目录要能被“剪枝”
        dirs, files = [], []
        for e in entries:
            if e.is_dir():
                # 如果 e 这个目录命中排除，直接跳过（剪枝）
                if match_any(e.relative_to(root), excludes) or e.name in EXCLUDE_DIR_NAMES:
                    continue
                dirs.append(e)
            else:
                if match_any(e.relative_to(root), excludes):
                    continue
... (truncated)
```

### `main.py`

```text
```

### `README.md`

```text
TBD
```

### `requirements.txt`

```text
Appium-Python-Client==5.2.4
attrs==25.3.0
certifi==2025.8.3
charset-normalizer==3.4.3
diskcache==5.6.3
filelock==3.19.1
fsspec==2025.9.0
h11==0.16.0
hf-xet==1.1.10
huggingface-hub==0.35.3
idna==3.10
imageio==2.37.0
Jinja2==3.1.6
joblib==1.5.2
lazy_loader==0.4
llama_cpp_python==0.3.16
MarkupSafe==3.0.3
networkx==3.5
numpy==2.2.6
opencv-contrib-python==4.12.0.88
outcome==1.3.0.post0
packaging==25.0
pillow==11.3.0
PySocks==1.7.1
PyYAML==6.0.2
regex==2025.9.18
requests==2.32.5
safetensors==0.6.2
scikit-image==0.25.2
scikit-learn==1.7.2
scipy==1.16.2
selenium==4.35.0
setuptools==80.9.0
sniffio==1.3.1
sortedcontainers==2.4.0
threadpoolctl==3.6.0
tifffile==2025.9.20
tokenizers==0.22.1
tqdm==4.67.1
transformers==4.57.1
trio==0.30.0
trio-websocket==0.12.2
typing_extensions==4.14.1
urllib3==2.5.0
websocket-client==1.8.0
wheel==0.45.1
wsproto==1.2.0
```

### `smoke.py`

```text
#!/usr/bin/env python3
import argparse, subprocess, time, re, sys

PKG = "com.google.ar.core.examples.java.hellorecordingplayback"
ACT = ".HelloRecordingPlaybackActivity"  # 也可用完整名：com.google.ar.core.examples.java.hellorecordingplayback.HelloRecordingPlaybackActivity

def run(cmd):
    return subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT).decode("utf-8","ignore")

def start_app(serial=None):
    base = f"adb -s {serial} " if serial else "adb "
    # 启动：包名/Activity 支持以点开头的简写
    run(base + f"shell am start -n {PKG}/{ACT}")

def top_component(serial=None):
    base = f"adb -s {serial} " if serial else "adb "
    # 优先从 activity dumpsys 抓 topResumedActivity，抓不到再看 window 的 mCurrentFocus
    out = run(base + "shell dumpsys activity")
    m = re.search(r'topResumedActivity.*? ([\w\.]+)/([\w\.$]+)', out)
    if not m:
        m = re.search(r'mResumedActivity.*? ([\w\.]+)/([\w\.$]+)', out)
    if m:
        return m.group(1), m.group(2)
    out = run(base + "shell dumpsys window")
    m = re.search(r'mCurrentFocus.*? ([\w\.]+)/([\w\.$]+)', out)
    if m:
        return m.group(1), m.group(2)
    return None, None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serial", help="设备序列号（adb devices 可查看，多设备时必须指定）")
    args = ap.parse_args()

    # 启动 App
    start_app(args.serial)
    time.sleep(3.0)  # 等摄像头/AR会话初始化

    # 校验前台是否在目标包
    tpkg, tact = top_component(args.serial)
    if tpkg is None:
        print("✖ 读取前台 Activity 失败（权限或系统限制）。")
        sys.exit(2)

    print(f"前台：{tpkg}/{tact}")
    if tpkg == PKG:
        print("✓ SMOKE OK：App 已启动并处于前台。")
        sys.exit(0)
    else:
        print("✖ SMOKE FAIL：前台不在目标包。")
        sys.exit(1)

if __name__ == "__main__":
    main()

  
```

### `v0_ar_monkey_adb.py`

```text
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v0_ar_monkey_adb_baseline.py
最小可用 ADB Monkey 基线（零预设、零授权、零启动）：
- 事件：tap / double-tap / long-press / swipe / key(back/home/appswitch/vol)
- 坐标全屏随机（可设 margin），不依赖 UI 树/权限/前置准备
- 自动选择唯一已连接设备；多设备或无设备时给出提示
- 轻量 JSONL 日志（可选）

python3 v0_ar_monkey_adb.py --rounds 1000 --out run.jsonl
"""
import argparse, json, os, random, re, shlex, subprocess, time

# === 新增：接入验证统计 ===
from common.verify_motion import MotionVerifier

# ---------------- ADB helpers ----------------
def adb(cmd, serial=None, check=True):
    base = ["adb"]
    if serial: base += ["-s", serial]
    if isinstance(cmd, str): cmd = shlex.split(cmd)
    return subprocess.run(base + cmd, capture_output=True, text=True, check=check).stdout

def list_adb_devices():
    out = adb("devices", check=True)
    lines = [l.strip() for l in out.splitlines()[1:] if l.strip()]
    devs = []
    for line in lines:
        parts = line.split()
        if len(parts) >= 2:
            serial, state = parts[0], parts[1]
            devs.append((serial, state))
    return [(s, st) for (s, st) in devs]

def auto_pick_serial(user_serial=None):
    if user_serial:
        return user_serial
    devs = list_adb_devices()
    ready = [(s, st) for s, st in devs if st == "device"]
    if len(ready) == 1:
        return ready[0][0]
    if len(ready) == 0:
        if not devs:
            raise SystemExit("未检测到设备。请连接设备或开启ADB over Wi-Fi；或使用 --serial 指定。")
        states = ", ".join([f"{s}({st})" for s, st in devs])
        raise SystemExit(f"检测到设备但不可用：{states}。请解锁授权或重连；或使用 --serial 指定。")
    lst = ", ".join([s for s, _ in ready])
    raise SystemExit(f"检测到多台设备：{lst}。请使用 --serial 选择其一。")

def get_size(serial=None):
    try:
        out = adb("shell wm size", serial)
        m = re.search(r"Physical size:\s*(\d+)x(\d+)", out)
        if m: return int(m.group(1)), int(m.group(2))
    except subprocess.CalledProcessError:
        pass
    out = adb("shell dumpsys window displays", serial)
    m = re.search(r"init=(\d+)x(\d+)", out)
    if not m: raise RuntimeError("无法获取屏幕分辨率")
    return int(m.group(1)), int(m.group(2))

def top_app(serial=None):
    try:
        out = adb("shell dumpsys activity top", serial, check=False)
        m = re.search(r" ([\w\.]+)/([\w\.$]+)", out)
        if m: return m.group(1), m.group(2)
    except Exception:
        pass
    return None, None

# ---------------- touch/key actions ----------------
def tap(x, y, serial=None): adb(f"shell input tap {x} {y}", serial, check=False)
def swipe(x1, y1, x2, y2, dur_ms, serial=None): adb(f"shell input swipe {x1} {y1} {x2} {y2} {dur_ms}", serial, check=False)
def long_press(x, y, hold_ms, serial=None): swipe(x, y, x, y, hold_ms, serial)
def keyevent(code, serial=None): adb(f"shell input keyevent {code}", serial, check=False)

KEY_BACK, KEY_HOME, KEY_APP_SWITCH = 4, 3, 187
KEY_VOL_UP, KEY_VOL_DOWN = 24, 25

def rand_xy(W, H, margin_ratio):
    L = int(W * margin_ratio); T = int(H * margin_ratio)
    R = W - L; B = H - T
    return random.randint(L, R-1), random.randint(T, B-1)

# === 新增：v0 仅打印 overall 的小工具 ===
def _print_overall(ver: MotionVerifier, prefix: str = ""):
    snap = ver.snapshot()
    o = snap["overall"]
    print(f"{prefix}overall ok={o['ok']:4d}  total={o['total']:4d}  succ={o['rate']:.3f}", flush=True)

def run_monkey(
    serial=None,
    rounds=500,
    sleep_min=0.10, sleep_max=0.60,
    margin_ratio=0.0,
    swipe_dur_ms=(150, 800),
    long_ms=(600, 1500),
    doubletap_gap=(0.05, 0.18),
    seed=None,
    out_jsonl=None,
    # === 新增：验证统计打印间隔（秒）；0 表示不定时打印，仅结束时打印一次 ===
    verify_print_sec: float = 10.0,
):
    if seed is not None: random.seed(seed)
    W, H = get_size(serial)
    print(f"[baseline] screen={W}x{H} margin_ratio={margin_ratio}")

    weights = {
        "tap": 5, "swipe": 4, "long": 1, "double": 1,
        "back": 1, "home": 1, "appswitch": 1, "volup": 1, "voldown": 1
    }
    ops = list(weights.keys()); ws = [weights[k] for k in ops]
    key_pool = {"back": KEY_BACK, "home": KEY_HOME, "appswitch": KEY_APP_SWITCH,
                "volup": KEY_VOL_UP, "voldown": KEY_VOL_DOWN}

    logf = open(out_jsonl, "a", encoding="utf-8") if out_jsonl else None
    def log(event):
        if logf:
            logf.write(json.dumps(event, ensure_ascii=False) + "\n"); logf.flush()
... (truncated)
```

### `v0_ar_monkey_appium.py`

```text
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v0_ar_monkey_appium.py
Appium baseline（随机手势），接入 verify_motion（含 tap）与 timing：
- 启动前清空 logcat（adb logcat -c），采集从“当前时刻”（-T 0）开始
- 定时打印与最终打印：place/drag/pinch/rotate/tap + overall
- 统计：程序总时长 / 模拟操作时长 / 平均每操作耗时（= sim_runtime / rounds）

用法示例：
  python v0_ar_monkey_appium.py --pkg com.google.ar.sceneform.samples.hellosceneform \
    --activity auto --rounds 50 --print-interval 10
"""

import argparse, random, time, sys, threading, subprocess
from common.device import make_driver, get_window_size, resolve_main_activity
from common.actions import tap, drag_line, long_press, pinch_or_zoom, rotate
from common.policy_random import step as random_step

from common.verify_motion import MotionVerifier
from common.timing import Timing

# ----------------- 工具 -----------------
def clear_logcat():
    subprocess.run(["adb", "logcat", "-c"], check=False)

def periodic_print(ver: MotionVerifier, interval_sec: int, stop_evt: threading.Event):
    while not stop_evt.is_set():
        time.sleep(interval_sec)
        if stop_evt.is_set():
            break
        print("[tick]\n" + ver.summary_str(), flush=True)

# ----------------- 主流程 -----------------
def run_monkey(pkg="com.rooom.app",
               activity="auto",
               serial=None,
               rounds=250,
               sleep_min=0.25, sleep_max=0.85,
               safe=(0.12, 0.18, 0.88, 0.88),
               drag_ms=(300, 900), long_ms=(700, 1500),
               pinch_start=80, pinch_end=220,
               rotate_radius=(160, 260), rotate_angle=(30, 90), rotate_steps=8,
               warmup_wait=3.0, seed=None,
               print_interval=10,
               overall_include_tap=False):

    if seed is not None:
        random.seed(seed)

    # 计时：程序总时长从此开始
    tim = Timing()

    # 清空旧日志，启动 verify_motion（从当前时刻），可带 serial（多设备时可用）
    clear_logcat()
    ver = MotionVerifier(include_tap_in_overall=overall_include_tap)
    ver.start_from_adb(serial=serial, extra_logcat_args=["-T", "0"])

    # 定时打印线程
    stop_evt = threading.Event()
    printer = None
    if print_interval and print_interval > 0:
        printer = threading.Thread(target=periodic_print, args=(ver, print_interval, stop_evt), daemon=True)
        printer.start()

    # 解析/启动 Appium driver
    if activity in (None, "auto"):
        act = resolve_main_activity(pkg, serial)
        activity = act if act else None
        print(f"[resolve] {pkg} -> activity: {activity or 'Intent-Launcher'}")

    drv = make_driver(pkg=pkg, activity=activity, serial=serial, warmup_wait=warmup_wait)
    W, H = get_window_size(drv)
    L, T, R, B = int(W * safe[0]), int(H * safe[1]), int(W * safe[2]), int(H * safe[3])
    print(f"[v0-appium] {pkg}/{activity or 'auto'}, screen {W}x{H}, safe=({L},{T})~({R},{B})")

    # 模拟操作计时开始
    tim.start_sim()

    try:
        # 维持你现有的“随机策略”接口：每轮调用 common.policy_random.step(drv,W,H)
        for i in range(1, rounds + 1):
            try:
                msg = random_step(drv, W, H)  # 内部会随机从 tap/drag/long/pinch/zoom/rotate 里选并执行
                print(f"[{i:03d}/{rounds}] {msg}")
            except Exception as e:
                print(f"[{i:03d}/{rounds}] error: {e}")
            time.sleep(random.uniform(sleep_min, sleep_max))
    finally:
        # 模拟操作计时结束
        tim.stop_sim()
        try:
            drv.quit()
        except Exception:
            pass
        # 停止定时打印
        stop_evt.set()
        if printer:
            printer.join(timeout=1.0)
        # 设置总操作数（用 rounds 作为“发起的手势次数”）
        tim.set_ops_total(rounds)

        # 打印完整统计与时间
        print("[final]\n" + ver.summary_str(), flush=True)
        print(tim.summary_str(), flush=True)

        # 收尾 verify_motion
        ver.stop()

def main():
    ap = argparse.ArgumentParser(description="v0 Appium AR-Monkey (random gestures) + verify_motion + timing")
    ap.add_argument("--serial", help="可选，多设备时指定")
    ap.add_argument("--pkg", default="com.rooom.app")
    ap.add_argument("--activity", default="auto")
    ap.add_argument("--rounds", type=int, default=250)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--print-interval", type=int, default=10, help="每隔 N 秒打印一次统计；0=只在结束打印")
    ap.add_argument("--overall-include-tap", action="store_true", help="把 tap 纳入 overall（place 同时也会有 tap，谨慎开启）")
    args = ap.parse_args()

... (truncated)
```

### `v1_ar_monkey_appium.py`

```text
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v1_ar_monkey_appium.py — Appium (CV targeted + act-then-verify)
+ verify_motion(basis=tap) + timing

改动要点：
- 启动前 adb logcat -c，Logcat 采集从 -T 0 开始
- 接入 common.verify_motion（含 tap），overall 基于 tap：overall = tap+drag+pinch+rotate
  （place 视作 tap 的特例，仅展示不入 overall）
- 接入 common.timing：总时长 / 模拟时长 / 平均每操作耗时
- 支持 --print-interval 定时打印完整汇总；结束时打印 [final]/[compare]/[timing]

CV 与操作验证：
- 保留原 v1：cv.strategy_riri_invariant.locate 定位
- 仍使用 cv.verify_motion.verify_action 做动作后的图像验证（不影响统计口径）

python v1_ar_monkey_appium.py --pkg com.google.ar.sceneform.samples.hellosceneform \
    --activity auto --rounds 50 --print-interval 10
"""

import argparse
import random
import sys
import time
import os
import csv
import threading
import subprocess

from common.device import make_driver, get_window_size, resolve_main_activity, capture_bgr
from common.actions import tap, pinch_or_zoom, rotate, drag_line
from common.policy_random import step as random_step  # 若你备用随机策略仍需要
from cv.strategy_riri_invariant import locate as cv_locate
from cv.verify_motion import verify_action  # 保留：动作后验证

from common.verify_motion import MotionVerifier
from common.timing import Timing

# ----------------- 工具 -----------------
def _ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000.0

def clear_logcat(serial: str = None):
    cmd = ["adb"]
    if serial:
        cmd += ["-s", serial]
    cmd += ["logcat", "-c"]
    subprocess.run(cmd, check=False)

def periodic_print(ver: MotionVerifier, interval_sec: int, stop_evt: threading.Event):
    while not stop_evt.is_set():
        time.sleep(interval_sec)
        if stop_evt.is_set():
            break
        print("[tick]\n" + ver.summary_str(), flush=True)

def _map_img_to_window(x, y, w_img, h_img, W_win, H_win):
    # 若纵横比接近：按轴向缩放；否则按等比缩放 + 居中偏移（兼容 letterbox/状态栏）
    ar_img = w_img / float(h_img)
    ar_win = W_win / float(H_win)
    if abs(ar_img - ar_win) < 0.03:
        sx = W_win / float(w_img)
        sy = H_win / float(h_img)
        return int(x * sx), int(y * sy)
    else:
        s = min(W_win / float(w_img), H_win / float(h_img))
        x_off = int((W_win - w_img * s) * 0.5)
        y_off = int((H_win - h_img * s) * 0.5)
        return int(x * s + x_off), int(y * s + y_off)

def _map_bbox_to_window(x, y, w, h, w_img, h_img, W_win, H_win):
    x1, y1 = _map_img_to_window(x, y, w_img, h_img, W_win, H_win)
    x2, y2 = _map_img_to_window(x + w, y + h, w_img, h_img, W_win, H_win)
    X, Y = min(x1, x2), min(y1, y2)
    Wb, Hb = max(1, abs(x2 - x1)), max(1, abs(y2 - y1))
    return X, Y, Wb, Hb

# ----------------- 主流程 -----------------
def run_monkey_v1(
    pkg="com.rooom.app",
    activity="auto",
    serial=None,
    rounds=250,
    sleep_min=0.25,
    sleep_max=0.85,
    # 动作参数
    rotate_steps=8,
    warmup_wait=3.0,
    seed=None,
    drag_ms=(300, 900),
    # —— CV 的“公共”参数：与具体策略无关 ——
    cv_min_area_ratio=0.002,
    cv_downsample_w=640,
    # CSV
    log_csv=None,
    # 预触摸
    prime_tap=True,
    prime_pause_ms=80,
    # 验证（动作后）
    enable_verify=True,
    verify_wait_ms=140,
    drag_min_px=8.0,
    drag_dir_cos=0.6,
    verify_min_frac=0.5,
    pinch_scale_thr=0.10,
    rotate_min_deg=15.0,
    # 打印
    print_interval=10,
):
    """
    v1 主循环（CV 定位 → 执行动作 → 图像验证），并用 verify_motion 统计操作成功率（basis=tap）
    """
    if seed is not None:
        random.seed(seed)

    # 计时从此刻开始
    tim = Timing()

    # 解析 Activity 并启动驱动
... (truncated)
```

### `v2_ar_monkey_appium.py`

```text
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v2_ar_monkey_appium.py — Appium (CV targeted + act-then-verify)
+ verify_motion(basis=tap) + timing

改动要点：
- 启动前 adb logcat -c，Logcat 采集从 -T 0 开始
- 接入 common.verify_motion（含 tap），overall 基于 tap：overall = tap+drag+pinch+rotate
  （place 视作 tap 的特例，仅展示不入 overall）
- 接入 common.timing：总时长 / 模拟时长 / 平均每操作耗时
- 支持 --print-interval 定时打印完整汇总；结束时打印 [final]/[compare]/[timing]

CV 与操作验证：
- 保留原 v2：cv.strategy_riri_invariant.locate 定位
- 仍使用 cv.verify_motion.verify_action 做动作后的图像验证（不影响统计口径）

python v2_ar_monkey_appium.py --pkg com.google.ar.sceneform.samples.hellosceneform \
    --activity auto --rounds 50 --print-interval 10
"""

import argparse
import random
import sys
import time
import os
import csv
import threading
import subprocess

from common.device import make_driver, get_window_size, resolve_main_activity, capture_bgr
from common.actions import tap, pinch_or_zoom, rotate, drag_line
from common.policy_random import step as random_step  # 若你备用随机策略仍需要
from cv.strategy_yolo import locate as cv_locate
from cv.verify_motion import verify_action  # 保留：动作后验证

from common.verify_motion import MotionVerifier
from common.timing import Timing

# ----------------- 工具 -----------------
def _ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000.0

def clear_logcat(serial: str = None):
    cmd = ["adb"]
    if serial:
        cmd += ["-s", serial]
    cmd += ["logcat", "-c"]
    subprocess.run(cmd, check=False)

def periodic_print(ver: MotionVerifier, interval_sec: int, stop_evt: threading.Event):
    while not stop_evt.is_set():
        time.sleep(interval_sec)
        if stop_evt.is_set():
            break
        print("[tick]\n" + ver.summary_str(), flush=True)

def _map_img_to_window(x, y, w_img, h_img, W_win, H_win):
    # 若纵横比接近：按轴向缩放；否则按等比缩放 + 居中偏移（兼容 letterbox/状态栏）
    ar_img = w_img / float(h_img)
    ar_win = W_win / float(H_win)
    if abs(ar_img - ar_win) < 0.03:
        sx = W_win / float(w_img)
        sy = H_win / float(h_img)
        return int(x * sx), int(y * sy)
    else:
        s = min(W_win / float(w_img), H_win / float(h_img))
        x_off = int((W_win - w_img * s) * 0.5)
        y_off = int((H_win - h_img * s) * 0.5)
        return int(x * s + x_off), int(y * s + y_off)

def _map_bbox_to_window(x, y, w, h, w_img, h_img, W_win, H_win):
    x1, y1 = _map_img_to_window(x, y, w_img, h_img, W_win, H_win)
    x2, y2 = _map_img_to_window(x + w, y + h, w_img, h_img, W_win, H_win)
    X, Y = min(x1, x2), min(y1, y2)
    Wb, Hb = max(1, abs(x2 - x1)), max(1, abs(y2 - y1))
    return X, Y, Wb, Hb

# ----------------- 主流程 -----------------
def run_monkey_v2(
    pkg="com.rooom.app",
    activity="auto",
    serial=None,
    rounds=250,
    sleep_min=0.25,
    sleep_max=0.85,
    # 动作参数
    rotate_steps=8,
    warmup_wait=3.0,
    seed=None,
    drag_ms=(300, 900),
    # —— CV 的“公共”参数：与具体策略无关 ——
    cv_min_area_ratio=0.002,
    cv_downsample_w=640,
    # CSV
    log_csv=None,
    # 预触摸
    prime_tap=True,
    prime_pause_ms=80,
    # 验证（动作后）
    enable_verify=True,
    verify_wait_ms=140,
    drag_min_px=8.0,
    drag_dir_cos=0.6,
    verify_min_frac=0.5,
    pinch_scale_thr=0.10,
    rotate_min_deg=15.0,
    # 打印
    print_interval=10,
):
    """
    v2 主循环（CV 定位 → 执行动作 → 图像验证），并用 verify_motion 统计操作成功率（basis=tap）
    """
    if seed is not None:
        random.seed(seed)

    # 计时从此刻开始
    tim = Timing()

    # 解析 Activity 并启动驱动
... (truncated)
```

### `v3_ar_monkey_appium.py`

```text
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
v3_ar_monkey_appium.py — VLM定位 + SMART纠偏 + 目标导向动作（保持原有策略）
集成：
  - verify_motion（含 tap；overall 基于 tap）
  - timing（总时长 / 模拟时长 / 均值）
  - 启动前清空 logcat；采集从 -T 0 开始
  - 定时 [tick] 与最终 [final] / [compare] / [timing] 输出

原版要点保持：循环 rounds 次；每轮：截图→VLM定位→(可选)SMART纠偏→四选一动作（drag/pinch/zoom/rotate）

python v3_ar_monkey_appium.py --pkg com.google.ar.sceneform.samples.hellosceneform \
    --activity auto --rounds 50 --print-interval 10
"""

import os
import io
import csv
import time
import json
import base64
import argparse
import random
import threading
import subprocess
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageOps
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Qwen25VLChatHandler

from common.device import make_driver, get_window_size, capture_bgr
from common.actions import tap, pinch_or_zoom, drag_line, rotate, smart_correct_rule_based, smart_correct_model_driver

# 新增：统计&计时
from common.verify_motion import MotionVerifier
from common.timing import Timing

# ---------- 默认参数 ----------
DEFAULT_MODEL   = "../llama_models/qwen2_5/Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf"
DEFAULT_MMPROJ  = "../llama_models/qwen2_5/mmproj-F16.gguf"
DEFAULT_PKG     = "com.rooom.app"
DEFAULT_MAXSIDE = 896

# ---------- 工具 ----------
def clear_logcat(serial: str = None):
    cmd = ["adb"]
    if serial:
        cmd += ["-s", serial]
    cmd += ["logcat", "-c"]
    subprocess.run(cmd, check=False)

def periodic_print(ver: MotionVerifier, interval_sec: int, stop_evt: threading.Event):
    while not stop_evt.is_set():
        time.sleep(interval_sec)
        if stop_evt.is_set():
            break
        print("[tick]\n" + ver.summary_str(), flush=True)

# ---------- 图像 -> data:URI ----------
def bgr_to_data_uri(bgr, *, encode="png", apply_exif=True, jpeg_quality=90):
    rgb = bgr[:, :, ::-1]
    img = Image.fromarray(rgb)
    if apply_exif:
        img = ImageOps.exif_transpose(img)
    buf = io.BytesIO()
    if encode == "jpeg":
        img.save(buf, format="JPEG", quality=jpeg_quality, optimize=True, progressive=False)
        mime = "image/jpeg"
    else:
        img.save(buf, format="PNG")
        mime = "image/png"
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

# ---------- 提示词（加入屏幕分辨率） ----------
def build_messages(image_uri: str, screen_w: int, screen_h: int):
    sys_msg = "你是一个只输出JSON的多模态助手。不要解释，不要多余文本。"
    user_text = (
        f"这是一张来自一台手机的**全屏截图**，屏幕分辨率为 {screen_w}x{screen_h} 像素。\n"
        "请识别图像中的“主要 AR 物体”（最显著/居中/占比最大的虚拟对象），并返回其中心坐标与边界框。\n"
        "要求：所有坐标/尺寸均以**当前这张截图**为基准，使用归一化坐标（范围 0~1），并保留≤3位小数。\n"
        "只返回以下 JSON，放在 <json> 与 </json> 之间：\n"
        "{\n"
        '  "name": "string",\n'
        '  "coords": {"cx": float, "cy": float},\n'
        '  "bbox": {"x": float, "y": float, "w": float, "h": float},\n'
        '  "confidence": float\n'
        "}\n"
        "<json>"
    )
    return [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": image_uri}},
            {"type": "text", "text": user_text}
        ]}
    ]

def extract_first_json(s: str) -> Optional[str]:
    if not s:
        return None
    s = s.split("<|im_end|>")[0]
    start = s.find("{"); end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return s[start:end+1]

def ask_and_parse(llm: Llama, messages, retries: int = 2):
    last = ""
    for _ in range(retries + 1):
        out = llm.create_chat_completion(
            messages=messages,
            temperature=0.1,
            max_tokens=512,
            stop=["</json>"],
        )
... (truncated)
```

### `yolo_adb_screencap.py`

```text
#!/usr/bin/env python3
import argparse
import io
import os
import sys
import time
import json
import subprocess
from datetime import datetime
from typing import Optional, Tuple

try:
    from PIL import Image
except ImportError:
    print("Pillow is required. Install with: pip install pillow", file=sys.stderr)
    sys.exit(1)

try:
    from ultralytics import YOLO
except ImportError:
    print("Ultralytics YOLO is required. Install with: pip install ultralytics", file=sys.stderr)
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    print("numpy is required. Install with: pip install numpy", file=sys.stderr)
    sys.exit(1)

try:
    import cv2
except ImportError:
    cv2 = None  # optional for saving annotated images


def adb_cmd(serial: Optional[str], *args: str) -> list:
    cmd = ["adb"]
    if serial:
        cmd += ["-s", serial]
    cmd += list(args)
    return cmd


def check_adb(serial: Optional[str]) -> None:
    # Ensure adb is available and device is connected
    try:
        subprocess.run(["adb", "version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("ERROR: adb not found. Please install Android Platform Tools and ensure 'adb' is in PATH.", file=sys.stderr)
        sys.exit(2)

    # Check device connectivity
    out = subprocess.run(adb_cmd(serial, "get-state"), capture_output=True, text=True)
    if out.returncode != 0 or out.stdout.strip() not in {"device", "authorizing"}:
        print("ERROR: Device not connected or unauthorized. Run `adb devices` and allow USB debugging.", file=sys.stderr)
        sys.exit(3)


def grab_screenshot(serial: Optional[str]):
    """
    Grab a PNG screenshot via ADB and return as a PIL Image.
    """
    p = subprocess.run(adb_cmd(serial, "exec-out", "screencap", "-p"), capture_output=True)
    if p.returncode != 0 or not p.stdout:
        raise RuntimeError(f"adb screencap failed: {p.stderr.decode(errors='ignore')}")
    data = p.stdout
    from PIL import Image
    img = Image.open(io.BytesIO(data)).convert("RGB")
    return img


def save_annotated(res, out_path: str) -> None:
    """
    Save an annotated image using result.plot(). Requires cv2 for best fidelity.
    """
    img = res.plot()  # numpy array (BGR)
    if cv2 is None:
        from PIL import Image as _Image
        img_rgb = img[..., ::-1]
        _Image.fromarray(img_rgb).save(out_path)
    else:
        cv2.imwrite(out_path, img)


def main():
    ap = argparse.ArgumentParser(description="Every N seconds capture Android screenshot via ADB and run YOLO inference.")
    ap.add_argument("--model", type=str, default="yolo11n.pt", help="YOLO model path or name (e.g., yolo11n.pt)")
    ap.add_argument("--serial", type=str, default=None, help="ADB device serial (optional; default: first device)")
    ap.add_argument("--interval", type=float, default=5.0, help="Seconds between captures")
    ap.add_argument("--outdir", type=str, default="yolo_snaps", help="Directory to save annotated results")
    ap.add_argument("--conf", type=float, default=0.1, help="Confidence threshold")
    ap.add_argument("--device", type=str, default=None, help="Inference device, e.g. 'cpu', 'mps', 'cuda:0' (optional)")
    ap.add_argument("--save", action="store_true", help="Save annotated images (PNG) and JSON summaries")
    ap.add_argument("--max", type=int, default=0, help="Max iterations (0 = infinite)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    check_adb(args.serial)

    print(f"[YOLO] Loading model: {args.model}", flush=True)
    model = YOLO(args.model)

    try:
        model.fuse()
    except Exception:
        pass

    iter_count = 0
    print("[RUN] Start loop. Press Ctrl+C to stop.", flush=True)
    try:
        while True:
            t0 = time.time()
            img = grab_screenshot(args.serial)

            # Predict
            pred_t0 = time.time()
            results = model(img, conf=args.conf, device=args.device, verbose=False)
            pred_ms = (time.time() - pred_t0) * 1000

            res = results[0]
... (truncated)
```

### `yolo_scratch.py`

```text
"""
python yolo_scratch.py
"""

from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("cv/yolo/best.pt")

# Define path to the image file
source = "val/Screenshot_20251023-165222.png"

# Run inference on the source
results = model(source, conf=0.05, iou=0.45, imgsz=960, verbose=False)  # list of Results objects


for result in results:
    boxes = result.boxes.xyxy  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    print(
        f"boxes: {boxes}\n"
        f"masks: {masks}\n"
        f"keypoints: {keypoints}\n"
        f"probs: {probs}\n"
        f"obb: {obb}\n"
    )
    result.show()  # display to screen
    result.save(filename="result.jpg")
    
```

### `yolo_val.py`

```text
from ultralytics import YOLO

# Load a model
model = YOLO("cv/yolo/best.pt")

# Customize validation settings
metrics = model.val(data="cv/yolo/coco8.yaml", imgsz=640)
```

## Excludes

```
.git
.svn
.hg
.DS_Store
__pycache__
*.pyc
*.pyo
*.pyd
.idea
.vscode
.venv
.venv311
venv
env
ENV
.conda
conda
.env
build
dist
.mypy_cache
.pytest_cache
.ruff_cache
node_modules
runs
outputs
logs
*.pt
*.onnx
*.ckpt
*.mp4
*.mov
*.avi
*.png
*.jpg
*.jpeg
*.webp
*.zip
*.tar
*.gz
*.7z
**/.venv/**
**/.venv311/**
```
