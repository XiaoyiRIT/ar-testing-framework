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

def double_tap(driver, x, y, tap_interval_ms=100, press_ms=60):
    """
    Perform a double-tap gesture at (x, y).
    tap_interval_ms: time between the two taps
    press_ms: duration of each tap press
    """
    a = ActionBuilder(driver)
    f1 = a.add_pointer_input(kind="touch", name="finger1")

    # First tap
    f1.create_pointer_move(duration=0, x=int(x), y=int(y))
    f1.create_pointer_down(button=0)
    f1.create_pause(press_ms/1000.0)
    f1.create_pointer_up(button=0)

    # Interval between taps
    f1.create_pause(tap_interval_ms/1000.0)

    # Second tap
    f1.create_pointer_move(duration=0, x=int(x), y=int(y))
    f1.create_pointer_down(button=0)
    f1.create_pause(press_ms/1000.0)
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
    driver, W, H, det: dict,
    zoom_cfg=None, edge_cfg=None
) -> str:
    """
    规则层纠偏（缩放/回中），输入为检测 det：
    det = {"coords":{"cx","cy"}, "bbox":{"x","y","w","h"}, "confidence":float, "name":str}
    返回动作描述字符串（用于日志）。
    """
    if zoom_cfg is None:
        zoom_cfg = {"min_area": 0.05, "max_area": 0.35, "zoom_step": 0.22}
    if edge_cfg is None:
        edge_cfg = {"edge_tol": 0.08, "drag_frac": 0.20, "duration_ms": 600}

    actions = []
    bbox = det.get("bbox") or {}
    try:
        x = float(bbox.get("x", -1)); y = float(bbox.get("y", -1))
        w = float(bbox.get("w", -1)); h = float(bbox.get("h", -1))
        cx = float(det["coords"]["cx"]); cy = float(det["coords"]["cy"])
    except Exception:
        return ""

    # 无 bbox，用中心点靠边判定是否回正
    if min(x, y, w, h) < 0:
        edge = edge_cfg["edge_tol"]
        if (cx < edge or cy < edge or (1-cx) < edge or (1-cy) < edge):
            sx, sy = int(cx * W), int(cy * H)
            tx, ty = int(0.5 * W), int(0.5 * H)
            drag_line(driver, sx, sy, tx, ty, duration_ms=edge_cfg["duration_ms"])
            actions.append(f"recenter_by_center drag ({sx},{sy})->({tx},{ty})")
        return "; ".join(actions)

    # 1) 缩放
    area = max(0.0, min(1.0, w*h))
    cx_px, cy_px = int(cx*W), int(cy*H)
    if area > zoom_cfg["max_area"]:
        start = int(max(80, min(W, H) * (zoom_cfg["zoom_step"] + 0.15)))
        end   = int(max(30, start * 0.5))
        pinch_or_zoom(driver, cx_px, cy_px, start_dist=start, end_dist=end, duration_ms=700)
        actions.append(f"zoom_out area={area:.3f} start={start}->end={end}")
    elif area < zoom_cfg["min_area"]:
        start = int(max(60, min(W, H) * 0.10))
        end   = int(max(120, int(start * (1.0 + zoom_cfg['zoom_step']*2))))
        pinch_or_zoom(driver, cx_px, cy_px, start_dist=start, end_dist=end, duration_ms=700)
        actions.append(f"zoom_in area={area:.3f} start={start}->end={end}")

    # 2) 边缘回正
    left, top, right, bottom = x, y, x+w, y+h
    tol = edge_cfg["edge_tol"]
    if (left < tol) or (top < tol) or ((1-right) < tol) or ((1-bottom) < tol):
        sx, sy = int(cx * W), int(cy * H)
        tx = int(sx + (0.5*W - sx) * edge_cfg["drag_frac"])
        ty = int(sy + (0.5*H - sy) * edge_cfg["drag_frac"])
        drag_line(driver, sx, sy, tx, ty, duration_ms=edge_cfg["duration_ms"])
        actions.append(f"recenter drag ({sx},{sy})->({tx},{ty})")

    return "; ".join(actions)

def smart_correct_model_driver(driver, W, H, bgr, det) -> str:
    s = det.get("smart") or {}
    act = (s.get("action") or "none").lower()
    if act == "none":
        return ""
    if act in ("zoom_in","zoom_out"):
        cx = float(det["coords"]["cx"]); cy = float(det["coords"]["cy"])
        cx_px, cy_px = int(cx*W), int(cy*H)
        k = float(s.get("strength", 0.5))
        k = max(0.0, min(1.0, k))
        base = int(max(60, min(W,H)*0.12))
        if act == "zoom_in":
            start, end = base, int(base*(1.8 + 1.0*k))
            pinch_or_zoom(driver, cx_px, cy_px, start_dist=start, end_dist=end, duration_ms=700)
            return f"model_zoom_in k={k:.2f} {start}->{end}"
        else:
            start, end = int(base*(2.2 + 1.0*k)), int(base*0.6)
            pinch_or_zoom(driver, cx_px, cy_px, start_dist=start, end_dist=end, duration_ms=700)
            return f"model_zoom_out k={k:.2f} {start}->{end}"
    if act == "recenter":
        v = s.get("vector") or {}
        dx = float(v.get("dx", 0.0)); dy = float(v.get("dy", 0.0))
        cx = float(det["coords"]["cx"]); cy = float(det["coords"]["cy"])
        sx, sy = int(cx*W), int(cy*H)
        # 将 [-1,1] 的向量映射到屏幕的 20% 距离（可调）
        tx = int(sx + dx * 0.2 * W)
        ty = int(sy + dy * 0.2 * H)
        drag_line(driver, sx, sy, tx, ty, duration_ms=600)
        return f"model_recenter vec=({dx:.2f},{dy:.2f}) ({sx},{sy})->({tx},{ty})"
    return ""

