# common/policy_random.py
# ------------------------------------------------------------
# Description:
#   Random gesture policy used by v0 and as fallback in v1.
#   Generates coordinates + calls coordinate-only actions.
# ------------------------------------------------------------

# src/common/policy_random.py
import random
from typing import Tuple
from .actions import tap, drag_line, long_press, pinch_or_zoom, rotate


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
