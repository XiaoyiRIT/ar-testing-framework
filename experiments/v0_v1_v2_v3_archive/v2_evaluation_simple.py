#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v2_evaluation_simple.py — Simple Frame-Diff Evaluation Script

基于 v2_evaluation.py 的流程，移除复杂 CV 验证逻辑，仅在手势前后取帧，
通过帧差 + SSIM 判断操作是否生效，再与 Ground Truth 对比统计。
"""

import argparse
import csv
import json
import os
import random
import subprocess
import sys
import threading
import time

import cv2
import numpy as np

# 添加项目根目录和 src 目录到 Python 路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from src.common.device import capture_bgr, get_window_size, make_driver, resolve_main_activity
from src.common.actions import (
    double_tap,
    drag_line,
    flick,
    long_press,
    pinch_or_zoom,
    rotate,
    swipe,
    tap,
    triple_tap,
    two_finger_tap,
)
from cv.strategy_yolo import locate as cv_locate
from src.common.verify_motion import MotionVerifier
from src.common.timing import Timing


# ----------------- 工具 -----------------
def _ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000.0


def execute_operation(drv, op_name, cx, cy, W, H, Ww=None, Hw=None, w_img_box=None, h_img_box=None,
                      drag_ms=(300, 900), rotate_steps=8):
    """
    执行指定的操作

    Returns:
        str: 操作描述信息
    """
    if op_name == "tap":
        tap(drv, cx, cy, press_ms=random.randint(40, 120))
        return f"tap at ({cx},{cy})"

    elif op_name == "double_tap":
        interval_ms = random.randint(80, 150)
        double_tap(drv, cx, cy, tap_interval_ms=interval_ms)
        return f"double_tap at ({cx},{cy}) interval={interval_ms}ms"

    elif op_name == "triple_tap":
        interval_ms = random.randint(80, 150)
        triple_tap(drv, cx, cy, tap_interval_ms=interval_ms)
        return f"triple_tap at ({cx},{cy}) interval={interval_ms}ms"

    elif op_name == "long_press":
        hold_ms = random.randint(800, 1200)
        long_press(drv, cx, cy, hold_ms=hold_ms)
        return f"long_press at ({cx},{cy}) hold={hold_ms}ms"

    elif op_name == "drag":
        if Ww is None or Hw is None:
            Ww = Hw = 100  # default
        max_step = min(int(min(W, H) / 3), int(0.6 * max(Ww, Hw)))
        dist = max(40, max_step)
        dx, dy = random.choice([(dist, 0), (-dist, 0), (0, dist), (0, -dist)])
        x2 = max(1, min(W - 2, cx + dx))
        y2 = max(1, min(H - 2, cy + dy))
        dur = random.randint(*drag_ms)
        drag_line(drv, cx, cy, x2, y2, duration_ms=dur)
        return f"drag from ({cx},{cy}) to ({x2},{y2})"

    elif op_name == "swipe":
        # Fast swipe - short duration
        max_step = min(int(min(W, H) / 3), 200)
        dist = random.randint(80, max_step)
        dx, dy = random.choice([(dist, 0), (-dist, 0), (0, dist), (0, -dist)])
        x2 = max(1, min(W - 2, cx + dx))
        y2 = max(1, min(H - 2, cy + dy))
        swipe(drv, cx, cy, x2, y2, duration_ms=random.randint(100, 180))
        return f"swipe from ({cx},{cy}) to ({x2},{y2})"

    elif op_name == "flick":
        # Very fast flick - very short distance
        dist = random.randint(30, 80)
        dx, dy = random.choice([(dist, 0), (-dist, 0), (0, dist), (0, -dist)])
        x2 = max(1, min(W - 2, cx + dx))
        y2 = max(1, min(H - 2, cy + dy))
        flick(drv, cx, cy, x2, y2, duration_ms=random.randint(60, 100))
        return f"flick from ({cx},{cy}) to ({x2},{y2})"

    elif op_name == "pinch_in":
        if w_img_box is None or h_img_box is None:
            s, e = 150, 60  # default
        else:
            s = max(40, min(w_img_box // 2, h_img_box // 2, 220))
            e = max(20, min(s // 2, 120))
        pinch_or_zoom(drv, cx, cy, start_dist=s, end_dist=e,
                      duration_ms=random.randint(450, 900))
        return f"pinch_in at ({cx},{cy})"

    elif op_name == "rotate":
        if Ww is None or Hw is None:
            radius = 120  # default
        else:
            radius = max(60, min(int(max(Ww, Hw) / 2), 260))
        angle = random.choice([45, 60, 90])
        rotate(drv, cx, cy, radius=radius, angle_deg=angle,
               duration_ms=random.randint(600, 1100), steps=rotate_steps)
        return f"rotate at ({cx},{cy}) angle={angle}deg"

    elif op_name == "two_finger_tap":
        finger_dist = random.randint(80, 150)
        two_finger_tap(drv, cx, cy, finger_dist=finger_dist)
        return f"two_finger_tap at ({cx},{cy}) dist={finger_dist}"

    else:
        # Unknown operation - just tap
        tap(drv, cx, cy, press_ms=60)
        return f"unknown_op:{op_name} (fallback to tap)"


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


def _clamp_bbox(bbox, width, height):
    if not bbox:
        return None
    x, y, w, h = bbox
    x = max(0, min(int(x), width - 1))
    y = max(0, min(int(y), height - 1))
    w = max(1, min(int(w), width - x))
    h = max(1, min(int(h), height - y))
    return x, y, w, h


def _compute_ssim(gray_a, gray_b):
    gray_a = gray_a.astype(np.float32)
    gray_b = gray_b.astype(np.float32)
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    mu_a = cv2.GaussianBlur(gray_a, (11, 11), 1.5)
    mu_b = cv2.GaussianBlur(gray_b, (11, 11), 1.5)
    mu_a2 = mu_a * mu_a
    mu_b2 = mu_b * mu_b
    mu_ab = mu_a * mu_b

    sigma_a2 = cv2.GaussianBlur(gray_a * gray_a, (11, 11), 1.5) - mu_a2
    sigma_b2 = cv2.GaussianBlur(gray_b * gray_b, (11, 11), 1.5) - mu_b2
    sigma_ab = cv2.GaussianBlur(gray_a * gray_b, (11, 11), 1.5) - mu_ab

    numerator = (2 * mu_ab + c1) * (2 * sigma_ab + c2)
    denominator = (mu_a2 + mu_b2 + c1) * (sigma_a2 + sigma_b2 + c2)
    ssim_map = numerator / np.maximum(denominator, 1e-6)
    return float(np.mean(ssim_map))


def simple_frame_diff_ratio(pre_bgr, post_bgr, bbox=None, diff_threshold=18, downsample_w=320):
    if pre_bgr is None or post_bgr is None:
        return 0.0, 1.0, 0, 0

    h, w = pre_bgr.shape[:2]
    if bbox:
        x, y, bw, bh = _clamp_bbox(bbox, w, h)
        pre_bgr = pre_bgr[y:y + bh, x:x + bw]
        post_bgr = post_bgr[y:y + bh, x:x + bw]

    if downsample_w and pre_bgr.shape[1] > downsample_w:
        scale = downsample_w / float(pre_bgr.shape[1])
        new_h = max(1, int(pre_bgr.shape[0] * scale))
        pre_bgr = cv2.resize(pre_bgr, (downsample_w, new_h), interpolation=cv2.INTER_AREA)
        post_bgr = cv2.resize(post_bgr, (downsample_w, new_h), interpolation=cv2.INTER_AREA)

    pre_g = cv2.cvtColor(pre_bgr, cv2.COLOR_BGR2GRAY)
    post_g = cv2.cvtColor(post_bgr, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(pre_g, post_g)
    changed = diff > diff_threshold
    changed_pixels = int(np.count_nonzero(changed))
    total_pixels = int(changed.size)
    change_ratio = float(changed_pixels) / float(max(1, total_pixels))
    ssim_score = _compute_ssim(pre_g, post_g)
    return change_ratio, ssim_score, changed_pixels, total_pixels


# ----------------- Ground Truth Detection -----------------
def get_recent_logcat_lines(serial: str = None, num_lines: int = 20):
    """获取最近的 logcat 行"""
    cmd = ["adb"]
    if serial:
        cmd += ["-s", serial]
    cmd += ["logcat", "-d", "-s", "AR_OP:D"]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=2)
        lines = result.stdout.strip().split("\n")
        return lines[-num_lines:] if len(lines) > num_lines else lines
    except Exception:
        return []


def check_ground_truth(serial: str, op_name: str):
    """
    检查最近的 logcat，看是否有对应操作的成功记录

    Args:
        serial: 设备序列号
        op_name: 操作名称 (pinch_in, drag, rotate, long_press, double_tap)

    Returns:
        bool: True 表示 logcat 中有对应操作的成功记录
    """
    op_map = {
        "pinch_in": "pinch",
        "pinch_out": "pinch",
        "zoom_in": "pinch",
        "zoom_out": "pinch",
        "drag": "drag",
        "rotate": "rotate",
        "long_press": "long_press_end",
        "double_tap": "double_tap",
        "tap": "tap",
    }

    target_kind = op_map.get(op_name, op_name)
    recent_lines = get_recent_logcat_lines(serial, num_lines=15)

    for line in reversed(recent_lines):
        if "AR_OP" in line and "{" in line:
            try:
                json_start = line.index("{")
                json_str = line[json_start:]
                data = json.loads(json_str)

                kind = data.get("kind", "").strip().lower()
                ok = data.get("ok", False)

                if kind == target_kind and ok:
                    return True
            except Exception:
                continue

    return False


# ----------------- 主流程 -----------------
def run_evaluation(
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
    # CSV
    log_csv=None,
    # 预触摸
    prime_tap=False,
    prime_pause_ms=80,
    # 简单差分参数
    verify_wait_ms=140,
    diff_threshold=18,
    min_change_ratio=0.02,
    ssim_threshold=0.92,
    min_changed_pixels=0,
    diff_downsample_w=320,
    # 打印
    print_interval=10,
    # 操作类型
    supported_ops=None,
    unsupported_ops=None,
    negative_sample_ratio=0.5,
):
    """
    评估版主循环：执行操作 → 帧差+SSIM验证 → GT检测 → 对比 → 统计准确率
    """
    if seed is not None:
        random.seed(seed)

    if supported_ops is None:
        supported_ops = ["tap", "double_tap", "drag", "long_press", "pinch_in", "rotate"]
    if unsupported_ops is None:
        unsupported_ops = ["triple_tap", "swipe", "two_finger_tap", "flick"]

    all_operations = supported_ops + unsupported_ops
    operation_plan = []
    ops_per_type = rounds // len(all_operations)
    remainder = rounds % len(all_operations)

    for i, op in enumerate(all_operations):
        count = ops_per_type + (1 if i < remainder else 0)
        operation_plan.extend([op] * count)

    random.shuffle(operation_plan)
    negative_plan = [random.random() < negative_sample_ratio for _ in range(rounds)]

    print("[v2_eval_simple] Operation distribution:")
    from collections import Counter
    op_counts = Counter(operation_plan)
    for op, count in sorted(op_counts.items()):
        support_status = "✓ supported" if op in supported_ops else "✗ unsupported"
        print(f"  {op:18s}: {count:3d} times  ({support_status})")
    print(f"[v2_eval_simple] Negative samples: {sum(negative_plan)}/{rounds} "
          f"({100*sum(negative_plan)/rounds:.1f}%)")
    print(f"[v2_eval_simple] Random seed: {seed}")

    tim = Timing()

    if activity in (None, "auto"):
        act = resolve_main_activity(pkg, serial)
        activity = act if act else None
        print(f"[resolve] {pkg} -> activity: {activity or 'Intent-Launcher'}")

    drv = make_driver(pkg=pkg, activity=activity, serial=serial, warmup_wait=warmup_wait)
    W, H = get_window_size(drv)
    print(f"[v2_eval_simple] {pkg}/{activity or 'auto'}, screen {W}x{H}")

    clear_logcat(serial)
    ver = MotionVerifier(overall_basis="tap")
    extra = ["-T", "0"]
    ver.start_from_adb(serial=serial, extra_logcat_args=extra)

    stop_evt = threading.Event()
    printer = None
    if print_interval and print_interval > 0:
        printer = threading.Thread(target=periodic_print, args=(ver, print_interval, stop_evt), daemon=True)
        printer.start()

    csv_fp = csv_writer = None
    if log_csv:
        d = os.path.dirname(log_csv)
        if d:
            os.makedirs(d, exist_ok=True)
        csv_fp = open(log_csv, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_fp)
        csv_writer.writerow([
            "step", "detected", "diff_ratio", "ssim", "changed_px", "total_px",
            "cv_verified", "gt_verified", "cv_correct",
            "operation", "is_negative", "is_supported", "cx_img", "cy_img", "bbox_x", "bbox_y", "bbox_w", "bbox_h",
            "message", "cap_ms", "cv_ms", "action_ms", "verify&wait_ms", "total_ms"
        ])

    prev = capture_bgr(drv)
    tim.start_sim()

    cv_verified_count = 0
    gt_verified_count = 0
    tp_count = 0
    tn_count = 0
    fp_count = 0
    fn_count = 0

    from collections import defaultdict
    tp_by_op = defaultdict(int)
    tn_by_op = defaultdict(int)
    fp_by_op = defaultdict(int)
    fn_by_op = defaultdict(int)

    unsupported_total = 0
    unsupported_cv_reject = 0
    unsupported_cv_accept = 0

    try:
        for i in range(1, rounds + 1):
            T_loop = time.perf_counter()

            T = time.perf_counter()
            curr = capture_bgr(drv)
            t_cap = _ms(T)

            T = time.perf_counter()
            det = cv_locate(
                image_bgr=curr,
                debug_profile=True
            )
            t_cv = _ms(T)

            T = time.perf_counter()
            op_name = operation_plan[i - 1]
            is_negative = negative_plan[i - 1]
            is_supported = op_name in supported_ops

            act_name = None
            msg = ""
            bbox_win = None
            cx_out = cy_out = bx = by = bw = bh = ""
            detected = 1 if det is not None else 0
            cv_verified = 0
            gt_verified = 0
            cv_correct = 0
            diff_ratio = 0.0
            ssim_score = 1.0
            changed_pixels = 0
            total_pixels = 0
            roi_bbox = None

            if is_negative or det is None:
                h_img, w_img = curr.shape[:2]

                if det is not None and is_negative:
                    cx_img, cy_img = det["center"]
                    x_img, y_img, w_img_box, h_img_box = det["bbox"]

                    margin = 0.15
                    attempts = 0
                    while attempts < 10:
                        rand_cx_img = random.randint(int(w_img * margin), int(w_img * (1 - margin)))
                        rand_cy_img = random.randint(int(h_img * margin), int(h_img * (1 - margin)))

                        dist_x = abs(rand_cx_img - cx_img) / w_img
                        dist_y = abs(rand_cy_img - cy_img) / h_img
                        if dist_x > 0.2 or dist_y > 0.2:
                            break
                        attempts += 1

                    cx, cy = _map_img_to_window(rand_cx_img, rand_cy_img, w_img, h_img, W, H)
                    bbox_win = None
                    msg_prefix = f"NEGATIVE({op_name})"
                else:
                    cx = random.randint(int(W * 0.2), int(W * 0.8))
                    cy = random.randint(int(H * 0.2), int(H * 0.8))
                    bbox_win = None
                    msg_prefix = f"MISS/NEGATIVE({op_name})"

                act_name = op_name
                pre_act = curr
                op_msg = execute_operation(drv, op_name, cx, cy, W, H,
                                           drag_ms=drag_ms, rotate_steps=rotate_steps)
                msg = f"{msg_prefix}: {op_msg}"

            else:
                cx_img, cy_img = det["center"]
                x_img, y_img, w_img_box, h_img_box = det["bbox"]

                h_img, w_img = curr.shape[:2]
                cx, cy = _map_img_to_window(cx_img, cy_img, w_img, h_img, W, H)
                Xw, Yw, Ww, Hw = _map_bbox_to_window(x_img, y_img, w_img_box, h_img_box, w_img, h_img, W, H)
                bbox_win = (Xw, Yw, Ww, Hw)
                roi_bbox = (x_img, y_img, w_img_box, h_img_box)

                cx_out, cy_out, bx, by, bw, bh = cx_img, cy_img, x_img, y_img, w_img_box, h_img_box

                if prime_tap and op_name != "tap":
                    tap(drv, cx, cy, press_ms=random.randint(40, 120))
                    if prime_pause_ms > 0:
                        time.sleep(prime_pause_ms / 1000.0)

                act_name = op_name
                pre_act = curr

                op_msg = execute_operation(drv, op_name, cx, cy, W, H, Ww, Hw, w_img_box, h_img_box,
                                           drag_ms=drag_ms, rotate_steps=rotate_steps)

                support_tag = "✓" if is_supported else "✗"
                msg = f"{support_tag} {'tap+' if (prime_tap and op_name != 'tap') else ''}{op_msg} bbox={bbox_win}"

            t_action = _ms(T)

            T = time.perf_counter()
            if verify_wait_ms > 0:
                time.sleep(verify_wait_ms / 1000.0)
            post_act = capture_bgr(drv)

            diff_ratio, ssim_score, changed_pixels, total_pixels = simple_frame_diff_ratio(
                pre_act,
                post_act,
                bbox=roi_bbox,
                diff_threshold=diff_threshold,
                downsample_w=diff_downsample_w,
            )
            cv_verified = 1 if (
                diff_ratio >= min_change_ratio
                and ssim_score <= ssim_threshold
                and (min_changed_pixels <= 0 or changed_pixels >= min_changed_pixels)
            ) else 0
            if cv_verified:
                cv_verified_count += 1

            time.sleep(0.1)
            gt_ok = check_ground_truth(serial, act_name)
            gt_verified = 1 if gt_ok else 0
            if gt_verified:
                gt_verified_count += 1

            if not is_supported:
                unsupported_total += 1
                if cv_verified == 0 and gt_verified == 0:
                    unsupported_cv_reject += 1
                elif cv_verified == 1 and gt_verified == 0:
                    unsupported_cv_accept += 1

            if cv_verified == 1 and gt_verified == 1:
                tp_count += 1
                tp_by_op[act_name] += 1
                cv_correct = 1
            elif cv_verified == 0 and gt_verified == 0:
                tn_count += 1
                tn_by_op[act_name] += 1
                cv_correct = 1
            elif cv_verified == 1 and gt_verified == 0:
                fp_count += 1
                fp_by_op[act_name] += 1
                cv_correct = 0
            elif cv_verified == 0 and gt_verified == 1:
                fn_count += 1
                fn_by_op[act_name] += 1
                cv_correct = 0

            t_verify_wait = _ms(T)

            t_total = _ms(T_loop)
            if det is None:
                status = "MISS"
            else:
                status = (f"diff={diff_ratio:.4f} ssim={ssim_score:.4f} "
                          f"px={changed_pixels}/{total_pixels} "
                          f"CV={cv_verified} GT={gt_verified} {'✓' if cv_correct else '✗'}")
            print(
                f"[v2_eval_simple r{i:03d}] cap={t_cap:.1f}ms  cv={t_cv:.1f}ms  action={t_action:.1f}ms  "
                f"verify&wait={t_verify_wait:.1f}ms  TOTAL={t_total:.1f}ms  "
                f"{status}:{act_name}"
                + (f"  bbox={bbox_win}" if det is not None else "")
            )
            print(f"[{i:03d}/{rounds}] {msg}")

            if csv_writer:
                csv_writer.writerow([
                    i, detected, f"{diff_ratio:.4f}", f"{ssim_score:.4f}", changed_pixels, total_pixels,
                    cv_verified, gt_verified, cv_correct,
                    act_name, 1 if is_negative else 0, 1 if is_supported else 0,
                    cx_out, cy_out, bx, by, bw, bh,
                    msg, f"{t_cap:.1f}", f"{t_cv:.1f}", f"{t_action:.1f}",
                    f"{t_verify_wait:.1f}", f"{t_total:.1f}"
                ])

            prev = curr
            time.sleep(random.uniform(sleep_min, sleep_max))

        total_ops = tp_count + tn_count + fp_count + fn_count
        accuracy = (tp_count + tn_count) / max(1, total_ops)
        precision = tp_count / max(1, tp_count + fp_count)
        recall = tp_count / max(1, tp_count + fn_count)
        f1_score = 2 * precision * recall / max(0.001, precision + recall)

        unsupported_accuracy = unsupported_cv_reject / max(1, unsupported_total) if unsupported_total > 0 else 0

        print("\n" + "=" * 70)
        print("[EVALUATION RESULTS]")
        print("=" * 70)
        print(f"Total script operations:           {rounds}")
        print(f"CV evaluation operations:          {total_ops}  (all operations evaluated)")
        print()
        print("Overall CV/GT statistics:")
        print(f"  CV verified (change detected):   {cv_verified_count}/{rounds} "
              f"({100.0 * cv_verified_count / max(1, rounds):.1f}%)")
        print(f"  GT verified (app confirmed):     {gt_verified_count}/{rounds} "
              f"({100.0 * gt_verified_count / max(1, rounds):.1f}%)")

        print("\n" + "-" * 70)
        print("[Confusion Matrix - Frame Diff + SSIM Performance]")
        print("-" * 70)
        print(f"Based on {total_ops} operations (all supported + unsupported + negative samples)")
        print()
        print(f"True Positive (TP):   {tp_count:4d}  (CV=1, GT=1) ✓ Correct detection")
        print(f"True Negative (TN):   {tn_count:4d}  (CV=0, GT=0) ✓ Correct rejection")
        print(f"False Positive (FP):  {fp_count:4d}  (CV=1, GT=0) ✗ False alarm")
        print(f"False Negative (FN):  {fn_count:4d}  (CV=0, GT=1) ✗ Missed detection")

        if fn_count > 0 or fp_count > 0:
            print("\n" + "-" * 70)
            print("[Detailed Breakdown by Operation Type]")
            print("-" * 70)

            if fn_count > 0:
                print(f"\n❌ False Negative ({fn_count} total) - CV missed these operations:")
                for op, count in sorted(fn_by_op.items(), key=lambda x: -x[1]):
                    percentage = 100.0 * count / fn_count
                    print(f"  {op:20s}: {count:3d} ({percentage:5.1f}%)")

            if fp_count > 0:
                print(f"\n⚠️  False Positive ({fp_count} total) - CV incorrectly detected:")
                for op, count in sorted(fp_by_op.items(), key=lambda x: -x[1]):
                    percentage = 100.0 * count / fp_count
                    print(f"  {op:20s}: {count:3d} ({percentage:5.1f}%)")

            if tp_count > 0:
                print(f"\n✓ True Positive ({tp_count} total) - CV correctly detected:")
                for op, count in sorted(tp_by_op.items(), key=lambda x: -x[1]):
                    percentage = 100.0 * count / tp_count
                    print(f"  {op:20s}: {count:3d} ({percentage:5.1f}%)")

        print()
        print("TN breakdown:")
        print(f"  - Unsupported operations:  ~{unsupported_cv_reject}")
        print(f"  - Negative samples:        ~{tn_count - unsupported_cv_reject}")
        print("  - Failed CV operations:    (included above)")

        print("\n" + "-" * 60)
        print("[Overall Metrics]")
        print("-" * 60)
        print(f"Accuracy:  {accuracy:.4f} ({100.0 * accuracy:.2f}%)  - (TP+TN)/(TP+TN+FP+FN)")
        print(f"Precision: {precision:.4f} ({100.0 * precision:.2f}%)  - TP/(TP+FP)")
        print(f"Recall:    {recall:.4f} ({100.0 * recall:.2f}%)  - TP/(TP+FN)")
        print(f"F1-Score:  {f1_score:.4f} ({100.0 * f1_score:.2f}%)")

        if unsupported_total > 0:
            print("\n" + "-" * 70)
            print("[Unsupported Operations - False Positive Test]")
            print("-" * 70)
            print(f"Total unsupported ops:        {unsupported_total}")
            print(f"Correctly rejected (CV=0):    {unsupported_cv_reject}  ({100.0 * unsupported_accuracy:.2f}%)")
            print(f"Incorrectly accepted (CV=1):  {unsupported_cv_accept}  "
                  f"({100.0 * (1 - unsupported_accuracy):.2f}%)")
            print("Expected rejection rate:      100% (app should not respond)")

        print("=" * 70)

    finally:
        stop_evt.set()
        if printer:
            printer.join(timeout=1.0)
        tim.stop_sim()
        if csv_fp:
            csv_fp.close()
        try:
            drv.quit()
        except Exception:
            pass

        tim.set_ops_total(rounds)
        snap = ver.snapshot()
        print("\n[final]\n" + ver.summary_str(), flush=True)

        counted_attempts = sum([
            snap["tap"]["total"],
            snap["drag"]["total"],
            snap["pinch"]["total"],
            snap["rotate"]["total"],
            snap["long_press"]["total"],
            snap["double_tap"]["total"]
        ])
        missing = rounds - counted_attempts

        print("\n[statistics]")
        print(f"  Script executed:     {rounds} operations")
        print(f"  Logcat recorded:     {counted_attempts} operations")
        print(f"  Difference:          {abs(missing)} operations")

        if missing < 0:
            print("\n  Note: Logcat recorded MORE operations because:")
            print("    - Prime tap (--prime_tap=1) adds extra tap events before each non-tap operation")
            print("    - Place operations trigger additional tap events")
            print("    - Some operations may trigger multiple app responses")
        elif missing > 0:
            print("\n  Note: Logcat recorded FEWER operations because:")
            print("    - Negative samples (operations outside AR objects) may not trigger app responses")
            print("    - Unsupported operations do not generate logcat events")
            print("    - Some operations may have failed silently")

        print(flush=True)
        print(tim.summary_str(), flush=True)
        ver.stop()


def main():
    ap = argparse.ArgumentParser(description="v2 Simple Frame-Diff Evaluation Script")
    ap.add_argument("--serial", help="ADB device serial")
    ap.add_argument("--pkg", default="com.google.ar.sceneform.samples.hellosceneform")
    ap.add_argument("--activity", default="auto")
    ap.add_argument("--rounds", type=int, default=100)
    ap.add_argument("--seed", type=int)

    ap.add_argument("--prime_tap", type=int, default=0,
                    help="1=先轻触再追加手势；0=直接做手势（默认0）")
    ap.add_argument("--prime_pause_ms", type=int, default=80,
                    help="轻触后可选暂停毫秒数，0表示不暂停（默认80）")

    ap.add_argument("--verify_wait_ms", type=int, default=200,
                    help="操作后等待时间（ms），确保AR响应完成")
    ap.add_argument("--diff_threshold", type=int, default=18,
                    help="帧差像素阈值")
    ap.add_argument("--min_change_ratio", type=float, default=0.02,
                    help="判定变化的最小像素比例")
    ap.add_argument("--ssim_threshold", type=float, default=0.92,
                    help="SSIM阈值（低于该值判定有变化）")
    ap.add_argument("--min_changed_pixels", type=int, default=0,
                    help="变化像素最小数量（<=0 表示不启用）")
    ap.add_argument("--diff_downsample_w", type=int, default=320,
                    help="帧差计算的下采样宽度")

    ap.add_argument("--log_csv", default=None, help="CSV 日志路径")

    ap.add_argument("--print-interval", type=int, default=10,
                    help="每隔 N 秒打印一次统计；0=仅结束时打印")

    ap.add_argument("--supported_ops", type=str,
                    default="tap,double_tap,drag,long_press,pinch_in,rotate",
                    help="逗号分隔的app支持的操作类型列表")
    ap.add_argument("--unsupported_ops", type=str,
                    default="triple_tap,swipe,two_finger_tap,flick",
                    help="逗号分隔的app不支持的操作类型列表（用于测试False Positive）")
    ap.add_argument("--negative_sample_ratio", type=float, default=0.5,
                    help="Negative sample比例（在AR物体外操作的比例，0.0-1.0）")

    args = ap.parse_args()

    supported_ops = [op.strip() for op in args.supported_ops.split(",") if op.strip()]
    unsupported_ops = [op.strip() for op in args.unsupported_ops.split(",") if op.strip()]

    run_evaluation(
        pkg=args.pkg,
        activity=args.activity,
        serial=args.serial,
        rounds=args.rounds,
        seed=args.seed,
        log_csv=args.log_csv,
        prime_tap=bool(args.prime_tap),
        prime_pause_ms=args.prime_pause_ms,
        verify_wait_ms=args.verify_wait_ms,
        diff_threshold=args.diff_threshold,
        min_change_ratio=args.min_change_ratio,
        ssim_threshold=args.ssim_threshold,
        min_changed_pixels=args.min_changed_pixels,
        diff_downsample_w=args.diff_downsample_w,
        print_interval=args.print_interval,
        supported_ops=supported_ops,
        unsupported_ops=unsupported_ops,
        negative_sample_ratio=args.negative_sample_ratio,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[Interrupted]")
        sys.exit(130)
