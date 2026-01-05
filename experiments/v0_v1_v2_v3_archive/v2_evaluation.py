#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v2_evaluation.py — CV Algorithm Evaluation Script

基于 v2_ar_monkey_appium.py，增加以下功能：
1. 添加 long_press 和 double_tap 操作
2. 每次操作后检测 Ground Truth（从 logcat 获取真实操作结果）
3. 对比 CV 验证结果 vs Ground Truth
4. 计算准确率指标

Ground Truth 定义：
- True Positive (TP): CV说有运动 AND logcat确实记录了对应操作
- True Negative (TN): CV说没运动 AND logcat也没记录
- False Positive (FP): CV说有运动 BUT logcat没记录（app不支持或失败）
- False Negative (FN): CV说没运动 BUT logcat有记录（CV漏检）

准确率 = (TP + TN) / (TP + TN + FP + FN)
"""

import argparse
import random
import sys
import time
import os
import csv
import threading
import subprocess
import json

# 添加项目根目录和 src 目录到 Python 路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from src.common.device import make_driver, get_window_size, resolve_main_activity, capture_bgr
from src.common.actions import tap, pinch_or_zoom, rotate, drag_line, long_press, double_tap
from cv.strategy_yolo import locate as cv_locate
from src.verifier.backends.motion_similarity import verify_action

from src.common.verify_motion import MotionVerifier
from src.common.timing import Timing

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

# ----------------- Ground Truth Detection -----------------
def get_recent_logcat_lines(serial: str = None, num_lines: int = 20):
    """获取最近的 logcat 行"""
    cmd = ["adb"]
    if serial:
        cmd += ["-s", serial]
    cmd += ["logcat", "-d", "-s", "AR_OP:D"]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=2)
        lines = result.stdout.strip().split('\n')
        return lines[-num_lines:] if len(lines) > num_lines else lines
    except Exception:
        return []

def check_ground_truth(serial: str, op_name: str, time_window_sec: float = 2.0):
    """
    检查最近的 logcat，看是否有对应操作的成功记录

    Args:
        serial: 设备序列号
        op_name: 操作名称 (pinch_in, drag, rotate, long_press, double_tap)
        time_window_sec: 时间窗口（秒）

    Returns:
        bool: True 表示 logcat 中有对应操作的成功记录
    """
    # 操作名映射：脚本操作名 -> logcat kind 名
    op_map = {
        "pinch_in": "pinch",
        "pinch_out": "pinch",
        "zoom_in": "pinch",
        "zoom_out": "pinch",
        "drag": "drag",
        "rotate": "rotate",
        "long_press": "long_press_end",  # 完整的长按结束
        "double_tap": "double_tap",
        "tap": "tap",
    }

    target_kind = op_map.get(op_name, op_name)

    # 获取最近的 logcat 行
    recent_lines = get_recent_logcat_lines(serial, num_lines=15)

    # 从后往前检查（最新的在最后）
    for line in reversed(recent_lines):
        if "AR_OP" in line and "{" in line:
            try:
                json_start = line.index("{")
                json_str = line[json_start:]
                data = json.loads(json_str)

                kind = data.get("kind", "").strip().lower()
                ok = data.get("ok", False)

                # 找到对应操作且成功
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
    # CV 参数
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
    # 操作类型
    operations=None,
):
    """
    评估版主循环：执行操作 → CV验证 → GT检测 → 对比 → 统计准确率
    """
    if seed is not None:
        random.seed(seed)

    # 默认操作类型
    if operations is None:
        operations = ["drag", "rotate", "pinch_in", "long_press", "double_tap"]

    # 计时从此刻开始
    tim = Timing()

    # 解析 Activity 并启动驱动
    if activity in (None, "auto"):
        act = resolve_main_activity(pkg, serial)
        activity = act if act else None
        print(f"[resolve] {pkg} -> activity: {activity or 'Intent-Launcher'}")

    drv = make_driver(pkg=pkg, activity=activity, serial=serial, warmup_wait=warmup_wait)
    W, H = get_window_size(drv)
    print(f"[v2_eval] {pkg}/{activity or 'auto'}, screen {W}x{H}")
    print(f"[v2_eval] Operations: {operations}")

    # 统计器：清空旧日志，从"当前时刻"开始；overall 基于 tap
    clear_logcat(serial)
    ver = MotionVerifier(overall_basis="tap")
    extra = ["-T", "0"]
    ver.start_from_adb(serial=serial, extra_logcat_args=extra)

    # 定时打印线程
    stop_evt = threading.Event()
    printer = None
    if print_interval and print_interval > 0:
        printer = threading.Thread(target=periodic_print, args=(ver, print_interval, stop_evt), daemon=True)
        printer.start()

    # CSV
    csv_fp = csv_writer = None
    if log_csv:
        d = os.path.dirname(log_csv)
        if d:
            os.makedirs(d, exist_ok=True)
        csv_fp = open(log_csv, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_fp)
        csv_writer.writerow([
            "step", "detected", "cv_verified", "gt_verified", "cv_correct",
            "operation", "cx_img", "cy_img", "bbox_x", "bbox_y", "bbox_w", "bbox_h",
            "message", "cap_ms", "cv_ms", "action_ms", "verify&wait_ms", "total_ms"
        ])

    # 先抓 prev
    prev = capture_bgr(drv)

    # 模拟操作计时开始
    tim.start_sim()

    # 评估统计
    cv_verified_count = 0
    gt_verified_count = 0
    tp_count = 0  # True Positive: CV=1, GT=1
    tn_count = 0  # True Negative: CV=0, GT=0
    fp_count = 0  # False Positive: CV=1, GT=0
    fn_count = 0  # False Negative: CV=0, GT=1

    try:
        for i in range(1, rounds + 1):
            T_loop = time.perf_counter()

            # 1) 抓屏
            T = time.perf_counter()
            curr = capture_bgr(drv)
            t_cap = _ms(T)

            # 2) CV 定位
            T = time.perf_counter()
            det = cv_locate(
                image_bgr=curr,
                debug_profile=True
            )
            t_cv = _ms(T)

            # 3) 决策 & 注入
            T = time.perf_counter()
            act_name = None
            msg = ""
            bbox_win = None
            cx_out = cy_out = bx = by = bw = bh = ""
            detected = 1 if det is not None else 0
            cv_verified = 0
            gt_verified = 0
            cv_correct = 0

            if det is not None:
                # 图像空间（用于 verify）
                cx_img, cy_img = det["center"]
                x_img, y_img, w_img_box, h_img_box = det["bbox"]

                # 窗口空间（用于手势注入与日志展示）
                h_img, w_img = curr.shape[:2]
                cx, cy = _map_img_to_window(cx_img, cy_img, w_img, h_img, W, H)
                Xw, Yw, Ww, Hw = _map_bbox_to_window(x_img, y_img, w_img_box, h_img_box, w_img, h_img, W, H)
                bbox_win = (Xw, Yw, Ww, Hw)

                # CSV 写"截图坐标"，便于离线核对
                cx_out, cy_out, bx, by, bw, bh = cx_img, cy_img, x_img, y_img, w_img_box, h_img_box

                # 先轻触（可关）
                if prime_tap:
                    tap(drv, cx, cy, press_ms=random.randint(40, 120))
                    if prime_pause_ms > 0:
                        time.sleep(prime_pause_ms / 1000.0)

                # 在目标动作中随机挑一个
                op_name = random.choice(operations)
                act_name = op_name

                # 关键：pre_act 直接用 curr，减少一次抓屏
                pre_act = curr

                if op_name == "pinch_in":
                    s = max(40, min(w_img_box//2, h_img_box//2, 220))
                    e = max(20, min(s//2, 120))
                    pinch_or_zoom(drv, cx, cy, start_dist=s, end_dist=e,
                                  duration_ms=random.randint(450, 900))
                    msg = f"{'tap+' if prime_tap else ''}pinch_in at ({cx},{cy}) bbox={bbox_win}"

                elif op_name == "rotate":
                    radius = max(60, min(int(max(Ww, Hw)/2), 260))
                    angle = random.choice([45, 60, 90])
                    rotate(drv, cx, cy, radius=radius, angle_deg=angle,
                           duration_ms=random.randint(600, 1100), steps=rotate_steps)
                    msg = f"{'tap+' if prime_tap else ''}rotate at ({cx},{cy}) bbox={bbox_win}"

                elif op_name == "drag":
                    max_step = min(int(min(W, H) / 3), int(0.6 * max(Ww, Hw)))
                    dist = max(40, max_step)
                    dx, dy = random.choice([(dist, 0), (-dist, 0), (0, dist), (0, -dist)])
                    x2 = max(1, min(W - 2, cx + dx))
                    y2 = max(1, min(H - 2, cy + dy))
                    dur = random.randint(*drag_ms)
                    drag_line(drv, cx, cy, x2, y2, duration_ms=dur)
                    msg = f"{'tap+' if prime_tap else ''}drag from ({cx},{cy}) to ({x2},{y2}) bbox={bbox_win}"

                elif op_name == "long_press":
                    hold_ms = random.randint(800, 1200)
                    long_press(drv, cx, cy, hold_ms=hold_ms)
                    msg = f"{'tap+' if prime_tap else ''}long_press at ({cx},{cy}) hold={hold_ms}ms bbox={bbox_win}"

                elif op_name == "double_tap":
                    interval_ms = random.randint(80, 150)
                    double_tap(drv, cx, cy, tap_interval_ms=interval_ms)
                    msg = f"{'tap+' if prime_tap else ''}double_tap at ({cx},{cy}) interval={interval_ms}ms bbox={bbox_win}"

            else:
                # MISS：未命中目标
                act_name = "miss"
                msg = "MISS"

            t_action = _ms(T)

            # 4) （可选）动作后等待 + CV验证
            T = time.perf_counter()
            if enable_verify and det is not None:
                if verify_wait_ms > 0:
                    time.sleep(verify_wait_ms / 1000.0)
                post_act = capture_bgr(drv)

                # CV 验证
                ok = verify_action(
                    act_name,
                    pre_act,
                    post_act,
                    (cx_img, cy_img),
                    (x_img, y_img, w_img_box, h_img_box),
                    {
                        "scale_thr": pinch_scale_thr,
                        "min_frac": verify_min_frac,
                        "min_deg": rotate_min_deg,
                        "min_motion_px": drag_min_px,
                        "min_dir_cos": drag_dir_cos,
                    }
                )
                cv_verified = 1 if ok else 0
                if cv_verified:
                    cv_verified_count += 1

                # Ground Truth 检测
                # 等待一小段时间确保 logcat 已输出
                time.sleep(0.1)
                gt_ok = check_ground_truth(serial, act_name, time_window_sec=2.0)
                gt_verified = 1 if gt_ok else 0
                if gt_verified:
                    gt_verified_count += 1

                # 计算 TP/TN/FP/FN
                if cv_verified == 1 and gt_verified == 1:
                    tp_count += 1
                    cv_correct = 1
                elif cv_verified == 0 and gt_verified == 0:
                    tn_count += 1
                    cv_correct = 1
                elif cv_verified == 1 and gt_verified == 0:
                    fp_count += 1
                    cv_correct = 0
                elif cv_verified == 0 and gt_verified == 1:
                    fn_count += 1
                    cv_correct = 0

            t_verify_wait = _ms(T)

            # 5) 打印阶段耗时
            t_total = _ms(T_loop)
            status = "MISS" if det is None else f"CV={cv_verified} GT={gt_verified} {'✓' if cv_correct else '✗'}"
            print(
                f"[v2_eval r{i:03d}] cap={t_cap:.1f}ms  cv={t_cv:.1f}ms  action={t_action:.1f}ms  "
                f"verify&wait={t_verify_wait:.1f}ms  TOTAL={t_total:.1f}ms  "
                f"{status}:{act_name}"
                + (f"  bbox={bbox_win}" if det is not None else "")
            )

            # 6) 一行摘要
            print(f"[{i:03d}/{rounds}] {msg}")

            # 7) CSV
            if csv_writer:
                csv_writer.writerow([
                    i, detected, cv_verified, gt_verified, cv_correct,
                    act_name, cx_out, cy_out, bx, by, bw, bh,
                    msg, f"{t_cap:.1f}", f"{t_cv:.1f}", f"{t_action:.1f}",
                    f"{t_verify_wait:.1f}", f"{t_total:.1f}"
                ])

            # 8) 准备下一轮
            prev = curr

            # 9) 等待下轮
            time.sleep(random.uniform(sleep_min, sleep_max))

        # 计算评估指标
        total_ops = tp_count + tn_count + fp_count + fn_count
        accuracy = (tp_count + tn_count) / max(1, total_ops)
        precision = tp_count / max(1, tp_count + fp_count)
        recall = tp_count / max(1, tp_count + fn_count)
        f1_score = 2 * precision * recall / max(0.001, precision + recall)

        print("\n" + "="*60)
        print("[EVALUATION RESULTS]")
        print("="*60)
        print(f"Total operations: {total_ops}")
        print(f"CV verified: {cv_verified_count}/{rounds} ({100.0*cv_verified_count/max(1,rounds):.1f}%)")
        print(f"GT verified: {gt_verified_count}/{rounds} ({100.0*gt_verified_count/max(1,rounds):.1f}%)")
        print("-" * 60)
        print(f"True Positive (TP):   {tp_count:4d}  (CV=1, GT=1) ✓")
        print(f"True Negative (TN):   {tn_count:4d}  (CV=0, GT=0) ✓")
        print(f"False Positive (FP):  {fp_count:4d}  (CV=1, GT=0) ✗ CV误判")
        print(f"False Negative (FN):  {fn_count:4d}  (CV=0, GT=1) ✗ CV漏检")
        print("-" * 60)
        print(f"Accuracy:  {accuracy:.4f} ({100.0*accuracy:.2f}%)")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1_score:.4f}")
        print("="*60)

    finally:
        # 停止定时打印线程
        stop_evt.set()
        if printer:
            printer.join(timeout=1.0)
        # 模拟时长停止
        tim.stop_sim()
        # CSV 关闭
        if csv_fp:
            csv_fp.close()
        # 关闭 driver
        try:
            drv.quit()
        except Exception:
            pass

        # 平均每操作耗时：用 rounds 作为发起的手势数
        tim.set_ops_total(rounds)

        # 打印统计
        snap = ver.snapshot()
        print("\n[final]\n" + ver.summary_str(), flush=True)

        # 以 basis=tap 口径统计到的"尝试数"
        counted_attempts = sum([
            snap["tap"]["total"],
            snap["drag"]["total"],
            snap["pinch"]["total"],
            snap["rotate"]["total"],
            snap["long_press"]["total"],
            snap["double_tap"]["total"]
        ])
        missing = rounds - counted_attempts
        print(f"[compare] counted_attempts={counted_attempts}  rounds={rounds}  missing={missing}", flush=True)

        print(tim.summary_str(), flush=True)

        # 结束采集
        ver.stop()


def main():
    ap = argparse.ArgumentParser(description="v2 CV Algorithm Evaluation Script")
    ap.add_argument("--serial", help="ADB device serial")
    ap.add_argument("--pkg", default="com.google.ar.sceneform.samples.hellosceneform")
    ap.add_argument("--activity", default="auto")
    ap.add_argument("--rounds", type=int, default=100)
    ap.add_argument("--seed", type=int)

    # CV 策略参数
    ap.add_argument("--cv_min_area_ratio", type=float, default=0.002)
    ap.add_argument("--cv_downsample_w", type=int, default=None)

    # 先轻触（可选）
    ap.add_argument("--prime_tap", type=int, default=1,
                    help="1=先轻触再追加手势；0=直接做手势（默认1）")
    ap.add_argument("--prime_pause_ms", type=int, default=80,
                    help="轻触后可选暂停毫秒数，0表示不暂停（默认80）")

    # 验证调参
    ap.add_argument("--verify_wait_ms", type=int, default=200,
                    help="操作后等待时间（ms），确保AR响应完成")
    ap.add_argument("--drag_min_px", type=float, default=8.0)
    ap.add_argument("--drag_dir_cos", type=float, default=0.6)
    ap.add_argument("--verify_min_frac", type=float, default=0.5)
    ap.add_argument("--pinch_scale_thr", type=float, default=0.10)
    ap.add_argument("--rotate_min_deg", type=float, default=15.0)

    # CSV
    ap.add_argument("--log_csv", default=None, help="CSV 日志路径")

    # 打印
    ap.add_argument("--print-interval", type=int, default=10,
                    help="每隔 N 秒打印一次统计；0=仅结束时打印")

    # 操作类型
    ap.add_argument("--operations", type=str, default="drag,rotate,pinch_in,long_press,double_tap",
                    help="逗号分隔的操作类型列表")

    args = ap.parse_args()

    # 解析操作列表
    ops = [op.strip() for op in args.operations.split(",")]

    run_evaluation(
        pkg=args.pkg,
        activity=args.activity,
        serial=args.serial,
        rounds=args.rounds,
        seed=args.seed,
        cv_min_area_ratio=args.cv_min_area_ratio,
        cv_downsample_w=args.cv_downsample_w,
        log_csv=args.log_csv,
        prime_tap=bool(args.prime_tap),
        prime_pause_ms=args.prime_pause_ms,
        verify_wait_ms=args.verify_wait_ms,
        drag_min_px=args.drag_min_px,
        drag_dir_cos=args.drag_dir_cos,
        verify_min_frac=args.verify_min_frac,
        pinch_scale_thr=args.pinch_scale_thr,
        rotate_min_deg=args.rotate_min_deg,
        print_interval=args.print_interval,
        operations=ops,
    )

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[Interrupted]")
        sys.exit(130)
