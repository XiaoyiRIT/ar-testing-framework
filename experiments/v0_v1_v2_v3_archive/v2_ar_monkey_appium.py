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
    if activity in (None, "auto"):
        act = resolve_main_activity(pkg, serial)
        activity = act if act else None
        print(f"[resolve] {pkg} -> activity: {activity or 'Intent-Launcher'}")

    drv = make_driver(pkg=pkg, activity=activity, serial=serial, warmup_wait=warmup_wait)
    W, H = get_window_size(drv)
    print(f"[v2] {pkg}/{activity or 'auto'}, screen {W}x{H}")

    # 统计器：清空旧日志，从“当前时刻”开始；overall 基于 tap
    clear_logcat(serial)
    ver = MotionVerifier(overall_basis="tap")
    extra = ["-T", "0"]
    # 如需限定 tag：verify_motion 内部已按 AR_OP 解析；这里仍使用 -s 保持高效
    extra = ["-T", "0"]  # 已经在 Popen 命令里加了 -s TAG:D
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
            "step","detected","verified",
            "cx_img","cy_img","bbox_x","bbox_y","bbox_w","bbox_h",
            "message","cap_ms","cv_ms","action_ms","verify&wait_ms","total_ms"
        ])

    # 先抓 prev
    prev = capture_bgr(drv)

    # 模拟操作计时开始
    tim.start_sim()

    verified_hits = 0
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
            verified = 0

            if det is not None:
                # —— 图像空间（用于 verify）——
                cx_img, cy_img = det["center"]
                x_img, y_img, w_img_box, h_img_box = det["bbox"]

                # —— 窗口空间（用于手势注入与日志展示）——
                h_img, w_img = curr.shape[:2]
                cx, cy = _map_img_to_window(cx_img, cy_img, w_img, h_img, W, H)
                Xw, Yw, Ww, Hw = _map_bbox_to_window(x_img, y_img, w_img_box, h_img_box, w_img, h_img, W, H)
                bbox_win = (Xw, Yw, Ww, Hw)

                # CSV 写“截图坐标”，便于离线核对
                cx_out, cy_out, bx, by, bw, bh = cx_img, cy_img, x_img, y_img, w_img_box, h_img_box

                # 先轻触（可关）
                if prime_tap:
                    tap(drv, cx, cy, press_ms=random.randint(40, 120))
                    if prime_pause_ms > 0:
                        time.sleep(prime_pause_ms / 1000.0)

                # 在目标动作中随机挑一个
                op_name = random.choice(["pinch_in", "rotate", "drag"])
                act_name = op_name

                # 关键：pre_act 直接用 curr，减少一次抓屏
                pre_act = curr

                if op_name == "pinch_in":
                    s = max(40, min(w_img_box//2, h_img_box//2, 220))
                    e = max(20, min(s//2, 120))
                    pinch_or_zoom(drv, cx, cy, start_dist=s, end_dist=e,
                                  duration_ms=random.randint(450, 900))
                    msg = f"{'tap+' if prime_tap else ''}pinch at ({cx},{cy}) bbox={bbox_win}"

                elif op_name == "rotate":
                    radius = max(60, min(int(max(Ww, Hw)/2), 260))
                    angle = random.choice([45, 60, 90])
                    rotate(drv, cx, cy, radius=radius, angle_deg=angle,
                           duration_ms=random.randint(600, 1100), steps=rotate_steps)
                    msg = f"{'tap+' if prime_tap else ''}rotate at ({cx},{cy}) bbox={bbox_win}"

                else:  # drag
                    max_step = min(int(min(W, H) / 3), int(0.6 * max(Ww, Hw)))
                    dist = max(40, max_step)
                    dx, dy = random.choice([(dist, 0), (-dist, 0), (0, dist), (0, -dist)])
                    x2 = max(1, min(W - 2, cx + dx))
                    y2 = max(1, min(H - 2, cy + dy))
                    dur = random.randint(*drag_ms)
                    drag_line(drv, cx, cy, x2, y2, duration_ms=dur)
                    msg = f"{'tap+' if prime_tap else ''}drag from ({cx},{cy}) to ({x2},{y2}) bbox={bbox_win}"

            else:
                # MISS：未命中目标（这里不降级用随机手势；保持 v2 口径）
                act_name = "miss"
                msg = "MISS"

            t_action = _ms(T)

            # 4) （可选）动作后等待 + 验证（使用图像坐标）
            T = time.perf_counter()
            if enable_verify and det is not None:
                if verify_wait_ms > 0:
                    time.sleep(verify_wait_ms / 1000.0)
                post_act = capture_bgr(drv)
                ok = verify_action(
                    act_name,
                    pre_act,               # before (image)
                    post_act,              # after  (image)
                    (cx_img, cy_img),      # center in image coords
                    (x_img, y_img, w_img_box, h_img_box),  # bbox in image coords
                    {
                        "scale_thr": pinch_scale_thr,
                        "min_frac": verify_min_frac,
                        "min_deg": rotate_min_deg,
                        "min_motion_px": drag_min_px,
                        "min_dir_cos": drag_dir_cos,
                    }
                )
                verified = 1 if ok else 0
                if verified:
                    verified_hits += 1
            t_verify_wait = _ms(T)

            # 5) 打印阶段耗时（bbox 用窗口坐标展示）
            t_total = _ms(T_loop)
            print(
                f"[v2 r{i:03d}] cap={t_cap:.1f}ms  cv={t_cv:.1f}ms  action={t_action:.1f}ms  "
                f"verify&wait={t_verify_wait:.1f}ms  TOTAL={t_total:.1f}ms  "
                f"{'HIT' if det is not None else 'MISS'}:{act_name}"
                + (f"  bbox={bbox_win}" if det is not None else "")
            )

            # 6) 与旧版兼容的一行摘要
            print(f"[{verified_hits:03d}/{i:03d}/{rounds}] {msg}")

            # 7) CSV（记录图像坐标，避免与 verify 不一致）
            if csv_writer:
                csv_writer.writerow([
                    i, 1 if det is not None else 0, verified,
                    cx_out, cy_out, bx, by, bw, bh,
                    msg, f"{t_cap:.1f}", f"{t_cv:.1f}", f"{t_action:.1f}", f"{t_verify_wait:.1f}", f"{t_total:.1f}"
                ])

            # 8) 准备下一轮
            prev = curr

            # 9) 等待下轮
            time.sleep(random.uniform(sleep_min, sleep_max))

        print(f"[v2] verified hits: {verified_hits}/{rounds} ({(100.0*verified_hits/max(1, rounds)):.1f}%)")

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
        print("[final]\n" + ver.summary_str(), flush=True)

        # 以 basis=tap 口径统计到的“尝试数”（tap+drag+pinch+rotate）
        counted_attempts = (
            snap["tap"]["total"] +
            snap["drag"]["total"] +
            snap["pinch"]["total"] +
            snap["rotate"]["total"]
        )
        missing = rounds - counted_attempts
        print(f"[compare] counted_attempts={counted_attempts}  rounds={rounds}  missing={missing} "
              f"(likely long-press or no-op)", flush=True)

        print(tim.summary_str(), flush=True)

        # 结束采集
        ver.stop()


def main():
    ap = argparse.ArgumentParser(description="v2 Appium AR-Monkey (CV targeted + act-then-verify) + verify_motion(basis=tap) + timing")
    ap.add_argument("--serial", help="ADB device serial（可选；单设备可不填）")
    ap.add_argument("--pkg", default="com.rooom.app")
    ap.add_argument("--activity", default="auto")
    ap.add_argument("--rounds", type=int, default=250)
    ap.add_argument("--seed", type=int)

    # CV 策略参数
    ap.add_argument("--cv_min_area_ratio", type=float, default=0.002)
    ap.add_argument("--cv_downsample_w", type=int, default=None)


    # 先轻触（可选）
    ap.add_argument("--prime_tap", type=int, default=0,
                    help="1=先轻触再追加手势；0=直接做手势（默认1）")
    ap.add_argument("--prime_pause_ms", type=int, default=80,
                    help="轻触后可选暂停毫秒数，0表示不暂停（默认80）")

    # 验证调参（可选）
    ap.add_argument("--verify_wait_ms", type=int, default=140)
    ap.add_argument("--drag_min_px", type=float, default=8.0)
    ap.add_argument("--drag_dir_cos", type=float, default=0.6)
    ap.add_argument("--verify_min_frac", type=float, default=0.5)
    ap.add_argument("--pinch_scale_thr", type=float, default=0.10)
    ap.add_argument("--rotate_min_deg", type=float, default=15.0)

    # CSV
    ap.add_argument("--log_csv", default=None, help="可选的 per-step CSV 日志路径")

    # 打印
    ap.add_argument("--print-interval", type=int, default=10, help="每隔 N 秒打印一次统计；0=仅结束时打印")

    args = ap.parse_args()

    run_monkey_v2(
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
    )

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[Interrupted]")
        sys.exit(130)
