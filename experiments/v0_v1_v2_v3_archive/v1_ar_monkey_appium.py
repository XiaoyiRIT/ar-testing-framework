#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v1_ar_monkey_appium.py — Appium (CV targeted + act-then-verify)
+ MotionVerifier(日志 ground truth) + Timing
+ 新版 CV Verifier（src.verifier.motion_similarity 后端）

改动要点：
- 顶层不再用顶层 common/，改为 src.common.*
- 不再使用 cv.verify_motion.verify_action，
  改为统一通过 src.verifier.verifier.Verifier.verify()
- 仍然使用：
    * cv.strategy_riri_invariant.locate 作为 AR 目标检测策略
    * common.verify_motion.MotionVerifier 读取 Sceneform 日志作为 100% GT
- 通过 Verifier.verify() 打印 CV 检测结果（你在 verifier.py 中已添加 print）

用法示例（在项目根目录 program 下）：
  python v1_ar_monkey_appium.py \
      --pkg com.google.ar.sceneform.samples.hellosceneform \
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

from src.common.device import make_driver, get_window_size, resolve_main_activity, capture_bgr
from src.common.actions import tap, pinch_or_zoom, rotate, drag_line
from src.common.policy_random import step as random_step  # 若你备用随机策略仍需要
from cv.strategy_riri_invariant import locate as cv_locate

from src.common.verify_motion import MotionVerifier
from src.common.timing import Timing
from src.verifier.verifier import Verifier  # 新版 CV Verifier（motion_similarity 后端）


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


def _map_vec_window_to_img(dx_win: float, dy_win: float,
                           w_img: int, h_img: int,
                           W_win: int, H_win: int):
    """
    近似逆变换：已知 window 坐标系下的位移向量 (dx_win, dy_win)，
    估计在 screenshot (img) 坐标系下的位移向量 (dx_img, dy_img)。

    注意：只考虑缩放，不考虑偏移，对 drag 这种局部位移来说足够近似。
    """
    ar_img = w_img / float(h_img)
    ar_win = W_win / float(H_win)
    if abs(ar_img - ar_win) < 0.03:
        sx = W_win / float(w_img)
        sy = H_win / float(h_img)
        return dx_win / sx, dy_win / sy
    else:
        s = min(W_win / float(w_img), H_win / float(h_img))
        return dx_win / s, dy_win / s


# ----------------- 主流程 -----------------
def run_monkey_v1(pkg="com.rooom.app",
                  activity="auto",
                  serial=None,
                  rounds=250,
                  sleep_min=0.25, sleep_max=0.85,
                  safe=(0.12, 0.18, 0.88, 0.88),
                  drag_ms=(300, 900),
                  pinch_start=80, pinch_end=220,
                  rotate_radius=(160, 260), rotate_angle=(30, 90), rotate_steps=8,
                  warmup_wait=3.0, seed=None,
                  print_interval=10,
                  overall_include_tap=False,
                  # v1: AR 目标检测 & 验证参数
                  detect_min_area_ratio=0.0015,
                  detect_angle_thr=20.0,
                  detect_min_flow=0.08,
                  detect_cam_thr=0.01,
                  verify_wait_ms=500,
                  drag_min_px=40.0,
                  drag_dir_cos=0.6,
                  pinch_scale_thr=0.08,
                  rotate_min_deg=20.0,
                  verify_min_frac=0.20,
                  out_csv="v1_trials.csv",
                  ):
    """
    v1: CV 定位 + act-then-verify + 日志 GT
    - 使用 strategy_riri_invariant 做 AR object 定位
    - 使用 src.verifier.verifier.Verifier 做 CV 后验验证（motion_similarity 后端）
    - 使用 MotionVerifier 通过 logcat 提取 GT
    """

    if seed is not None:
        random.seed(seed)

    # 计时：程序总时长从此开始
    tim = Timing()

    # 清空旧日志，启动 MotionVerifier（从当前时刻），可带 serial（多设备时可用）
    clear_logcat(serial=serial)
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
    print(f"[v1-appium] {pkg}/{activity or 'auto'}, screen {W}x{H}, safe=({L},{T})~({R},{B})")

    # 模拟操作计时开始
    tim.start_sim()

    # ---- 新版 CV Verifier：基于 motion_similarity 后端 ----
    # 这里把 tau_move 固定为 0.03（即可视为“至少移动 3% 对角线长度”），
    # 其他阈值从命令行参数继承，方便你后续调参。
    verifier_cfg = {
        "thresholds": {
            "tau_move": 0.03,             # 对角线 3% 的最小位移阈值
            "tau_rot_deg": rotate_min_deg,
            "tau_scale": pinch_scale_thr,
            "min_frac": verify_min_frac,
        }
    }
    cv_verifier = Verifier.from_cfg(verifier_cfg)

    # 输出 CSV
    csv_f = None
    csv_w = None
    if out_csv:
        csv_f = open(out_csv, "w", newline="", encoding="utf-8")
        csv_w = csv.writer(csv_f)
        csv_w.writerow([
            "round", "op", "x_img", "y_img", "w_img", "h_img",
            "x_win", "y_win", "w_win", "h_win",
            "verified", "verified_hits", "detect_ms", "act_ms", "verify_ms"
        ])

    prev = None
    verified_hits = 0

    try:
        for i in range(1, rounds + 1):
            print(f"\n[round {i}/{rounds}]", flush=True)

            # 1) 截图 + CV 定位
            t0 = time.perf_counter()
            curr = capture_bgr(drv)
            detect_ms = _ms(t0)
            if curr is None:
                print("[error] capture_bgr returned None, skip round.", flush=True)
                time.sleep(0.5)
                continue

            h_img, w_img = curr.shape[:2]
            print(f"[detect] frame size = {w_img}x{h_img}, detect_min_area_ratio={detect_min_area_ratio}", flush=True)

            t0 = time.perf_counter()
            det = cv_locate(
                curr,
                prev_bgr=prev,
                min_area_ratio=detect_min_area_ratio,
                max_w=720,
                angle_thr_deg=detect_angle_thr,
                min_flow_mag=detect_min_flow,
                camera_move_thr=detect_cam_thr,
                debug_profile=True,
            )
            detect_ms += _ms(t0)

            if det is None:
                print("[detect] no AR target found, random tap safety area", flush=True)
                x = random.randint(L, R - 1)
                y = random.randint(T, B - 1)
                tap(drv, x, y)
                prev = curr
                time.sleep(random.uniform(sleep_min, sleep_max))
                continue

            (cx_img, cy_img) = det["center"]
            (x_img, y_img, w_box_img, h_box_img) = det["bbox"]
            print(f"[detect] center={det['center']}, bbox={det['bbox']}", flush=True)

            # 2) 选择操作类型
            op = random.choice(["drag", "pinch", "rotate", "tap"])
            # op 可以在这里做更多约束，例如根据 bbox 大小/位置来选

            # 3) 将 bbox 映射到 window 坐标，生成实际注入操作
            X, Y, Wb, Hb = _map_bbox_to_window(x_img, y_img, w_box_img, h_box_img, w_img, h_img, W, H)
            cx_win = X + Wb // 2
            cy_win = Y + Hb // 2

            print(f"[op] op={op}, center_win=({cx_win},{cy_win}), bbox_win=({X},{Y},{Wb},{Hb})", flush=True)

            # 将 drag 的 window 位移向量也转换到 image 坐标系，用于 CV 校验
            drag_dx_img = 0.0
            drag_dy_img = 0.0

            # 4) 执行动作
            pre_act = curr.copy()
            t_act0 = time.perf_counter()
            if op == "drag":
                # 选择一个简单的 drag 方向
                dx = random.randint(int(drag_min_px), int(drag_min_px * 1.8))
                dy = 0
                x2 = max(0, min(W - 1, cx_win + dx))
                y2 = max(0, min(H - 1, cy_win + dy))
                drag_line(drv, cx_win, cy_win, x2, y2,
                          ms=random.randint(drag_ms[0], drag_ms[1]))
                # 将 window 位移向量映射回 image 坐标系
                drag_dx_img, drag_dy_img = _map_vec_window_to_img(
                    x2 - cx_win, y2 - cy_win, w_img, h_img, W, H
                )
                act_name = "drag"

            elif op == "pinch":
                # 在 bbox 周围做 pinch_in
                pinch_or_zoom(drv,
                              cx_win, cy_win,
                              start_dist=pinch_start,
                              end_dist=pinch_end,
                              pinch_in=True,
                              ms=600)
                act_name = "pinch_in"

            elif op == "rotate":
                r = random.randint(rotate_radius[0], rotate_radius[1])
                ang = random.uniform(rotate_angle[0], rotate_angle[1])
                rotate(drv, cx_win, cy_win, radius=r, angle=ang, steps=rotate_steps)
                act_name = "rotate"

            else:
                tap(drv, cx_win, cy_win)
                act_name = "tap"

            act_ms = _ms(t_act0)

            # 5) CV 验证（基于 Verifier + motion_similarity）
            verified = 0
            verify_ms = 0.0

            if verify_wait_ms > 0:
                time.sleep(verify_wait_ms / 1000.0)

            t_ver0 = time.perf_counter()
            post_act = capture_bgr(drv)
            verify_ms = _ms(t_ver0)

            if post_act is not None and act_name in ("drag", "pinch_in", "rotate"):
                region = {
                    "center_xy": (float(cx_img), float(cy_img)),
                    "bbox": (float(x_img), float(y_img),
                             float(w_box_img), float(h_box_img)),
                }
                params = {}
                if act_name == "drag":
                    params["dx"] = float(drag_dx_img)
                    params["dy"] = float(drag_dy_img)
                elif act_name == "pinch_in":
                    # 对于模糊 pinch 操作，scale_sign < 0 表示 pinch_in
                    params["scale_sign"] = -1

                ok = cv_verifier.verify(
                    op=act_name,
                    pre_bgr=pre_act,
                    post_bgr=post_act,
                    region=region,
                    params=params,
                )
                verified = 1 if ok else 0
                if verified:
                    verified_hits += 1

            print(f"[verify] act={act_name}, verified={bool(verified)}, "
                  f"verified_hits={verified_hits}, detect_ms={detect_ms:.1f}, "
                  f"act_ms={act_ms:.1f}, verify_ms={verify_ms:.1f}", flush=True)

            # 6) 将 CV 结果喂给 MotionVerifier，以便和日志 GT 做对比
            #    kind 使用 act_name（drag / pinch_in / rotate / tap），
            #    MotionVerifier 内部会做归一化到 drag/pinch/rotate/tap。
            ver.feed(kind=act_name, ok=bool(verified))

            # 7) 写 CSV
            if csv_w is not None:
                csv_w.writerow([
                    i, act_name,
                    x_img, y_img, w_box_img, h_box_img,
                    X, Y, Wb, Hb,
                    verified, verified_hits,
                    f"{detect_ms:.2f}", f"{act_ms:.2f}", f"{verify_ms:.2f}"
                ])

            prev = post_act if post_act is not None else curr

            # 8) 休眠
            time.sleep(random.uniform(sleep_min, sleep_max))

    finally:
        # 停止定时打印
        stop_evt.set()
        if printer is not None:
            printer.join(timeout=1.0)

        # 模拟操作结束
        tim.stop_sim()
        tim.set_ops_total(rounds)

        # 打印最终统计
        print("\n[final]\n" + ver.summary_str(), flush=True)
        print("\n[compare]\n" + ver.summary_compare_str(), flush=True)
        print("\n[timing]\n" + tim.summary_str(), flush=True)

        if csv_f is not None:
            csv_f.close()

        try:
            drv.quit()
        except Exception:
            pass


def main(argv=None):
    p = argparse.ArgumentParser(description="v1 AR Monkey (Appium + CV + log GT)")
    p.add_argument("--pkg", type=str, required=True, help="App package name")
    p.add_argument("--activity", type=str, default="auto", help="Main activity (or 'auto')")
    p.add_argument("--serial", type=str, default=None, help="ADB serial (optional)")
    p.add_argument("--rounds", type=int, default=50, help="Number of operations")
    p.add_argument("--sleep-min", type=float, default=0.25)
    p.add_argument("--sleep-max", type=float, default=0.85)
    p.add_argument("--safe", type=float, nargs=4, default=(0.12, 0.18, 0.88, 0.88))
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--print-interval", type=int, default=10)
    p.add_argument("--overall-include-tap", action="store_true")

    # v1 CV detect & verify params
    p.add_argument("--detect-min-area-ratio", type=float, default=0.0015)
    p.add_argument("--detect-angle-thr", type=float, default=20.0)
    p.add_argument("--detect-min-flow", type=float, default=0.08)
    p.add_argument("--detect-cam-thr", type=float, default=0.01)
    p.add_argument("--verify-wait-ms", type=int, default=500)
    p.add_argument("--drag-min-px", type=float, default=40.0)
    p.add_argument("--drag-dir-cos", type=float, default=0.6)
    p.add_argument("--pinch-scale-thr", type=float, default=0.08)
    p.add_argument("--rotate-min-deg", type=float, default=20.0)
    p.add_argument("--verify-min-frac", type=float, default=0.20)
    p.add_argument("--out-csv", type=str, default="v1_trials.csv")

    args = p.parse_args(argv)

    run_monkey_v1(
        pkg=args.pkg,
        activity=args.activity,
        serial=args.serial,
        rounds=args.rounds,
        sleep_min=args.sleep_min,
        sleep_max=args.sleep_max,
        safe=tuple(args.safe),
        seed=args.seed,
        print_interval=args.print_interval,
        overall_include_tap=args.overall_include_tap,
        detect_min_area_ratio=args.detect_min_area_ratio,
        detect_angle_thr=args.detect_angle_thr,
        detect_min_flow=args.detect_min_flow,
        detect_cam_thr=args.detect_cam_thr,
        verify_wait_ms=args.verify_wait_ms,
        drag_min_px=args.drag_min_px,
        drag_dir_cos=args.drag_dir_cos,
        pinch_scale_thr=args.pinch_scale_thr,
        rotate_min_deg=args.rotate_min_deg,
        verify_min_frac=args.verify_min_frac,
        out_csv=args.out_csv,
    )


if __name__ == "__main__":
    sys.exit(main())
