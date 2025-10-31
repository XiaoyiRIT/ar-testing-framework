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

    run_monkey(pkg=args.pkg,
               activity=args.activity,
               serial=args.serial,
               rounds=args.rounds,
               seed=args.seed,
               print_interval=args.print_interval,
               overall_include_tap=args.overall_include_tap)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[Interrupted]")
        sys.exit(130)
