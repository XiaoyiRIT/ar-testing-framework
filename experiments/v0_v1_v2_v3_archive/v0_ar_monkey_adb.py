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

    # === 新增：启动 AR_OP 验证统计（后台 logcat 解析） ===
    ver = MotionVerifier()
    ver.start_from_adb(serial=serial)
    next_tick = time.time() + verify_print_sec if verify_print_sec and verify_print_sec > 0 else None

    try:
        for i in range(1, rounds + 1):
            op = random.choices(ops, weights=ws, k=1)[0]
            ev = {"i": i, "t": time.time(), "op": op}

            try:
                if op == "tap":
                    x, y = rand_xy(W, H, margin_ratio); tap(x, y, serial); ev.update({"x": x, "y": y})
                elif op == "double":
                    x, y = rand_xy(W, H, margin_ratio); tap(x, y, serial)
                    time.sleep(random.uniform(*doubletap_gap)); tap(x, y, serial)
                    ev.update({"x": x, "y": y})
                elif op == "long":
                    x, y = rand_xy(W, H, margin_ratio); hold = random.randint(*long_ms)
                    long_press(x, y, hold, serial); ev.update({"x": x, "y": y, "hold_ms": hold})
                elif op == "swipe":
                    x1, y1 = rand_xy(W, H, margin_ratio)
                    if random.random() < 0.5:
                        x2, y2 = rand_xy(W, H, margin_ratio)
                    else:
                        dx = random.randint(-W, W); dy = random.randint(-H, H)
                        x2 = min(max(1, x1 + dx), W-1); y2 = min(max(1, y1 + dy), H-1)
                    dur = random.randint(*swipe_dur_ms); swipe(x1, y1, x2, y2, dur, serial)
                    ev.update({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "dur_ms": dur})
                else:
                    code = key_pool[op]; keyevent(code, serial); ev.update({"keycode": code})
            except Exception as e:
                ev.update({"error": f"{type(e).__name__}:{e}"})

            pkg, act = top_app(serial)
            ev.update({"top": f"{pkg}/{act}" if pkg else None})
            log(ev)
            time.sleep(random.uniform(sleep_min, sleep_max))

            # === 新增：定时仅打印 overall 成功率（v0基线口径） ===
            if next_tick and time.time() >= next_tick:
                _print_overall(ver, prefix="[tick]  ")
                next_tick = time.time() + verify_print_sec

    finally:
        # === 新增：结束前打印最终 overall，并停止采集线程 ===
        _print_overall(ver, prefix="[final] ")
        ver.stop()
        if logf: logf.close()
        print("[baseline] done.")

def main():
    ap = argparse.ArgumentParser(description="Pure ADB Monkey Baseline (zero preset)")
    ap.add_argument("--serial", help="adb -s SERIAL；留空将自动检测唯一设备")
    ap.add_argument("--rounds", type=int, default=500)
    ap.add_argument("--sleep-min", type=float, default=0.10)
    ap.add_argument("--sleep-max", type=float, default=0.60)
    ap.add_argument("--margin-ratio", type=float, default=0.0)
    ap.add_argument("--swipe-dur", type=int, nargs=2, default=[150, 800], metavar=("MIN", "MAX"))
    ap.add_argument("--long-ms", type=int, nargs=2, default=[600, 1500], metavar=("MIN", "MAX"))
    ap.add_argument("--double-gap", type=float, nargs=2, default=[0.05, 0.18], metavar=("MIN", "MAX"))
    ap.add_argument("--seed", type=int)
    ap.add_argument("--out", help="事件日志 JSONL 路径")
    # === 新增：验证统计定时打印（秒） ===
    ap.add_argument("--verify-print-sec", type=float, default=10.0, help="每隔 N 秒打印一次 overall 成功率；0 表示仅结束打印")
    args = ap.parse_args()

    serial = auto_pick_serial(args.serial)
    run_monkey(
        serial=serial,
        rounds=args.rounds,
        sleep_min=args.sleep_min,
        sleep_max=args.sleep_max,
        margin_ratio=args.margin_ratio,
        swipe_dur_ms=tuple(args.swipe_dur),
        long_ms=tuple(args.long_ms),
        doubletap_gap=tuple(args.double_gap),
        seed=args.seed,
        out_jsonl=args.out,
        verify_print_sec=args.verify_print_sec,   # 新增参数传递
    )

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[中断] 手动退出")
