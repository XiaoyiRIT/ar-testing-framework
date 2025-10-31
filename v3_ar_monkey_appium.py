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
        text = out["choices"][0]["message"]["content"] or ""
        last = text
        jtxt = extract_first_json(text)
        if jtxt:
            try:
                det = json.loads(jtxt)
                def c01(v):
                    try: return max(0.0, min(1.0, float(v)))
                    except: return 0.5
                cx = c01(det.get("coords", {}).get("cx", 0.5))
                cy = c01(det.get("coords", {}).get("cy", 0.5))
                bx = c01(det.get("bbox", {}).get("x", 0.0))
                by = c01(det.get("bbox", {}).get("y", 0.0))
                bw = c01(det.get("bbox", {}).get("w", 1.0))
                bh = c01(det.get("bbox", {}).get("h", 1.0))
                conf = float(det.get("confidence", 0.0))
                return {
                    "name": det.get("name", "object"),
                    "coords": {"cx": cx, "cy": cy},
                    "bbox": {"x": bx, "y": by, "w": bw, "h": bh},
                    "confidence": conf
                }
            except Exception:
                messages[-1]["content"][-1]["text"] += "\n上次JSON语法有误，请修正后仅输出JSON。<json>"
        else:
            messages[-1]["content"][-1]["text"] += "\n请仅在<json>中输出JSON。<json>"
    raise RuntimeError(f"LLM输出无法解析JSON，原始片段: {last[:200]}")

def norm_to_px(cx: float, cy: float, W: int, H: int) -> Tuple[int, int]:
    x = max(0, min(W - 1, int(round(cx * W))))
    y = max(0, min(H - 1, int(round(cy * H))))
    return x, y

# ---------- 单轮逻辑 ----------
def run_round(driver, W, H, llm, round_idx: int, csv_writer=None, args=None, last_smart_ts=[0]):
    t0 = time.time()

    # 1) 截图 & VLM 定位
    bgr = capture_bgr(driver)
    data_uri = bgr_to_data_uri(bgr, encode=("jpeg" if args.jpeg else "png"))
    messages = build_messages(data_uri, screen_w=W, screen_h=H)
    det = ask_and_parse(llm, messages, retries=2)

    # （可选）精准 tap（默认关闭；仅在 --precise_tap 开启时执行）
    cx = float(det["coords"]["cx"]); cy = float(det["coords"]["cy"])
    x, y = norm_to_px(cx, cy, W, H)
    if args and args.precise_tap:
        tap(driver, x, y, press_ms=90)

    # 纠偏冷却与阈值配置
    zoom_cfg = {"min_area": args.min_area, "max_area": args.max_area, "zoom_step": 0.22}
    edge_cfg = {"edge_tol": args.edge_tol, "drag_frac": args.drag_frac, "duration_ms": 600}
    now = time.time() * 1000.0
    smart_desc = ""

    # 2) SMART：pre/replace/model/off
    if args.smart_mode != "off":
        do_smart = (now - last_smart_ts[0] >= (args.smart_cooldown_ms or 0))
        if do_smart:
            if args.smart_mode in ("pre","replace"):
                smart_desc = smart_correct_rule_based(driver, W, H, det, zoom_cfg=zoom_cfg, edge_cfg=edge_cfg)
            elif args.smart_mode == "model":
                smart_desc = smart_correct_model_driver(driver, W, H, det)
            if smart_desc:
                print(f"[SMART] {smart_desc}")
                last_smart_ts[0] = now

    # 3) 基于目标坐标的“随机”单步动作（drag/pinch/zoom/rotate 四选一）
    policy_desc = ""
    if args.smart_mode != "replace":
        op = random.choice(["drag", "pinch", "zoom", "rotate"])

        if op == "drag":
            r_frac = random.uniform(0.15, 0.30)
            radius = int(min(W, H) * r_frac)
            theta = random.uniform(0.0, 2*np.pi)
            tx = max(0, min(W - 1, x + int(radius * np.cos(theta))))
            ty = max(0, min(H - 1, y + int(radius * np.sin(theta))))
            drag_line(driver, x, y, tx, ty, duration_ms=600)
            policy_desc = f"drag ({x},{y})->({tx},{ty}) dur=600ms"

        elif op == "pinch":
            base = int(max(60, 0.12 * min(W, H)))
            start = int(base * random.uniform(2.0, 2.8))
            end   = int(base * random.uniform(0.5, 0.8))   # pinch
            pinch_or_zoom(driver, x, y, start_dist=start, end_dist=end, duration_ms=700)
            policy_desc = f"pinch center=({x},{y}) {start}->{end} dur=700ms"

        elif op == "zoom":
            base = int(max(60, 0.12 * min(W, H)))
            start = int(base * random.uniform(0.6, 1.2))
            end   = int(base * random.uniform(1.8, 3.0))   # zoom
            pinch_or_zoom(driver, x, y, start_dist=start, end_dist=end, duration_ms=700)
            policy_desc = f"zoom center=({x},{y}) {start}->{end} dur=700ms"

        else:  # rotate
            base_r = int(max(80, 0.16 * min(W, H)))
            angle  = int(random.uniform(20, 45))
            rotate(driver, x, y, radius=int(0.16*min(W,H)), angle_deg=30, duration_ms=700, steps=12)
            policy_desc = f"rotate center=({x},{y}) r={base_r} angle={angle} dur=700ms"

    dt = time.time() - t0
    rec = {
        "round": round_idx,
        "name": det.get("name",""),
        "confidence": det.get("confidence",""),
        "cx": cx, "cy": cy, "px": x, "py": y,
        "bbox_x": det.get("bbox", {}).get("x", ""),
        "bbox_y": det.get("bbox", {}).get("y", ""),
        "bbox_w": det.get("bbox", {}).get("w", ""),
        "bbox_h": det.get("bbox", {}).get("h", ""),
        "smart_mode": args.smart_mode if args else "",
        "smart_desc": smart_desc,
        "policy_desc": policy_desc,
        "elapsed_s": round(dt, 3),
    }
    print(f"[ROUND {round_idx}] name={rec['name']} conf={rec['confidence']} "
          f"smart='{smart_desc}' policy='{policy_desc}' elapsed={rec['elapsed_s']}s")

    if csv_writer is not None:
        csv_writer.writerow(rec)

    return rec

# ---------- 主函数 ----------
def main():
    ap = argparse.ArgumentParser(description="v3：VLM定位 + SMART纠偏 + 目标导向动作 + verify_motion(basis=tap) + timing")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--mmproj", default=DEFAULT_MMPROJ)
    ap.add_argument("--pkg", default=DEFAULT_PKG)
    ap.add_argument("--activity", default="auto")
    ap.add_argument("--max_side", type=int, default=DEFAULT_MAXSIDE)
    ap.add_argument("--jpeg", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--rounds", type=int, default=100)
    ap.add_argument("--sleep-ms", type=int, default=300)
    ap.add_argument("--log-csv", type=str, default="")
    ap.add_argument("--smart-mode", choices=["pre","replace","off","model"], default="pre")
    ap.add_argument("--smart-cooldown-ms", type=int, default=1200)
    ap.add_argument("--precise_tap", action="store_true")
    ap.add_argument("--min-area", type=float, default=0.05)
    ap.add_argument("--max-area", type=float, default=0.35)
    ap.add_argument("--edge-tol", type=float, default=0.08)
    ap.add_argument("--drag-frac", type=float, default=0.20)
    # 新增：定时打印
    ap.add_argument("--print-interval", type=int, default=10, help="每隔 N 秒打印一次统计；0=仅结束时打印")

    args = ap.parse_args()
    os.environ.setdefault("GGML_METAL_USE_GRAPH", "0")

    # === 计时启动 ===
    tim = Timing()

    # === 设备与模型 ===
    driver = make_driver(pkg=args.pkg, activity=args.activity, serial=None, warmup_wait=2.0)
    W, H = get_window_size(driver)

    chat_handler = Qwen25VLChatHandler(clip_model_path=args.mmproj)
    llm = Llama(
        model_path=args.model,
        chat_handler=chat_handler,
        n_ctx=8192,
        n_threads=8,
        n_gpu_layers=0 if args.cpu else -1,
        verbose=False,
    )

    csv_writer = None
    csv_file = None
    if args.log_csv:
        fieldnames = ["round","name","confidence","cx","cy","px","py",
                      "bbox_x","bbox_y","bbox_w","bbox_h",
                      "smart_mode","smart_desc","policy_desc","elapsed_s"]
        csv_file = open(args.log_csv, "w", newline="", encoding="utf-8")
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        csv_writer.writeheader()

    # === 统计器：清空旧日志，从“当前时刻”开始；overall 基于 tap ===
    clear_logcat()
    ver = MotionVerifier(overall_basis="tap")
    ver.start_from_adb(extra_logcat_args=["-T", "0"])

    # 定时打印线程
    stop_evt = threading.Event()
    printer = None
    if args.print_interval and args.print_interval > 0:
        printer = threading.Thread(target=periodic_print, args=(ver, args.print_interval, stop_evt), daemon=True)
        printer.start()

    # === 模拟操作计时开始 ===
    tim.start_sim()

    try:
        for r in range(1, args.rounds + 1):
            try:
                run_round(driver, W, H, llm, r, csv_writer=csv_writer, args=args)
            except Exception as e:
                print(f"[ROUND {r}] ERROR: {e}")
            if args.sleep_ms > 0 and r < args.rounds:
                time.sleep(args.sleep_ms / 1000.0)
    finally:
        # 结束打印/回收
        stop_evt.set()
        if printer:
            printer.join(timeout=1.0)
        tim.stop_sim()
        try:
            if csv_file: csv_file.close()
        except Exception:
            pass
        try:
            driver.quit()
        except Exception:
            pass

        # 统一口径：发起的操作数 = rounds
        tim.set_ops_total(args.rounds)

        # 完整统计输出
        snap = ver.snapshot()
        print("[final]\n" + ver.summary_str(), flush=True)

        # 以 basis=tap 口径统计到的“尝试数”（tap+drag+pinch+rotate）
        counted_attempts = (
            snap["tap"]["total"] +
            snap["drag"]["total"] +
            snap["pinch"]["total"] +
            snap["rotate"]["total"]
        )
        missing = args.rounds - counted_attempts
        print(f"[compare] counted_attempts={counted_attempts}  rounds={args.rounds}  missing={missing} "
              f"(若缺口>0，通常是 long-press 或未被 AR_OP 记录的动作)", flush=True)

        print(tim.summary_str(), flush=True)

        # 停止采集
        ver.stop()

if __name__ == "__main__":
    main()
