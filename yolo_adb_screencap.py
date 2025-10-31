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

            # === Selection per user requirement ===
            sel_idx = None
            # 1) try classification top1 (if probs exists, typical for classification tasks)
            try:
                if hasattr(res, "probs") and res.probs is not None and getattr(res.probs, "top1", None) is not None:
                    sel_idx = int(res.probs.top1)
            except Exception:
                sel_idx = None

            # 2) fall back to detection: highest-confidence box
            try:
                boxes = res.boxes
                if sel_idx is None and boxes is not None and boxes.conf is not None and len(boxes) > 0:
                    conf_arr = boxes.conf.detach().cpu().numpy()
                    sel_idx = int(conf_arr.argmax())
            except Exception:
                sel_idx = None

            # 3) compute center from boxes.xyxy[sel_idx]
            center_str = "n/a"
            if sel_idx is not None and hasattr(res, "boxes") and res.boxes is not None and len(res.boxes) > sel_idx:
                xyxy_sel = res.boxes.xyxy[sel_idx].detach().cpu().numpy().tolist()
                x1, y1, x2, y2 = map(float, xyxy_sel)
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                center_str = f"({cx:.1f},{cy:.1f})"
                print(f"[SELECT] idx={sel_idx} box=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}) center={center_str}", flush=True)
            else:
                print("[SELECT] no valid selection", flush=True)

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Minimal summary
            boxes_count = int(len(res.boxes)) if getattr(res, "boxes", None) is not None else 0
            print(f"[{ts}] boxes={boxes_count} infer={pred_ms:.1f}ms center={center_str}", flush=True)

            if args.save:
                img_path = os.path.join(args.outdir, f"{ts}.png")
                save_annotated(res, img_path)
                meta = {
                    "time": ts,
                    "boxes": boxes_count,
                    "selected_index": sel_idx,
                    "center": center_str
                }
                with open(os.path.join(args.outdir, f"{ts}.json"), "w", encoding="utf-8") as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)

            iter_count += 1
            if args.max and iter_count >= args.max:
                break

            elapsed = time.time() - t0
            sleep_s = max(0.0, args.interval - elapsed)
            time.sleep(sleep_s)
    except KeyboardInterrupt:
        print("\n[EXIT] Interrupted by user.")
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(4)


if __name__ == "__main__":
    main()
