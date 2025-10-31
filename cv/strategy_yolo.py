# cv/strategy_yolo.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Any, Dict, Optional
import time
import os
import tempfile

import numpy as np
import cv2
from ultralytics import YOLO

_MODEL = None
_MODEL_PATH = None

def _lazy_load(model_path: str):
    global _MODEL, _MODEL_PATH
    if _MODEL is None or _MODEL_PATH != model_path:
        _MODEL = YOLO(model_path)
        _MODEL_PATH = model_path

def _pick_index(res) -> Optional[int]:
    boxes = res.boxes
    probs = getattr(res, "probs", None)
    if boxes is None or boxes.xywh is None or len(boxes) == 0:
        return None
    # 优先你指定的策略：probs.top1 -> 否则按最高 conf
    if probs is not None and hasattr(probs, "top1") and probs.top1 is not None:
        idx = int(probs.top1)
        if 0 <= idx < len(boxes):
            return idx
    confs = boxes.conf.cpu().numpy()
    return int(confs.argmax())

def locate(
    image_bgr: np.ndarray,
    *,
    model_path: str = "cv/yolo/best.pt",
    save_format: str = ".png",     # ".png" 或 ".jpg" 都可；默认 .png
    debug_profile: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    最接近 Ultralytics sample 的用法：
    1) 把 ndarray 截图原样写成临时 PNG/JPG
    2) 直接把文件路径传给 YOLO(...)，不设 conf/imgsz/classes 等额外参数
    """
    assert image_bgr is not None and image_bgr.ndim == 3 and image_bgr.shape[2] == 3, "expect HxWx3 BGR"
    if image_bgr.dtype != np.uint8:
        image_bgr = np.clip(image_bgr, 0, 255).astype(np.uint8)

    _lazy_load(model_path)
    t0 = time.perf_counter()

    # 1) 写临时文件（不做任何 resize/预处理）
    #    使用 cv2.imwrite 避免二次编码/解码差异；suffix 控制 png/jpg
    tmp = tempfile.NamedTemporaryFile(prefix="yolo_frame_", suffix=save_format, delete=False)
    tmp_path = tmp.name
    tmp.close()
    ok = cv2.imwrite(tmp_path, image_bgr)
    if not ok:
        if debug_profile:
            print("[yolo] cv2.imwrite failed")
        try:
            os.remove(tmp_path)
        finally:
            return None

    # 2) 以“路径”方式推理（不传 conf/imgsz/classes）
    try:
        t1 = time.perf_counter()
        results = _MODEL(tmp_path, conf=0.05, iou=0.45, imgsz=960, verbose=False)
        infer_ms = (time.perf_counter() - t1) * 1000.0
    finally:
        # 清理临时图片
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    if not results or len(results) == 0:
        if debug_profile:
            print(f"[yolo] empty results  io={(t1 - t0)*1000:.1f}ms  infer={infer_ms:.1f}ms")
        return None

    res = results[0]
    boxes = getattr(res, "boxes", None)
    if boxes is None or boxes.xywh is None or len(boxes) == 0:
        if debug_profile:
            print(f"[yolo] no boxes  infer={infer_ms:.1f}ms")
        return None

    # 选择一个框
    idx = _pick_index(res)
    if idx is None:
        if debug_profile:
            print("[yolo] selection failed (no valid index)")
        return None

    # 直接用 Ultralytics 的 xywh（中心点坐标 + 宽高）
    xywh = boxes.xywh.cpu().numpy()
    cx_f, cy_f, w_f, h_f = map(float, xywh[idx])

    cx, cy = int(round(cx_f)), int(round(cy_f))
    x = int(round(cx_f - 0.5 * w_f))
    y = int(round(cy_f - 0.5 * h_f))
    w = int(round(w_f))
    h = int(round(h_f))

    out = {
        "center": (cx, cy),
        "bbox": (x, y, w, h),
        "meta": {
            "model_path": _MODEL_PATH,
            "conf": float(boxes.conf[idx]),
            "cls": int(boxes.cls[idx]) if boxes.cls is not None else None,
            "infer_ms": infer_ms,
            "input": "path",      # 标记这次是“路径输入”
            "format": save_format
        },
    }

    if debug_profile:
        total_ms = (time.perf_counter() - t0) * 1000.0
        print(f"[yolo] idx={idx} center={out['center']} bbox={out['bbox']} infer={infer_ms:.1f}ms total={total_ms:.1f}ms [path-input:{save_format}]")

    return out
