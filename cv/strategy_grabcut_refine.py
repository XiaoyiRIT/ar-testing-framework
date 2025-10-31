# cv/strategy_grabcut_refine.py
# ------------------------------------------------------------
# Saliency pre-detection + GrabCut refinement.
# Exports: locate(curr_bgr, prev_bgr=None, min_area_ratio=0.002, max_w=640)
# Requires: OpenCV with ximgproc/saliency (falls back to gradient if absent).
# ------------------------------------------------------------
import cv2, numpy as np
from typing import Optional, Dict, Any

def _resize_keep(src, max_w=640):
    h, w = src.shape[:2]
    if w <= max_w: return src, 1.0
    s = max_w / w
    return cv2.resize(src, (int(w*s), int(h*s)), cv2.INTER_AREA), s

def _saliency(gray):
    # Try OpenCV saliency; fallback to spectral residual approximation
    try:
        sal = cv2.saliency.StaticSaliencyFineGrained_create()
        ok, m = sal.computeSaliency(gray)
        if ok and m is not None: 
            m = (m*255).astype("uint8")
            return m
    except Exception:
        pass
    # Fallback: spectral residual (simple)
    g = cv2.GaussianBlur(gray, (0,0), 2.0)
    spec = np.fft.fft2(g.astype(np.float32))
    logA = np.log(np.abs(spec)+1e-6)
    phase = np.angle(spec)
    avg = cv2.boxFilter(logA, ddepth=-1, ksize=(9,9))
    resid = logA - avg
    recon = np.exp(resid + 1j*phase)
    m = np.fft.ifft2(recon)
    m = np.abs(m)
    m = cv2.GaussianBlur(m, (0,0), 2.0)
    m = cv2.normalize(m, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
    return m

def _largest_bbox(bin_img, min_area):
    cnts, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    cand = [(cv2.contourArea(c), cv2.boundingRect(c)) for c in cnts if cv2.contourArea(c) >= min_area]
    if not cand: return None
    return sorted(cand, key=lambda z: z[0], reverse=True)[0][1]

def locate(curr_bgr, prev_bgr=None, min_area_ratio:float=0.002, max_w:int=640) -> Optional[Dict[str, Any]]:
    img, scale = _resize_keep(curr_bgr, max_w)
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sal = _saliency(gray)
    sal = cv2.GaussianBlur(sal, (7,7), 0)
    th = cv2.threshold(sal, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), 1)
    min_area = int(h*w*min_area_ratio)
    rect = _largest_bbox(th, min_area)
    if rect is None:
        return None
    x,y,ww,hh = rect
    # Expand rect slightly as GrabCut initialization
    pad = int(0.08 * max(ww, hh))
    x0 = max(0, x - pad); y0 = max(0, y - pad)
    x1 = min(w-1, x + ww + pad); y1 = min(h-1, y + hh + pad)
    rect = (x0, y0, x1-x0, y1-y0)

    # GrabCut refinement
    mask = np.zeros((h,w), np.uint8)
    bgdModel = np.zeros((1,65), np.float64)
    fgdModel = np.zeros((1,65), np.float64)
    try:
        cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        # If grabcut fails, fall back to rectangle
        fin = (x0,y0,x1-x0,y1-y0)
        inv = 1.0/scale
        X,Y,W,H = [int(v*inv) for v in fin]
        cx, cy = X + W//2, Y + H//2
        return {"center": (cx, cy), "bbox": (X, Y, W, H)}

    fgMask = np.where((mask==cv2.GC_FGD) | (mask==cv2.GC_PR_FGD), 255, 0).astype('uint8')
    fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8), 1)
    rect2 = _largest_bbox(fgMask, min_area)
    if rect2 is None:
        rect2 = rect
    X,Y,W,H = rect2
    inv = 1.0/scale
    X,Y,W,H = int(X*inv), int(Y*inv), int(W*inv), int(H*inv)
    cx, cy = X + W//2, Y + H//2
    return {"center": (cx, cy), "bbox": (X, Y, W, H)}
