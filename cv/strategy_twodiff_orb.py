# cv/strategy_twodiff_orb.py
# ------------------------------------------------------------
# Two-frame differencing + ORB alignment.
# Exports: locate(curr_bgr, prev_bgr=None, min_area_ratio=0.002, max_w=640)
# ------------------------------------------------------------
import cv2, numpy as np
from typing import Optional, Dict, Any

def _resize_keep(src, max_w=640):
    h, w = src.shape[:2]
    if w <= max_w: return src, 1.0
    s = max_w / w
    return cv2.resize(src, (int(w*s), int(h*s)), cv2.INTER_AREA), s

def _align_orb(prev_bgr, curr_bgr, max_kp=800):
    prev_g = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
    curr_g = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=max_kp)
    k1, d1 = orb.detectAndCompute(prev_g, None)
    k2, d2 = orb.detectAndCompute(curr_g, None)
    if d1 is None or d2 is None: return prev_bgr
    m = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(d1, d2)
    m = sorted(m, key=lambda x: x.distance)[:200]
    if len(m) < 10: return prev_bgr
    src = np.float32([k1[a.queryIdx].pt for a in m]).reshape(-1,1,2)
    dst = np.float32([k2[a.trainIdx].pt for a in m]).reshape(-1,1,2)
    H, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if H is None: return prev_bgr
    h, w = curr_bgr.shape[:2]
    return cv2.warpAffine(prev_bgr, H, (w, h))

def locate(curr_bgr, prev_bgr=None, min_area_ratio=0.002, max_w=640) -> Optional[Dict[str, Any]]:
    if prev_bgr is None: return None
    p_small, s1 = _resize_keep(prev_bgr, max_w)
    c_small, s2 = _resize_keep(curr_bgr, max_w)
    if abs(s1 - s2) > 1e-6: return None
    scale = s1
    p_aln = _align_orb(p_small, c_small)
    pb = cv2.GaussianBlur(p_aln, (5,5), 0)
    cb = cv2.GaussianBlur(c_small, (5,5), 0)
    diff = cv2.absdiff(cb, pb)
    g = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), 1)
    th = cv2.dilate(th, np.ones((5,5), np.uint8), 1)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    h, w = th.shape[:2]
    min_area = (w*h) * min_area_ratio
    cand = [(cv2.contourArea(c), cv2.boundingRect(c)) for c in cnts if cv2.contourArea(c) >= min_area]
    if not cand: return None
    _, (x, y, ww, hh) = sorted(cand, key=lambda z: z[0], reverse=True)[0]
    inv = 1.0 / scale
    x, y, ww, hh = int(x*inv), int(y*inv), int(ww*inv), int(hh*inv)
    cx, cy = x + ww//2, y + hh//2
    return {"center": (cx, cy), "bbox": (x, y, ww, hh)}
