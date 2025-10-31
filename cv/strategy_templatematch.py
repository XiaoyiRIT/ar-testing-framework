# cv/strategy_templatematch.py
# ------------------------------------------------------------
# Self-template from prev frame + NCC matching (with ECC alignment).
# Exports: locate(curr_bgr, prev_bgr=None, min_area_ratio=0.002, max_w=640)
# Behavior:
#   - If prev_bgr is provided, we find the most salient/textured patch in prev
#     (Harris corner density) and treat it as a template, then locate it in curr.
#   - If prev_bgr is None, falls back to edge-density contour selection.
# ------------------------------------------------------------
import cv2, numpy as np
from typing import Optional, Dict, Any

def _resize_keep(src, max_w=640):
    h, w = src.shape[:2]
    if w <= max_w: return src, 1.0
    s = max_w / w
    return cv2.resize(src, (int(w*s), int(h*s)), cv2.INTER_AREA), s

def _best_patch(gray, ksize=48, stride=24):
    h, w = gray.shape[:2]
    resp = cv2.cornerHarris(gray, 2, 3, 0.04)
    resp = cv2.normalize(resp, None, 0, 1.0, cv2.NORM_MINMAX)
    best = None
    for y in range(0, h-ksize, stride):
        for x in range(0, w-ksize, stride):
            s = resp[y:y+ksize, x:x+ksize].mean()
            if best is None or s > best[0]:
                best = (s, (x,y,ksize,ksize))
    return best[1] if best else None

def _align_ecc(ref_gray, mov_gray):
    warp = np.eye(2,3, dtype=np.float32)
    try:
        _, warp = cv2.findTransformECC(ref_gray, mov_gray, warp, cv2.MOTION_EUCLIDEAN,
                                       criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT, 30, 1e-4))
    except cv2.error:
        pass
    return warp

def locate(curr_bgr, prev_bgr=None, min_area_ratio:float=0.002, max_w:int=640) -> Optional[Dict[str, Any]]:
    img, scale = _resize_keep(curr_bgr, max_w)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = g.shape[:2]

    if prev_bgr is not None:
        pre, _ = _resize_keep(prev_bgr, max_w)
        pre_g = cv2.cvtColor(pre, cv2.COLOR_BGR2GRAY)
        warp = _align_ecc(pre_g, g)
        pre_g = cv2.warpAffine(pre_g, warp, (w, h), flags=cv2.INTER_LINEAR+cv2.WARP_INVERSE_MAP)
        # choose a textured patch in prev
        rect = _best_patch(pre_g, ksize=max(24, min(h,w)//10), stride=max(12, min(h,w)//20))
        if rect is None:
            rect = (w//3, h//3, w//3, h//3)
        x,y,ww,hh = rect
        tmpl = pre_g[y:y+hh, x:x+ww]
        res = cv2.matchTemplate(g, tmpl, cv2.TM_CCOEFF_NORMED)
        _, maxv, _, maxloc = cv2.minMaxLoc(res)
        X, Y = maxloc
        W, H = ww, hh
    else:
        # Fallback: edge density region
        edges = cv2.Canny(g, 60, 160)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), 1)
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: return None
        min_area = int(h*w*min_area_ratio)
        cand = [(cv2.contourArea(c), cv2.boundingRect(c)) for c in cnts if cv2.contourArea(c) >= min_area]
        if not cand: return None
        X,Y,W,H = sorted(cand, key=lambda z: z[0], reverse=True)[0][1]

    inv = 1.0/scale
    X,Y,W,H = int(X*inv), int(Y*inv), int(W*inv), int(H*inv)
    cx, cy = X + W//2, Y + H//2
    return {"center": (cx, cy), "bbox": (X, Y, W, H)}
