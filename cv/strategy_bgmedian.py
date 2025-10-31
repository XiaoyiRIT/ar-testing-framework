# cv/strategy_bgmedian.py
# ------------------------------------------------------------
# Background subtraction (MOG2/median-ish) + contour selection.
# Exports: locate(curr_bgr, prev_bgr=None, min_area_ratio=0.002, max_w=640)
# Notes:
#   - Stateless wrapper: builds a pseudo background by lightly blurring
#     and subtracting; if prev_bgr is given, we align roughly by ECC; otherwise
#     we just threshold high-gradient regions as foreground proxy.
# ------------------------------------------------------------
import cv2, numpy as np
from typing import Optional, Dict, Any

def _resize_keep(src, max_w=640):
    h, w = src.shape[:2]
    if w <= max_w: return src, 1.0
    s = max_w / w
    return cv2.resize(src, (int(w*s), int(h*s)), cv2.INTER_AREA), s

def _align_ecc(ref_gray, mov_gray):
    # ECC homography alignment (robust for small camera shakes)
    warp = np.eye(2,3, dtype=np.float32)
    try:
        cc, warp = cv2.findTransformECC(ref_gray, mov_gray, warp, cv2.MOTION_EUCLIDEAN,
                                        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 1e-4))
    except cv2.error:
        pass
    return warp

def locate(curr_bgr, prev_bgr=None, min_area_ratio:float=0.002, max_w:int=640) -> Optional[Dict[str, Any]]:
    img, scale = _resize_keep(curr_bgr, max_w)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.bilateralFilter(g, 5, 35, 35)

    if prev_bgr is not None:
        ref, _ = _resize_keep(prev_bgr, max_w)
        refg = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
        warp = _align_ecc(refg, g)
        refg = cv2.warpAffine(refg, warp, (g.shape[1], g.shape[0]), flags=cv2.INTER_LINEAR+cv2.WARP_INVERSE_MAP)
        fg = cv2.absdiff(g, refg)
    else:
        # fall back: high-frequency as proxy for foreground
        fg = cv2.GaussianBlur(g, (0,0), 1.5)
        fg = cv2.Laplacian(fg, cv2.CV_16S, ksize=3)
        fg = cv2.convertScaleAbs(fg)

    # Post-process
    th = cv2.threshold(fg, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), 1)
    th = cv2.dilate(th, np.ones((5,5), np.uint8), 1)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None

    h, w = th.shape[:2]
    min_area = (w*h) * min_area_ratio
    cand = [(cv2.contourArea(c), cv2.boundingRect(c)) for c in cnts if cv2.contourArea(c) >= min_area]
    if not cand: return None

    _, (x, y, ww, hh) = sorted(cand, key=lambda z: z[0], reverse=True)[0]
    inv = 1.0/scale
    x, y, ww, hh = int(x*inv), int(y*inv), int(ww*inv), int(hh*inv)
    cx, cy = x + ww//2, y + hh//2
    return {"center": (cx, cy), "bbox": (x, y, ww, hh)}
