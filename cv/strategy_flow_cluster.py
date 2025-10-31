# cv/strategy_flow_cluster.py
# ------------------------------------------------------------
# Sparse optical flow (KLT) + motion vector clustering.
# Exports: locate(curr_bgr, prev_bgr, min_area_ratio=0.002, max_w=640)
# Notes:
#   - Requires prev_bgr; if absent, returns None.
# ------------------------------------------------------------
import cv2, numpy as np
from typing import Optional, Dict, Any

def _resize_keep(src, max_w=640):
    h, w = src.shape[:2]
    if w <= max_w: return src, 1.0
    s = max_w / w
    return cv2.resize(src, (int(w*s), int(h*s)), cv2.INTER_AREA), s

def locate(curr_bgr, prev_bgr=None, min_area_ratio:float=0.002, max_w:int=640) -> Optional[Dict[str, Any]]:
    if prev_bgr is None:
        return None
    post, scale = _resize_keep(curr_bgr, max_w)
    pre,  _    = _resize_keep(prev_bgr, max_w)
    pre_g = cv2.cvtColor(pre,  cv2.COLOR_BGR2GRAY)
    post_g= cv2.cvtColor(post, cv2.COLOR_BGR2GRAY)

    # good features to track
    pts = cv2.goodFeaturesToTrack(pre_g, maxCorners=600, qualityLevel=0.01, minDistance=7, blockSize=7)
    if pts is None:
        return None
    p1, st, err = cv2.calcOpticalFlowPyrLK(pre_g, post_g, pts, None, winSize=(21,21), maxLevel=3,
                                           criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    good_new = p1[st==1]; good_old = pts[st==1].reshape(-1,2)
    v = good_new - good_old
    mag = np.linalg.norm(v, axis=1)
    if len(mag) < 10:
        return None
    # filter small motions
    thr = np.percentile(mag, 70)
    sel = v[np.where(mag >= thr)]
    loc = good_new[np.where(mag >= thr)]
    if len(sel) < 6:
        return None

    # k-means (k=2) on end-point positions to pick dominant cluster
    Z = np.float32(loc)
    K = 2 if len(Z) >= 8 else 1
    if K > 1:
        criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
        ret, labels, centers = cv2.kmeans(Z, K, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
        # pick cluster with more members
        k = np.argmax(np.bincount(labels.flatten()))
        pts_k = Z[labels.flatten()==k]
    else:
        pts_k = Z

    x,y,w,h = cv2.boundingRect(pts_k.astype(np.float32))
    inv = 1.0/scale
    X,Y,W,H = int(x*inv), int(y*inv), int(w*inv), int(h*inv)
    cx, cy = X + W//2, Y + H//2
    return {"center": (cx, cy), "bbox": (X, Y, W, H)}
