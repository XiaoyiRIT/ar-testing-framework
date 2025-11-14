# cv/verify_motion.py
# ------------------------------------------------------------
# （暂时弃用）
# Description:
#   Post-action verification utilities for AR interactions.
#   Given pre/post frames + (center, bbox) + op metadata,
#   verify whether the intended gesture truly occurred.
#   v1+ version with:
#     (1) background motion compensation (camera shake removal)
#     (2) similarity transform fitting (scale/rotation/translation)
# ------------------------------------------------------------
from typing import Tuple, Optional, Dict, Any
import numpy as np
import cv2

Point = Tuple[int, int]
BBox  = Tuple[int, int, int, int]

# ------------------------------
# Utility helpers
# ------------------------------
def _to_gray(img_bgr: np.ndarray) -> np.ndarray:
    if img_bgr.ndim == 3:
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return img_bgr

def _clip_bbox(bbox: BBox, w: int, h: int) -> BBox:
    x, y, bw, bh = bbox
    x = int(max(0, min(x, w-1)))
    y = int(max(0, min(y, h-1)))
    bw = int(max(1, min(bw, w - x)))
    bh = int(max(1, min(bh, h - y)))
    return (x, y, bw, bh)

def _good_points(gray: np.ndarray, bbox: BBox, max_pts=120):
    """GoodFeatures within bbox area."""
    x, y, w, h = _clip_bbox(bbox, gray.shape[1], gray.shape[0])
    mask = np.zeros_like(gray, np.uint8)
    mask[y:y+h, x:x+w] = 255
    pts = cv2.goodFeaturesToTrack(
        gray, maxCorners=max_pts, qualityLevel=0.01, minDistance=6, mask=mask
    )
    if pts is None:
        return None
    return pts.reshape(-1, 2).astype(np.float32)

def _track(pre_g: np.ndarray, post_g: np.ndarray, pts: np.ndarray):
    """PyrLK optical flow, return matched p0, p1 arrays."""
    nxt, st, err = cv2.calcOpticalFlowPyrLK(
        pre_g, post_g, pts, None,
        winSize=(21, 21), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)
    )
    if nxt is None or st is None:
        return None, None
    ok = (st.reshape(-1) == 1)
    p0 = pts[ok]
    p1 = nxt.reshape(-1, 2)[ok]
    if len(p0) == 0:
        return None, None
    return p0, p1

def _bg_compensate(pre_g: np.ndarray, post_g: np.ndarray, bbox: BBox) -> np.ndarray:
    """Estimate background affine (camera motion) using points OUTSIDE bbox, then warp post_g."""
    H, W = pre_g.shape
    x, y, w, h = _clip_bbox(bbox, W, H)

    mask = np.ones_like(pre_g, np.uint8) * 255
    mask[y:y+h, x:x+w] = 0  # exclude target region

    pts_bg = cv2.goodFeaturesToTrack(
        pre_g, maxCorners=300, qualityLevel=0.01, minDistance=6, mask=mask
    )
    if pts_bg is None or len(pts_bg) < 20:
        return post_g  # fallback

    p0, p1 = _track(pre_g, post_g, pts_bg.reshape(-1, 2).astype(np.float32))
    if p0 is None or len(p0) < 12:
        return post_g

    # Fit partial affine (rotation+scale+translation)
    M, inliers = cv2.estimateAffinePartial2D(
        p1, p0, method=cv2.RANSAC, ransacReprojThreshold=3.0, maxIters=2000, confidence=0.99
    )
    if M is None:
        return post_g
    return cv2.warpAffine(post_g, M, (W, H), flags=cv2.INTER_LINEAR)

def _fit_similarity(p0: np.ndarray, p1: np.ndarray, center: Optional[Tuple[float, float]]=None):
    """
    Robustly fit similarity transform from p0 -> p1 using partial affine, then decompose.
    Return dict: {'scale': s, 'theta': rad, 'tx': tx, 'ty': ty, 'inlier_ratio': r, 'M': 2x3}
    """
    if len(p0) < 6:
        return None

    a0 = p0.copy().astype(np.float32)
    a1 = p1.copy().astype(np.float32)

    if center is not None:
        cx, cy = center
        a0 -= np.array([[cx, cy]], dtype=np.float32)
        a1 -= np.array([[cx, cy]], dtype=np.float32)

    M, inliers = cv2.estimateAffinePartial2D(
        a0, a1, method=cv2.RANSAC, ransacReprojThreshold=2.5, maxIters=2000, confidence=0.99
    )
    if M is None:
        return None

    # Decompose similarity parameters from 2x3 affine: [[a, b, tx],[c, d, ty]]
    a, b, tx = M[0, 0], M[0, 1], M[0, 2]
    c, d, ty = M[1, 0], M[1, 1], M[1, 2]
    # For similarity, c ≈ -b, d ≈ a. Use average.
    # Scale and rotation from the rotation-scale matrix [[a, b],[-b, a]]
    s = float(np.sqrt(max(1e-12, ((a + d) * 0.5)**2 + ((b - c) * 0.5)**2)))
    # Use atan2 of average (b, a) form
    theta = float(np.arctan2((b - c) * 0.5, (a + d) * 0.5))

    r = float(inliers.sum() / len(a0)) if inliers is not None else 1.0
    return dict(scale=s, theta=theta, tx=float(tx), ty=float(ty), inlier_ratio=r, M=M)

# ------------------------------
# Verifiers
# ------------------------------
def verify_drag(
    pre_bgr: np.ndarray,
    post_bgr: np.ndarray,
    center_xy: Point,
    bbox: BBox,
    start_xy: Point,
    end_xy: Point,
    min_motion_px: float = 12.0,
    min_dir_cos: float = 0.6,
    min_frac: float = 0.5,
) -> bool:
    """Verify drag by checking if majority flows align with intended direction and magnitude."""
    pre_g  = _to_gray(pre_bgr)
    post_g = _to_gray(post_bgr)
    # Background compensation
    post_gc = _bg_compensate(pre_g, post_g, bbox)

    pts = _good_points(pre_g, bbox)
    if pts is None or len(pts) < 8:
        return False
    p0, p1 = _track(pre_g, post_gc, pts)
    if p0 is None or len(p0) < 8:
        return False

    v  = np.array(end_xy, dtype=np.float32) - np.array(start_xy, dtype=np.float32)
    nv = float(np.linalg.norm(v)) + 1e-6
    u  = v / nv  # intended direction unit vector

    d   = (p1 - p0)                      # actual flow
    mag = np.linalg.norm(d, axis=1)      # displacement magnitude
    # direction cosine with intended direction
    cos = (d @ u) / (np.linalg.norm(d, axis=1) + 1e-6)

    ok = (mag >= min_motion_px) & (cos >= min_dir_cos)
    frac = float(np.mean(ok)) if len(ok) else 0.0

    if frac >= min_frac:
        return True

    # Fallback: use similarity translation estimation (more tolerant)
    sim = _fit_similarity(p0, p1, center=None)
    if sim is None:
        return False
    trans = np.array([sim['tx'], sim['ty']], dtype=np.float32)
    proj = float(np.dot(trans, u))
    if np.linalg.norm(trans) >= min_motion_px and proj / (np.linalg.norm(trans)+1e-6) >= min_dir_cos:
        # also require inlier ratio to be reasonable
        return sim['inlier_ratio'] >= min_frac
    return False

def verify_pinch(
    pre_bgr: np.ndarray,
    post_bgr: np.ndarray,
    center_xy: Point,
    bbox: BBox,
    pinch_in: bool,
    scale_thr: float = 0.10,
    min_frac: float = 0.5,
) -> bool:
    """Verify pinch by similarity scale around center."""
    pre_g  = _to_gray(pre_bgr)
    post_g = _to_gray(post_bgr)
    post_gc = _bg_compensate(pre_g, post_g, bbox)

    pts = _good_points(pre_g, bbox)
    if pts is None or len(pts) < 8:
        return False
    p0, p1 = _track(pre_g, post_gc, pts)
    if p0 is None or len(p0) < 8:
        return False

    sim = _fit_similarity(p0, p1, center=center_xy)
    if sim is None:
        return False

    s = sim['scale']
    if pinch_in and (s <= 1.0 - scale_thr) and sim['inlier_ratio'] >= min_frac:
        return True
    if (not pinch_in) and (s >= 1.0 + scale_thr) and sim['inlier_ratio'] >= min_frac:
        return True

    # Fallback using radial ratios (median) around center
    c = np.array(center_xy, dtype=np.float32)
    r0 = np.linalg.norm(p0 - c, axis=1) + 1e-6
    r1 = np.linalg.norm(p1 - c, axis=1) + 1e-6
    ratio = np.median(r1 / r0)
    if pinch_in:
        return (ratio <= 1.0 - scale_thr) and (sim['inlier_ratio'] >= min_frac*0.8)
    else:
        return (ratio >= 1.0 + scale_thr) and (sim['inlier_ratio'] >= min_frac*0.8)

def verify_rotate(
    pre_bgr: np.ndarray,
    post_bgr: np.ndarray,
    center_xy: Point,
    bbox: BBox,
    min_deg: float = 15.0,
    min_frac: float = 0.5,
) -> bool:
    """Verify rotation by similarity theta around center (degrees)."""
    pre_g  = _to_gray(pre_bgr)
    post_g = _to_gray(post_bgr)
    post_gc = _bg_compensate(pre_g, post_g, bbox)

    pts = _good_points(pre_g, bbox)
    if pts is None or len(pts) < 8:
        return False
    p0, p1 = _track(pre_g, post_gc, pts)
    if p0 is None or len(p0) < 8:
        return False

    sim = _fit_similarity(p0, p1, center=center_xy)
    if sim is None:
        return False
    theta_deg = abs(sim['theta'] * 180.0 / np.pi)
    if (theta_deg >= min_deg) and (sim['inlier_ratio'] >= min_frac):
        return True

    # Fallback: compute median angular change around center
    c = np.array(center_xy, dtype=np.float32)
    a0 = np.arctan2(p0[:,1]-c[1], p0[:,0]-c[0])
    a1 = np.arctan2(p1[:,1]-c[1], p1[:,0]-c[0])
    dtheta = np.rad2deg(np.unwrap(a1 - a0))
    if np.median(np.abs(dtheta)) >= min_deg * 1.2:
        return True
    return False

# ------------------------------
# Public API
# ------------------------------
def verify_action(
    op: str,
    pre_bgr: np.ndarray,
    post_bgr: np.ndarray,
    center_xy: Point,
    bbox: BBox,
    extra: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Verify result of an operation.
    Supported ops: 'drag', 'pinch_in', 'pinch_out', 'rotate'
    """
    extra = extra or {}
    if op == "drag":
        start_xy = extra.get("start_xy", center_xy)
        end_xy   = extra.get("end_xy", center_xy)
        return verify_drag(
            pre_bgr, post_bgr, center_xy, bbox,
            start_xy=start_xy, end_xy=end_xy,
            min_motion_px=extra.get("min_motion_px", 12.0),
            min_dir_cos=extra.get("min_dir_cos", 0.6),
            min_frac=extra.get("min_frac", 0.5),
        )
    if op == "pinch_in":
        return verify_pinch(
            pre_bgr, post_bgr, center_xy, bbox,
            pinch_in=True,
            scale_thr=extra.get("scale_thr", 0.10),
            min_frac=extra.get("min_frac", 0.5),
        )
    if op == "pinch_out":
        return verify_pinch(
            pre_bgr, post_bgr, center_xy, bbox,
            pinch_in=False,
            scale_thr=extra.get("scale_thr", 0.10),
            min_frac=extra.get("min_frac", 0.5),
        )
    if op == "rotate":
        return verify_rotate(
            pre_bgr, post_bgr, center_xy, bbox,
            min_deg=extra.get("min_deg", 15.0),
            min_frac=extra.get("min_frac", 0.5),
        )
    raise ValueError(f"Unsupported op: {op}")
