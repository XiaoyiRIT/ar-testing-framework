# Project Context

- **Path**: `/Users/yangxiaoyi/Desktop/phd/project/newproject/program`
- **Time (UTC)**: 2025-12-05 16:54:15Z
- **Git Branch**: main
- **Git Commit**: 735d5be

## Directory Tree (depth=4)

```
program/
├── configs/
│   └── ad.yaml
├── cv/
│   ├── yolo/
│   │   ├── .split/
│   │   ├── rc/
│   │   │   ├── yolo_app_ood.sh
│   │   │   ├── yolo_app_ood_kfold.sh
│   │   │   ├── yolo_app_ood_leave1out.sh
│   │   │   ├── yolo_obj_ood.sh
│   │   │   ├── yolo_obj_ood_kfold.sh
│   │   │   ├── yolo_random.sh
│   │   │   ├── yolo_scene_ood.sh
│   │   │   ├── yolo_scene_ood_kfold.sh
│   │   │   ├── yolo_scene_ood_leave1out.sh
│   │   │   └── yolo_test.sh
│   │   ├── splits_leave1out/
│   │   │   ├── app-ood_11/
│   │   │   ├── app-ood_21/
│   │   │   └── scene-ood_bedroom_low_small_indoor/
│   │   ├── testimages/
│   │   ├── check_leaveoneout_runs.py
│   │   ├── check_splits_groups.py
│   │   ├── check_splits_mutual_exclusion.py
│   │   ├── crossval_eval.py
│   │   ├── data_stat.csv
│   │   ├── data_stat_sample.csv
│   │   ├── eval_yolo_center_hit_debug.py
│   │   ├── inspect_app_column.py
│   │   ├── leaveoneout_eval.py
│   │   ├── myar2.yaml
│   │   ├── myar_scene.yaml
│   │   ├── myar_simp.yaml
│   │   ├── suggest_k.py
│   │   ├── sync_data_stat_with_images.py
│   │   ├── test_yolo_train.py
│   │   ├── test_yolo_train_dynsplit.py
│   │   └── yolo_train_leave_one_group.py
│   ├── strategy_bgmedian.py
│   ├── strategy_flow_cluster.py
│   ├── strategy_FoELS.py
│   ├── strategy_grabcut_refine.py
│   ├── strategy_riri_invariant.py
│   ├── strategy_templatematch.py
│   ├── strategy_twodiff_orb.py
│   └── strategy_yolo.py
├── docs/
├── experiments/
│   ├── notebooks/
│   │   └── framework_status.md
│   ├── playground/
│   │   ├── delete_images_without_labels.py
│   │   ├── diagnose_orphan_labels.py
│   │   ├── find_orphan_labels.py
│   │   └── fix_label_to_zero.py
│   └── v0_v1_v2_v3_archive/
│       ├── smoke.py
│       ├── v0_ar_monkey_adb.py
│       ├── v0_ar_monkey_appium.py
│       ├── v1_ar_monkey_appium.py
│       ├── v2_ar_monkey_appium.py
│       └── v3_ar_monkey_appium.py
├── scripts/
│   └── run_discovery.sh
├── src/
│   ├── common/
│   │   ├── __init__.py
│   │   ├── actions.py
│   │   ├── device.py
│   │   ├── locator_iface.py
│   │   ├── policy_random.py
│   │   ├── timing.py
│   │   └── verify_motion.py
│   ├── detector/
│   │   ├── __init__.py
│   │   └── yolo_detector.py
│   ├── discovery/
│   │   ├── __init__.py
│   │   └── run_discovery.py
│   ├── executor/
│   │   ├── __init__.py
│   │   └── appium_executor.py
│   ├── policy/
│   │   ├── __init__.py
│   │   └── policy.py
│   ├── sampler/
│   │   ├── __init__.py
│   │   └── default_sampler.py
│   ├── verifier/
│   │   ├── backends/
│   │   │   └── motion_similarity.py
│   │   ├── __init__.py
│   │   └── verifier.py
│   ├── __init__.py
│   └── __main__.py
├── tests/
├── .gitignore
├── chat_context.md
├── config.yaml
├── gen_project_context.py
├── project_context.md
├── project_status_summary.md
├── README.md
├── requirements.txt
├── yolo_adb_screencap.py
├── yolo_scratch.py
└── yolo_val.py
```

## File Stats

- Files counted (excluded patterns applied): **102**
- Total size: **2.0 MB**

### Top 15 Largest Files (excluded patterns applied)

| Size | Path |
|---:|---|
| 453.1 KB | `cv/yolo/data_stat.csv` |
| 298.9 KB | `cv/yolo/splits_leave1out/app-ood_21/train.txt` |
| 293.6 KB | `cv/yolo/splits_leave1out/scene-ood_bedroom_low_small_indoor/train.txt` |
| 293.1 KB | `cv/yolo/splits_leave1out/app-ood_11/train.txt` |
| 0.0 B | `chat_context.md` |
| 92.7 KB | `project_context.md` |
| 34.8 KB | `cv/yolo/runs/predict/exp4/results.csv` |
| 26.2 KB | `cv/yolo/test_yolo_train_dynsplit.py` |
| 18.8 KB | `cv/yolo/test_yolo_train.py` |
| 17.7 KB | `cv/yolo/eval_yolo_center_hit_debug.py` |
| 17.2 KB | `cv/strategy_FoELS.py` |
| 14.7 KB | `experiments/v0_v1_v2_v3_archive/v1_ar_monkey_appium.py` |
| 14.7 KB | `experiments/v0_v1_v2_v3_archive/v2_ar_monkey_appium.py` |
| 13.9 KB | `experiments/v0_v1_v2_v3_archive/v3_ar_monkey_appium.py` |
| 13.4 KB | `cv/yolo/yolo_train_leave_one_group.py` |

## Recent Commits

```
735d5be Update
a91cf4d Big update
d18c655 Create README.md
7143e3e init: import existing project
```

## File Snippets (first 120 lines)

### `chat_context.md`

```text
```

### `cv/strategy_bgmedian.py`

```text
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
```

### `cv/strategy_flow_cluster.py`

```text
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
```

### `cv/strategy_FoELS.py`

```text
# cv/strategy_FoELS.py  (clean & robust, relaxed-by-default)
import cv2, time, numpy as np
from typing import Optional, Dict, Any, Tuple

# -------------------- small utils --------------------
def _ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000.0

def _resize_keep(src: np.ndarray, max_w: int) -> Tuple[np.ndarray, float]:
    h, w = src.shape[:2]
    if w <= max_w: return src, 1.0
    s = max_w / w
    dst = cv2.resize(src, (int(w*s), int(h*s)), cv2.INTER_AREA)
    return dst, s

def _to_gray(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr

def _unit(v: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True) + eps
    return v / n

# -------------------- flows --------------------
def _flow_farneback(prev_g: np.ndarray, curr_g: np.ndarray) -> np.ndarray:
    # 轻微平滑以稳住纹理（对低纹理/噪声有帮助）
    prev_g = cv2.GaussianBlur(prev_g, (3,3), 0)
    curr_g = cv2.GaussianBlur(curr_g, (3,3), 0)
    flow = cv2.calcOpticalFlowFarneback(prev_g, curr_g, None,
                                        pyr_scale=0.5, levels=4, winsize=21,
                                        iterations=5, poly_n=5, poly_sigma=1.2,
                                        flags=0)
    return flow.astype(np.float32)

def _tvl1_available() -> bool:
    return hasattr(cv2, "optflow") and (
        hasattr(cv2.optflow, "DualTVL1OpticalFlow_create") or
        hasattr(cv2.optflow, "createOptFlow_DualTVL1")
    )

def _flow_tvl1(prev_g: np.ndarray, curr_g: np.ndarray) -> np.ndarray:
    if not _tvl1_available():
        raise RuntimeError("cv2.optflow TV-L1 not available")
    try:
        tvl1 = cv2.optflow.DualTVL1OpticalFlow_create()
    except AttributeError:
        tvl1 = cv2.optflow.createOptFlow_DualTVL1()
    return tvl1.calc(prev_g, curr_g, None)

# -------------------- geometry --------------------
def _intersect(p: np.ndarray, d: np.ndarray, q: np.ndarray, e: np.ndarray, eps: float = 1e-6):
    A = np.stack([d, -e], axis=1)
    b = (q - p)
    det = A[0,0]*A[1,1] - A[0,1]*A[1,0]
    if abs(det) < eps: return None
    invA = np.array([[ A[1,1], -A[0,1]],
                     [-A[1,0],  A[0,0]]], dtype=np.float32) / det
    t = (invA @ b)[0]
    return p + t * d

# -------------------- FoE estimation --------------------
def _foe_ransac(flow: np.ndarray, mask: np.ndarray,
                trials: int, ang_tol_deg: float,
                min_mag: float, rng=np.random.default_rng(0)) -> Optional[Dict[str, Any]]:
    H, W, _ = flow.shape
    ys, xs = np.where(mask)
    if len(xs) < 150: return None

    vec_all = flow[ys, xs]; mag_all = np.linalg.norm(vec_all, axis=1)
    keep = mag_all >= min_mag
    if keep.sum() < 120: return None

    xs = xs[keep]; ys = ys[keep]; vec = vec_all[keep]; mag = mag_all[keep]
    # 幅值分位裁剪 + 子采样
    ql, qh = np.quantile(mag, [0.20, 0.98])
    sel = (mag >= ql) & (mag <= qh)
    xs = xs[sel]; ys = ys[sel]; vec = vec[sel]
    MMAX = 4000
    if xs.size > MMAX:
        idx = rng.choice(xs.size, size=MMAX, replace=False)
        xs = xs[idx]; ys = ys[idx]; vec = vec[idx]
    if xs.size < 120: return None

    pts  = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
    dirs = _unit(vec)
    M = xs.size
    cos_thr = np.cos(np.deg2rad(ang_tol_deg))
    best = None
    stop_hits = int(0.65 * M)
    xm_lo, xm_hi = -0.4*W, 1.4*W
    ym_lo, ym_hi = -0.4*H, 1.4*H

    for _ in range(trials):
        i, j = rng.choice(M, size=2, replace=False)
        p, q = pts[i], pts[j]; d, e = dirs[i], dirs[j]
        foe = _intersect(p, d, q, e)
        if foe is None: continue
        if not (xm_lo <= foe[0] <= xm_hi and ym_lo <= foe[1] <= ym_hi): continue
        vF = pts - foe; vFn = _unit(vF)
        cosang = np.sum(vFn * dirs, axis=1)
        inliers = (cosang > cos_thr)
        score = int(inliers.sum())
        if (best is None) or (score > best["score"]):
            sign_votes = np.sign(np.sum(np.sum(vF[inliers] * vec[inliers], axis=1)))
            sign = 1.0 if sign_votes >= 0 else -1.0
            best = dict(foe=foe, inliers=inliers, score=score, sign=sign)
            if score >= stop_hits: break

    if best is None: return None
    if best["score"] / float(M) < 0.18: return None
    return best

def _foe_lsq(flow: np.ndarray, mask: np.ndarray, min_mag: float) -> Optional[Dict[str, Any]]:
    """
    线性最小二乘 FoE 备选：最小化 Σ|| (I - dd^T)(e - p) ||^2
    推导： (ΣP) e = ΣP p, 其中 P = I - dd^T
    """
    ys, xs = np.where(mask)
    if xs.size < 80: return None
    vec = flow[ys, xs]; mag = np.linalg.norm(vec, axis=1)
    keep = mag >= min_mag
... (truncated)
```

### `cv/strategy_grabcut_refine.py`

```text
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
```

### `cv/strategy_riri_invariant.py`

```text
# cv/strategy_riri_invariant.py
import cv2, time, numpy as np
from typing import Optional, Dict, Any, Tuple

# —— 小工具：与 strategy_FoELS 风格一致 —— #
def _ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000.0

def _resize_keep(src: np.ndarray, max_w: int) -> Tuple[np.ndarray, float]:
    h, w = src.shape[:2]
    if w <= max_w: return src, 1.0
    s = max_w / float(w)
    dst = cv2.resize(src, (int(w*s), int(h*s)), cv2.INTER_AREA)
    return dst, s

def _to_gray(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr

def _flow_farneback(prev_g: np.ndarray, curr_g: np.ndarray) -> np.ndarray:
    prev_g = cv2.GaussianBlur(prev_g, (3,3), 0)
    curr_g = cv2.GaussianBlur(curr_g, (3,3), 0)
    flow = cv2.calcOpticalFlowFarneback(prev_g, curr_g, None,
                                        pyr_scale=0.5, levels=4, winsize=21,
                                        iterations=5, poly_n=5, poly_sigma=1.2,
                                        flags=0)
    return flow.astype(np.float32)

def _unit(v: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True) + eps
    return v / n

# —— 估计 FoE（复用简化 LSQ，细节参考 strategy_FoELS） —— #
def _foe_lsq(flow: np.ndarray, mask: np.ndarray, min_mag: float = 0.10):
    ys, xs = np.where(mask)
    if xs.size < 80: return None
    vec = flow[ys, xs]; mag = np.linalg.norm(vec, axis=1)
    keep = mag >= min_mag
    if keep.sum() < 60: return None
    xs = xs[keep]; ys = ys[keep]; vec = vec[keep]
    pts  = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
    dirs = _unit(vec)

    S = np.zeros((2,2), np.float32)
    b = np.zeros((2,), np.float32)
    I = np.eye(2, dtype=np.float32)
    for (p, d) in zip(pts, dirs):
        P = I - np.outer(d, d)
        S += P; b += P @ p
    S += 1e-6 * I
    try:
        foe = np.linalg.solve(S, b)
    except np.linalg.LinAlgError:
        return None
    # 符号判定：径向是否发散（与 strategy_FoELS 类似）
    vF = pts - foe
    sign_votes = np.sign(np.sum(np.sum(vF * vec, axis=1)))
    sign = 1.0 if sign_votes >= 0 else -1.0
    return dict(foe=foe, sign=sign)

# —— 主入口：与 strategy_FoELS.locate 同名同参 —— #
def locate(curr_bgr: np.ndarray,
           prev_bgr: Optional[np.ndarray] = None,
           min_area_ratio: float = 0.0015,
           max_w: int = 720,
           angle_thr_deg: float = 20.0,
           min_flow_mag: float = 0.08,
           camera_move_thr: float = 0.01,
           debug_profile: bool = False) -> Optional[Dict[str, Any]]:

    if prev_bgr is None:
        if debug_profile: print("[RIRI:None] no prev frame")
        return None

    Tall = time.perf_counter()

    # 1) resize / gray
    T = time.perf_counter()
    c_small, s  = _resize_keep(curr_bgr, max_w)
    p_small, s2 = _resize_keep(prev_bgr, max_w)
    t_resize = _ms(T)
    if abs(s - s2) > 1e-6:
        if debug_profile: print(f"[RIRI:None] resize scale mismatch s={s:.4f}, s2={s2:.4f}")
        return None

    T = time.perf_counter()
    c_g = _to_gray(c_small); p_g = _to_gray(p_small)
    t_gray = _ms(T)

    # 2) flow
    T = time.perf_counter()
    flow = _flow_farneback(p_g, c_g)
    t_flow = _ms(T)

    # 3) 简单“相机在动”检查：统计全图/低先验区域的光流比例
    T = time.perf_counter()
    mag = np.linalg.norm(flow, axis=-1)
    ratio_move = float((mag > min_flow_mag).mean())
    t_cam = _ms(T)
    if ratio_move < camera_move_thr:
        if debug_profile: print(f"[RIRI:None] camera static ratio={ratio_move:.3f} < thr={camera_move_thr}")
        return None

    # 4) 估计 FoE（不固定 FoE，逐帧估计以构造“查找图”）
    T = time.perf_counter()
    mask = mag > min_flow_mag
    foe_est = _foe_lsq(flow, mask, min_mag=min_flow_mag)
    if foe_est is None:
        if debug_profile: print("[RIRI:None] LSQ FoE failed")
        return None
    foe, sign = foe_est["foe"], foe_est["sign"]
    t_foe = _ms(T)

    # 5) 比值不变量残差（角度差在 -90~90 度内等价于 arctan 比值差）
    T = time.perf_counter()
    Hs, Ws = c_g.shape[:2]
    X, Y = np.meshgrid(np.arange(Ws, dtype=np.float32),
                       np.arange(Hs, dtype=np.float32))
    vF = np.stack([(X - foe[0]) * sign, (Y - foe[1]) * sign], axis=-1)  # 期望径向（带正负）
    F  = flow

... (truncated)
```

### `cv/strategy_templatematch.py`

```text
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
```

### `cv/strategy_twodiff_orb.py`

```text
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
```

### `cv/strategy_yolo.py`

```text
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
... (truncated)
```

### `cv/yolo/check_leaveoneout_runs.py`

```text
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
检查 leave-one-group-out 的 YOLO 训练结果是否完整。

用法示例：
  python check_leaveoneout_runs.py \
    --project runs/app_ood_leave1out \
    --ood-type app-ood \
    --k 22
"""

import argparse
from pathlib import Path
from typing import Dict, List


def find_run_dir_for_idx(project: Path, ood_type: str, idx: int) -> List[Path]:
    """
    在 project 目录下查找名称以 "{ood_type}_idx{idx}_" 开头的子目录。
    例如：app-ood_idx2_11
    """
    prefix = f"{ood_type}_idx{idx}_"
    matches: List[Path] = []
    for d in project.iterdir():
        if d.is_dir() and d.name.startswith(prefix):
            matches.append(d)
    return matches


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check incomplete leave-one-group-out YOLO runs."
    )
    parser.add_argument(
        "--project",
        type=str,
        required=True,
        help="YOLO 训练的 project 目录（与 yolo_train_leave_one_group.py 中的 --project 一致）。",
    )
    parser.add_argument(
        "--ood-type",
        type=str,
        required=True,
        choices=["app-ood", "scene-ood"],
        help="当前检查的 OOD 类型。",
    )
    parser.add_argument(
        "--k",
        type=int,
        required=True,
        help="总的 group 数量（例如 app 一共有 22 个）。",
    )
    parser.add_argument(
        "--check-best",
        action="store_true",
        help="额外检查每个 run 的 weights/best.pt 是否存在（默认只检查目录存在）。",
    )

    args = parser.parse_args()

    project = Path(args.project)
    if not project.exists():
        print(f"[ERROR] project 目录不存在: {project}")
        return

    missing_run: List[int] = []
    multi_run: Dict[int, List[str]] = {}
    no_best: List[int] = []
    ok: List[int] = []

    for idx in range(args.k):
        runs = find_run_dir_for_idx(project, args.ood_type, idx)
        if not runs:
            missing_run.append(idx)
            continue
        if len(runs) > 1:
            multi_run[idx] = [str(d) for d in runs]

        # 默认认为只要目录存在，就算“有训练记录”
        run_dir = runs[0]

        if args.check-best:
            best_path = run_dir / "weights" / "best.pt"
            if not best_path.exists():
                no_best.append(idx)
            else:
                ok.append(idx)
        else:
            ok.append(idx)

    print("\n========== CHECK SUMMARY ==========")
    print(f"project   : {project}")
    print(f"ood_type  : {args.ood_type}")
    print(f"k (total) : {args.k}")

    print(f"\n[OK] 有训练目录的 index: {sorted(ok)}")

    if missing_run:
        print(f"\n[WARN] 缺少 run 目录的 index: {missing_run}")
    else:
        print("\n[OK] 所有 index 都有对应的 run 目录。")

    if multi_run:
        print("\n[WARN] 某些 index 对应多个 run 目录（可能重复运行）:")
        for idx, paths in multi_run.items():
            print(f"  idx={idx}:")
            for p in paths:
                print(f"    - {p}")

    if args.check-best:
        if no_best:
            print(f"\n[WARN] 没有 weights/best.pt 的 index: {no_best}")
        else:
            print("\n[OK] 所有 run 都包含 weights/best.pt。")

    print("\n========== END ==========")


... (truncated)
```

### `cv/yolo/check_splits_groups.py`

```text
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
检查 .split（或指定 splits 目录）下每个 split 子目录中：
1. train/val/test 是否在“文件级别”互斥；
2. 指定的 group 列（如 App/Object/Scene）在“group 级别”是否互斥。

- 文件级别：同一图片路径是否同时出现在 train/val/test；
- group 级别：例如 App 维度上，train 与 test 是否共享同一个 App。

使用示例：

python check_splits_groups.py \
  --split-outdir .split \
  --data-csv data_stat.csv \
  --group-cols App,Object,Scene

如只想检查 App：

python check_splits_groups.py \
  --split-outdir .split \
  --data-csv data_stat.csv \
  --group-cols App
"""

import argparse
from pathlib import Path
from typing import Dict, Set, Tuple, List

import pandas as pd


# ---------- 通用小工具 ----------

def load_list(path: Path) -> Set[str]:
    """读取一个 txt 文件，每行一个路径，返回去重后的 set。"""
    if not path.exists():
        return set()
    lines = path.read_text(encoding="utf-8").splitlines()
    return {ln.strip() for ln in lines if ln.strip()}


def _find_filename_column(df: pd.DataFrame) -> str:
    """根据列名自动推测 filename 列。"""
    lower_map = {c.lower(): c for c in df.columns}
    for key in ["filename", "file", "image", "img"]:
        if key in lower_map:
            return lower_map[key]
    # 没有匹配到时，用第一列，但给出提醒
    print(f"[WARN] 未找到典型 filename 列，默认使用第一列: {df.columns[0]!r}")
    return df.columns[0]


# ---------- group 映射构建 ----------

def build_name_to_groups(
    data_csv: Path,
    group_cols: List[str],
) -> Tuple[Dict[str, Dict[str, str]], List[str]]:
    """
    从 data_csv 读取 filename 和指定 group 列，构建映射：

    name_to_groups[basename][group_col] = group_id

    返回:
      - name_to_groups: { "xxx.jpg": { "App": "xxxApp", "Object": "obj1", ... }, ... }
      - valid_group_cols: 实际在 CSV 中存在的 group 列（自动过滤不存在的）
    """
    df = pd.read_csv(data_csv)
    if df.empty:
        raise RuntimeError(f"CSV 为空: {data_csv}")

    filename_col = _find_filename_column(df)
    lower_map = {c.lower(): c for c in df.columns}

    # 过滤出真正存在的 group 列
    valid_group_cols: List[str] = []
    for g in group_cols:
        if g in df.columns:
            valid_group_cols.append(g)
        elif g.lower() in lower_map:
            valid_group_cols.append(lower_map[g.lower()])
        else:
            print(f"[WARN] group 列 {g!r} 不在 CSV 中，跳过该列。")

    if not valid_group_cols:
        print("[WARN] 未找到任何有效的 group 列，不会进行 group 级别检查。")
        return {}, []

    print(f"[INFO] 使用 filename 列: {filename_col!r}")
    print(f"[INFO] 使用 group 列: {valid_group_cols}")

    name_to_groups: Dict[str, Dict[str, str]] = {}

    for _, row in df.iterrows():
        raw_name = str(row[filename_col])
        basename = Path(raw_name).name  # 仅使用文件名部分
        grp_dict = name_to_groups.setdefault(basename, {})
        for col in valid_group_cols:
            grp_dict[col] = str(row[col])

    return name_to_groups, valid_group_cols


# ---------- 单个 split 目录检查 ----------

def check_one_split_dir_files(split_dir: Path) -> Tuple[bool, str]:
    """
    检查单个 split 子目录中 train/val/test 在“文件级别”是否互斥。

    返回:
      ok, msg
        ok = True  : 互斥
        ok = False : 存在重复文件
    """
    train_txt = split_dir / "train.txt"
    val_txt = split_dir / "val.txt"
    test_txt = split_dir / "test.txt"

... (truncated)
```

### `cv/yolo/check_splits_mutual_exclusion.py`

```text
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
检查 .split（或指定 splits 目录）下每个 split 子目录中
train/val/test 的数据是否互斥。

- 若 train/val/test 之间有图片路径重叠，则视为“数据泄露”，打印该目录名。
- 默认只检查文件级别（同一图片路径是否被重复使用）。
- 使用方式示例：

  python check_splits_mutual_exclusion.py --split-outdir .split

"""

import argparse
from pathlib import Path
from typing import Set, Tuple


def load_list(path: Path) -> Set[str]:
    """读取一个 txt 文件，每行一个路径，返回去重后的 set。"""
    if not path.exists():
        return set()
    lines = path.read_text(encoding="utf-8").splitlines()
    return {ln.strip() for ln in lines if ln.strip()}


def check_one_split_dir(split_dir: Path) -> Tuple[bool, str]:
    """
    检查单个 split 子目录中 train/val/test 是否互斥。

    返回值:
      (ok, msg)
        ok = True  : 互斥（没有重叠）
        ok = False : 存在重叠（数据泄露），msg 中包含细节
    """
    train_txt = split_dir / "train.txt"
    val_txt = split_dir / "val.txt"
    test_txt = split_dir / "test.txt"

    # 没有 train.txt 的目录直接跳过（可能是 last.yaml 等）
    if not train_txt.exists():
        return True, "no train.txt, skipped"

    train_set = load_list(train_txt)
    val_set = load_list(val_txt)
    test_set = load_list(test_txt)

    # 至少有一个集合为空时，也照样检查，但一般视为“没有泄露”
    inter_tr_va = train_set & val_set
    inter_tr_te = train_set & test_set
    inter_va_te = val_set & test_set

    leak = False
    parts = []

    if inter_tr_va:
        leak = True
        parts.append(f"train ∩ val = {len(inter_tr_va)}")
    if inter_tr_te:
        leak = True
        parts.append(f"train ∩ test = {len(inter_tr_te)}")
    if inter_va_te:
        leak = True
        parts.append(f"val ∩ test = {len(inter_va_te)}")

    if leak:
        detail = "; ".join(parts)
        return False, detail

    return True, "OK"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="检查 splits 目录下 train/val/test 是否互斥（文件级别）。"
    )
    parser.add_argument(
        "--split-outdir",
        type=str,
        default=".split",
        help="保存各个 split 子目录的根目录（默认 .split）",
    )

    args = parser.parse_args()
    split_root = Path(args.split_outdir)

    if not split_root.exists():
        raise FileNotFoundError(f"split-outdir 不存在: {split_root}")

    leak_dirs = []
    checked = 0
    skipped = 0

    print(f"[INFO] 检查 splits 根目录: {split_root}")

    for d in sorted(split_root.iterdir(), key=lambda p: p.name):
        if not d.is_dir():
            continue

        ok, msg = check_one_split_dir(d)
        if msg == "no train.txt, skipped":
            skipped += 1
            continue

        checked += 1
        if ok:
            print(f"[OK]    {d.name}: {msg}")
        else:
            print(f"[LEAK]  {d.name}: {msg}")
            leak_dirs.append(d.name)

    print("\n========== SUMMARY ==========")
    print(f"已检查 split 子目录数量: {checked}")
    print(f"跳过（无 train.txt）数量: {skipped}")
    if not leak_dirs:
        print("✅ 未发现 train/val/test 之间的文件级数据泄露。")
    else:
        print("⚠ 发现以下目录存在数据泄露（train/val/test 有重复图片）：")
... (truncated)
```

### `cv/yolo/crossval_eval.py`

```text
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
对 K 折训练好的 YOLO 模型进行一键式 cross validation 评测。

逻辑：
- 根据 split-mode / seed / kfold / fold-index 规则，自动匹配：
  - .split 下对应的 test.txt
  - runs/train 下对应的 best.pt
- 对每个 fold 调用 eval_yolo_center_hit_debug.py 做评测
- 解析其中的 Success Rate，最后输出每折结果 + mean/std

使用示例（app-ood, k=5）：

python crossval_eval.py \
  --split-mode app-ood \
  --kfold 5 \
  --seed 10 \
  --weights-stem yolo12n \
  --split-outdir .split \
  --train-project runs/train \
  --eval-script eval_yolo_center_hit_debug.py \
  --labels-root /home/xy3371/Yolo/datasets/myar/labels \
  --class-id 0 --imgsz 640 --conf 0.05

"""

import argparse
import math
import re
import statistics
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple


def find_split_dir(
    split_outdir: Path,
    mode: str,
    seed: int,
    kfold: int,
    fold_index: int,
) -> Path:
    """在 split_outdir 下找到当前 fold 对应的目录."""
    prefix = f"{mode}_s{seed}_k{kfold}_f{fold_index}_"
    candidates = [
        d for d in split_outdir.iterdir()
        if d.is_dir() and d.name.startswith(prefix)
    ]
    if not candidates:
        raise FileNotFoundError(
            f"在 {split_outdir} 下找不到前缀为 {prefix!r} 的 split 目录，"
            f"请确认已经跑过对应 fold 的训练。"
        )
    # 如有多个，按名字排序选最后一个（时间戳更大）
    candidates.sort(key=lambda p: p.name)
    chosen = candidates[-1]
    print(f"[INFO] fold={fold_index}: 使用 split 目录: {chosen}")
    return chosen


def build_exp_name(
    mode: str,
    weights_stem: str,
    split_dir_name: str,
) -> str:
    """根据 split 目录名推导训练 exp 目录名."""
    # split_dir_name 例如: "app-ood_s42_k5_f0_1763082385"
    if not split_dir_name.startswith(f"{mode}_"):
        raise ValueError(
            f"split 目录名 {split_dir_name!r} 不以 {mode + '_'} 开头，无法推导 exp name"
        )
    suffix = split_dir_name[len(mode) + 1 :]  # 去掉 "app-ood_"
    exp_name = f"{mode}_{weights_stem}_{suffix}"
    return exp_name


def run_single_eval(
    eval_script: Path,
    image_list: Path,
    labels_root: Path,
    weights: Path,
    class_id: int,
    imgsz: int,
    conf: float,
    iou: float,
    project: Path,
    name: str,
    save_vis: bool = False,
) -> float:
    """调用 eval_yolo_center_hit_debug.py 做一次评测，并解析 Success Rate."""
    cmd = [
        "python",
        str(eval_script),
        "--image_list",
        str(image_list),
        "--labels-root",
        str(labels_root),
        "--weights",
        str(weights),
        "--class-id",
        str(class_id),
        "--imgsz",
        str(imgsz),
        "--conf",
        str(conf),
        "--iou",
        str(iou),
        "--project",
        str(project),
        "--name",
        name,
    ]
    if save_vis:
        cmd.append("--save-vis")

    print(f"[CMD] {' '.join(cmd)}")

    proc = subprocess.run(
... (truncated)
```

### `cv/yolo/eval_yolo_center_hit_debug.py`

```text
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
YOLO 评测脚本（中心点命中 GT）+ 调试信息 + 可视化错误样本 + Top-K 试错机制

功能：
1. 从 --image_list 或 --test-dir 读取待评测图片；
2. 检查：
   - list 文件是否存在；
   - list 中有多少“原始行 / 非空行”；
   - 有多少路径实际存在文件；
3. 对于每张图片：
   - 读取对应 label（YOLO txt，class cx cy w h）中的指定 class（--class-id）；
   - 用 YOLO 模型预测该图片；
   - 对指定 class：
       * 按置信度排序，取前 K（由 --topk 控制，默认 1）；
       * 只要前 K 个 box 中有任意一个的中心点落在 GT bbox 内，就算预测成功；
4. 输出详细统计 + CSV 报告 + （可选）可视化结果：
   - vis/success    : 命中 GT 的样本
   - vis/fail       : 有 GT 和预测，但中心未命中的样本
   - vis/no_label   : 没有 GT label 的样本
   - vis/no_pred    : 有 GT 但该类没有预测框的样本

示例：
python eval_yolo_center_hit_debug.py \
  --image_list .split/scene-ood_s11_k0_f0_1763737407/test.txt \
  --labels-root /home/xy3371/Yolo/datasets/myar/labels \
  --weights runs/train/scene-ood_yolo12n_s11_k0_f0_1763737407/weights/best.pt \
  --class-id 0 --imgsz 640 --conf 0.05 --topk 3 --save-vis
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


# ----------------- 工具函数 -----------------


def pick_device(prefer: str = "auto") -> str:
    import torch
    import platform

    if prefer != "auto":
        return prefer
    if platform.system() == "Darwin" and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def debug_load_list(list_file: Path) -> List[Path]:
    print(f"[DEBUG] 准备读取 image_list: {list_file}")
    if not list_file.exists():
        print(f"[ERROR] image_list 文件不存在：{list_file}")
        return []

    raw_lines: List[str] = list_file.read_text(encoding="utf-8").splitlines()
    print(f"[DEBUG] list 文件原始行数（包含空行）: {len(raw_lines)}")

    stripped = [ln.strip() for ln in raw_lines]
    non_empty = [ln for ln in stripped if ln]
    print(f"[DEBUG] list 文件非空行数: {len(non_empty)}")

    images: List[Path] = []
    missing: List[str] = []

    for ln in non_empty:
        p = Path(ln)
        if not p.is_absolute():
            # 相对路径按当前工作目录解析
            p = p.resolve()
        if p.exists():
            images.append(p)
        else:
            missing.append(str(p))

    print(f"[DEBUG] 实际存在的图片数量: {len(images)}")
    print(f"[DEBUG] 丢失的图片数量: {len(missing)}")

    if images:
        print("[DEBUG] 前 5 个有效图片路径示例：")
        for p in images[:5]:
            print(f"         {p}")
    if missing:
        print("[WARN] 前 5 个不存在的图片路径示例：")
        for s in missing[:5]:
            print(f"         {s}")

    return images


def draw_vis(
    img_path: Path,
    save_path: Path,
    gt_xyxy: Optional[Tuple[float, float, float, float]],
    pred_xyxy: Optional[Tuple[float, float, float, float]],
    pred_center: Optional[Tuple[float, float]],
) -> None:
    """
    可视化一张图片：
    - 绿色框：GT bbox
    - 红色框：预测 bbox（top-1）
    - 红点：用于评估的预测中心（可以是 top-1 或命中的那个）
    """
    im = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(im)

    # GT 框（绿色）
    if gt_xyxy is not None:
... (truncated)
```

### `cv/yolo/inspect_app_column.py`

```text
import pandas as pd
import unicodedata

df = pd.read_csv("data_stat.csv")

print("=== 原始唯一值列表（repr 显示隐藏字符）===\n")
unique_vals = df["App"].astype(str).unique()
for v in unique_vals:
    print(repr(v))

print("\n=== 清洗前，每个 App 值的计数 ===\n")
print(df["App"].astype(str).value_counts())

print("\n=== 显示每个 App 字符的 Unicode 码位（帮助定位隐藏字符）===\n")

def show_unicode(s):
    return " ".join(f"{c}({ord(c):04X})" for c in s)

for v in unique_vals:
    print(f"{repr(v)} → {show_unicode(v)}")
```

### `cv/yolo/leaveoneout_eval.py`

```text
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Leave-One-Group-Out YOLO evaluator

功能：
- 针对 leave-one-out 版本的 app-ood / scene-ood 训练结果，
  自动遍历对应 split 目录和训练权重，调用 eval_yolo_center_hit_debug.py，
  解析每个 group 的 Success Rate，最后输出整体统计（mean / std）。

假设目录结构：
  splits_root/
    app-ood_1/
      test.txt
      labels_root.txt
    app-ood_2/
      test.txt
      labels_root.txt
    ...
  train_project/
    app-ood_idx0_1/
      weights/best.pt
    app-ood_idx1_2/
      weights/best.pt
    ...

使用示例（app-ood）：
  python leaveoneout_eval.py \
    --ood-type app-ood \
    --splits-dir splits_leave1out_app \
    --train-project runs/app_ood_leave1out \
    --eval-script eval_yolo_center_hit_debug.py \
    --class-id 0 --imgsz 640 --conf 0.05 --iou 0.6 --topk 1

使用示例（scene-ood）：
  python leaveoneout_eval.py \
    --ood-type scene-ood \
    --splits-dir splits_leave1out_scene \
    --train-project runs/scene_ood_leave1out \
    --eval-script eval_yolo_center_hit_debug.py \
    --class-id 0 --imgsz 640 --conf 0.05 --iou 0.6 --topk 1
"""

import argparse
import re
import statistics
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def find_split_dirs_for_ood(splits_root: Path, ood_type: str) -> List[Path]:
    """
    在 splits_root 下查找当前 ood_type 对应的所有 split 目录。
    目录名形如：app-ood_11 / scene-ood_LivingRoom 等。
    """
    prefix = f"{ood_type}_"
    dirs = [
        d for d in splits_root.iterdir()
        if d.is_dir() and d.name.startswith(prefix)
    ]
    dirs.sort(key=lambda p: p.name)
    return dirs


def find_train_dir_for_group(
    train_project: Path,
    ood_type: str,
    safe_group: str,
) -> Optional[Path]:
    """
    在 train_project 下查找给定 group 对应的训练目录。

    假定训练目录名形如：
      app-ood_idx2_11
    即：以 "{ood_type}_idx" 开头，以 "_{safe_group}" 结尾。
    """
    prefix = f"{ood_type}_idx"
    suffix = f"_{safe_group}"

    candidates: List[Path] = []
    for d in train_project.iterdir():
        if not d.is_dir():
            continue
        name = d.name
        if name.startswith(prefix) and name.endswith(suffix):
            candidates.append(d)

    if not candidates:
        return None

    # 如有多个（比如重复训练），按名称排序选最后一个
    candidates.sort(key=lambda p: p.name)
    return candidates[-1]


def run_single_eval(
    eval_script: Path,
    image_list: Path,
    labels_root: Path,
    weights: Path,
    class_id: int,
    imgsz: int,
    conf: float,
    iou: float,
    topk: int,
    project: Path,
    name: str,
    save_vis: bool = False,
) -> float:
    """
    调用 eval_yolo_center_hit_debug.py 做一次评测，并解析 Success Rate。
    """

    cmd = [
        "python",
        str(eval_script),
        "--image_list",
        str(image_list),
... (truncated)
```

### `cv/yolo/suggest_k.py`

```text
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根据 data_stat.csv 和某个 group 列（App/Object/Scene 等），
自动推荐一个较合适的 K，用于 GroupKFold。

示例用法：
python suggest_k.py --data-csv data_stat.csv --group-col App
python suggest_k.py --data-csv data_stat.csv --group-col Scene
python suggest_k.py --data-csv data_stat.csv --group-col Object

注意：
- 本脚本的评估逻辑和训练脚本中的 KFold 保持一致：
  * 在 “unique groups” 上做 GroupKFold（每个 group 权重相同）
  * 再用 group -> 样本数 来计算每折的验证集样本比例
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


# ----------------- 数据加载：group+样本数 -----------------


def load_group_counts(
    data_csv: Path,
    group_col: str,
) -> Tuple[np.ndarray, Dict[str, int], int]:
    """
    从 CSV 中读取 group 列，返回：

    - unique_groups: np.ndarray[str]，所有不重复 group id
    - group_counts: dict[group -> 样本数量]
    - total_samples: int，总样本数（行数）

    会做一层列名的大小写容错。
    """
    df = pd.read_csv(data_csv)
    if df.empty:
        raise RuntimeError(f"CSV 为空：{data_csv}")

    # 列名大小写容错
    lower_map = {c.lower(): c for c in df.columns}
    if group_col not in df.columns:
        if group_col.lower() not in lower_map:
            raise ValueError(f"CSV 中找不到列 {group_col!r}，实际列有：{list(df.columns)}")
        group_col = lower_map[group_col.lower()]

    groups_series = df[group_col].astype(str)
    total_samples = int(len(groups_series))

    # group -> 样本数量
    group_counts_series = groups_series.value_counts()
    unique_groups = group_counts_series.index.astype(str).to_numpy()
    group_counts = {g: int(group_counts_series[g]) for g in group_counts_series.index}

    return unique_groups, group_counts, total_samples


# ----------------- 评估某个 K -----------------


def evaluate_k_for_groups(
    unique_groups: np.ndarray,
    group_counts: Dict[str, int],
    total_samples: int,
    k: int,
    target_val_ratio: float,
    min_val_samples: int,
    min_val_groups: int,
    min_train_groups: int,
) -> Optional[Dict]:
    """
    在 unique_groups 上做一次 GroupKFold 评估，并基于“样本数”统计每折的验证集比例。

    参数：
    - unique_groups: 所有 group id（每个只出现一次）
    - group_counts: group -> 样本数量
    - total_samples: 数据集中总样本数
    - k: 候选的 K
    - 其它参数为各种约束/目标

    返回：
    - 若不满足约束（val 过小 / group 太少等），返回 None
    - 否则返回统计信息 dict：
      {
        "k", "score",
        "val_ratio_mean", "val_ratio_std",
        "min_val_size", "min_val_groups", "min_train_groups"
      }
    """
    num_groups = len(unique_groups)
    if k > num_groups:
        # K 不能大于 group 数
        return None

    # 在 “group 级别” 上做 GroupKFold
    gkf = GroupKFold(n_splits=k)
    dummy_X = np.zeros(num_groups)
    folds = gkf.split(dummy_X, y=None, groups=unique_groups)

    val_sample_sizes: List[int] = []
    val_group_counts: List[int] = []
    train_group_counts: List[int] = []

    for train_idx, val_idx in folds:
        # 当前折的 group id
        train_groups = unique_groups[train_idx]
        val_groups = unique_groups[val_idx]

        # 样本数：把这些 group 对应的样本数加起来
        val_size = int(sum(group_counts[g] for g in val_groups))
        train_size = int(sum(group_counts[g] for g in train_groups))

        val_sample_sizes.append(val_size)
... (truncated)
```

### `cv/yolo/sync_data_stat_with_images.py`

```text
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
同步 data_stat.csv 与实际 images 目录。

功能：
- 从 data_stat.csv 中删除那些在 images_root 下找不到对应图片文件的行。
- 仅根据“文件名”匹配（不看路径），即：
  - data_stat.csv 里的 filename 列如果是 "foo.jpg" 或 "subdir/foo.jpg"，
  - 都会被当成 "foo.jpg" 来和 images_root 下的实际文件名匹配。

用法示例：

python sync_data_stat_with_images.py \
  --csv data_stat.csv \
  --images-root /Users/yangxiaoyi/datasets/myar/images \
  --out data_stat_updated.csv

如果你想直接覆盖原文件（记得自己先备份）：
python sync_data_stat_with_images.py \
  --csv data_stat.csv \
  --images-root /Users/yangxiaoyi/datasets/myar/images \
  --inplace
"""

import argparse
from pathlib import Path
from typing import Set

import pandas as pd

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def collect_existing_filenames(images_root: Path) -> Set[str]:
    """收集 images_root 下所有图片的“文件名”（不含路径），用于匹配 CSV 的 filename 列。"""
    existing = set()
    for p in images_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            existing.add(p.name)  # 只保留文件名本身
    return existing


def find_filename_column(df: pd.DataFrame) -> str:
    """在 CSV 中找到 filename 列（大小写不敏感）。"""
    lower_map = {c.lower(): c for c in df.columns}
    for key in ["filename", "file", "image", "img"]:
        if key in lower_map:
            return lower_map[key]
    raise ValueError(
        f"data_stat.csv 中找不到 filename 列，实际列为：{list(df.columns)}。\n"
        f"请确保有一列名为 filename / Filename 等类似名称。"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="根据实际 images 目录，清理 data_stat.csv 中已经不存在的图片行。"
    )
    parser.add_argument("--csv", type=str, required=True, help="data_stat.csv 路径")
    parser.add_argument(
        "--images-root",
        type=str,
        required=True,
        help="图片根目录（会递归扫描其中所有图片文件）",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="输出 CSV 路径（默认不填则需要使用 --inplace 覆盖原文件）",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="直接覆盖原 CSV（请自行做好备份）。与 --out 互斥。",
    )

    args = parser.parse_args()

    csv_path = Path(args.csv)
    images_root = Path(args.images_root)

    if not csv_path.exists():
        raise FileNotFoundError(f"找不到 CSV 文件：{csv_path}")
    if not images_root.exists():
        raise FileNotFoundError(f"找不到图片根目录：{images_root}")

    if args.inplace and args.out is not None:
        raise ValueError("--inplace 与 --out 不能同时使用")

    # 1) 读取 CSV
    df = pd.read_csv(csv_path)
    fn_col = find_filename_column(df)

    print(f"[INFO] 读取 CSV：{csv_path}")
    print(f"[INFO] 使用列作为 filename：{fn_col!r}")
    n_before = len(df)

    # 2) 收集现有图片文件名
    existing_fns = collect_existing_filenames(images_root)
    print(f"[INFO] images_root 下找到图片数量：{len(existing_fns)}")

    # 3) 过滤：仅保留在 existing_fns 中的行
    def row_has_image(x: str) -> bool:
        if not isinstance(x, str):
            return False
        # 如果 CSV 里 filename 含有子路径，只取最后一段
        name = Path(x).name
        return name in existing_fns

    mask = df[fn_col].apply(row_has_image)
    df_kept = df[mask].copy()
    n_after = len(df_kept)
    n_removed = n_before - n_after

    print(f"[INFO] 原始行数：{n_before}")
    print(f"[INFO] 保留行数：{n_after}")
    print(f"[INFO] 删除行数：{n_removed}")
... (truncated)
```

### `cv/yolo/test_yolo_train.py`

```text
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
python test_yolo_train.py train --data myar_scene.yaml --weights yolo11n.pt --epochs 200 --imgsz 640
python test_yolo_train.py val   --data myar2.yaml --weights runs/train/exp/weights/best.pt
python test_yolo_train.py test  --test-dir /Users/yangxiaoyi/datasets/myar_scene/images/test --weights runs/train/exp5/weights/best.pt --conf 0.05

新增：有标签评测（中心点命中 GT 判成功）
python test_yolo_train.py eval --data myar2.yaml --weights runs/train/exp/weights/best.pt \
    --split test --class-id 0 --conf 0.05 --report-latex --save-vis
    
python test_yolo_train.py eval \
  --test-dir /Users/yangxiaoyi/datasets/myar_scene/images/test \
  --labels-dir /Users/yangxiaoyi/datasets/myar_scene/labels/test \
  --weights runs/train/exp5/weights/best.pt \
  --class-id 0 --conf 0.05 --report-latex --save-vis


也支持显式目录：
python test_yolo_train.py eval --test-dir path/to/images --labels-dir path/to/labels --weights ... --class-id 0
"""

import argparse
import sys
import platform
from tqdm import tqdm
from pathlib import Path
from ultralytics import YOLO
import time
from datetime import timedelta
import csv
import math
from typing import List, Tuple, Optional, Dict

from PIL import Image  # 用于读取图像宽高
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

def pick_device(prefer="auto"):
    # 简单设备选择：优先 MPS（Apple Silicon），其次 CUDA，最后 CPU
    if prefer != "auto":
        return prefer
    try:
        import torch
        if platform.system() == "Darwin" and torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"

def ensure_yaml(data_yaml: str):
    p = Path(data_yaml)
    if not p.exists():
        print(f"[WARN] data yaml 不存在：{p.resolve()}，请确认路径。")
    return str(p)

def cmd_train(args):
    device = pick_device(args.device)
    print(f"[INFO] device = {device}")

    model = YOLO(args.weights)

    # 计时开始
    _t0 = time.perf_counter()

    results = model.train(
        data=ensure_yaml(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        lr0=args.lr0,
        patience=args.patience,
        seed=args.seed,
        workers=args.workers,
        project=args.project,
        name=args.name,
        pretrained=True
    )

    # 计时结束并输出
    elapsed_s = time.perf_counter() - _t0
    avg_per_epoch = elapsed_s / max(1, args.epochs)
    print(results)
    print(f"[TIME] training wall time = {timedelta(seconds=int(elapsed_s))} "
          f"({elapsed_s:.2f}s), avg/epoch = {avg_per_epoch:.2f}s")

def cmd_val(args):
    device = pick_device(args.device)
    print(f"[INFO] device = {device}")

    model = YOLO(args.weights)
    metrics = model.val(
        data=ensure_yaml(args.data),
        split="val",              # 依赖 data.yaml 的 val
        imgsz=args.imgsz,
        device=device,
        iou=0.7,                  # 可按需调整
        conf=0.001,               # 评测用低阈值，做 PR 曲线更稳
        project=args.project,
        name=args.name
    )
    print(metrics)                # dict-like，含 mAP50、mAP50-95 等

def cmd_test(args):
    device = pick_device(args.device)
    print(f"[INFO] device = {device}")

    model = YOLO(args.weights)

    test_dir = Path(args.test_dir)
    # 统计测试集图片数量（递归）
    total_imgs = sum(1 for p in test_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS)

    if total_imgs == 0:
        print(f"[WARN] 在 {test_dir} 未找到图片（支持扩展名：{sorted(IMAGE_EXTS)}）")
        return
... (truncated)
```

### `cv/yolo/test_yolo_train_dynsplit.py`

```text
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dynamic split + YOLO training/testing script.

功能概述
--------
1. 从 data_stat.csv 动态划分 train/val/test：
   - random              : 随机划分样本（不看 group）
   - app-ood             : 按 App 列分组做 OOD
   - object-ood          : 按 Object 列分组做 OOD
   - scene-ood           : 按 Scene 列分组做 OOD
   - group-ood           : 按任意指定列分组做 OOD (--group-col)

2. 支持 group 级 K 折 (GroupKFold)，用于 OOD 稳定评估：
   - kfold=0 或 1 : 不使用 K 折
   - kfold>1      : 按 group 做 K 折，在当前折中再根据样本数切出 train/test

3. 生成的 data.yaml 固定为 2 类：
   - 0: AR_Object
   - 1: UI_Element

4. 训练输出目录自动命名为：
   <split-mode>_<weights-stem>_s<seed>_k<kfold>_f<fold>_<timestamp>
   例如：
   app-ood_yolo11n_s42_k0_f0_1763082385
   避免多任务/slurm 并发写同一 exp 目录。

5. 提供一个简单的 test 子命令做无标签批量推理。

用法示例：

# 1) 随机分割（不复制文件；从 CSV 读取文件名并写入三份 list.txt）
python test_yolo_train_dynsplit.py train \
  --images-root /Users/yangxiaoyi/datasets/myar/images \
  --labels-root /Users/yangxiaoyi/datasets/myar/labels \
  --data-csv data_stat.csv \
  --split-mode random --ratios 0.7,0.15,0.15 \
  --weights yolo11n.pt --epochs 100 --imgsz 640

# 2) App-OOD（按 app 组互斥）+ 5 折交叉验证第0折（train/val 来自 K 折；test 额外保留一批 app 组）
python test_yolo_train_dynsplit.py train \
  --images-root /Users/yangxiaoyi/datasets/myar/images \
  --labels-root /Users/yangxiaoyi/datasets/myar/labels \
  --data-csv data_stat.csv \
  --split-mode app-ood --group-col App \
  --ratios 0.7,0.15,0.15 --seed 10\
  --weights yolo11n.pt --epochs 100 --imgsz 640
  
  --kfold 5 --fold-index 0 

# 3) 仅评测（中心点命中 GT 算成功），从 list 文件读入
python test_yolo_train_dynsplit.py eval \
  --image-list split/random_s42_k0_f0_1763082385/test.txt \
  --labels-root /Users/yangxiaoyi/datasets/myar/labels \
  --weights runs/train/exp4/weights/best.pt \
  --class-id 0 --imgsz 640 --conf 0.05 --save-vis
"""

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from ultralytics import YOLO


# ----------------- 基本工具 -----------------


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def pick_device(prefer: str = "auto") -> str:
    import torch
    import platform

    if prefer != "auto":
        return prefer
    if platform.system() == "Darwin" and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@dataclass
class SplitLists:
    train: List[str]
    val: List[str]
    test: List[str]


# ----------------- 路径解析与 CSV 读取 -----------------

def _find_existing_split_yaml(args) -> Optional[Path]:
    """
    对于 KFold（kfold > 1），尝试在 split-outdir 下复用已有的 split 目录。
    返回其 data.yaml 路径；若不存在则返回 None。
    """
    # 只在 kfold>1 时尝试复用；random/单次划分不复用，保持原语义
    if not args.kfold or args.kfold <= 1:
        return None

    split_outdir = Path(args.split_outdir)
    if not split_outdir.exists():
        return None

    mode = args.split_mode
    seed = args.seed
    kfold = args.kfold
    fold_index = args.fold_index

    prefix = f"{mode}_s{seed}_k{kfold}_f{fold_index}_"
    candidates = [
... (truncated)
```

### `cv/yolo/yolo_train_leave_one_group.py`

```text
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
YOLO Leave-One-Group-Out Training Script
----------------------------------------

功能摘要：
- 只保留 train + test，不再有单独 val set。
- k = group（App 或 Scene）的去重个数。
- Leave-One-Group-Out：每次选一个 group 做 test，其余全部做 train。
- 两种运行模式：
  1) 不给 --kfold-index：只打印 k 和所有 group 的 index + 样本行数，直接退出；
  2) 给 --kfold-index：构建对应的 train/test/data.yaml，并（可选）启动 YOLO 训练。
- 支持 --dryrun：生成所有 split 文件和 data.yaml，但不真正启动训练。
- group 列统一使用 str.strip() 清洗，避免因空格等字符导致计数不一致。

python yolo_train_leave_one_group.py \
    --data-csv data_stat.csv \
    --ood-type scene-ood \
    --kfold-index 2 \
    --images-root /Users/yangxiaoyi/datasets/myar/images \
    --labels-root /Users/yangxiaoyi/datasets/myar/labels \
    --model yolov12n.pt \
    --dryrun

"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from ultralytics import YOLO


# -------------------- Utility functions --------------------


def pick_device(prefer: str = "auto") -> str:
    import torch
    import platform

    if prefer != "auto":
        return prefer
    if platform.system() == "Darwin" and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def set_global_seed(seed: int) -> None:
    """仅影响训练随机性，不影响数据划分。"""
    import random
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_csv(data_csv: Path) -> pd.DataFrame:
    if not data_csv.exists():
        raise FileNotFoundError(f"CSV not found: {data_csv}")
    df = pd.read_csv(data_csv)
    if df.empty:
        raise RuntimeError(f"CSV is empty: {data_csv}")
    return df


def _find_filename_column(df: pd.DataFrame) -> str:
    """自动推断存放文件名的列（参考 dynsplit 的风格）"""
    lower_map = {c.lower(): c for c in df.columns}
    for key in ["filename", "file", "image", "img"]:
        if key in lower_map:
            return lower_map[key]
    print(f"[WARN] 未找到 filename/file/image/img 列，默认使用第一列: {df.columns[0]}")
    return df.columns[0]


def _guess_group_col(df: pd.DataFrame, ood_type: str, group_col_arg: Optional[str]) -> str:
    """根据 ood_type 自动推断 group 列（App / Scene），或使用显式指定的列名。"""
    if group_col_arg is not None:
        if group_col_arg not in df.columns:
            raise ValueError(
                f"指定的 group 列 {group_col_arg!r} 不在 CSV 中。\n可用列: {list(df.columns)}"
            )
        return group_col_arg

    lower_map = {c.lower(): c for c in df.columns}

    if ood_type == "app-ood":
        for key in ["app", "app_id", "appname", "package", "package_name"]:
            if key in lower_map:
                return lower_map[key]
        raise ValueError("无法自动找到 App 列，请使用 --group-col 显式指定。")

    if ood_type == "scene-ood":
        for key in ["scene", "scene_id"]:
            if key in lower_map:
                return lower_map[key]
        raise ValueError("无法自动找到 Scene 列，请使用 --group-col 显式指定。")

    raise ValueError(f"不支持的 ood-type={ood_type}")


def _resolve_image_path(images_root: Path, name: str) -> Optional[str]:
    """根据 CSV 中的文件名解析真实图片路径（绝对路径）"""
    p = Path(name)

    # 绝对路径且存在
    if p.is_absolute() and p.exists():
        return str(p.resolve())

    # 相对于 images_root
... (truncated)
```

### `experiments/notebooks/framework_status.md`

```text
📁 AR Action Discovery Framework – Current Progress Summary

（可直接用于新对话的上下文）

1. 项目目标简述

本项目的最终目标是构建一个 自动化 AR Interaction Event Generation + Action Discovery 系统，完成：

AR 对象检测（YOLO）

手势采样（tap / drag / rotate / pinch）

多指手势注入（Appium dispatchGesture）

操作前后场景分析（FoELS + optical flow + geometry + SSIM）

N/M 策略统计操作支持情况

输出 Action Support Matrix（JSONL）

当前工程已完成基础骨架，但正式版本代码在 src/ 下尚未补齐。
已有代码大部分还在 common/、cv/、experiments/ 中。

2. 📦 当前项目目录结构（已改造成正式工程）

已包含：

src/：正式版模块的目标目录（仍是空骨架，需要逐个补齐代码）

common/、cv/、experiments/：旧版/实验代码（用于迁移参考）

configs/ad.yaml：Action Discovery 的配置文件

scripts/run_discovery.sh：运行入口脚本

3. 📌 模块状态与待办事项
3.1 Detector（YOLO Detector）
已有

YOLO 训练、推理代码在 cv/strategy_yolo.py

YOLO 模型、数据集等资源完整

待完成（src/detector/yolo_detector.py）

创建 YOLODetector 类

初始化模型（from ultralytics import YOLO）

detect(frame_bgr) → 返回统一格式：

{
  "objects": [
    {"id": 0, "cls": "AR_Object", "bbox": [x, y, w, h], "center_xy": [cx, cy], "score": 0.92}
  ]
}

3.2 Executor（Appium 多指手势执行器）
已有

完整的手势执行逻辑在 experiments/v3_ar_monkey_appium.py

输入事件相关工具：common/device.py、common/actions.py

待完成（src/executor/appium_executor.py）

封装 Appium driver 建立逻辑

snapshot_screen() → BGR numpy array

perform(op, region, params) → 调用多指手势注入执行一次操作

所有截图/执行都由该类统一管理

3.3 Sampler（采样模块）
已有

随机策略在 common/policy_random.py

待完成（src/sampler/default_sampler.py）

增加 sample(op, region) 方法

drag：方向/距离

rotate：角度/半径

pinch：scale_sign（in/out）

tap：抖动半径

3.4 Verifier（多证据验证器）
已有

src/verifier/backends/motion_similarity.py：几何+光流验证的完整后端

common/verify_motion.py：旧版（参考）

待完成（src/verifier/verifier.py）

将 YOLO bbox/center 与前后帧交给 motion_similarity

构造 extra 参数（像素阈值等）

返回 success, evidence, metrics

未来可加入 FoELS / SSIM / optical flow 多通道融合

3.5 Policy（N/M 判定）
已有

src/policy/policy.py 已有基本骨架

待完成

确保 decide_support(op, trial_results) 正常返回布尔值即可（简单部分）

3.6 Discovery（总控流程）
已有

... (truncated)
```

### `experiments/playground/delete_images_without_labels.py`

```text
# -*- coding: utf-8 -*-
"""
删除没有对应 label 的图片：
- 支持递归遍历 images_dir
- 支持 labels_dir 与 images_dir 分离（YOLO 常见结构）
- 匹配规则：图片 basename == 标签 basename（忽略扩展名与大小写）
- 识别的图片扩展名：.jpg .jpeg .png .bmp .webp

用法示例：
python delete_images_without_labels.py myar/images/train myar/labels/train
# 若 labels 就在同一目录（旁边同名 .txt），则：
python delete_images_without_labels.py /path/to/images
"""

import sys
from pathlib import Path
from typing import Optional, Tuple, Set

IMAGE_EXTS: Set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def collect_label_stems(labels_root: Path) -> Set[str]:
    stems: Set[str] = set()
    for p in labels_root.rglob("*.txt"):
        stems.add(p.stem.lower())
    return stems

def delete_orphan_images(images_root: Path, labels_root: Optional[Path] = None) -> Tuple[int, int]:
    deleted, kept = 0, 0

    if labels_root is None:
        # 同目录：图片旁边找同名 .txt
        for img in images_root.rglob("*"):
            if img.is_file() and img.suffix.lower() in IMAGE_EXTS:
                txt = img.with_suffix(".txt")
                if not txt.exists():
                    try:
                        img.unlink()
                        deleted += 1
                        print(f"DELETE: {img}")
                    except Exception as e:
                        print(f"ERROR deleting {img}: {e}")
                else:
                    kept += 1
        return deleted, kept

    # 分离目录：预收集 labels 的 basename
    label_stems = collect_label_stems(labels_root)
    if not label_stems:
        print(f"WARNING: labels_dir `{labels_root}` 中未发现任何 .txt 标签文件。")

    for img in images_root.rglob("*"):
        if img.is_file() and img.suffix.lower() in IMAGE_EXTS:
            if img.stem.lower() not in label_stems:
                try:
                    img.unlink()
                    deleted += 1
                    print(f"DELETE: {img}")
                except Exception as e:
                    print(f"ERROR deleting {img}: {e}")
            else:
                kept += 1

    return deleted, kept

def main() -> None:
    if len(sys.argv) < 2:
        print("用法：python delete_images_without_labels.py <images_dir> [labels_dir]")
        sys.exit(1)

    images_dir = Path(sys.argv[1]).resolve()
    labels_dir = Path(sys.argv[2]).resolve() if len(sys.argv) >= 3 else None

    if not images_dir.is_dir():
        print(f"❌ images_dir 无效：{images_dir}")
        sys.exit(1)
    if labels_dir is not None and not labels_dir.is_dir():
        print(f"❌ labels_dir 无效：{labels_dir}")
        sys.exit(1)

    deleted, kept = delete_orphan_images(images_dir, labels_dir)
    print(f"\n✅ 处理完成：删除 {deleted} 张图片，保留 {kept} 张图片。")

if __name__ == "__main__":
    main()
```

### `experiments/playground/diagnose_orphan_labels.py`

```text
# -*- coding: utf-8 -*-
"""
诊断 labels 比 images 多 1 的原因：
- 递归收集两边的“基名集合”（忽略扩展名、大小写）
- 打印数量统计与差集
- 兼容 .TXT/.JPG 等大小写
- 可选择忽略诸如 classes.txt 等“非标注”文本

用法：
  仅诊断：
    python diagnose_orphan_labels.py /path/to/labels /path/to/images
  忽略某些文件名（逗号分隔，基名或完整文件名都可）：
    python diagnose_orphan_labels.py myar/images/train myar/labels/train --ignore classes.txt,readme.txt
"""

import sys
from pathlib import Path
from typing import Set, List

IMAGE_EXTS: Set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LABEL_EXT: str = ".txt"

def parse_ignore_list(arg: str) -> Set[str]:
    items = set()
    for s in arg.split(","):
        s = s.strip()
        if s:
            items.add(s.lower())
    return items

def collect_image_stems(images_root: Path) -> Set[str]:
    stems: Set[str] = set()
    for p in images_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            stems.add(p.stem.lower())
    return stems

def collect_label_stems(labels_root: Path, ignore: Set[str]) -> Set[str]:
    stems: Set[str] = set()
    for p in labels_root.rglob("*"):
        if p.is_file() and p.suffix.lower() == LABEL_EXT:
            # 忽略名单支持：完整文件名 或 仅基名
            name_l = p.name.lower()
            stem_l = p.stem.lower()
            if name_l in ignore or stem_l in ignore:
                continue
            stems.add(stem_l)
    return stems

def main() -> None:
    if len(sys.argv) < 3:
        print("用法：python diagnose_orphan_labels.py <labels_dir> <images_dir> [--ignore name1,name2]")
        sys.exit(1)

    labels_dir = Path(sys.argv[1]).resolve()
    images_dir = Path(sys.argv[2]).resolve()
    ignore: Set[str] = set()
    if len(sys.argv) >= 4 and sys.argv[3].startswith("--ignore"):
        parts = sys.argv[3].split("=", 1)
        if len(parts) == 2:
            ignore = parse_ignore_list(parts[1])
        elif len(sys.argv) >= 5:
            ignore = parse_ignore_list(sys.argv[4])

    if not labels_dir.is_dir():
        print(f"❌ labels_dir 无效：{labels_dir}")
        sys.exit(1)
    if not images_dir.is_dir():
        print(f"❌ images_dir 无效：{images_dir}")
        sys.exit(1)

    img_stems = collect_image_stems(images_dir)
    lab_stems = collect_label_stems(labels_dir, ignore)

    print("—— 统计 ——")
    print(f"images_dir: {images_dir}")
    print(f"labels_dir: {labels_dir}")
    print(f"忽略名单: {sorted(ignore) if ignore else '（无）'}")
    print(f"图片数量（按基名去重）: {len(img_stems)}")
    print(f"标签数量（按基名去重）: {len(lab_stems)}")
    diff_lab = sorted(lab_stems - img_stems)
    diff_img = sorted(img_stems - lab_stems)

    print("\n—— labels有而images没有（疑似多余label）——")
    if diff_lab:
        for s in diff_lab[:50]:
            print("  +", s)
        if len(diff_lab) > 50:
            print(f"  ... 以及 {len(diff_lab)-50} 个更多")
    else:
        print("  无")

    print("\n—— images有而labels没有（疑似缺失label的图片）——")
    if diff_img:
        for s in diff_img[:50]:
            print("  -", s)
        if len(diff_img) > 50:
            print(f"  ... 以及 {len(diff_img)-50} 个更多")
    else:
        print("  无")

    # 小结判断
    print("\n—— 结论 ——")
    if diff_lab:
        print(f"labels 确实多出 {len(diff_lab)} 个基名。")
    elif diff_img:
        print(f"labels 并不比 images 多；相反缺少 {len(diff_img)} 个标签。")
    else:
        print("两边基名数量一致。如果你仍看到“文件数相差 1”，很可能是被计入了非图片/非标签文件（如 classes.txt、README.txt、.DS_Store），或统计时跨了 train/val 分区。")

if __name__ == "__main__":
    main()
```

### `experiments/playground/find_orphan_labels.py`

```text
# -*- coding: utf-8 -*-
"""
在指定 labels_dir 下递归查找“没有对应同名图片”的 .txt 标签文件。
- 默认只列出这些“孤儿 label”
- 传 --delete 则会直接删除它们
- 支持常见图片扩展名：.jpg .jpeg .png .bmp .webp

用法：
  仅列出：
    python find_orphan_labels.py myar/images/train myar/labels/train
  直接删除：
    python find_orphan_labels.py /path/to/labels /path/to/images --delete
"""

import sys
from pathlib import Path
from typing import Set, Tuple

IMAGE_EXTS: Set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def collect_image_stems(images_root: Path) -> Set[str]:
    stems: Set[str] = set()
    for p in images_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            stems.add(p.stem.lower())
    return stems

def find_orphan_labels(labels_root: Path, images_root: Path, do_delete: bool = False) -> Tuple[int, int]:
    image_stems = collect_image_stems(images_root)
    if not image_stems:
        print(f"WARNING: images_dir `{images_root}` 中未发现任何图片文件。")

    total, orphans = 0, 0
    for txt in labels_root.rglob("*.txt"):
        total += 1
        if txt.stem.lower() not in image_stems:
            orphans += 1
            if do_delete:
                try:
                    txt.unlink()
                    print(f"DELETE: {txt}")
                except Exception as e:
                    print(f"ERROR deleting {txt}: {e}")
            else:
                print(f"ORPHAN: {txt}")

    return orphans, total

def main() -> None:
    if len(sys.argv) < 3:
        print("用法：python find_orphan_labels.py <labels_dir> <images_dir> [--delete]")
        sys.exit(1)

    labels_dir = Path(sys.argv[1]).resolve()
    images_dir = Path(sys.argv[2]).resolve()
    do_delete = (len(sys.argv) >= 4 and sys.argv[3] == "--delete")

    if not labels_dir.is_dir():
        print(f"❌ labels_dir 无效：{labels_dir}")
        sys.exit(1)
    if not images_dir.is_dir():
        print(f"❌ images_dir 无效：{images_dir}")
        sys.exit(1)

    orphans, total = find_orphan_labels(labels_dir, images_dir, do_delete)
    action = "删除" if do_delete else "发现"
    print(f"\n✅ 完成：{action} 孤儿 label {orphans} 个 / 标签总数 {total} 个。")

if __name__ == "__main__":
    main()
```

### `experiments/playground/fix_label_to_zero.py`

```text
import os

def fix_labels_to_zero(folder):
    """
    批量将 YOLO 标签文件中的类别 ID 改为 0
    folder: 存放 .txt 标签文件的目录
    """
    count_files = 0
    count_lines = 0

    for filename in os.listdir(folder):
        if filename.endswith(".txt"):
            path = os.path.join(folder, filename)
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    parts[0] = "0"  # 把第一个数字改成 0
                    new_lines.append(" ".join(parts) + "\n")
                    count_lines += 1
                else:
                    # 跳过空行或格式异常的行
                    new_lines.append(line)

            with open(path, "w", encoding="utf-8") as f:
                f.writelines(new_lines)

            count_files += 1

    print(f"✅ 已修改 {count_files} 个文件，共 {count_lines} 行标签为 0。")

if __name__ == "__main__":
    folder = input("请输入标签文件所在的文件夹路径: ").strip()
    if os.path.isdir(folder):
        fix_labels_to_zero(folder)
    else:
        print("❌ 路径不存在，请检查输入。")
```

### `experiments/v0_v1_v2_v3_archive/smoke.py`

```text
#!/usr/bin/env python3
import argparse, subprocess, time, re, sys

PKG = "com.google.ar.core.examples.java.hellorecordingplayback"
ACT = ".HelloRecordingPlaybackActivity"  # 也可用完整名：com.google.ar.core.examples.java.hellorecordingplayback.HelloRecordingPlaybackActivity

def run(cmd):
    return subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT).decode("utf-8","ignore")

def start_app(serial=None):
    base = f"adb -s {serial} " if serial else "adb "
    # 启动：包名/Activity 支持以点开头的简写
    run(base + f"shell am start -n {PKG}/{ACT}")

def top_component(serial=None):
    base = f"adb -s {serial} " if serial else "adb "
    # 优先从 activity dumpsys 抓 topResumedActivity，抓不到再看 window 的 mCurrentFocus
    out = run(base + "shell dumpsys activity")
    m = re.search(r'topResumedActivity.*? ([\w\.]+)/([\w\.$]+)', out)
    if not m:
        m = re.search(r'mResumedActivity.*? ([\w\.]+)/([\w\.$]+)', out)
    if m:
        return m.group(1), m.group(2)
    out = run(base + "shell dumpsys window")
    m = re.search(r'mCurrentFocus.*? ([\w\.]+)/([\w\.$]+)', out)
    if m:
        return m.group(1), m.group(2)
    return None, None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serial", help="设备序列号（adb devices 可查看，多设备时必须指定）")
    args = ap.parse_args()

    # 启动 App
    start_app(args.serial)
    time.sleep(3.0)  # 等摄像头/AR会话初始化

    # 校验前台是否在目标包
    tpkg, tact = top_component(args.serial)
    if tpkg is None:
        print("✖ 读取前台 Activity 失败（权限或系统限制）。")
        sys.exit(2)

    print(f"前台：{tpkg}/{tact}")
    if tpkg == PKG:
        print("✓ SMOKE OK：App 已启动并处于前台。")
        sys.exit(0)
    else:
        print("✖ SMOKE FAIL：前台不在目标包。")
        sys.exit(1)

if __name__ == "__main__":
    main()

  
```

### `experiments/v0_v1_v2_v3_archive/v0_ar_monkey_adb.py`

```text
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
... (truncated)
```

### `experiments/v0_v1_v2_v3_archive/v0_ar_monkey_appium.py`

```text
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v0_ar_monkey_appium.py
Appium baseline（随机手势），接入 verify_motion（含 tap）与 timing：
- 启动前清空 logcat（adb logcat -c），采集从“当前时刻”（-T 0）开始
- 定时打印与最终打印：place/drag/pinch/rotate/tap + overall
- 统计：程序总时长 / 模拟操作时长 / 平均每操作耗时（= sim_runtime / rounds）

用法示例：
  python v0_ar_monkey_appium.py --pkg com.google.ar.sceneform.samples.hellosceneform \
    --activity auto --rounds 50 --print-interval 10
"""

import argparse, random, time, sys, threading, subprocess
from common.device import make_driver, get_window_size, resolve_main_activity
from common.actions import tap, drag_line, long_press, pinch_or_zoom, rotate
from common.policy_random import step as random_step

from common.verify_motion import MotionVerifier
from common.timing import Timing

# ----------------- 工具 -----------------
def clear_logcat():
    subprocess.run(["adb", "logcat", "-c"], check=False)

def periodic_print(ver: MotionVerifier, interval_sec: int, stop_evt: threading.Event):
    while not stop_evt.is_set():
        time.sleep(interval_sec)
        if stop_evt.is_set():
            break
        print("[tick]\n" + ver.summary_str(), flush=True)

# ----------------- 主流程 -----------------
def run_monkey(pkg="com.rooom.app",
               activity="auto",
               serial=None,
               rounds=250,
               sleep_min=0.25, sleep_max=0.85,
               safe=(0.12, 0.18, 0.88, 0.88),
               drag_ms=(300, 900), long_ms=(700, 1500),
               pinch_start=80, pinch_end=220,
               rotate_radius=(160, 260), rotate_angle=(30, 90), rotate_steps=8,
               warmup_wait=3.0, seed=None,
               print_interval=10,
               overall_include_tap=False):

    if seed is not None:
        random.seed(seed)

    # 计时：程序总时长从此开始
    tim = Timing()

    # 清空旧日志，启动 verify_motion（从当前时刻），可带 serial（多设备时可用）
    clear_logcat()
    ver = MotionVerifier(include_tap_in_overall=overall_include_tap)
    ver.start_from_adb(serial=serial, extra_logcat_args=["-T", "0"])

    # 定时打印线程
    stop_evt = threading.Event()
    printer = None
    if print_interval and print_interval > 0:
        printer = threading.Thread(target=periodic_print, args=(ver, print_interval, stop_evt), daemon=True)
        printer.start()

    # 解析/启动 Appium driver
    if activity in (None, "auto"):
        act = resolve_main_activity(pkg, serial)
        activity = act if act else None
        print(f"[resolve] {pkg} -> activity: {activity or 'Intent-Launcher'}")

    drv = make_driver(pkg=pkg, activity=activity, serial=serial, warmup_wait=warmup_wait)
    W, H = get_window_size(drv)
    L, T, R, B = int(W * safe[0]), int(H * safe[1]), int(W * safe[2]), int(H * safe[3])
    print(f"[v0-appium] {pkg}/{activity or 'auto'}, screen {W}x{H}, safe=({L},{T})~({R},{B})")

    # 模拟操作计时开始
    tim.start_sim()

    try:
        # 维持你现有的“随机策略”接口：每轮调用 common.policy_random.step(drv,W,H)
        for i in range(1, rounds + 1):
            try:
                msg = random_step(drv, W, H)  # 内部会随机从 tap/drag/long/pinch/zoom/rotate 里选并执行
                print(f"[{i:03d}/{rounds}] {msg}")
            except Exception as e:
                print(f"[{i:03d}/{rounds}] error: {e}")
            time.sleep(random.uniform(sleep_min, sleep_max))
    finally:
        # 模拟操作计时结束
        tim.stop_sim()
        try:
            drv.quit()
        except Exception:
            pass
        # 停止定时打印
        stop_evt.set()
        if printer:
            printer.join(timeout=1.0)
        # 设置总操作数（用 rounds 作为“发起的手势次数”）
        tim.set_ops_total(rounds)

        # 打印完整统计与时间
        print("[final]\n" + ver.summary_str(), flush=True)
        print(tim.summary_str(), flush=True)

        # 收尾 verify_motion
        ver.stop()

def main():
    ap = argparse.ArgumentParser(description="v0 Appium AR-Monkey (random gestures) + verify_motion + timing")
    ap.add_argument("--serial", help="可选，多设备时指定")
    ap.add_argument("--pkg", default="com.rooom.app")
    ap.add_argument("--activity", default="auto")
    ap.add_argument("--rounds", type=int, default=250)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--print-interval", type=int, default=10, help="每隔 N 秒打印一次统计；0=只在结束打印")
    ap.add_argument("--overall-include-tap", action="store_true", help="把 tap 纳入 overall（place 同时也会有 tap，谨慎开启）")
    args = ap.parse_args()

... (truncated)
```

### `experiments/v0_v1_v2_v3_archive/v1_ar_monkey_appium.py`

```text
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v1_ar_monkey_appium.py — Appium (CV targeted + act-then-verify)
+ verify_motion(basis=tap) + timing

改动要点：
- 启动前 adb logcat -c，Logcat 采集从 -T 0 开始
- 接入 common.verify_motion（含 tap），overall 基于 tap：overall = tap+drag+pinch+rotate
  （place 视作 tap 的特例，仅展示不入 overall）
- 接入 common.timing：总时长 / 模拟时长 / 平均每操作耗时
- 支持 --print-interval 定时打印完整汇总；结束时打印 [final]/[compare]/[timing]

CV 与操作验证：
- 保留原 v1：cv.strategy_riri_invariant.locate 定位
- 仍使用 cv.verify_motion.verify_action 做动作后的图像验证（不影响统计口径）

python v1_ar_monkey_appium.py --pkg com.google.ar.sceneform.samples.hellosceneform \
    --activity auto --rounds 50 --print-interval 10
"""

import argparse
import random
import sys
import time
import os
import csv
import threading
import subprocess

from common.device import make_driver, get_window_size, resolve_main_activity, capture_bgr
from common.actions import tap, pinch_or_zoom, rotate, drag_line
from common.policy_random import step as random_step  # 若你备用随机策略仍需要
from cv.strategy_riri_invariant import locate as cv_locate
from cv.verify_motion import verify_action  # 保留：动作后验证

from common.verify_motion import MotionVerifier
from common.timing import Timing

# ----------------- 工具 -----------------
def _ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000.0

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

def _map_img_to_window(x, y, w_img, h_img, W_win, H_win):
    # 若纵横比接近：按轴向缩放；否则按等比缩放 + 居中偏移（兼容 letterbox/状态栏）
    ar_img = w_img / float(h_img)
    ar_win = W_win / float(H_win)
    if abs(ar_img - ar_win) < 0.03:
        sx = W_win / float(w_img)
        sy = H_win / float(h_img)
        return int(x * sx), int(y * sy)
    else:
        s = min(W_win / float(w_img), H_win / float(h_img))
        x_off = int((W_win - w_img * s) * 0.5)
        y_off = int((H_win - h_img * s) * 0.5)
        return int(x * s + x_off), int(y * s + y_off)

def _map_bbox_to_window(x, y, w, h, w_img, h_img, W_win, H_win):
    x1, y1 = _map_img_to_window(x, y, w_img, h_img, W_win, H_win)
    x2, y2 = _map_img_to_window(x + w, y + h, w_img, h_img, W_win, H_win)
    X, Y = min(x1, x2), min(y1, y2)
    Wb, Hb = max(1, abs(x2 - x1)), max(1, abs(y2 - y1))
    return X, Y, Wb, Hb

# ----------------- 主流程 -----------------
def run_monkey_v1(
    pkg="com.rooom.app",
    activity="auto",
    serial=None,
    rounds=250,
    sleep_min=0.25,
    sleep_max=0.85,
    # 动作参数
    rotate_steps=8,
    warmup_wait=3.0,
    seed=None,
    drag_ms=(300, 900),
    # —— CV 的“公共”参数：与具体策略无关 ——
    cv_min_area_ratio=0.002,
    cv_downsample_w=640,
    # CSV
    log_csv=None,
    # 预触摸
    prime_tap=True,
    prime_pause_ms=80,
    # 验证（动作后）
    enable_verify=True,
    verify_wait_ms=140,
    drag_min_px=8.0,
    drag_dir_cos=0.6,
    verify_min_frac=0.5,
    pinch_scale_thr=0.10,
    rotate_min_deg=15.0,
    # 打印
    print_interval=10,
):
    """
    v1 主循环（CV 定位 → 执行动作 → 图像验证），并用 verify_motion 统计操作成功率（basis=tap）
    """
    if seed is not None:
        random.seed(seed)

    # 计时从此刻开始
    tim = Timing()

    # 解析 Activity 并启动驱动
... (truncated)
```

### `experiments/v0_v1_v2_v3_archive/v2_ar_monkey_appium.py`

```text
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v2_ar_monkey_appium.py — Appium (CV targeted + act-then-verify)
+ verify_motion(basis=tap) + timing

改动要点：
- 启动前 adb logcat -c，Logcat 采集从 -T 0 开始
- 接入 common.verify_motion（含 tap），overall 基于 tap：overall = tap+drag+pinch+rotate
  （place 视作 tap 的特例，仅展示不入 overall）
- 接入 common.timing：总时长 / 模拟时长 / 平均每操作耗时
- 支持 --print-interval 定时打印完整汇总；结束时打印 [final]/[compare]/[timing]

CV 与操作验证：
- 保留原 v2：cv.strategy_riri_invariant.locate 定位
- 仍使用 cv.verify_motion.verify_action 做动作后的图像验证（不影响统计口径）

python v2_ar_monkey_appium.py --pkg com.google.ar.sceneform.samples.hellosceneform \
    --activity auto --rounds 50 --print-interval 10
"""

import argparse
import random
import sys
import time
import os
import csv
import threading
import subprocess

from common.device import make_driver, get_window_size, resolve_main_activity, capture_bgr
from common.actions import tap, pinch_or_zoom, rotate, drag_line
from common.policy_random import step as random_step  # 若你备用随机策略仍需要
from cv.strategy_yolo import locate as cv_locate
from cv.verify_motion import verify_action  # 保留：动作后验证

from common.verify_motion import MotionVerifier
from common.timing import Timing

# ----------------- 工具 -----------------
def _ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000.0

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

def _map_img_to_window(x, y, w_img, h_img, W_win, H_win):
    # 若纵横比接近：按轴向缩放；否则按等比缩放 + 居中偏移（兼容 letterbox/状态栏）
    ar_img = w_img / float(h_img)
    ar_win = W_win / float(H_win)
    if abs(ar_img - ar_win) < 0.03:
        sx = W_win / float(w_img)
        sy = H_win / float(h_img)
        return int(x * sx), int(y * sy)
    else:
        s = min(W_win / float(w_img), H_win / float(h_img))
        x_off = int((W_win - w_img * s) * 0.5)
        y_off = int((H_win - h_img * s) * 0.5)
        return int(x * s + x_off), int(y * s + y_off)

def _map_bbox_to_window(x, y, w, h, w_img, h_img, W_win, H_win):
    x1, y1 = _map_img_to_window(x, y, w_img, h_img, W_win, H_win)
    x2, y2 = _map_img_to_window(x + w, y + h, w_img, h_img, W_win, H_win)
    X, Y = min(x1, x2), min(y1, y2)
    Wb, Hb = max(1, abs(x2 - x1)), max(1, abs(y2 - y1))
    return X, Y, Wb, Hb

# ----------------- 主流程 -----------------
def run_monkey_v2(
    pkg="com.rooom.app",
    activity="auto",
    serial=None,
    rounds=250,
    sleep_min=0.25,
    sleep_max=0.85,
    # 动作参数
    rotate_steps=8,
    warmup_wait=3.0,
    seed=None,
    drag_ms=(300, 900),
    # —— CV 的“公共”参数：与具体策略无关 ——
    cv_min_area_ratio=0.002,
    cv_downsample_w=640,
    # CSV
    log_csv=None,
    # 预触摸
    prime_tap=True,
    prime_pause_ms=80,
    # 验证（动作后）
    enable_verify=True,
    verify_wait_ms=140,
    drag_min_px=8.0,
    drag_dir_cos=0.6,
    verify_min_frac=0.5,
    pinch_scale_thr=0.10,
    rotate_min_deg=15.0,
    # 打印
    print_interval=10,
):
    """
    v2 主循环（CV 定位 → 执行动作 → 图像验证），并用 verify_motion 统计操作成功率（basis=tap）
    """
    if seed is not None:
        random.seed(seed)

    # 计时从此刻开始
    tim = Timing()

    # 解析 Activity 并启动驱动
... (truncated)
```

### `experiments/v0_v1_v2_v3_archive/v3_ar_monkey_appium.py`

```text
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
... (truncated)
```

### `gen_project_context.py`

```text
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成工程结构的 Markdown 文件（项目“快照说明”），便于在对话中共享上下文。
- 输出：默认写入 project_context.md
- 目录树：可设最大深度
- 排除：默认忽略 .git、虚拟环境、缓存与大文件等，可自定义
- 可选摘录：按通配符挑选关键文件，截取前 N 行作为上下文片段

python gen_project_context.py --out chat_context.md --tree-depth 4

"""

import argparse
import datetime as dt
import os
import sys
import subprocess
from pathlib import Path
from typing import Iterable, List, Tuple

# 目录名级别的硬排除（出现即跳过整个子树）
EXCLUDE_DIR_NAMES = {".git", ".svn", ".hg", ".venv", ".venv311", "venv", "env", "ENV", ".conda", "conda"}
# 文件后缀级别的硬排除（可选）
EXCLUDE_FILE_SUFFIXES = {".pt", ".onnx", ".ckpt", ".mp4", ".mov", ".avi", ".zip", ".tar", ".gz", ".7z"}

DEFAULT_EXCLUDES = [
    ".git", ".svn", ".hg",
    ".DS_Store",
    "__pycache__", "*.pyc", "*.pyo", "*.pyd",
    ".idea", ".vscode",
    ".venv", ".venv311", "venv", "env", "ENV", ".conda", "conda", ".env",
    "build", "dist", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "node_modules",
    "runs", "outputs", "logs",
    # 大文件常见后缀（仅用于目录树与统计时忽略；不影响真实代码）
    "*.pt", "*.onnx", "*.ckpt",
    "*.mp4", "*.mov", "*.avi",
    "*.png", "*.jpg", "*.jpeg", "*.webp",
    "*.zip", "*.tar", "*.gz", "*.7z",
]

DEFAULT_SNIPPETS = [
    "README.md",
    "*.md",
    "*.py",
    "requirements.txt",
    "pyproject.toml",
    "setup.cfg",
]

def parse_args():
    p = argparse.ArgumentParser(description="生成工程结构 Markdown 文档")
    p.add_argument("--root", type=str, default=".", help="工程根目录（默认当前目录）")
    p.add_argument("--out", type=str, default="project_context.md", help="输出文件名")
    p.add_argument("--tree-depth", type=int, default=3, help="目录树最大深度（默认 3）")
    p.add_argument("--exclude", action="append", default=[], help="追加排除模式（可多次）")
    p.add_argument("--no-default-excludes", action="store_true", help="不使用默认排除列表")
    p.add_argument("--snippet-glob", action="append", default=[], help="需要摘录片段的通配符（可多次）")
    p.add_argument("--snippet-lines", type=int, default=120, help="每个文件摘录的最大行数")
    p.add_argument("--topn", type=int, default=15, help="体积 Top-N 文件列表条数")
    return p.parse_args()

def is_git_repo(root: Path) -> bool:
    return (root / ".git").exists()

def run_cmd(cmd: List[str], cwd: Path) -> str:
    try:
        out = subprocess.check_output(cmd, cwd=str(cwd), stderr=subprocess.DEVNULL, text=True, timeout=5)
        return out.strip()
    except Exception:
        return ""

def gather_git_info(root: Path) -> Tuple[str, str, str]:
    if not is_git_repo(root):
        return ("N/A", "N/A", "N/A")
    branch = run_cmd(["git", "rev-parse", "--abbrev-ref", "HEAD"], root) or "N/A"
    commit = run_cmd(["git", "rev-parse", "--short", "HEAD"], root) or "N/A"
    latest = run_cmd(["git", "log", "--oneline", "-n", "5"], root)
    return (branch, commit, latest or "N/A")

def match_any(path: Path, patterns: Iterable[str]) -> bool:
    from fnmatch import fnmatch
    rel = str(path).replace("\\", "/")
    name = path.name
    # 1) 通配匹配：支持 **/dir/** 形式
    if any(fnmatch(rel, pat) or fnmatch(name, pat) for pat in patterns):
        return True
    # 2) 目录名硬排除：只要任何一段路径命中就排除
    for part in path.parts:
        if part in EXCLUDE_DIR_NAMES:
            return True
    # 3) 文件后缀硬排除
    if path.is_file() and path.suffix.lower() in EXCLUDE_FILE_SUFFIXES:
        return True
    return False

def iter_tree(root: Path, max_depth: int, excludes: List[str]) -> List[str]:
    lines = []
    root = root.resolve()

    def walk(d: Path, depth: int, prefix: str = ""):
        if depth > max_depth:
            return
        try:
            entries = sorted(d.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        except PermissionError:
            return

        # 先分组，目录要能被“剪枝”
        dirs, files = [], []
        for e in entries:
            if e.is_dir():
                # 如果 e 这个目录命中排除，直接跳过（剪枝）
                if match_any(e.relative_to(root), excludes) or e.name in EXCLUDE_DIR_NAMES:
                    continue
                dirs.append(e)
            else:
                if match_any(e.relative_to(root), excludes):
                    continue
... (truncated)
```

### `project_context.md`

```text
# Project Context

- **Path**: `/Users/yangxiaoyi/Desktop/phd/project/newproject/program`
- **Time (UTC)**: 2025-10-31 17:06:01Z
- **Git Branch**: main
- **Git Commit**: d18c655

## Directory Tree (depth=3)

```
program/
├── common/
│   ├── actions.py
│   ├── device.py
│   ├── locator_iface.py
│   ├── policy_random.py
│   ├── timing.py
│   └── verify_motion.py
├── cv/
│   ├── yolo/
│   │   ├── testimages/
│   │   ├── coco.yaml
│   │   ├── coco8.yaml
│   │   ├── myar2.yaml
│   │   ├── myar_simp.yaml
│   │   └── test_yolo_train.py
│   ├── strategy_bgmedian.py
│   ├── strategy_flow_cluster.py
│   ├── strategy_FoELS.py
│   ├── strategy_grabcut_refine.py
│   ├── strategy_riri_invariant.py
│   ├── strategy_templatematch.py
│   ├── strategy_twodiff_orb.py
│   ├── strategy_yolo.py
│   └── verify_motion.py
├── val/
├── .gitignore
├── config.yaml
├── gen_project_context.py
├── main.py
├── README.md
├── requirements.txt
├── smoke.py
├── temp.java
├── v0_ar_monkey_adb.py
├── v0_ar_monkey_appium.py
├── v1_ar_monkey_appium.py
├── v2_ar_monkey_appium.py
├── v3_ar_monkey_appium.py
├── yolo_adb_screencap.py
├── yolo_scratch.py
└── yolo_val.py
```

## File Stats

- Files counted (excluded patterns applied): **9995**
- Total size: **1.7 MB**

### Top 15 Largest Files (excluded patterns applied)

| Size | Path |
|---:|---|
| 17.2 KB | `cv/strategy_FoELS.py` |
| 14.8 KB | `temp.java` |
| 14.7 KB | `v1_ar_monkey_appium.py` |
| 14.7 KB | `v2_ar_monkey_appium.py` |
| 13.9 KB | `v3_ar_monkey_appium.py` |
| 12.3 KB | `cv/yolo/runs/train/exp3/results.csv` |
| 11.7 KB | `cv/yolo/runs/train/exp2/results.csv` |
| 10.0 KB | `cv/verify_motion.py` |
| 9.9 KB | `cv/yolo/runs/train/exp4/results.csv` |
| 9.4 KB | `gen_project_context.py` |
| 8.6 KB | `v0_ar_monkey_adb.py` |
| 8.2 KB | `common/actions.py` |
| 8.1 KB | `common/verify_motion.py` |
| 6.7 KB | `yolo_adb_screencap.py` |
| 6.4 KB | `cv/yolo/test_yolo_train.py` |

## Recent Commits

```
d18c655 Create README.md
7143e3e init: import existing project
```

## File Snippets (first 120 lines)

### `common/actions.py`

```text
# common/actions.py
# ------------------------------------------------------------
# Description:
#   Coordinate-only gesture primitives implemented with
#   Selenium ActionBuilder (touch pointers). This mirrors
#   your working v0 semantics to avoid API differences across
#   Selenium/Appium versions.
# ------------------------------------------------------------

import math
from selenium.webdriver.common.actions.action_builder import ActionBuilder
from selenium.webdriver.common.actions.pointer_input import PointerInput
from selenium.webdriver.common.actions.interaction import Interaction

def tap(driver, x, y, press_ms=60):
    a  = ActionBuilder(driver)
    f1 = a.add_pointer_input(kind="touch", name="finger1")
    f1.create_pointer_move(duration=0, x=int(x), y=int(y))
    f1.create_pointer_down(button=0)
    f1.create_pause(press_ms/1000.0)
    f1.create_pointer_up(button=0)
    a.perform()

def long_press(driver, x, y, hold_ms=900):
    a  = ActionBuilder(driver)
    f1 = a.add_pointer_input(kind="touch", name="finger1")
    f1.create_pointer_move(duration=0, x=int(x), y=int(y))
    f1.create_pointer_down(button=0)
    f1.create_pause(hold_ms/1000.0)
... (truncated)
```

### `project_status_summary.md`

```text

---

# 🚀 **Action Discovery Framework – 全局工程进度总结（截至当前对话）**

本文件总结了当前 AR 自动化测试框架（Action Discovery Framework）的全部代码结构、完成状态与待办事项，用于新的 ChatGPT 对话作为上下文输入。

---

# 1. 🎯 项目目标概述

本工程旨在构建一个 **用于 AR 应用的自动化 Action Discovery 工具**，通过视觉检测、操作采样、手势执行、变化验证与策略判定，最终输出：

* **Action Support Matrix（支持矩阵）**
* 可用于后续 LLM 驱动智能脚本生成的基础能力图谱

系统核心流程如下：

1. **YOLODetector**：检测 AR 物体位置（bbox / center）
2. **DefaultSampler**：根据物体与操作生成执行参数
3. **AppiumExecutor**：对 Android 设备注入手势
4. **Verifier**：通过视觉证据（motion similarity 等）判断操作是否成功
5. **NMPolicy**：N/M 判定，决定该目标是否“支持”某个动作
6. **run_discovery**：主循环，将上述模块整合为完整 Action Discovery 流程

---

# 2. 📁 代码目录结构（当前有效结构）

```
program/
├── src/
│   ├── __init__.py
│   ├── __main__.py
│   ├── common/
│   │   ├── __init__.py
│   │   ├── actions.py
│   │   ├── device.py
│   │   ├── locator_iface.py
│   │   ├── policy_random.py
│   │   ├── timing.py
│   │   └── verify_motion.py
│   │
│   ├── detector/
│   │   ├── __init__.py
│   │   └── yolo_detector.py
│   │
│   ├── sampler/
│   │   ├── __init__.py
│   │   └── default_sampler.py
│   │
│   ├── executor/
│   │   ├── __init__.py
│   │   └── appium_executor.py
│   │
│   ├── verifier/
│   │   ├── __init__.py
│   │   ├── backends/
│   │   │   └── motion_similarity.py
│   │   └── verifier.py
│   │
│   ├── policy/
│   │   ├── __init__.py
│   │   └── policy.py
│   │
│   └── discovery/
│       ├── __init__.py
│       └── run_discovery.py
│
├── configs/
│   └── ad.yaml
│
├── scripts/
│   └── run_discovery.sh
│
├── experiments/
│   └── v0_v1_v2_v3_archive/
│       └── v2_ar_monkey_appium.py  (参考用)
│
├── README.md
└── requirements.txt
```

---

# 3. 🧩 各模块完成度与当前状态

## ✔️ **3.1 Detector (YOLODetector)**

* 已完成：`yolo_detector.py`
* 功能：加载 YOLO 模型、输出 bbox + center + score
* 输出格式标准化完毕

## ✔️ **3.2 Sampler (DefaultSampler)**

* 已完成：`default_sampler.py`
* 支持动作：

  * tap/single_tap
  * drag/drag_short/drag_long
  * rotate/rotate_cw/rotate_ccw
  * pinch/pinch_in/pinch_out/zoom_in/zoom_out
* 所有参数都规范化并可通过 cfg 调整

## ✔️ **3.3 Executor (AppiumExecutor)**

* 已完成：`appium_executor.py`
* 与 `common/actions.py` 与 `common/device.py` 完全对齐
* 支持动作家族：tap / drag / rotate / pinch（自动处理别名）

## ✔️ **3.4 Verifier**

* 已完成：

  * `verifier/backends/motion_similarity.py`（已有）
  * `verifier.py`（新写）
* 功能：将操作映射到 motion-based 几何验证
* 支持：

  * drag：方向/幅度
... (truncated)
```

### `README.md`

```text
TBD
```

### `requirements.txt`

```text
Appium-Python-Client==5.2.4
attrs==25.3.0
certifi==2025.8.3
charset-normalizer==3.4.3
diskcache==5.6.3
filelock==3.19.1
fsspec==2025.9.0
h11==0.16.0
hf-xet==1.1.10
huggingface-hub==0.35.3
idna==3.10
imageio==2.37.0
Jinja2==3.1.6
joblib==1.5.2
lazy_loader==0.4
llama_cpp_python==0.3.16
MarkupSafe==3.0.3
networkx==3.5
numpy==2.2.6
opencv-contrib-python==4.12.0.88
outcome==1.3.0.post0
packaging==25.0
pillow==11.3.0
PySocks==1.7.1
PyYAML==6.0.2
regex==2025.9.18
requests==2.32.5
safetensors==0.6.2
scikit-image==0.25.2
scikit-learn==1.7.2
scipy==1.16.2
selenium==4.35.0
setuptools==80.9.0
sniffio==1.3.1
sortedcontainers==2.4.0
threadpoolctl==3.6.0
tifffile==2025.9.20
tokenizers==0.22.1
tqdm==4.67.1
transformers==4.57.1
trio==0.30.0
trio-websocket==0.12.2
typing_extensions==4.14.1
urllib3==2.5.0
websocket-client==1.8.0
wheel==0.45.1
wsproto==1.2.0
```

### `src/__init__.py`

```text
# src/__init__.py
```

### `src/__main__.py`

```text
# src/__main__.py
# -*- coding: utf-8 -*-

import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import yaml  # 需安装 pyyaml

# === 你自己的模块实现（按你的目录结构替换/补齐） ===
from src.discovery.run_discovery import run_once
from src.verifier.verifier import Verifier
from src.policy.policy import NMPolicy

from src.detector import YOLODetector
from src.executor import AppiumExecutor
from src.sampler import DefaultSampler


def load_cfg(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 组装输出目录与 JSONL 文件路径
    out_dir = cfg.get("runtime", {}).get("out_dir", None)
    if not out_dir:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        out_dir = f"runs/exp-{stamp}"
        cfg.setdefault("runtime", {})["out_dir"] = out_dir

    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    cfg["runtime"]["out_jsonl"] = str(out_dir_path / "trials.jsonl")
    cfg["runtime"]["support_jsonl"] = str(out_dir_path / "support.jsonl")

    # 默认项补齐
    cfg.setdefault("policy", {}).setdefault("N", 10)
    cfg["policy"].setdefault("M", 2)
    cfg.setdefault("thresholds", {}).setdefault("tau_move", 0.03)
    cfg["thresholds"].setdefault("tau_rot_deg", 8.0)
    cfg["thresholds"].setdefault("tau_ssim", 0.08)
    cfg.setdefault("runtime", {}).setdefault("post_wait_s", 0.4)

    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    # 如果你使用了 torch 或其他框架，可以在这里补充它们的随机种子


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="ActionDiscovery",
        description="Run Action Discovery pipeline (detector → sampler → executor → verifier → policy).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--cfg", type=str, required=True, help="Path to YAML config (e.g., configs/ad.yaml)")
    p.add_argument("--seed", type=int, default=42, help="Global random seed")
    p.add_argument(
        "--ops",
        type=str,
        default="drag,rotate,tap",
        help="Comma-separated ops to run (subset of: tap,drag,rotate,pinch_in,pinch_out)",
    )
    p.add_argument("--out-dir", type=str, default=None, help="Override runtime.out_dir")
    p.add_argument("--device-id", type=str, default=None, help="Override device id if needed")
    p.add_argument("--dry-run", action="store_true", help="Load everything but skip executing gestures")
    return p.parse_args(argv)


def make_components(cfg: dict, device_id_override: str | None = None):
    detector = YOLODetector.from_cfg(cfg)
    sampler = DefaultSampler.from_cfg(cfg)
    dev_id = device_id_override or cfg.get("device", None)
    executor = AppiumExecutor(cfg=cfg, device_id=dev_id)

    driver = executor.driver
    return driver, detector, sampler, executor


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    cfg = load_cfg(args.cfg)

    if args.out_dir:
        cfg["runtime"]["out_dir"] = args.out_dir
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)
        cfg["runtime"]["out_jsonl"] = str(Path(args.out_dir) / "trials.jsonl")
        cfg["runtime"]["support_jsonl"] = str(Path(args.out_dir) / "support.jsonl")

    if args.device_id:
        cfg["device"] = args.device_id

    cfg["seed"] = args.seed
    set_seed(args.seed)

    # 仅在需要时将 ops 覆盖到 cfg（给 sampler/executor 可见）
    ops = [s.strip() for s in args.ops.split(",") if s.strip()]
    cfg["ops"] = ops

    # 打印一次关键配置（便于追踪）
    print(json.dumps(
        {
            "out_dir": cfg["runtime"]["out_dir"],
            "out_jsonl": cfg["runtime"]["out_jsonl"],
            "support_jsonl": cfg["runtime"]["support_jsonl"],
            "ops": ops,
            "seed": cfg["seed"],
        },
        indent=2,
        ensure_ascii=False,
... (truncated)
```

### `src/common/__init__.py`

```text
# src/common/__init__.py

from . import actions, device, locator_iface, policy_random, timing, verify_motion

__all__ = [
    "actions",
    "device",
    "locator_iface",
    "policy_random",
    "timing",
    "verify_motion",
]
```

### `src/common/actions.py`

```text
# common/actions.py
# ------------------------------------------------------------
# Description:
#   Coordinate-only gesture primitives implemented with
#   Selenium ActionBuilder (touch pointers). This mirrors
#   your working v0 semantics to avoid API differences across
#   Selenium/Appium versions.
# ------------------------------------------------------------

import math
from selenium.webdriver.common.actions.action_builder import ActionBuilder
from selenium.webdriver.common.actions.pointer_input import PointerInput
from selenium.webdriver.common.actions.interaction import Interaction

def tap(driver, x, y, press_ms=60):
    a  = ActionBuilder(driver)
    f1 = a.add_pointer_input(kind="touch", name="finger1")
    f1.create_pointer_move(duration=0, x=int(x), y=int(y))
    f1.create_pointer_down(button=0)
    f1.create_pause(press_ms/1000.0)
    f1.create_pointer_up(button=0)
    a.perform()

def long_press(driver, x, y, hold_ms=900):
    a  = ActionBuilder(driver)
    f1 = a.add_pointer_input(kind="touch", name="finger1")
    f1.create_pointer_move(duration=0, x=int(x), y=int(y))
    f1.create_pointer_down(button=0)
    f1.create_pause(hold_ms/1000.0)
    f1.create_pointer_up(button=0)
    a.perform()

def drag_line(driver, x1, y1, x2, y2, duration_ms=600):
    a  = ActionBuilder(driver)
    f1 = a.add_pointer_input(kind="touch", name="finger1")
    f1.create_pointer_move(duration=0, x=int(x1), y=int(y1))
    f1.create_pointer_down(button=0)
    f1.create_pause(0.03)
    f1.create_pointer_move(duration=duration_ms, x=int(x2), y=int(y2))
    f1.create_pointer_up(button=0)
    a.perform()

def pinch_or_zoom(driver, cx, cy, start_dist=60, end_dist=220, duration_ms=600):
    """
    If end_dist < start_dist => pinch; else => zoom.
    Uses diagonal symmetry (↙↗).
    """
    dx0 = dy0 = start_dist // 2
    dx1 = dy1 = end_dist   // 2
    s1 = (int(cx - dx0), int(cy - dy0)); e1 = (int(cx - dx1), int(cy - dy1))
    s2 = (int(cx + dx0), int(cy + dy0)); e2 = (int(cx + dx1), int(cy + dy1))

    a  = ActionBuilder(driver)
    f1 = a.add_pointer_input(kind="touch", name="finger1")
    f2 = a.add_pointer_input(kind="touch", name="finger2")

    f1.create_pointer_move(0, x=s1[0], y=s1[1])
    f2.create_pointer_move(0, x=s2[0], y=s2[1])
    f1.create_pointer_down(button=0)
    f2.create_pointer_down(button=0)
    f1.create_pointer_move(duration_ms, x=e1[0], y=e1[1])
    f2.create_pointer_move(duration_ms, x=e2[0], y=e2[1])
    f1.create_pointer_up(button=0)
    f2.create_pointer_up(button=0)
    a.perform()

def rotate(
    driver,
    cx, cy,
    radius=200,
    angle_deg=60,
    duration_ms=800,
    steps=8,
    direction="ccw",   # "ccw" 逆时针 | "cw" 顺时针
    start_deg=0        # 可选：两指起始基准角（第二指为 start_deg+180）
):
    """
    Two-finger rotation around (cx, cy) with both fingers sweeping the SAME angular direction.
    - direction: "ccw" (counter-clockwise) or "cw" (clockwise)
    - start_deg: starting angle for finger1; finger2 starts at start_deg + 180
    Screen coords notice: y 轴向下增长，若方向与预期相反，切换 direction 即可。
    """
    sgn = +1 if str(direction).lower() == "ccw" else -1

    def pos(deg):
        r = math.radians(deg)
        return int(cx + radius * math.cos(r)), int(cy + radius * math.sin(r))

    # 起点：两指相差 180°
    f1_start = float(start_deg)
    f2_start = float(start_deg + 180.0)

    a  = ActionBuilder(driver)
    f1 = a.add_pointer_input(kind="touch", name="finger1")
    f2 = a.add_pointer_input(kind="touch", name="finger2")

    x1, y1 = pos(f1_start)
    x2, y2 = pos(f2_start)
    f1.create_pointer_move(0, x=x1, y=y1)
    f2.create_pointer_move(0, x=x2, y=y2)
    f1.create_pointer_down(button=0)
    f2.create_pointer_down(button=0)

    # 分段同步走弧线
    seg = max(1, int(duration_ms / max(1, int(steps))))
    for i in range(1, steps + 1):
        delta = sgn * (angle_deg / steps) * i  # 两指同向，同幅度
        nx1, ny1 = pos(f1_start + delta)
        nx2, ny2 = pos(f2_start + delta)
        f1.create_pointer_move(seg, x=nx1, y=ny1)
        f2.create_pointer_move(seg, x=nx2, y=ny2)

    f1.create_pointer_up(button=0)
    f2.create_pointer_up(button=0)
    a.perform()

    
    # common/actions.py

def smart_correct_rule_based(
... (truncated)
```

### `src/common/device.py`

```text
# common/device.py
# ------------------------------------------------------------
# Description:
#   Appium driver bootstrap (UiAutomator2Options) + adb helpers
#   + simple utilities (window size, screenshot if需要扩展).
#   Mirrors your working connection semantics.
# ------------------------------------------------------------

import subprocess, cv2, numpy as np, time
from typing import Optional
from appium import webdriver
from appium.options.android import UiAutomator2Options
from selenium.common.exceptions import WebDriverException


def _try_connect(url, options):
    try:
        drv = webdriver.Remote(url, options=options)
        return drv
    except WebDriverException as e:
        # 典型 404/UnknownCommandError：base-path 不对
        msg = str(e)
        if "UnknownCommandError" in msg or "requested resource could not be found" in msg or "404" in msg:
            return None
        # 其他异常直接抛
        raise

def sh(cmd: str) -> str:
    return subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT).decode("utf-8", "ignore")

def adb(cmd: str, serial: Optional[str] = None) -> str:
    base = f"adb -s {serial} " if serial else "adb "
    return sh(base + cmd)

def one_device_or_none() -> Optional[str]:
    out = sh("adb devices")
    lines = [l for l in out.strip().splitlines() if "\tdevice" in l and not l.startswith("List")]
    return lines[0].split("\t")[0] if len(lines) == 1 else None

def resolve_main_activity(pkg: str, serial: Optional[str] = None) -> Optional[str]:
    intent = 'intent:#Intent;action=android.intent.action.MAIN;category=android.intent.category.LAUNCHER;end'
    try:
        out = adb(f'shell cmd package query-activities --brief "{intent}"', serial)
        for line in out.splitlines():
            line = line.strip()
            if line.startswith(pkg + "/"):
                return line.split("/", 1)[1]
    except Exception:
        pass
    return None

def make_driver(pkg: str,
                activity: Optional[str],
                serial: Optional[str],
                warmup_wait: float = 3.0):
    """Create driver with your v0-style capabilities."""
    caps = {
        "platformName": "Android",
        "appium:automationName": "UiAutomator2",
        "appium:deviceName": serial or (one_device_or_none() or "Android"),
        "appium:udid": serial if serial else one_device_or_none(),
        "appium:noReset": True,
        "appium:autoGrantPermissions": True,
        "appium:newCommandTimeout": 300,
        "appium:appPackage": pkg,
        "appium:disableWindowAnimation": True,
    }

    if activity and activity not in ("auto", None):
        caps["appium:appActivity"] = activity
        caps["appium:appWaitActivity"] = "*"
    else:
        caps["appium:intentAction"]   = "android.intent.action.MAIN"
        caps["appium:intentCategory"] = "android.intent.category.LAUNCHER"
        caps["appium:intentFlags"]    = "0x10200000"
        caps["appium:appWaitActivity"] = "*"

    options = UiAutomator2Options().load_capabilities(caps)
    
    # 允许从环境或 config 里传 server_url；没有就用默认
    server_url = "http://127.0.0.1:4723"  # 你也可以从 config 传进来
    # 组装两条候选：根路径 与 /wd/hub（去重）
    candidates = []
    if server_url.rstrip("/").endswith("/wd/hub"):
        root = server_url.rstrip("/").rsplit("/wd/hub", 1)[0]
        candidates = [server_url, root]
    else:
        hub = server_url.rstrip("/") + "/wd/hub"
        candidates = [server_url, hub]

    last_err = None
    for url in candidates:
        drv = _try_connect(url, options)
        if drv is not None:
            print(f"[device] connected via {url}")
            time.sleep(warmup_wait)
            return drv
    # 两条都失败，抛出更友好的错误
    raise RuntimeError(
        f"Failed to create session. Tried: {candidates}. "
        "Check Appium base path (root vs /wd/hub) and that uiautomator2 driver is installed "
        "(appium driver install uiautomator2)."
    )

def get_window_size(driver):
    s = driver.get_window_size()
    return int(s["width"]), int(s["height"])

def capture_bgr(driver):
    png = driver.get_screenshot_as_png()
    arr = np.frombuffer(png, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)
```

### `src/common/locator_iface.py`

```text
# common/locator_iface.py
# ------------------------------------------------------------
# Description:
#   Target locator interface. Implementations return an object
#   with "center" and "bbox" for the detected AR object.
# ------------------------------------------------------------

from typing import Optional, Tuple, Dict, Any
import numpy as np

class ITargetLocator:
    def reset(self) -> None:
        """Reset internal state if any."""
        ...

    def locate(self, curr_bgr: np.ndarray, prev_bgr: Optional[np.ndarray] = None
              ) -> Optional[Dict[str, Any]]:
        """
        Return:
          {"center": (cx, cy), "bbox": (x, y, w, h)}  or  None if not found.
        """
        ...
```

### `src/common/policy_random.py`

```text
# common/policy_random.py
# ------------------------------------------------------------
# Description:
#   Random gesture policy used by v0 and as fallback in v1.
#   Generates coordinates + calls coordinate-only actions.
# ------------------------------------------------------------

import random
from typing import Tuple
from common.actions import tap, drag_line, long_press, pinch_or_zoom, rotate

def _rand_point(w: int, h: int, margin_ratio: float = 0.05) -> Tuple[int,int]:
    L = int(w*margin_ratio); T = int(h*margin_ratio)
    R = w - L - 1;          B = h - T - 1
    return random.randint(L, R), random.randint(T, B)

def step(driver,
         w: int, h: int,
         safe_box=(0.12, 0.18, 0.88, 0.88),
         drag_ms=(300, 900), long_ms=(700, 1500),
         pinch_start=80, pinch_end=220,
         rotate_radius=(160, 260), rotate_angle=(30, 90), rotate_steps=8):
    """执行一次随机动作，返回字符串描述便于日志记录。"""
    L, T, R, B = int(w*safe_box[0]), int(h*safe_box[1]), int(w*safe_box[2]), int(h*safe_box[3])
    op = random.choice(["tap","drag","long","pinch","zoom","rotate"])
    #op = random.choice(["drag","pinch","zoom","rotate"])

    if op == "tap":
        x = random.randint(L, R); y = random.randint(T, B)
        tap(driver, x, y, press_ms=random.randint(40, 120))
        return f"tap({x},{y})"

    if op == "drag":
        x1 = random.randint(L, R); y1 = random.randint(T, B)
        x2 = random.randint(L, R); y2 = random.randint(T, B)
        dur = random.randint(*drag_ms)
        drag_line(driver, x1, y1, x2, y2, duration_ms=dur)
        return f"drag({x1},{y1}->{x2},{y2},{dur}ms)"

    if op == "long":
        x = random.randint(L, R); y = random.randint(T, B)
        hold = random.randint(*long_ms)
        long_press(driver, x, y, hold_ms=hold)
        return f"long({x},{y},{hold}ms)"

    if op in ("pinch", "zoom"):
        cx = random.randint(L + 40, R - 40)
        cy = random.randint(T + 40, B - 40)
        s, e = (pinch_end, pinch_start) if op == "pinch" else (pinch_start, pinch_end)
        pinch_or_zoom(driver, cx, cy, start_dist=s, end_dist=e,
                      duration_ms=random.randint(450, 900))
        return f"{op}({cx},{cy},{s}->{e})"

    # rotate
    cx = random.randint(L + 60, R - 60)
    cy = random.randint(T + 60, B - 60)
    radius = random.randint(*rotate_radius)
    angle  = random.randint(*rotate_angle)
    rotate(driver, cx, cy, radius=radius, angle_deg=angle,
           duration_ms=random.randint(600, 1100), steps=rotate_steps)
    return f"rotate({cx},{cy},r={radius},θ={angle}°)"
```

### `src/common/timing.py`

```text
# -*- coding: utf-8 -*-
"""
common/timing.py
--------------------------------
通用计时工具，用于 v0/v1/v3 统计：
- 程序总时长（含准备阶段）
- 模拟操作时长（你控制开始/结束）
- 平均每个操作耗时（传入总操作数；可选分手势统计）
- v0 场景可从 monkey 的标准输出解析 "Events injected: N"

用法要点：
- 创建 Timing() 即启动“程序总时长”计时
- 在模拟操作开始处调用 .start_sim()
- 在模拟操作结束处调用 .stop_sim()
- 结束前调用 .set_ops_total(n)（或 .set_ops_by_kind({...})），然后打印 .summary_str()
"""

import time
import re
from typing import Optional, Dict, IO

_MONKEY_INJECTED_RE = re.compile(r"Events injected:\s*(\d+)", re.IGNORECASE)

class Timing:
    def __init__(self) -> None:
        self._t_prog_start = time.time()
        self._t_sim_start: Optional[float] = None
        self._t_sim_end: Optional[float] = None
        self._ops_total: Optional[int] = None
        self._ops_by_kind: Dict[str, int] = {}   # 可选：{"place":N, "drag":M, ...}

    # --- 时钟控制 ---
    def start_sim(self) -> None:
        if self._t_sim_start is None:
            self._t_sim_start = time.time()

    def stop_sim(self) -> None:
        if self._t_sim_start is not None and self._t_sim_end is None:
            self._t_sim_end = time.time()

    # --- 事件数设置 ---
    def set_ops_total(self, n: int) -> None:
        self._ops_total = max(0, int(n))

    def set_ops_by_kind(self, d: Dict[str, int]) -> None:
        # 允许只传有意义的 key，例如 {"drag": 120, "pinch": 0}
        clean = {}
        for k, v in (d or {}).items():
            try:
                clean[str(k)] = max(0, int(v))
            except Exception:
                pass
        self._ops_by_kind = clean
        # 若未设置总数，自动合计
        if self._ops_total is None:
            self._ops_total = sum(clean.values())

    # --- 从 monkey 输出解析注入事件数（v0 专用，可选） ---
    @staticmethod
    def parse_monkey_injected_from_stream(stream: IO[str]) -> Optional[int]:
        """
        读取 monkey 的 stdout（文本流），找到 'Events injected: N'。
        若未找到返回 None。你可以一边读取一边把每行丢给本函数解析。
        """
        for line in stream:
            m = _MONKEY_INJECTED_RE.search(line)
            if m:
                try:
                    return int(m.group(1))
                except Exception:
                    pass
        return None

    # --- 统计取值 ---
    def total_runtime(self) -> float:
        return time.time() - self._t_prog_start

    def sim_runtime(self) -> float:
        if self._t_sim_start is None:
            return 0.0
        end = self._t_sim_end if self._t_sim_end is not None else time.time()
        return max(0.0, end - self._t_sim_start)

    def avg_time_per_op(self) -> float:
        ops = (self._ops_total or 0)
        return 0.0 if ops <= 0 else self.sim_runtime() / float(ops)

    def avg_time_per_kind(self) -> Dict[str, float]:
        if not self._ops_by_kind:
            return {}
        rt = self.sim_runtime()
        return {k: (0.0 if v <= 0 else rt / float(v)) for k, v in self._ops_by_kind.items()}

    # --- 输出 ---
    def summary_dict(self) -> Dict[str, object]:
        out = {
            "total_runtime_sec": round(self.total_runtime(), 6),
            "sim_runtime_sec": round(self.sim_runtime(), 6),
            "ops_total": int(self._ops_total or 0),
            "avg_time_per_op_sec": round(self.avg_time_per_op(), 6),
        }
        if self._ops_by_kind:
            out["ops_by_kind"] = {k: int(v) for k, v in self._ops_by_kind.items()}
            out["avg_time_per_kind_sec"] = {k: round(v, 6) for k, v in self.avg_time_per_kind().items()}
        return out

    def summary_str(self) -> str:
        d = self.summary_dict()
        lines = []
        lines.append("[timing]")
        lines.append("  total_runtime_sec       = {:.3f}".format(d["total_runtime_sec"]))
        lines.append("  sim_runtime_sec         = {:.3f}".format(d["sim_runtime_sec"]))
        lines.append("  ops_total               = {}".format(d["ops_total"]))
        lines.append("  avg_time_per_op_sec     = {:.6f}".format(d["avg_time_per_op_sec"]))
        if "ops_by_kind" in d:
            lines.append("  ops_by_kind             = {}".format(d["ops_by_kind"]))
            lines.append("  avg_time_per_kind_sec   = {}".format(d["avg_time_per_kind_sec"]))
        return "\n".join(lines)
```

### `src/common/verify_motion.py`

```text
# -*- coding: utf-8 -*-
"""
common/verify_motion.py
--------------------------------
- 解析 TAG=AR_OP 的单行 JSON（place/drag/pinch/rotate/tap）
- 计算 per-kind 与 overall 成功率
- overall 的口径可配置：overall_basis ∈ {"place","tap"}（默认 "place"）
  * "place": overall = place + drag + pinch + rotate（tap 不计，避免 place+tap 双计）
  * "tap":   overall = tap   + drag + pinch + rotate（place 完全当作 tap 的特例，不计入 overall）

注：place 在 app 里发生时通常也会有一条 tap；若选 "tap" 作为口径，place 仍单独打印，但不会进 overall。
"""

import json
import re
import time
import threading
import subprocess
from collections import Counter
from typing import Optional, Dict, Any, List

TAG = "AR_OP"
JSON_RE = re.compile(r'\bAR_OP\b[^{]*({.*})')  # 兼容 "AR_OP   :" / "AR_OP(1234):" 等

KINDS_ALL = ("place_start", "place_ok", "place_fail", "drag", "pinch", "rotate", "tap")

def _safe_rate(ok: int, total: int) -> float:
    return 0.0 if total <= 0 else ok / float(total)

def parse_line(line: str) -> Optional[Dict[str, Any]]:
    if "AR_OP" not in line:
        return None
    m = JSON_RE.search(line)
    if not m:
        return None
    try:
        evt = json.loads(m.group(1))
        k = evt.get("kind")
        if isinstance(k, str):
            evt["kind"] = k.strip().lower()
        return evt
    except Exception:
        return None

class MotionStats:
    """线程安全计数与成功率。"""
    def __init__(self, overall_basis: str = "place") -> None:
        self._cnt = Counter()
        self._t0 = time.time()
        self._lock = threading.Lock()
        self._pending_place = 0
        self._overall_basis = overall_basis if overall_basis in ("place","tap") else "place"

    def set_overall_basis(self, basis: str) -> None:
        if basis not in ("place","tap"):
            basis = "place"
        with self._lock:
            self._overall_basis = basis

    def reset(self) -> None:
        with self._lock:
            self._cnt.clear()
            self._t0 = time.time()
            self._pending_place = 0

    def feed_event(self, evt: Dict[str, Any]) -> None:
        kind = str(evt.get("kind", "")).strip().lower()
        ok = bool(evt.get("ok", False))
        if kind not in KINDS_ALL:
            return
        with self._lock:
            if kind == "place_start":
                self._cnt["place_start_total"] += 1
                self._pending_place += 1
                return
            if kind == "place_ok":
                if self._pending_place > 0:
                    self._cnt["place_ok_total"] += 1
                    self._pending_place -= 1
                return
            if kind == "place_fail":
                if self._pending_place > 0:
                    self._pending_place -= 1
                self._cnt["place_fail_total"] += 1
                return
            # 其余：drag/pinch/rotate/tap
            self._cnt[f"{kind}_total"] += 1
            if ok:
                self._cnt[f"{kind}_ok"] += 1

    def feed_line(self, line: str) -> None:
        evt = parse_line(line)
        if evt:
            self.feed_event(evt)

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            c = dict(self._cnt)
            elapsed = time.time() - self._t0
            basis = self._overall_basis

        place_ok  = c.get("place_ok_total", 0)
        place_try = c.get("place_start_total", 0)
        drag_ok, drag_total = c.get("drag_ok", 0), c.get("drag_total", 0)
        pin_ok,  pin_total  = c.get("pinch_ok", 0), c.get("pinch_total", 0)
        rot_ok,  rot_total  = c.get("rotate_ok", 0), c.get("rotate_total", 0)
        tap_ok,  tap_total  = c.get("tap_ok", 0),  c.get("tap_total", 0)

        if basis == "tap":
            ok_sum  = tap_ok + drag_ok + pin_ok + rot_ok
            try_sum = tap_total + drag_total + pin_total + rot_total
        else:  # "place"
            ok_sum  = place_ok + drag_ok + pin_ok + rot_ok
            try_sum = place_try + drag_total + pin_total + rot_total

        return {
            "elapsed_sec": elapsed,
            "counts": c,
            "place":   {"ok": place_ok, "total": place_try, "rate": _safe_rate(place_ok, place_try)},
            "drag":    {"ok": drag_ok,  "total": drag_total, "rate": _safe_rate(drag_ok, drag_total)},
... (truncated)
```

### `src/detector/__init__.py`

```text
# src/detector/__init__.py
from .yolo_detector import YOLODetector

__all__ = ["YOLODetector"]
```

### `src/detector/yolo_detector.py`

```text
# src/detector/yolo_detector.py
# -*- coding: utf-8 -*-
"""
YOLO-based detector for AR objects and UI elements.

This module wraps an Ultralytics YOLO model and exposes a simple
`detect(frame_bgr)` interface that returns a list of detected objects
in a unified format, e.g.:

{
    "objects": [
        {
            "id": 0,
            "cls": "AR_Object",
            "cls_id": 0,
            "score": 0.92,
            "bbox": [x, y, w, h],       # absolute pixels
            "center_xy": [cx, cy]       # absolute pixels
        },
        ...
    ],
    "meta": {
        "img_shape": [H, W, C],
        "num_dets": N
    }
}
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from ultralytics import YOLO


class YOLODetector:
    """
    Wrapper around Ultralytics YOLO for AR object / UI element detection.
    """

    def __init__(
        self,
        weights: str,
        imgsz: int = 640,
        conf: float = 0.25,
        max_det: int = 100,
        device: Optional[str] = None,
        classes: Optional[List[int]] = None,
        class_map: Optional[Dict[int, str]] = None,
    ) -> None:
        """
        Args:
            weights: Path to YOLO weights file (e.g. .pt).
            imgsz: Inference image size (short side).
            conf: Confidence threshold.
            max_det: Maximum number of detections.
            device: Device string for Ultralytics (e.g. 'cpu', '0', '0,1').
            classes: Optional list of class indices to keep.
            class_map: Optional mapping from class id to name, e.g. {0: 'AR_Object', 1: 'UI_Element'}.
        """
        self.weights = str(weights)
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.max_det = int(max_det)
        self.device = device
        self.classes = classes

        # 默认类别映射：你可以在 cfg 中覆盖
        if class_map is None:
            # 假设大多数情况下是二类：AR_Object / UI_Element
            class_map = {
                0: "AR_Object",
                1: "UI_Element",
            }
        self.class_map = class_map

        # 加载 YOLO 模型
        self.model = YOLO(self.weights)

    # --------------------------------------------------------------------- #
    # 工厂方法：从 config 字典构造（方便在 __main__.py 中使用）
    # --------------------------------------------------------------------- #
    @classmethod
    def from_cfg(cls, cfg: Dict[str, Any]) -> "YOLODetector":
        """
        Build YOLODetector from a config dict like cfg['yolo'].

        Expected cfg structure (example):

        yolo:
          weights: "cv/yolo/runs/detect/train/weights/best.pt"
          imgsz: 640
          conf: 0.05
          max_det: 100
          device: "0"
          classes: [0, 1]
          class_map:
            0: "AR_Object"
            1: "UI_Element"
        """
        yolo_cfg = cfg.get("yolo", {})

        weights = yolo_cfg.get("weights", "best.pt")
        imgsz = int(yolo_cfg.get("imgsz", 640))
        conf = float(yolo_cfg.get("conf", 0.25))
        max_det = int(yolo_cfg.get("max_det", 100))
        device = yolo_cfg.get("device", None)

        classes = yolo_cfg.get("classes", None)
        if classes is not None:
            classes = [int(c) for c in classes]

        class_map_cfg = yolo_cfg.get("class_map", None)
        if class_map_cfg is not None:
            # YAML 读出来可能是 {0: 'AR_Object'} 或 {'0': 'AR_Object'}
            class_map = {int(k): str(v) for k, v in class_map_cfg.items()}
        else:
            class_map = None
... (truncated)
```

### `src/discovery/__init__.py`

```text
```

### `src/discovery/run_discovery.py`

```text
# src/discovery/run_discovery.py
# -*- coding: utf-8 -*-
"""
Action Discovery main loop.

Detector  -> YOLODetector       (src/detector/yolo_detector.py)
Sampler   -> DefaultSampler     (src/sampler/default_sampler.py)
Executor  -> AppiumExecutor     (src/executor/appium_executor.py)
Verifier  -> Verifier           (src/verifier/verifier.py)
Policy    -> NMPolicy           (src/policy/policy.py)

入口函数:
    run_once(drv, detector, sampler, executor, cfg) -> bool
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from src.policy import NMPolicy
from src.verifier import Verifier


@dataclass
class DiscoveryConfig:
    ops: List[str]
    N: int
    M: int
    post_wait_s: float
    max_targets: int
    out_jsonl: Path
    support_jsonl: Path


def _build_cfg(cfg: Dict[str, Any]) -> DiscoveryConfig:
    runtime = cfg.get("runtime", {})
    policy_cfg = cfg.get("policy", {})

    ops = cfg.get("ops") or runtime.get("ops") or ["drag", "rotate", "tap"]
    if isinstance(ops, str):
        ops = [s.strip() for s in ops.split(",") if s.strip()]

    N = int(policy_cfg.get("N", 10))
    M = int(policy_cfg.get("M", 2))
    post_wait_s = float(runtime.get("post_wait_s", 0.4))
    max_targets = int(runtime.get("max_targets", 3))

    out_jsonl = Path(runtime.get("out_jsonl", "runs/exp/trials.jsonl"))
    support_jsonl = Path(runtime.get("support_jsonl", "runs/exp/support.jsonl"))
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    return DiscoveryConfig(
        ops=ops,
        N=N,
        M=M,
        post_wait_s=post_wait_s,
        max_targets=max_targets,
        out_jsonl=out_jsonl,
        support_jsonl=support_jsonl,
    )


# --------------------------------------------------------------------- #
# Target selection
# --------------------------------------------------------------------- #
def _select_targets(det_result: Dict[str, Any], dc: DiscoveryConfig) -> List[Dict[str, Any]]:
    """
    从 detector 输出中选出若干 target。

    det_result 期望格式:
        {
          "objects": [
             {"id": 0, "cls": "AR_Object", "bbox": [...], "center_xy": [...], "score": ...},
             ...
          ],
          "meta": {...}
        }
    """
    objs = det_result.get("objects", []) or []
    if not objs:
        return []

    ar_objs = [o for o in objs if str(o.get("cls", "")).lower() == "ar_object"]
    if not ar_objs:
        ar_objs = objs  # 兜底: 没有标明 AR_Object 时直接用所有对象

    targets: List[Dict[str, Any]] = []
    for i, obj in enumerate(ar_objs[: dc.max_targets]):
        bbox = obj.get("bbox", [0.0, 0.0, 0.0, 0.0])
        cx, cy = obj.get("center_xy", [bbox[0] + bbox[2] / 2.0, bbox[1] + bbox[3] / 2.0])

        targets.append(
            {
                "target_id": i,
                "det_id": obj.get("id", i),
                "cls": obj.get("cls", ""),
                "score": float(obj.get("score", 0.0)),
                "bbox": [float(b) for b in bbox],
                "center_xy": [float(cx), float(cy)],
            }
        )
    return targets


# --------------------------------------------------------------------- #
# Main entry
# --------------------------------------------------------------------- #
def run_once(
    drv: Any,
    detector: Any,
    sampler: Any,
    executor: Any,
    cfg: Dict[str, Any],
) -> bool:
... (truncated)
```

### `src/executor/__init__.py`

```text
# src/executor/__init__.py
from .appium_executor import AppiumExecutor

__all__ = ["AppiumExecutor"]
```

### `src/executor/appium_executor.py`

```text
# src/executor/appium_executor.py
# -*- coding: utf-8 -*-
"""
Appium-based executor for AR interaction events.

This module wraps your existing Appium + gesture utilities
in `common/device.py` and `common/actions.py` and exposes:

- snapshot_screen() -> np.ndarray (BGR image)
- perform(op, region, params) -> None

`region` is expected to be a dict like:
{
    "target_id": int,
    "bbox": [x, y, w, h],         # usually in window coords
    "center_xy": [cx, cy],
    "region_type": "on_object" | "off_object",
    ...
}

`params` is an op-specific parameter dict generated by the sampler.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..common import actions, device


@dataclass
class ExecutorConfig:
    """Lightweight config holder for the executor."""
    device_serial: Optional[str]
    app_package: str
    app_activity: Optional[str]
    warmup_wait: float = 3.0

    default_tap_ms: int = 80
    default_drag_ms: int = 300
    default_rotate_ms: int = 400
    default_pinch_ms: int = 400


class AppiumExecutor:
    """
    High-level executor that wraps an Appium driver and your gesture
    primitives (tap/drag/pinch/rotate).

    Design:
      - Executor only understands a few primitive action *families*:
        'tap', 'drag', 'rotate', 'pinch'.
      - High-level ops like 'drag_short', 'rotate_cw', 'pinch_small'
        should be reduced to these families via ACTION_FAMILY mapping.
      - All semantic differences (short/long/clockwise/...) should be
        handled in the sampler by generating different `params`.
    """

    # 映射高层 op 名称 -> 原子 family
    # 需要新增高层动作时，只改这里即可（以及 sampler 中的参数生成逻辑）
    ACTION_FAMILY: Dict[str, str] = {
        "tap": "tap",
        "single_tap": "tap",

        "drag": "drag",
        "drag_short": "drag",
        "drag_long": "drag",

        "rotate": "rotate",
        "rotate_cw": "rotate",
        "rotate_ccw": "rotate",

        "pinch": "pinch",
        "pinch_in": "pinch",
        "pinch_out": "pinch",
        "zoom_in": "pinch",
        "zoom_out": "pinch",
    }

    def __init__(
        self,
        cfg: Dict[str, Any],
        device_id: Optional[str] = None,
        driver: Optional[Any] = None,
    ) -> None:
        """
        Args:
            cfg: Global config dict (parsed from YAML).
            device_id: Optional override for device serial/udid.
            driver: Optional existing Appium driver.
        """
        # 设备串号优先级：显式传入 > cfg["device_serial"] > cfg["device"]
        serial = device_id or cfg.get("device_serial") or cfg.get("device")

        app_cfg = cfg.get("app", {})
        pkg = app_cfg.get("package")
        activity = app_cfg.get("activity", "auto")

        if not pkg:
            raise ValueError(
                "AppiumExecutor: cfg['app']['package'] is required to launch the app."
            )

        self.config = ExecutorConfig(
            device_serial=serial,
            app_package=pkg,
            app_activity=activity,
            warmup_wait=float(cfg.get("runtime", {}).get("warmup_wait", 3.0)),
            default_tap_ms=int(cfg.get("runtime", {}).get("default_tap_ms", 80)),
            default_drag_ms=int(cfg.get("runtime", {}).get("default_drag_ms", 300)),
            default_rotate_ms=int(cfg.get("runtime", {}).get("default_rotate_ms", 400)),
            default_pinch_ms=int(cfg.get("runtime", {}).get("default_pinch_ms", 400)),
        )

        if driver is not None:
            self.driver = driver
        else:
... (truncated)
```

### `src/policy/__init__.py`

```text
# src/policy/__init__.py
from .policy import NMPolicy

__all__ = ["NMPolicy"]
```

### `src/policy/policy.py`

```text
# src/policy/policy.py
# -*- coding: utf-8 -*-
"""
N/M policy for deciding whether an op is "supported" on a target.

Usage:
    policy = NMPolicy.from_cfg(cfg)
    support = policy.decide_support(results)  # results: list[bool]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List


@dataclass
class NMPolicy:
    N: int = 10
    M: int = 2

    @classmethod
    def from_cfg(cls, cfg: Dict[str, Any]) -> "NMPolicy":
        pc = cfg.get("policy", {})
        return cls(
            N=int(pc.get("N", 10)),
            M=int(pc.get("M", 2)),
        )

    def decide_support(self, results: List[bool]) -> bool:
        """
        Args:
            results: list of booleans for one (target, op) pair.

        Returns:
            True if the op is considered supported (>= M successes),
            False otherwise.
        """
        if not results:
            return False
        ok_count = sum(1 for r in results if r)
        return ok_count >= self.M
```

### `src/sampler/__init__.py`

```text
# src/sampler/__init__.py
from .default_sampler import DefaultSampler

__all__ = ["DefaultSampler"]
```

### `src/sampler/default_sampler.py`

```text
# src/sampler/default_sampler.py
# -*- coding: utf-8 -*-
"""
Default sampler for AR interaction events.

This module is responsible for generating operation-specific parameter
dicts for actions such as:

- tap, single_tap
- drag, drag_short, drag_long
- rotate, rotate_cw, rotate_ccw
- pinch, pinch_in, pinch_out, zoom_in, zoom_out

The Executor will interpret these parameters and call low-level gesture
primitives (tap/drag_line/rotate/pinch_or_zoom).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


@dataclass
class SamplerConfig:
    """Config for DefaultSampler with reasonable defaults."""
    seed: int = 42

    # tap
    tap_jitter_frac: float = 0.05      # center 抖动范围 ~ min(w,h)*tap_jitter_frac
    tap_duration_ms_min: int = 50
    tap_duration_ms_max: int = 120

    # drag
    drag_len_frac_min: float = 0.2     # 相对于目标 bbox 对角线的比例
    drag_len_frac_max: float = 0.6
    drag_duration_ms_min: int = 200
    drag_duration_ms_max: int = 400

    # rotate
    rotate_angle_min_deg: float = 20.0
    rotate_angle_max_deg: float = 90.0
    rotate_radius_frac: float = 0.6    # radius ~ min(w,h)*radius_frac
    rotate_steps_min: int = 6
    rotate_steps_max: int = 10
    rotate_duration_ms_min: int = 300
    rotate_duration_ms_max: int = 500

    # pinch / zoom
    pinch_radius_frac: float = 0.8     # 初始两指间距的一半 ~ min(w,h)*pinch_radius_frac
    pinch_base_scale_min: float = 0.15
    pinch_base_scale_max: float = 0.35
    pinch_duration_ms_min: int = 280
    pinch_duration_ms_max: int = 480


class DefaultSampler:
    """
    Default sampler that generates low-level parameters for different ops.

    Usage:
        sampler = DefaultSampler.from_cfg(cfg)
        params = sampler.sample("drag", region)
    """

    def __init__(self, config: SamplerConfig) -> None:
        self.config = config
        self.rng = random.Random(config.seed)

    # ------------------------------------------------------------------ #
    # 工厂方法：从全局 cfg 中构造
    # ------------------------------------------------------------------ #
    @classmethod
    def from_cfg(cls, cfg: Dict[str, Any]) -> "DefaultSampler":
        sampler_cfg = cfg.get("sampler", {})
        seed = int(cfg.get("seed", sampler_cfg.get("seed", 42)))

        sc = SamplerConfig(
            seed=seed,
            tap_jitter_frac=float(sampler_cfg.get("tap_jitter_frac", 0.05)),
            tap_duration_ms_min=int(sampler_cfg.get("tap_duration_ms_min", 50)),
            tap_duration_ms_max=int(sampler_cfg.get("tap_duration_ms_max", 120)),
            drag_len_frac_min=float(sampler_cfg.get("drag_len_frac_min", 0.2)),
            drag_len_frac_max=float(sampler_cfg.get("drag_len_frac_max", 0.6)),
            drag_duration_ms_min=int(sampler_cfg.get("drag_duration_ms_min", 200)),
            drag_duration_ms_max=int(sampler_cfg.get("drag_duration_ms_max", 400)),
            rotate_angle_min_deg=float(sampler_cfg.get("rotate_angle_min_deg", 20.0)),
            rotate_angle_max_deg=float(sampler_cfg.get("rotate_angle_max_deg", 90.0)),
            rotate_radius_frac=float(sampler_cfg.get("rotate_radius_frac", 0.6)),
            rotate_steps_min=int(sampler_cfg.get("rotate_steps_min", 6)),
            rotate_steps_max=int(sampler_cfg.get("rotate_steps_max", 10)),
            rotate_duration_ms_min=int(sampler_cfg.get("rotate_duration_ms_min", 300)),
            rotate_duration_ms_max=int(sampler_cfg.get("rotate_duration_ms_max", 500)),
            pinch_radius_frac=float(sampler_cfg.get("pinch_radius_frac", 0.8)),
            pinch_base_scale_min=float(sampler_cfg.get("pinch_base_scale_min", 0.15)),
            pinch_base_scale_max=float(sampler_cfg.get("pinch_base_scale_max", 0.35)),
            pinch_duration_ms_min=int(sampler_cfg.get("pinch_duration_ms_min", 280)),
            pinch_duration_ms_max=int(sampler_cfg.get("pinch_duration_ms_max", 480)),
        )
        return cls(sc)

    # ------------------------------------------------------------------ #
    # 公共接口：为给定 op & region 采样参数
    # ------------------------------------------------------------------ #
    def sample(self, op: str, region: Dict[str, Any]) -> Dict[str, Any]:
        op = op.lower()
        if op in ("tap", "single_tap"):
            return self._sample_tap(region)
        elif op.startswith("drag"):
            # drag, drag_short, drag_long, drag_diagonal, etc.
            return self._sample_drag(region, op=op)
        elif op.startswith("rotate"):
            # rotate, rotate_cw, rotate_ccw, rotate_small, etc.
            return self._sample_rotate(region, op=op)
        elif op.startswith("pinch") or op.startswith("zoom"):
            # pinch, pinch_in, pinch_out, zoom_in, zoom_out
            return self._sample_pinch(region, op=op)
        else:
... (truncated)
```

### `src/verifier/__init__.py`

```text
# src/verifier/__init__.py
from .verifier import Verifier

__all__ = ["Verifier"]
```

### `src/verifier/backends/motion_similarity.py`

```text
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
... (truncated)
```

### `src/verifier/verifier.py`

```text
# src/verifier/verifier.py
# -*- coding: utf-8 -*-
"""
High-level Verifier that wraps motion-based backends.

Currently uses:
    src/verifier/backends/motion_similarity.py

API:
    Verifier.from_cfg(cfg)
    Verifier.verify(op, pre_bgr, post_bgr, region, params) -> bool
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np

from src.verifier.backends import motion_similarity as ms


@dataclass
class VerifierConfig:
    tau_move: float = 0.03     # normalized (relative to image diagonal)
    tau_rot_deg: float = 8.0
    tau_scale: float = 0.10
    min_frac: float = 0.5      # 最少多少比例的 inlier 才算成功


class Verifier:
    def __init__(self, config: VerifierConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------ #
    # Factory from cfg
    # ------------------------------------------------------------------ #
    @classmethod
    def from_cfg(cls, cfg: Dict[str, Any]) -> "Verifier":
        thr = cfg.get("thresholds", {})
        vc = VerifierConfig(
            tau_move=float(thr.get("tau_move", 0.03)),
            tau_rot_deg=float(thr.get("tau_rot_deg", thr.get("tau_rot_deg", 8.0))),
            tau_scale=float(thr.get("tau_scale", 0.10)),
            min_frac=float(thr.get("min_frac", 0.5)),
        )
        return cls(vc)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def verify(
        self,
        op: str,
        pre_bgr: np.ndarray,
        post_bgr: np.ndarray,
        region: Dict[str, Any],
        params: Dict[str, Any],
    ) -> bool:
        """
        Args:
            op: high-level op name ('drag', 'drag_short', 'rotate', 'rotate_cw',
                'pinch_in', 'pinch_out', 'zoom_in', 'zoom_out', etc.)
            pre_bgr/post_bgr: HxWx3 uint8 images (BGR)
            region: dict with at least 'bbox' and 'center_xy'
            params: sampler-generated parameter dict

        Returns:
            bool: whether the action is considered successful on this target.
        """
        op = op.lower()
        bbox = region.get("bbox", [0.0, 0.0, 0.0, 0.0])
        center_xy = region.get(
            "center_xy",
            [bbox[0] + bbox[2] / 2.0, bbox[1] + bbox[3] / 2.0],
        )
        cx, cy = float(center_xy[0]), float(center_xy[1])
        bbox_tuple: Tuple[float, float, float, float] = (
            float(bbox[0]),
            float(bbox[1]),
            float(bbox[2]),
            float(bbox[3]),
        )

        h, w = pre_bgr.shape[:2]
        img_diag = float((w * w + h * h) ** 0.5)

        # ---- drag family ------------------------------------------------
        if op.startswith("drag"):
            dx = float(params.get("dx", 0.0))
            dy = float(params.get("dy", 0.0))
            start_xy = (cx, cy)
            end_xy = (cx + dx, cy + dy)

            extra = {
                "start_xy": start_xy,
                "end_xy": end_xy,
                "min_motion_px": self.config.tau_move * img_diag,
                "min_dir_cos": 0.6,
                "min_frac": self.config.min_frac,
            }
            return ms.verify_action(
                op="drag",
                pre_bgr=pre_bgr,
                post_bgr=post_bgr,
                center_xy=(cx, cy),
                bbox=bbox_tuple,
                extra=extra,
            )

        # ---- rotate family ---------------------------------------------- #
        if op.startswith("rotate"):
            extra = {
                "min_deg": self.config.tau_rot_deg,
                "min_frac": self.config.min_frac,
            }
            return ms.verify_action(
                op="rotate",
                pre_bgr=pre_bgr,
... (truncated)
```

### `yolo_adb_screencap.py`

```text
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
... (truncated)
```

### `yolo_scratch.py`

```text
"""
python yolo_scratch.py
"""

from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("cv/yolo/best.pt")

# Define path to the image file
source = "val/Screenshot_20251023-165222.png"

# Run inference on the source
results = model(source, conf=0.05, iou=0.45, imgsz=960, verbose=False)  # list of Results objects


for result in results:
    boxes = result.boxes.xyxy  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    print(
        f"boxes: {boxes}\n"
        f"masks: {masks}\n"
        f"keypoints: {keypoints}\n"
        f"probs: {probs}\n"
        f"obb: {obb}\n"
    )
    result.show()  # display to screen
    result.save(filename="result.jpg")
    
```

### `yolo_val.py`

```text
from ultralytics import YOLO

# Load a model
model = YOLO("cv/yolo/best.pt")

# Customize validation settings
metrics = model.val(data="cv/yolo/coco8.yaml", imgsz=640)
```

## Excludes

```
.git
.svn
.hg
.DS_Store
__pycache__
*.pyc
*.pyo
*.pyd
.idea
.vscode
.venv
.venv311
venv
env
ENV
.conda
conda
.env
build
dist
.mypy_cache
.pytest_cache
.ruff_cache
node_modules
runs
outputs
logs
*.pt
*.onnx
*.ckpt
*.mp4
*.mov
*.avi
*.png
*.jpg
*.jpeg
*.webp
*.zip
*.tar
*.gz
*.7z
```
