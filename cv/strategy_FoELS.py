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
    if keep.sum() < 60: return None
    xs = xs[keep]; ys = ys[keep]; vec = vec[keep]
    pts  = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
    dirs = _unit(vec)

    # ΣP 与 ΣPp
    S = np.zeros((2,2), np.float32)
    b = np.zeros((2,), np.float32)
    I = np.eye(2, dtype=np.float32)
    for (p, d) in zip(pts, dirs):
        P = I - np.outer(d, d)  # 垂直于流向的投影
        S += P
        b += P @ p
    # 稳定性：加极小 Tikhonov
    S += 1e-6 * I
    try:
        foe = np.linalg.solve(S, b)
    except np.linalg.LinAlgError:
        return None
    # 符号：与 RANSAC 一致（用整体投票）
    vF = pts - foe
    sign_votes = np.sign(np.sum(np.sum(vF * vec, axis=1)))
    sign = 1.0 if sign_votes >= 0 else -1.0
    return dict(foe=foe, inliers=None, score=-1, sign=sign)

# -------------------- likelihood --------------------
def _likelihood(flow: np.ndarray, foe_xy: np.ndarray, sign: float,
                static_mag_ref: float, theta_th_deg: float, alpha_len: float) -> np.ndarray:
    H, W, _ = flow.shape
    X, Y = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    V = np.stack([X - foe_xy[0], Y - foe_xy[1]], axis=-1) * sign
    Vn = _unit(V); Fn = _unit(flow)
    cosang = np.clip(np.sum(Vn * Fn, axis=-1), -1.0, 1.0)
    #cos_th = np.cos(np.deg2rad(theta_th_deg))
    #Pa = np.clip((1.0 - cosang) / (1.0 - cos_th + 1e-6), 0.0, 1.0)
    Pa = (1.0 - cosang) * 0.5
    mag = np.linalg.norm(flow, axis=-1) + 1e-6
    dl = mag / max(1e-3, static_mag_ref)
    Fl = np.abs(np.log10(np.maximum(dl, 1e-6)))
    Fl = np.clip(Fl, 0.0, 1.0)
    PFoE = np.clip(Pa + alpha_len * Fl, 0.0, 1.0)
    return PFoE

# -------------------- public API --------------------
def _heavy_fallback(prev_src: np.ndarray, curr_src: np.ndarray,
                    prior_small: np.ndarray,
                    max_w_heavy: int = 720,
                    theta_th_deg: float = 25.0,
                    alpha_len: float = 0.30,
                    high_q: float = 0.80,
                    low_q: float = 0.60,
                    topk_frac: float = 0.10,
                    debug: bool = False) -> Optional[Dict[str, Any]]:
    """TV-L1 + 高分辨率 + 滞后阈值 + Top-K 掐尖；返回原图尺度 bbox 或 None"""
    def _resize_to_w(img, Wt):
        h, w = img.shape[:2]
        if w == Wt: return img, 1.0
        s = Wt / w
        return cv2.resize(img, (int(w*s), int(h*s)), cv2.INTER_AREA), s

    curr_h, s1 = _resize_to_w(curr_src, max_w_heavy)
    prev_h, s2 = _resize_to_w(prev_src,  max_w_heavy)
    if abs(s1 - s2) > 1e-6:
        return None

    cg = cv2.cvtColor(curr_h, cv2.COLOR_BGR2GRAY) if curr_h.ndim==3 else curr_h
    pg = cv2.cvtColor(prev_h,  cv2.COLOR_BGR2GRAY) if prev_h.ndim==3  else prev_h

    try:
        # 先试 Farneback（快很多）
        flow = _flow_farneback(pg, cg)
    except Exception:
        # 极端情况下再试 TV-L1
        flow = _flow_tvl1(pg, cg)

    Hs, Ws = cg.shape[:2]
    mag = np.linalg.norm(flow, axis=-1)

    foe_est = _foe_lsq(flow, np.ones_like(mag, dtype=bool), min_mag=0.08)
    if foe_est is None:
        return None
    foe, sign = foe_est["foe"], foe_est["sign"]

    pr = prior_small
    if pr.shape[:2] != (Hs, Ws):
        pr = cv2.resize(prior_small.astype(np.float32), (Ws, Hs), cv2.INTER_AREA)
    pr = np.clip(pr.astype(np.float32), 0.0, 1.0)

    ref_mag = max(float(np.median(mag)), 1e-3)
    pfoe = _likelihood(flow, foe_xy=foe, sign=sign,
                       static_mag_ref=ref_mag,
                       theta_th_deg=theta_th_deg,
                       alpha_len=alpha_len)
    post = np.clip(pr * pfoe, 0.0, 1.0)

    hi = float(np.quantile(post, high_q))
    lo = float(np.quantile(post, low_q))
    
    if hi >= 0.999 or lo >= 0.999:
        kth = float(np.quantile(post, 1.0 - topk_frac))  # 例如 10%“掐尖”
        bw = (post >= kth).astype(np.uint8)
    else:
        seeds = (post >= hi).astype(np.uint8)
        mask  = (post >= lo).astype(np.uint8)
        num, labels = cv2.connectedComponents(mask, connectivity=8)
        if num > 1 and seeds.any():
            keep = np.zeros_like(mask)
            for lid in range(1, num):
                if (seeds[labels == lid].any()):
                    keep[labels == lid] = 1
            bw = keep
        else:
            bw = seeds
    
    seeds = (post >= hi).astype(np.uint8)
    mask  = (post >= lo).astype(np.uint8)

    num, labels = cv2.connectedComponents(mask, connectivity=8)
    if num > 1 and seeds.any():
        keep = np.zeros_like(mask)
        for lid in range(1, num):
            if (seeds[labels == lid].any()):
                keep[labels == lid] = 1
        bw = keep
    else:
        bw = (post >= hi).astype(np.uint8)

    if cv2.countNonZero(bw) == 0:
        kth = float(np.quantile(post, 1.0 - topk_frac))
        bw = (post >= kth).astype(np.uint8)

    k = np.ones((3,3), np.uint8)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k, iterations=1)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k, iterations=1)

    num, _, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    if num <= 1:
        return None

    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h = (stats[idx, cv2.CC_STAT_LEFT],
                  stats[idx, cv2.CC_STAT_TOP],
                  stats[idx, cv2.CC_STAT_WIDTH],
                  stats[idx, cv2.CC_STAT_HEIGHT])

    inv = 1.0 / s1
    X = int(x * inv); Y = int(y * inv); Wb = int(w * inv); Hb = int(h * inv)
    CX = int((x + w//2) * inv); CY = int((y + h//2) * inv)

    if debug:
        print(f"[FoELS-HEAVY] hi/lo=({hi:.3f},{lo:.3f})  bbox={(X,Y,Wb,Hb)}")
    return {"center": (CX, CY), "bbox": (X, Y, Wb, Hb)}



def locate(curr_bgr: np.ndarray,
           prev_bgr: Optional[np.ndarray] = None,
           min_area_ratio: float = 0.0015,
           max_w: int = 720,
           seg_prior: Optional[np.ndarray] = None,
           flow_method: str = 'farneback',  # 'farneback' | 'tvl1'
           rand_trials: int = 240,
           theta_th_deg: float = 30.0,
           alpha_len: float = 0.05,
           camera_move_thr: float = 0.01,
           min_flow_mag: float = 0.12,
           min_static_prior: float = 0.2,
           post_thr: Optional[float] = None,  # None → adaptive
           debug_profile: bool = False) -> Optional[Dict[str, Any]]:

    if prev_bgr is None:
        if debug_profile: print("[FoELS:None] no prev frame")
        return None

    Tall = time.perf_counter()

    # 1) resize / gray
    T = time.perf_counter()
    c_small, s  = _resize_keep(curr_bgr, max_w)
    p_small, s2 = _resize_keep(prev_bgr, max_w)
    t_resize = _ms(T)
    if abs(s - s2) > 1e-6:
        if debug_profile: print(f"[FoELS:None] resize scale mismatch s={s:.4f}, s2={s2:.4f}")
        return None

    T = time.perf_counter()
    c_g = _to_gray(c_small); p_g = _to_gray(p_small)
    t_gray = _ms(T)

    # 2) flow
    T = time.perf_counter()
    backend = flow_method
    try:
        flow = _flow_tvl1(p_g, c_g) if flow_method == 'tvl1' else _flow_farneback(p_g, c_g)
    except Exception:
        flow = _flow_farneback(p_g, c_g); backend = 'farneback'
    t_flow = _ms(T)

    # 3) prior & camera-moving check
    T = time.perf_counter()
    Hs, Ws = c_small.shape[:2]
    mag = np.linalg.norm(flow, axis=-1)
    if seg_prior is None:
        prior = np.ones((Hs, Ws), np.float32)
    else:
        pr = seg_prior.astype(np.float32)
        if pr.shape[:2] != (Hs, Ws):
            pr = cv2.resize(pr, (Ws, Hs), interpolation=cv2.INTER_AREA)
        prior = np.clip(pr, 0.0, 1.0)

    static_mask = (prior <= min_static_prior)
    if static_mask.any():
        ratio_move = float((mag[static_mask] > min_flow_mag).mean())
    else:
        ratio_move = float((mag > min_flow_mag).mean())
    t_cam = _ms(T)

    if ratio_move < camera_move_thr:
        if debug_profile: print(f"[FoELS:None] camera static ratio={ratio_move:.3f} < thr={camera_move_thr}")
        return None

    # 4) FoE: RANSAC → fallback to LSQ
    T = time.perf_counter()
    rmask = static_mask if static_mask.any() else (mag > min_flow_mag)
    foe_est = _foe_ransac(flow, rmask, trials=rand_trials, ang_tol_deg=16.0, min_mag=min_flow_mag)
    if foe_est is None:
        foe_est = _foe_lsq(flow, rmask, min_mag=min_flow_mag)
        t_ransac = _ms(T)
        if foe_est is None:
            if debug_profile:
                print("[FoELS:None] RANSAC failed (no foe) & LSQ fallback failed")
                print(f"[FoELS] flow={t_flow:.1f}ms RANSAC=fail ({t_ransac:.1f}ms)")
            return None
    else:
        t_ransac = _ms(T)

    foe, sign = foe_est["foe"], foe_est["sign"]
    if debug_profile and foe_est.get("score", -1) >= 0:
        print(f"[FoELS] foe=({foe[0]:.1f},{foe[1]:.1f}), inliers={int(foe_est['inliers'].sum())}")

    # 5) likelihood
    T = time.perf_counter()
    sm = mag[static_mask] if static_mask.any() else mag
    if sm.size == 0: sm = mag
    q30, q70 = np.quantile(sm, [0.30, 0.70])
    ref_mag = float(np.median(sm[(sm >= q30) & (sm <= q70)]))
    ref_mag = max(ref_mag, 1e-3)
    pfoe = _likelihood(flow, foe_xy=foe, sign=sign,
                       static_mag_ref=ref_mag,
                       theta_th_deg=theta_th_deg,
                       alpha_len=alpha_len)
    t_like = _ms(T)

    # 6) posterior & adaptive threshold with progressive relax
    T = time.perf_counter()
    post = np.clip(prior * pfoe, 0.0, 1.0)
    base = float(np.median(post[static_mask])) if static_mask.any() else float(np.median(post))
    post_rel = np.clip(post - base, 0.0, 1.0)

    def _make_bw(q, floor):
        thr_a = float(np.quantile(post_rel, q))
        thr = max(floor, thr_a) if post_thr is None else float(post_thr)
        bw = (post_rel >= thr).astype(np.uint8)
        k = np.ones((3,3), np.uint8)
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k, iterations=1)
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k, iterations=1)
        return bw

    # 逐级放宽分位：0.90 → 0.85 → 0.80 → 0.75
    bw = None
    for (q, floor) in [(0.90, 0.55), (0.85, 0.55), (0.80, 0.50), (0.75, 0.50)]:
        bw_try = _make_bw(q, floor)
        if cv2.countNonZero(bw_try) > 0:
            bw = bw_try; break

    if bw is None:
        if debug_profile:
            print("[FoELS] no CC in light path -> HEAVY fallback")
        fb = _heavy_fallback(prev_bgr, curr_bgr, prior,
                             max_w_heavy=960, theta_th_deg=25.0, alpha_len=0.30,
                             high_q=0.80, low_q=0.60, topk_frac=0.10,
                             debug=debug_profile)
        return fb

    num, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    if num <= 1:
        if debug_profile:
            print("[FoELS] empty CC after binarize -> HEAVY fallback")
        fb = _heavy_fallback(prev_bgr, curr_bgr, prior,
                             max_w_heavy=960, theta_th_deg=25.0, alpha_len=0.30,
                             high_q=0.80, low_q=0.60, topk_frac=0.10,
                             debug=debug_profile)
        return fb

    # 7) largest CC & sanity filters
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h, a = (stats[idx, cv2.CC_STAT_LEFT],
                     stats[idx, cv2.CC_STAT_TOP],
                     stats[idx, cv2.CC_STAT_WIDTH],
                     stats[idx, cv2.CC_STAT_HEIGHT],
                     stats[idx, cv2.CC_STAT_AREA])
    area_img = bw.shape[0] * bw.shape[1]
    # —— 新增：如果>92%画面，视为无效，强制改用 Top-K 更窄的掐尖 —— #
    if a > 0.92 * area_img:
        kth = float(np.quantile(post, 0.98))   # 前 2% 更强硬
        bw = (post >= kth).astype(np.uint8)
        num, _, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
        if num <= 1:
            return None
        idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        x, y, w, h = (stats[idx, cv2.CC_STAT_LEFT],
                      stats[idx, cv2.CC_STAT_TOP],
                      stats[idx, cv2.CC_STAT_WIDTH],
                      stats[idx, cv2.CC_STAT_HEIGHT])
    min_area = int(area_img * min_area_ratio)
    if a < max(12, min_area):
        if debug_profile: print(f"[FoELS:None] CC too small {a} < {max(12, min_area)}")
        return None

    t_post = _ms(T)
    cx, cy = int(x + w//2), int(y + h//2)

    # 8) map back
    inv = 1.0 / s
    X = int(x*inv); Y = int(y*inv); Wb = int(w*inv); Hb = int(h*inv)
    CX = int(cx*inv); CY = int(cy*inv)

    if debug_profile:
        total = _ms(Tall)
        print(f"[FoELS] resize={t_resize:.1f}ms gray={t_gray:.1f}ms flow({backend})={t_flow:.1f}ms "
              f"camchk={t_cam:.1f}ms RANSAC={t_ransac:.1f}ms like={t_like:.1f}ms post+cc={t_post:.1f}ms "
              f"TOTAL={total:.1f}ms")

    return {"center": (CX, CY), "bbox": (X, Y, Wb, Hb)}
