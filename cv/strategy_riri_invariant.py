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

    # 只在足够流量处计算
    m = mag > min_flow_mag
    vx = F[...,0]; vy = F[...,1]
    ux = vF[...,0]; uy = vF[...,1]

    eps = 1e-6
    r_obs = np.arctan2(vy, vx + eps)
    r_exp = np.arctan2(uy, ux + eps)
    dtheta = np.abs(r_obs - r_exp)
    # 归一化到 [0,pi] 再折叠到 [0,pi/2]（方向无符号）
    dtheta = np.minimum(dtheta, np.pi - dtheta)
    dtheta_deg = (dtheta * 180.0 / np.pi)

    # 6) 阈值 + 形态学 + 最大连通域
    bw = (dtheta_deg >= angle_thr_deg) & m
    bw = bw.astype(np.uint8)
    k = np.ones((3,3), np.uint8)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k, iterations=1)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k, iterations=1)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    if num <= 1:
        if debug_profile:
            total = _ms(Tall)
            print(f"[RIRI] resize={t_resize:.1f}ms gray={t_gray:.1f}ms flow={t_flow:.1f}ms "
                  f"camchk={t_cam:.1f}ms foe={t_foe:.1f}ms post={_ms(T):.1f}ms TOTAL={total:.1f}ms :: no CC")
        return None

    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h, a = (stats[idx, cv2.CC_STAT_LEFT],
                     stats[idx, cv2.CC_STAT_TOP],
                     stats[idx, cv2.CC_STAT_WIDTH],
                     stats[idx, cv2.CC_STAT_HEIGHT],
                     stats[idx, cv2.CC_STAT_AREA])
    area_img = Hs * Ws
    if a < max(12, int(area_img * min_area_ratio)):
        if debug_profile: print(f"[RIRI:None] CC too small {a}")
        return None

    # 7) 映射回原图
    inv = 1.0 / s
    Xo = int(x * inv); Yo = int(y * inv); Wo = int(w * inv); Ho = int(h * inv)
    CX = int((x + w//2) * inv); CY = int((y + h//2) * inv)

    if debug_profile:
        total = _ms(Tall)
        print(f"[RIRI] resize={t_resize:.1f}ms gray={t_gray:.1f}ms flow={t_flow:.1f}ms "
              f"camchk={t_cam:.1f}ms foe={t_foe:.1f}ms post={_ms(T):.1f}ms TOTAL={total:.1f}ms "
              f"bbox={(Xo,Yo,Wo,Ho)}")

    return {"center": (CX, CY), "bbox": (Xo, Yo, Wo, Ho)}
