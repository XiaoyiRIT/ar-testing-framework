# src/verifier/verifier.py
from typing import Dict, Any, Tuple
import numpy as np
from .backends import motion_similarity as ms  # 你的 verify_motion.py

class Verifier:
    def __init__(self, tau_move=0.03, tau_rot_deg=8.0, tau_ssim_delta=0.08, use_diag_norm=False):
        self.tau_move = tau_move
        self.tau_rot_deg = tau_rot_deg
        self.tau_ssim_delta = tau_ssim_delta
        self.use_diag_norm = use_diag_norm

    def _norm_to_pixels(self, bbox, img_shape):
        H, W = img_shape[:2]
        if self.use_diag_norm:
            w, h = bbox[2], bbox[3]
            D = (w**2 + h**2) ** 0.5
            return max(1.0, D)
        else:
            return float(min(W, H))

    def verify(self, op_type: str,
               before_img: np.ndarray, after_img: np.ndarray,
               det_before: Dict[str,Any], det_after: Dict[str,Any],
               target_id: int, extra: Dict[str,Any]) -> Tuple[bool, Dict[str, Any], Dict[str, float]]:
        # 取目标框与中心（你已有 detector 的输出，这里略）
        bbox = extra["bbox"]          # (x,y,w,h)
        center = extra["center_xy"]   # (cx,cy)
        img_scale = self._norm_to_pixels(bbox, before_img.shape)

        # —— 几何证据（用你的后端） ——
        ms_extra = {}
        if op_type == "drag":
            ms_extra.update({
                "start_xy": extra["start_xy"],
                "end_xy":   extra["end_xy"],
                "min_motion_px": max(8.0, self.tau_move * img_scale),
                "min_dir_cos": 0.6,
                "min_frac": 0.5,
            })
            ms_op = "drag"
        elif op_type == "pinch":
            # 你后端区分 pinch_in/pinch_out，这里用 scale_sign 指定
            ms_op = "pinch_out" if extra.get("scale_sign", +1) > 0 else "pinch_in"
            ms_extra.update({
                "scale_thr": max(0.06, self.tau_ssim_delta),  # 初版：把 τ_ssim 当作纹理变化下限的近似替代
                "min_frac": 0.5,
            })
        elif op_type == "rotate":
            ms_op = "rotate"
            ms_extra.update({
                "min_deg": max(5.0, self.tau_rot_deg),  # 直接对齐你的角度阈值
                "min_frac": 0.5,
            })
        else:
            ms_op = op_type  # 兼容后续新op

        ok_geom = ms.verify_action(
            op=ms_op,
            pre_bgr=before_img, post_bgr=after_img,
            center_xy=center, bbox=bbox, extra=ms_extra
        )

        # —— 汇总（当前仅用几何证据，后续再接 FoELS/SSIM/光流像素统计） ——
        metrics = {"ok_geom": float(ok_geom)}
        success = ok_geom
        evidence = {"geom_backend": "motion_similarity", "bbox": bbox, "center": center}
        return success, evidence, metrics
