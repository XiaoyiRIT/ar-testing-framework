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
                post_bgr=post_bgr,
                center_xy=(cx, cy),
                bbox=bbox_tuple,
                extra=extra,
            )

        # ---- pinch / zoom family ---------------------------------------- #
        if op.startswith("pinch") or op.startswith("zoom"):
            # op: pinch_in / pinch_out / zoom_in / zoom_out / pinch
            if op in ("pinch_in", "zoom_out"):
                pinch_in = True
            elif op in ("pinch_out", "zoom_in"):
                pinch_in = False
            else:
                # 通过 scale_sign 决定 in/out
                scale_sign = int(params.get("scale_sign", -1))
                pinch_in = scale_sign < 0

            ms_op = "pinch_in" if pinch_in else "pinch_out"

            extra = {
                "scale_thr": self.config.tau_scale,
                "min_frac": self.config.min_frac,
            }
            return ms.verify_action(
                op=ms_op,
                pre_bgr=pre_bgr,
                post_bgr=post_bgr,
                center_xy=(cx, cy),
                bbox=bbox_tuple,
                extra=extra,
            )

        # ---- tap: 暂时不做视觉验证，直接返回 False --------------------- #
        # 后续你可以接 FoELS 或 SSIM 做「有无变化」检测。
        if op.startswith("tap"):
            return False

        # 未知 action：保守返回 False
        return False
