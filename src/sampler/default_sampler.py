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
            # 如果遇到未知操作，先保守返回一个空参数 + 默认 duration
            return {"duration_ms": 300}

    # ------------------------------------------------------------------ #
    # Helper: bbox / center
    # ------------------------------------------------------------------ #
    def _region_bbox_center(self, region: Dict[str, Any]) -> Tuple[float, float, float, float, float, float]:
        bbox = region.get("bbox", [0.0, 0.0, 0.0, 0.0])
        x, y, w, h = [float(v) for v in bbox]
        cx, cy = region.get("center_xy", [x + w / 2.0, y + h / 2.0])
        return x, y, w, h, float(cx), float(cy)

    # ------------------------------------------------------------------ #
    # tap 参数采样
    # ------------------------------------------------------------------ #
    def _sample_tap(self, region: Dict[str, Any]) -> Dict[str, Any]:
        x, y, w, h, cx, cy = self._region_bbox_center(region)

        jitter_r = min(w, h) * self.config.tap_jitter_frac
        theta = self.rng.uniform(0.0, 2.0 * math.pi)
        r = self.rng.uniform(0.0, jitter_r)

        dx = r * math.cos(theta)
        dy = r * math.sin(theta)

        duration_ms = self.rng.randint(
            self.config.tap_duration_ms_min,
            self.config.tap_duration_ms_max,
        )

        return {
            "dx": dx,
            "dy": dy,
            "duration_ms": duration_ms,
        }

    # ------------------------------------------------------------------ #
    # drag 参数采样
    # ------------------------------------------------------------------ #
    def _sample_drag(self, region: Dict[str, Any], op: str) -> Dict[str, Any]:
        x, y, w, h, cx, cy = self._region_bbox_center(region)

        # 基于 bbox 对角线确定拖动长度范围
        diag = math.sqrt(w * w + h * h)
        frac_min = self.config.drag_len_frac_min
        frac_max = self.config.drag_len_frac_max

        # 高层语义可以微调长度，例如 drag_short / drag_long
        if "short" in op:
            frac_max = (frac_min + frac_max) / 2.0
        elif "long" in op:
            frac_min = (frac_min + frac_max) / 2.0

        length = self.rng.uniform(frac_min * diag, frac_max * diag)

        # 拖动方向：完全随机，后续可以加入“优先进出屏幕”等策略
        theta = self.rng.uniform(0.0, 2.0 * math.pi)
        dx = length * math.cos(theta)
        dy = length * math.sin(theta)

        duration_ms = self.rng.randint(
            self.config.drag_duration_ms_min,
            self.config.drag_duration_ms_max,
        )

        return {
            "dx": dx,
            "dy": dy,
            "duration_ms": duration_ms,
        }

    # ------------------------------------------------------------------ #
    # rotate 参数采样
    # ------------------------------------------------------------------ #
    def _sample_rotate(self, region: Dict[str, Any], op: str) -> Dict[str, Any]:
        x, y, w, h, cx, cy = self._region_bbox_center(region)

        # 半径基于目标尺寸
        radius = min(w, h) * self.config.rotate_radius_frac

        # 角度正负控制方向，rotate_cw / rotate_ccw 可以覆盖/加强方向语义
        angle_min = self.config.rotate_angle_min_deg
        angle_max = self.config.rotate_angle_max_deg
        angle_mag = self.rng.uniform(angle_min, angle_max)

        # 默认随机方向
        sign = self.rng.choice([-1.0, 1.0])
        direction = "ccw" if sign > 0 else "cw"

        if op.endswith("cw"):
            sign = -1.0
            direction = "cw"
        elif op.endswith("ccw"):
            sign = 1.0
            direction = "ccw"

        angle_deg = sign * angle_mag

        steps = self.rng.randint(
            self.config.rotate_steps_min,
            self.config.rotate_steps_max,
        )
        duration_ms = self.rng.randint(
            self.config.rotate_duration_ms_min,
            self.config.rotate_duration_ms_max,
        )

        return {
            "angle_deg": angle_deg,
            "radius_px": radius,
            "steps": steps,
            "direction": direction,
            "start_deg": 0.0,
            "duration_ms": duration_ms,
        }

    # ------------------------------------------------------------------ #
    # pinch / zoom 参数采样
    # ------------------------------------------------------------------ #
    def _sample_pinch(self, region: Dict[str, Any], op: str) -> Dict[str, Any]:
        x, y, w, h, cx, cy = self._region_bbox_center(region)

        radius = min(w, h) * self.config.pinch_radius_frac

        base_scale = self.rng.uniform(
            self.config.pinch_base_scale_min,
            self.config.pinch_base_scale_max,
        )

        # 根据 op 决定方向：in / out / zoom_in / zoom_out 等
        if op in ("pinch_in", "zoom_out"):
            scale_sign = -1
        elif op in ("pinch_out", "zoom_in"):
            scale_sign = +1
        else:
            scale_sign = self.rng.choice([-1, 1])

        duration_ms = self.rng.randint(
            self.config.pinch_duration_ms_min,
            self.config.pinch_duration_ms_max,
        )

        # 这里不直接指定 start_dist/end_dist，交给 Executor 用 radius/scale_sign 计算
        return {
            "radius_px": radius,
            "base_scale": base_scale,
            "scale_sign": scale_sign,
            "duration_ms": duration_ms,
        }
