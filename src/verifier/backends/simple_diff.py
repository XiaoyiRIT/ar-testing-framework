# -*- coding: utf-8 -*-
"""
simple_diff.py
----------------------------------------------------------------------
Simple frame difference verifier for AR interactions.
Strategy: Compare pre-action and post-action frames to detect any visual change.

Philosophy:
  - If an operation succeeded, SOME visual change should occur
  - No need to identify WHAT changed (motion, appearance, etc.)
  - Just detect IF something changed

This avoids complex optical flow analysis and parameter tuning.
"""
from typing import Tuple, Optional, Dict, Any
import numpy as np
import cv2

Point = Tuple[int, int]
BBox = Tuple[int, int, int, int]


def _to_gray(img_bgr: np.ndarray) -> np.ndarray:
    """Convert BGR to grayscale."""
    if img_bgr.ndim == 3:
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return img_bgr


def verify_simple_change(
    pre_bgr: np.ndarray,
    post_bgr: np.ndarray,
    min_change_ratio: float = 0.01,
    diff_threshold: int = 15,
) -> bool:
    """
    Verify if any visual change occurred between two frames.

    Args:
        pre_bgr: Pre-action frame (BGR)
        post_bgr: Post-action frame (BGR)
        min_change_ratio: Minimum ratio of changed pixels (default 1%)
        diff_threshold: Pixel difference threshold (default 15)

    Returns:
        True if significant change detected, False otherwise
    """
    # Convert to grayscale
    pre_g = _to_gray(pre_bgr)
    post_g = _to_gray(post_bgr)

    # Compute absolute difference
    diff = cv2.absdiff(pre_g, post_g)

    # Count pixels with significant change
    changed_pixels = np.sum(diff > diff_threshold)
    total_pixels = diff.size

    change_ratio = changed_pixels / total_pixels

    return change_ratio >= min_change_ratio


def verify_local_change(
    pre_bgr: np.ndarray,
    post_bgr: np.ndarray,
    bbox: BBox,
    min_change_ratio: float = 0.15,
    diff_threshold: int = 20,
) -> bool:
    """
    Verify if visual change occurred in a specific region (bbox).

    Args:
        pre_bgr: Pre-action frame (BGR)
        post_bgr: Post-action frame (BGR)
        bbox: Region of interest (x, y, w, h)
        min_change_ratio: Minimum ratio of changed pixels in ROI (default 15%)
        diff_threshold: Pixel difference threshold (default 20)

    Returns:
        True if significant change in ROI, False otherwise
    """
    pre_g = _to_gray(pre_bgr)
    post_g = _to_gray(post_bgr)

    H, W = pre_g.shape
    x, y, w, h = bbox

    # Clip bbox to image bounds
    x = max(0, min(x, W-1))
    y = max(0, min(y, H-1))
    w = max(1, min(w, W - x))
    h = max(1, min(h, H - y))

    # Extract ROI
    roi_pre = pre_g[y:y+h, x:x+w]
    roi_post = post_g[y:y+h, x:x+w]

    # Compute difference in ROI
    diff = cv2.absdiff(roi_pre, roi_post)
    changed_pixels = np.sum(diff > diff_threshold)
    total_pixels = diff.size

    change_ratio = changed_pixels / max(1, total_pixels)

    return change_ratio >= min_change_ratio


def verify_action_simple(
    op: str,
    pre_bgr: np.ndarray,
    post_bgr: np.ndarray,
    center_xy: Optional[Point] = None,
    bbox: Optional[BBox] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Simple verification strategy: just check if frames changed.

    Strategy:
    1. For ALL operations, use the same simple logic
    2. Compute global frame difference
    3. If bbox provided, also check local change
    4. Return True if EITHER global or local change detected

    Args:
        op: Operation name (tap, drag, rotate, etc.)
        pre_bgr: Pre-action frame
        post_bgr: Post-action frame
        center_xy: Center point (unused in simple version)
        bbox: Bounding box (optional, for local check)
        extra: Extra parameters
            - global_min_change: Global change threshold (default 0.01 = 1%)
            - local_min_change: Local change threshold (default 0.15 = 15%)
            - diff_threshold: Pixel diff threshold (default 15)

    Returns:
        True if change detected, False otherwise
    """
    extra = extra or {}

    global_min_change = extra.get("global_min_change", 0.01)   # 1% global change
    local_min_change = extra.get("local_min_change", 0.15)     # 15% local change
    diff_threshold = extra.get("diff_threshold", 15)

    # Method 1: Global change (entire frame)
    global_changed = verify_simple_change(
        pre_bgr, post_bgr,
        min_change_ratio=global_min_change,
        diff_threshold=diff_threshold
    )

    if global_changed:
        return True

    # Method 2: Local change (if bbox provided)
    if bbox is not None:
        local_changed = verify_local_change(
            pre_bgr, post_bgr, bbox,
            min_change_ratio=local_min_change,
            diff_threshold=diff_threshold
        )
        if local_changed:
            return True

    return False


# Alias for consistency with other backends
verify_action = verify_action_simple
