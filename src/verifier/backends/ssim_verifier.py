# -*- coding: utf-8 -*-
"""
ssim_verifier.py
----------------------------------------------------------------------
SSIM (Structural Similarity Index) based verifier for AR interactions.

Strategy:
  - Use SSIM to measure structural similarity between pre/post frames
  - Lower SSIM = more change = operation likely succeeded
  - Simple, parameter-free, and intuitive

SSIM ranges from -1 to 1:
  - 1.0 = identical images
  - 0.5 = significant difference
  - 0.0 = very different
"""
from typing import Tuple, Optional, Dict, Any
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim

Point = Tuple[int, int]
BBox = Tuple[int, int, int, int]


def _to_gray(img_bgr: np.ndarray) -> np.ndarray:
    """Convert BGR to grayscale."""
    if img_bgr.ndim == 3:
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return img_bgr


def compute_ssim(
    img1: np.ndarray,
    img2: np.ndarray
) -> float:
    """
    Compute SSIM between two images.

    Returns:
        SSIM value (1.0 = identical, lower = more different)
    """
    # Convert to grayscale if needed
    gray1 = _to_gray(img1)
    gray2 = _to_gray(img2)

    # Compute SSIM
    score, _ = ssim(gray1, gray2, full=True)
    return float(score)


def compute_pixel_diff_ratio(
    img1: np.ndarray,
    img2: np.ndarray,
    threshold: int = 15
) -> float:
    """
    Compute ratio of pixels that changed significantly.

    Returns:
        Ratio of changed pixels (0.0 to 1.0)
    """
    gray1 = _to_gray(img1)
    gray2 = _to_gray(img2)

    diff = cv2.absdiff(gray1, gray2)
    changed_pixels = np.sum(diff > threshold)
    total_pixels = diff.size

    return changed_pixels / total_pixels


def verify_by_ssim(
    pre_bgr: np.ndarray,
    post_bgr: np.ndarray,
    max_ssim: float = 0.95,
) -> bool:
    """
    Verify change using SSIM.

    Args:
        pre_bgr: Pre-action frame
        post_bgr: Post-action frame
        max_ssim: Maximum SSIM for "changed" (default 0.95)
                  If SSIM < max_ssim, operation succeeded

    Returns:
        True if change detected (SSIM < threshold)
    """
    score = compute_ssim(pre_bgr, post_bgr)
    return score < max_ssim


def verify_by_pixel_diff(
    pre_bgr: np.ndarray,
    post_bgr: np.ndarray,
    min_change_ratio: float = 0.01,
    diff_threshold: int = 15
) -> bool:
    """
    Verify change using pixel difference ratio.

    Args:
        pre_bgr: Pre-action frame
        post_bgr: Post-action frame
        min_change_ratio: Minimum ratio of changed pixels (default 1%)
        diff_threshold: Pixel difference threshold (default 15)

    Returns:
        True if change detected
    """
    ratio = compute_pixel_diff_ratio(pre_bgr, post_bgr, diff_threshold)
    return ratio >= min_change_ratio


def verify_hybrid(
    pre_bgr: np.ndarray,
    post_bgr: np.ndarray,
    max_ssim: float = 0.95,
    min_change_ratio: float = 0.01,
    diff_threshold: int = 15
) -> Dict[str, Any]:
    """
    Hybrid verification using both SSIM and pixel diff.

    Returns:
        {
            'ssim': float,
            'pixel_diff_ratio': float,
            'ssim_pass': bool,
            'pixel_diff_pass': bool,
            'overall_pass': bool  # True if either method passed
        }
    """
    ssim_score = compute_ssim(pre_bgr, post_bgr)
    pixel_ratio = compute_pixel_diff_ratio(pre_bgr, post_bgr, diff_threshold)

    ssim_pass = ssim_score < max_ssim
    pixel_pass = pixel_ratio >= min_change_ratio

    return {
        'ssim': ssim_score,
        'pixel_diff_ratio': pixel_ratio,
        'ssim_pass': ssim_pass,
        'pixel_diff_pass': pixel_pass,
        'overall_pass': ssim_pass or pixel_pass
    }


def verify_action(
    op: str,
    pre_bgr: np.ndarray,
    post_bgr: np.ndarray,
    center_xy: Optional[Point] = None,
    bbox: Optional[BBox] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Unified verification interface.

    Args:
        op: Operation name (unused, all operations use same logic)
        pre_bgr: Pre-action frame
        post_bgr: Post-action frame
        center_xy: Center point (unused)
        bbox: Bounding box (unused in current version)
        extra: Extra parameters
            - method: 'ssim', 'pixel_diff', or 'hybrid' (default)
            - max_ssim: SSIM threshold (default 0.95)
            - min_change_ratio: Pixel diff threshold (default 0.01)
            - diff_threshold: Pixel value threshold (default 15)

    Returns:
        True if change detected
    """
    extra = extra or {}
    method = extra.get('method', 'hybrid')

    if method == 'ssim':
        max_ssim = extra.get('max_ssim', 0.95)
        return verify_by_ssim(pre_bgr, post_bgr, max_ssim)

    elif method == 'pixel_diff':
        min_change_ratio = extra.get('min_change_ratio', 0.01)
        diff_threshold = extra.get('diff_threshold', 15)
        return verify_by_pixel_diff(pre_bgr, post_bgr, min_change_ratio, diff_threshold)

    else:  # hybrid (default)
        max_ssim = extra.get('max_ssim', 0.95)
        min_change_ratio = extra.get('min_change_ratio', 0.01)
        diff_threshold = extra.get('diff_threshold', 15)
        result = verify_hybrid(pre_bgr, post_bgr, max_ssim, min_change_ratio, diff_threshold)
        return result['overall_pass']
