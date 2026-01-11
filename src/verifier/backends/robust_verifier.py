# -*- coding: utf-8 -*-
"""
robust_verifier.py
----------------------------------------------------------------------
Robust CV verification for AR interactions with three-step enhancement:
  1. Camera Motion Compensation (ECC Alignment)
  2. Adaptive Connected-Component Filtering
  3. Temporal Consistency Enforcement

Design Goals:
  - Reduce False Positives from camera shake, point cloud flicker, lighting changes
  - Cross-app generalization (no app-specific assumptions)
  - Modular design (each step can be enabled/disabled)

Author: Claude Code
Date: 2026-01-11
"""
from typing import Tuple, Optional, Dict, Any, List
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim

Point = Tuple[int, int]
BBox = Tuple[int, int, int, int]


# ==================== Step 1: Camera Motion Compensation ====================

def align_image_ecc(
    img_pre: np.ndarray,
    img_post: np.ndarray,
    warp_mode: int = cv2.MOTION_TRANSLATION,
    max_iterations: int = 50,
    termination_eps: float = 1e-3,
    use_center_crop: bool = True,
    crop_ratio: float = 0.8
) -> Tuple[np.ndarray, bool]:
    """
    Align post image to pre image using ECC algorithm.

    This compensates for camera shake/drift between frames.

    Args:
        img_pre: Reference image (BGR or grayscale)
        img_post: Image to align (BGR or grayscale)
        warp_mode: cv2.MOTION_TRANSLATION (default) or cv2.MOTION_EUCLIDEAN
        max_iterations: Max ECC iterations
        termination_eps: ECC termination threshold
        use_center_crop: Whether to use center region for alignment (more stable)
        crop_ratio: Center crop ratio if use_center_crop=True

    Returns:
        (aligned_post, success):
            - aligned_post: Aligned image (same size as img_post)
            - success: True if alignment succeeded, False if failed (returns original)
    """
    # Convert to grayscale
    gray_pre = cv2.cvtColor(img_pre, cv2.COLOR_BGR2GRAY) if img_pre.ndim == 3 else img_pre
    gray_post = cv2.cvtColor(img_post, cv2.COLOR_BGR2GRAY) if img_post.ndim == 3 else img_post

    # Optional: Use center crop for more stable alignment
    if use_center_crop:
        h, w = gray_pre.shape
        crop_h, crop_w = int(h * crop_ratio), int(w * crop_ratio)
        start_h, start_w = (h - crop_h) // 2, (w - crop_w) // 2
        gray_pre_align = gray_pre[start_h:start_h+crop_h, start_w:start_w+crop_w]
        gray_post_align = gray_post[start_h:start_h+crop_h, start_w:start_w+crop_w]
    else:
        gray_pre_align = gray_pre
        gray_post_align = gray_post

    # Initialize warp matrix
    if warp_mode == cv2.MOTION_TRANSLATION:
        warp_matrix = np.eye(2, 3, dtype=np.float32)
    else:  # MOTION_EUCLIDEAN
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iterations, termination_eps)

    try:
        # Run ECC algorithm
        _, warp_matrix = cv2.findTransformECC(
            gray_pre_align,
            gray_post_align,
            warp_matrix,
            warp_mode,
            criteria,
            inputMask=None,
            gaussFiltSize=1
        )

        # Apply transformation to full post image
        h, w = img_post.shape[:2]
        aligned_post = cv2.warpAffine(
            img_post,
            warp_matrix,
            (w, h),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
        )

        return aligned_post, True

    except cv2.error as e:
        # Alignment failed (e.g., insufficient texture)
        # Return original image
        return img_post, False


# ==================== Step 2: Adaptive Connected-Component Filtering ====================

def filter_fragmented_changes(
    diff_mask: np.ndarray,
    roi_area: Optional[int] = None,
    min_component_ratio: float = 0.01,
    max_components_threshold: int = 50,
    min_largest_ratio: float = 0.3,
    top_k: Optional[int] = None,
    fallback_on_over_filter: bool = True
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Filter out fragmented noise changes using connected-component analysis.

    This removes scattered changes from point clouds, mesh updates, rendering noise.

    Strategy:
        - Only filter if changes are highly fragmented (many small components)
        - Use adaptive thresholds based on ROI area
        - Failsafe: Revert if over-filtering is detected

    Args:
        diff_mask: Binary mask (0/255) of changed pixels
        roi_area: Reference area for adaptive thresholds (e.g., bbox area)
                  If None, uses full image area
        min_component_ratio: Min component size as ratio of ROI area (default 1%)
        max_components_threshold: If num_components > this, consider fragmented
        min_largest_ratio: If largest component < this ratio of total change, consider fragmented
        top_k: If set, keep only top-K largest components (alternative strategy)
        fallback_on_over_filter: Revert to original if filtered area drops too much

    Returns:
        (filtered_mask, stats):
            - filtered_mask: Filtered binary mask
            - stats: Dict with diagnostic info
    """
    if roi_area is None:
        roi_area = diff_mask.shape[0] * diff_mask.shape[1]

    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        diff_mask, connectivity=8
    )

    # stats columns: [x, y, width, height, area]
    # Label 0 is background, skip it
    component_areas = stats[1:, cv2.CC_STAT_AREA]

    if len(component_areas) == 0:
        # No components, return empty mask
        return diff_mask, {
            'num_components': 0,
            'total_area': 0,
            'filtered': False,
            'reason': 'no_components'
        }

    total_change_area = np.sum(diff_mask > 0)
    num_components = len(component_areas)
    largest_area = np.max(component_areas)
    largest_ratio = largest_area / total_change_area if total_change_area > 0 else 0

    # Gate 1: Check if changes are fragmented
    is_fragmented = (
        num_components > max_components_threshold or
        largest_ratio < min_largest_ratio
    )

    if not is_fragmented:
        # Changes are coherent, don't filter
        return diff_mask, {
            'num_components': num_components,
            'total_area': total_change_area,
            'largest_area': largest_area,
            'largest_ratio': largest_ratio,
            'filtered': False,
            'reason': 'not_fragmented'
        }

    # Gate 2: Apply adaptive filtering
    filtered_mask = np.zeros_like(diff_mask)

    if top_k is not None:
        # Strategy A: Keep top-K largest components
        top_k_indices = np.argsort(component_areas)[-top_k:] + 1  # +1 for label offset
        for label_idx in top_k_indices:
            filtered_mask[labels == label_idx] = 255
    else:
        # Strategy B: Keep components above size threshold
        min_component_area = int(roi_area * min_component_ratio)
        for i, area in enumerate(component_areas):
            if area >= min_component_area:
                label_idx = i + 1  # +1 for label offset
                filtered_mask[labels == label_idx] = 255

    filtered_area = np.sum(filtered_mask > 0)

    # Failsafe: Check for over-filtering
    if fallback_on_over_filter and filtered_area < 0.2 * total_change_area:
        # Filtered out >80% of changes, might be over-filtering a TP case
        # Revert to original
        return diff_mask, {
            'num_components': num_components,
            'total_area': total_change_area,
            'filtered_area': filtered_area,
            'filtered': False,
            'reason': 'failsafe_triggered'
        }

    return filtered_mask, {
        'num_components': num_components,
        'total_area': total_change_area,
        'filtered_area': filtered_area,
        'largest_area': largest_area,
        'largest_ratio': largest_ratio,
        'filtered': True,
        'reason': 'fragmented'
    }


# ==================== Step 3: Temporal Consistency Enforcement ====================

def verify_temporal_consistency(
    pre_frame: np.ndarray,
    post_frames: List[np.ndarray],
    verify_fn,
    min_consistent_frames: int = 3,
    strategy: str = 'majority_vote'
) -> Tuple[bool, Dict[str, Any]]:
    """
    Enforce temporal consistency by verifying across multiple post-operation frames.

    This suppresses transient changes from flicker, exposure adjustment, rendering jitter.

    Args:
        pre_frame: Pre-action frame
        post_frames: List of post-action frames (e.g., 5 consecutive frames)
        verify_fn: Function(pre, post) -> bool that returns True if change detected
        min_consistent_frames: Minimum frames that must show change (majority vote)
        strategy: 'majority_vote' or 'median_fusion'

    Returns:
        (verified, stats):
            - verified: True if change is temporally consistent
            - stats: Diagnostic info
    """
    if strategy == 'majority_vote':
        # Vote across frames
        votes = [verify_fn(pre_frame, post) for post in post_frames]
        num_positive = sum(votes)
        verified = num_positive >= min_consistent_frames

        return verified, {
            'strategy': 'majority_vote',
            'total_frames': len(post_frames),
            'positive_votes': num_positive,
            'min_required': min_consistent_frames,
            'verified': verified
        }

    elif strategy == 'median_fusion':
        # Compute median post frame, then verify once
        # Convert to float for median computation
        post_stack = np.stack([post.astype(np.float32) for post in post_frames], axis=0)
        median_post = np.median(post_stack, axis=0).astype(np.uint8)

        verified = verify_fn(pre_frame, median_post)

        return verified, {
            'strategy': 'median_fusion',
            'total_frames': len(post_frames),
            'verified': verified
        }

    else:
        raise ValueError(f"Unknown strategy: {strategy}")


# ==================== Integrated Robust Verification ====================

def verify_robust(
    pre_bgr: np.ndarray,
    post_bgr: np.ndarray,
    bbox: Optional[BBox] = None,
    # Step 1: Camera motion compensation
    enable_alignment: bool = True,
    alignment_mode: int = cv2.MOTION_TRANSLATION,
    # Step 2: Connected-component filtering
    enable_cc_filter: bool = True,
    min_component_ratio: float = 0.01,
    # Step 3: Temporal consistency (requires multiple post frames)
    post_frames: Optional[List[np.ndarray]] = None,
    enable_temporal: bool = False,
    min_consistent_frames: int = 3,
    # Base verification parameters
    max_ssim: float = 0.95,
    min_change_ratio: float = 0.01,
    diff_threshold: int = 15,
    # Debug output
    return_debug_info: bool = False
) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """
    Robust verification with three-step enhancement.

    Args:
        pre_bgr: Pre-action frame
        post_bgr: Post-action frame (primary)
        bbox: Optional bounding box (x, y, w, h) for ROI-based filtering
        enable_alignment: Enable camera motion compensation
        alignment_mode: cv2.MOTION_TRANSLATION or cv2.MOTION_EUCLIDEAN
        enable_cc_filter: Enable connected-component filtering
        min_component_ratio: Min component size ratio for filtering
        post_frames: List of post frames for temporal consistency (optional)
        enable_temporal: Enable temporal consistency check
        min_consistent_frames: Min frames for temporal majority vote
        max_ssim: SSIM threshold
        min_change_ratio: Pixel diff ratio threshold
        diff_threshold: Pixel value difference threshold
        return_debug_info: Return detailed debug info

    Returns:
        (verified, debug_info):
            - verified: True if change detected
            - debug_info: Dict with diagnostic info (if return_debug_info=True)
    """
    debug_info = {}

    # Step 1: Camera Motion Compensation
    if enable_alignment:
        post_aligned, align_success = align_image_ecc(
            pre_bgr, post_bgr, warp_mode=alignment_mode
        )
        debug_info['alignment'] = {
            'enabled': True,
            'success': align_success
        }
    else:
        post_aligned = post_bgr
        debug_info['alignment'] = {'enabled': False}

    # Convert to grayscale for diff computation
    gray_pre = cv2.cvtColor(pre_bgr, cv2.COLOR_BGR2GRAY)
    gray_post = cv2.cvtColor(post_aligned, cv2.COLOR_BGR2GRAY)

    # Compute base metrics
    ssim_score, ssim_map = ssim(gray_pre, gray_post, full=True)
    diff_abs = cv2.absdiff(gray_pre, gray_post)
    diff_mask = (diff_abs > diff_threshold).astype(np.uint8) * 255

    debug_info['base_metrics'] = {
        'ssim': float(ssim_score),
        'raw_change_area': int(np.sum(diff_mask > 0)),
        'raw_change_ratio': float(np.sum(diff_mask > 0) / diff_mask.size)
    }

    # Step 2: Connected-Component Filtering
    if enable_cc_filter:
        roi_area = None
        if bbox is not None:
            x, y, w, h = bbox
            roi_area = w * h

        filtered_mask, cc_stats = filter_fragmented_changes(
            diff_mask,
            roi_area=roi_area,
            min_component_ratio=min_component_ratio
        )

        # Recompute change ratio after filtering
        filtered_change_ratio = np.sum(filtered_mask > 0) / filtered_mask.size

        debug_info['cc_filter'] = cc_stats
        debug_info['cc_filter']['filtered_change_ratio'] = float(filtered_change_ratio)
    else:
        filtered_mask = diff_mask
        filtered_change_ratio = debug_info['base_metrics']['raw_change_ratio']
        debug_info['cc_filter'] = {'enabled': False}

    # Base verification decision
    ssim_pass = ssim_score < max_ssim
    pixel_pass = filtered_change_ratio >= min_change_ratio
    base_verified = ssim_pass or pixel_pass

    debug_info['base_decision'] = {
        'ssim_pass': ssim_pass,
        'pixel_pass': pixel_pass,
        'verified': base_verified
    }

    # Step 3: Temporal Consistency (optional)
    if enable_temporal and post_frames is not None and len(post_frames) > 1:
        # Define verification function for temporal check
        def verify_single_frame(pre, post):
            if enable_alignment:
                post, _ = align_image_ecc(pre, post, warp_mode=alignment_mode)
            gray_pre_tmp = cv2.cvtColor(pre, cv2.COLOR_BGR2GRAY)
            gray_post_tmp = cv2.cvtColor(post, cv2.COLOR_BGR2GRAY)
            ssim_tmp, _ = ssim(gray_pre_tmp, gray_post_tmp, full=True)
            diff_tmp = cv2.absdiff(gray_pre_tmp, gray_post_tmp)
            diff_mask_tmp = (diff_tmp > diff_threshold).astype(np.uint8) * 255

            if enable_cc_filter:
                diff_mask_tmp, _ = filter_fragmented_changes(
                    diff_mask_tmp, roi_area=roi_area, min_component_ratio=min_component_ratio
                )

            ratio_tmp = np.sum(diff_mask_tmp > 0) / diff_mask_tmp.size
            return (ssim_tmp < max_ssim) or (ratio_tmp >= min_change_ratio)

        temporal_verified, temporal_stats = verify_temporal_consistency(
            pre_bgr,
            post_frames,
            verify_single_frame,
            min_consistent_frames=min_consistent_frames
        )

        debug_info['temporal'] = temporal_stats
        final_verified = temporal_verified
    else:
        debug_info['temporal'] = {'enabled': False}
        final_verified = base_verified

    if return_debug_info:
        return final_verified, debug_info
    else:
        return final_verified, None


# ==================== Backward Compatible Interface ====================

def verify_action(
    op: str,
    pre_bgr: np.ndarray,
    post_bgr: np.ndarray,
    center_xy: Optional[Point] = None,
    bbox: Optional[BBox] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Unified verification interface (backward compatible with ssim_verifier.py).

    Args:
        op: Operation name
        pre_bgr: Pre-action frame
        post_bgr: Post-action frame
        center_xy: Center point (unused)
        bbox: Bounding box for ROI-based filtering
        extra: Extended parameters:
            - enable_alignment: bool (default True)
            - enable_cc_filter: bool (default True)
            - enable_temporal: bool (default False)
            - post_frames: List[np.ndarray] (for temporal consistency)
            - max_ssim: float (default 0.95)
            - min_change_ratio: float (default 0.01)
            - ... (see verify_robust for full list)

    Returns:
        True if change detected
    """
    extra = extra or {}

    # Extract parameters with defaults
    enable_alignment = extra.get('enable_alignment', True)
    enable_cc_filter = extra.get('enable_cc_filter', True)
    enable_temporal = extra.get('enable_temporal', False)
    post_frames = extra.get('post_frames', None)
    max_ssim = extra.get('max_ssim', 0.95)
    min_change_ratio = extra.get('min_change_ratio', 0.01)
    diff_threshold = extra.get('diff_threshold', 15)
    min_component_ratio = extra.get('min_component_ratio', 0.01)

    verified, _ = verify_robust(
        pre_bgr=pre_bgr,
        post_bgr=post_bgr,
        bbox=bbox,
        enable_alignment=enable_alignment,
        enable_cc_filter=enable_cc_filter,
        enable_temporal=enable_temporal,
        post_frames=post_frames,
        max_ssim=max_ssim,
        min_change_ratio=min_change_ratio,
        diff_threshold=diff_threshold,
        min_component_ratio=min_component_ratio,
        return_debug_info=False
    )

    return verified
