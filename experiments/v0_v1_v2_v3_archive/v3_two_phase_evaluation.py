#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v3_two_phase_evaluation.py — Two-Phase CV Evaluation Framework

Phase 1 (collect): Execute operations and save frames + metadata
Phase 2 (eval): Load saved data and run CV verification

Advantages:
1. Reproducibility: Save all frames for repeated experiments
2. Debuggability: Manually inspect frames
3. Fast iteration: Modify CV algorithm without re-executing operations
4. Simple CV: Use SSIM + pixel diff instead of complex optical flow

Usage:
    # Phase 1: Collect data
    python3 v3_two_phase_evaluation.py --mode collect --rounds 60 --output-dir ./data/run001

    # Phase 2: Evaluate with different CV methods
    python3 v3_two_phase_evaluation.py --mode eval --input-dir ./data/run001 --cv-method hybrid
    python3 v3_two_phase_evaluation.py --mode eval --input-dir ./data/run001 --cv-method ssim --max-ssim 0.90
"""

import argparse
import random
import sys
import time
import os
import csv
import json
import threading
import subprocess
from pathlib import Path
from typing import Dict, Any, List

# Add project paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from src.common.device import make_driver, get_window_size, resolve_main_activity, capture_bgr
from src.common.actions import (tap, pinch_or_zoom, rotate, drag_line, long_press, double_tap,
                                triple_tap, swipe, two_finger_tap, flick)
from cv.strategy_yolo import locate as cv_locate
from src.verifier.backends.ssim_verifier import verify_action, compute_ssim, compute_pixel_diff_ratio
from src.common.verify_motion import MotionVerifier
from src.common.timing import Timing
import cv2
import numpy as np


# ==================== Utility Functions ====================

def _ms(t0: float) -> float:
    """Convert time to milliseconds."""
    return (time.perf_counter() - t0) * 1000.0


def clear_logcat(serial=None):
    """Clear logcat buffer."""
    cmd = ["adb"]
    if serial:
        cmd += ["-s", serial]
    cmd += ["logcat", "-c"]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def check_ground_truth(serial, op_name, time_window_sec=2.0):
    """
    Check if operation was recorded in app logcat.

    Returns:
        True if operation succeeded (ok=true in logcat)
    """
    # Map operation names to logcat kinds
    op_map = {
        "tap": "tap",
        "double_tap": "double_tap",
        "long_press": "long_press_end",
        "drag": "drag",
        "pinch_in": "pinch",
        "pinch_out": "pinch",
        "rotate": "rotate",
    }

    target_kind = op_map.get(op_name)
    if not target_kind:
        return False  # Unsupported operation

    # Get recent logcat
    cmd = ["adb"]
    if serial:
        cmd += ["-s", serial]
    cmd += ["logcat", "-d", "-T", f"{time_window_sec}"]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        output = result.stdout
    except Exception:
        return False

    recent_lines = output.strip().split('\n')[-50:]  # Last 50 lines

    # Check for AR_OP with matching kind and ok=true
    for line in reversed(recent_lines):
        if "AR_OP" in line and "{" in line:
            try:
                json_start = line.index("{")
                json_str = line[json_start:]
                data = json.loads(json_str)

                kind = data.get("kind", "").strip().lower()
                ok = data.get("ok", False)

                if kind == target_kind and ok:
                    return True
            except Exception:
                continue

    return False


def execute_operation(drv, op_name, cx, cy, W, H, **kwargs):
    """Execute a single operation."""
    if op_name == "tap":
        tap(drv, cx, cy, press_ms=random.randint(40, 120))
        return f"tap at ({cx},{cy})"

    elif op_name == "double_tap":
        interval_ms = random.randint(80, 150)
        double_tap(drv, cx, cy, tap_interval_ms=interval_ms)
        return f"double_tap at ({cx},{cy}) interval={interval_ms}ms"

    elif op_name == "long_press":
        hold_ms = random.randint(800, 1200)
        long_press(drv, cx, cy, hold_ms=hold_ms)
        return f"long_press at ({cx},{cy}) hold={hold_ms}ms"

    elif op_name == "drag":
        drag_ms = kwargs.get('drag_ms', (300, 900))
        duration_ms = random.randint(*drag_ms)
        angle = random.uniform(0, 2 * 3.14159)
        distance = random.randint(50, min(W, H) // 3)
        ex = int(cx + distance * np.cos(angle))
        ey = int(cy + distance * np.sin(angle))
        ex = max(0, min(ex, W-1))
        ey = max(0, min(ey, H-1))
        drag_line(drv, cx, cy, ex, ey, duration_ms=duration_ms)
        return f"drag ({cx},{cy})→({ex},{ey}) {duration_ms}ms"

    elif op_name in ["pinch_in", "pinch_out"]:
        # pinch_in: fingers move closer (start_dist > end_dist)
        # pinch_out: fingers move apart (start_dist < end_dist)
        if op_name == "pinch_in":
            start_dist = 200
            end_dist = 60
        else:  # pinch_out
            start_dist = 60
            end_dist = 200
        pinch_or_zoom(drv, cx, cy, start_dist=start_dist, end_dist=end_dist)
        return f"{op_name} at ({cx},{cy}) {start_dist}→{end_dist}"

    elif op_name == "rotate":
        rotate_steps = kwargs.get('rotate_steps', 8)
        direction = random.choice(["cw", "ccw"])
        rotate(drv, cx, cy, steps=rotate_steps, direction=direction)
        return f"rotate at ({cx},{cy}) steps={rotate_steps} {direction}"

    # Unsupported operations
    elif op_name == "triple_tap":
        triple_tap(drv, cx, cy)
        return f"triple_tap at ({cx},{cy})"

    elif op_name == "swipe":
        # swipe needs start and end points
        angle = random.uniform(0, 2 * 3.14159)
        distance = random.randint(100, 200)
        ex = int(cx + distance * np.cos(angle))
        ey = int(cy + distance * np.sin(angle))
        ex = max(0, min(ex, W-1))
        ey = max(0, min(ey, H-1))
        swipe(drv, cx, cy, ex, ey, duration_ms=150)
        return f"swipe ({cx},{cy})→({ex},{ey})"

    elif op_name == "two_finger_tap":
        two_finger_tap(drv, cx, cy)
        return f"two_finger_tap at ({cx},{cy})"

    elif op_name == "flick":
        # flick needs start and end points
        angle = random.uniform(0, 2 * 3.14159)
        distance = random.randint(50, 100)  # Shorter distance than swipe
        ex = int(cx + distance * np.cos(angle))
        ey = int(cy + distance * np.sin(angle))
        ex = max(0, min(ex, W-1))
        ey = max(0, min(ey, H-1))
        flick(drv, cx, cy, ex, ey, duration_ms=80)
        return f"flick ({cx},{cy})→({ex},{ey})"

    return f"unknown operation: {op_name}"


# ==================== Phase 1: Collect ====================

def phase1_collect(args):
    """
    Phase 1: Execute operations and save frames + metadata.

    Saved structure:
        output_dir/
            metadata.json          # Overall run metadata
            operations.jsonl       # One JSON per line for each operation
            frames/
                op_000_pre.png
                op_000_post.png
                op_001_pre.png
                op_001_post.png
                ...
    """
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(exist_ok=True)

    print(f"\n{'='*70}")
    print(f"PHASE 1: COLLECT")
    print(f"{'='*70}")
    print(f"Output directory: {output_dir}")
    print(f"Rounds: {args.rounds}")
    print(f"Seed: {args.seed}")

    # Set random seed
    if args.seed is not None:
        random.seed(args.seed)

    # Default operation types
    supported_ops = ["tap", "double_tap", "drag", "long_press", "pinch_in", "rotate"]
    unsupported_ops = ["triple_tap", "swipe", "two_finger_tap", "flick"]
    all_operations = supported_ops + unsupported_ops

    # Generate operation plan (uniform distribution)
    operation_plan = []
    ops_per_type = args.rounds // len(all_operations)
    remainder = args.rounds % len(all_operations)

    for i, op in enumerate(all_operations):
        count = ops_per_type + (1 if i < remainder else 0)
        operation_plan.extend([op] * count)

    random.shuffle(operation_plan)

    # Generate negative sample plan
    negative_plan = [random.random() < args.negative_ratio for _ in range(args.rounds)]

    print(f"\nOperation distribution:")
    from collections import Counter
    op_counts = Counter(operation_plan)
    for op, count in sorted(op_counts.items()):
        support_status = "✓ supported" if op in supported_ops else "✗ unsupported"
        print(f"  {op:18s}: {count:3d} times  ({support_status})")
    print(f"Negative samples: {sum(negative_plan)}/{args.rounds} ({100*sum(negative_plan)/args.rounds:.1f}%)")

    # Connect to device
    pkg = args.package
    activity = args.activity
    if activity == "auto":
        act = resolve_main_activity(pkg, args.serial)
        activity = act if act else None
        print(f"\n[resolve] {pkg} -> activity: {activity or 'Intent-Launcher'}")

    drv = make_driver(pkg=pkg, activity=activity, serial=args.serial, warmup_wait=args.warmup_wait)
    W, H = get_window_size(drv)
    print(f"[device] {pkg}/{activity or 'auto'}, screen {W}x{H}")

    # Start logcat monitoring
    clear_logcat(args.serial)
    ver = MotionVerifier(overall_basis="tap")
    ver.start_from_adb(serial=args.serial, extra_logcat_args=["-T", "0"])

    # Metadata
    metadata = {
        'package': pkg,
        'activity': activity,
        'screen_size': [W, H],
        'rounds': args.rounds,
        'seed': args.seed,
        'negative_ratio': args.negative_ratio,
        'supported_ops': supported_ops,
        'unsupported_ops': unsupported_ops,
        'operation_plan': operation_plan,
        'negative_plan': negative_plan,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }

    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    # Operations log (JSONL format)
    operations_file = open(output_dir / 'operations.jsonl', 'w')

    try:
        prev = capture_bgr(drv)

        for i in range(1, args.rounds + 1):
            print(f"\n[{i:03d}/{args.rounds}] ", end='', flush=True)

            # Capture current frame
            curr = capture_bgr(drv)

            # Detect AR object
            det = cv_locate(image_bgr=curr, debug_profile=False)

            # Get operation from plan
            op_name = operation_plan[i - 1]
            is_negative = negative_plan[i - 1]
            is_supported = op_name in supported_ops

            # Determine operation location
            if is_negative or det is None:
                # Random location (negative sample or no AR object)
                cx = random.randint(int(W * 0.2), int(W * 0.8))
                cy = random.randint(int(H * 0.2), int(H * 0.8))
                bbox = None
                detected = False
            else:
                # On AR object
                cx_img, cy_img = det["center"]
                x_img, y_img, w_img_box, h_img_box = det["bbox"]
                h_img, w_img = curr.shape[:2]

                # Map to window coordinates
                cx = int(cx_img * W / w_img)
                cy = int(cy_img * H / h_img)

                bbox = [x_img, y_img, w_img_box, h_img_box]
                detected = True

            # Save pre-action frame
            pre_frame_path = frames_dir / f"op_{i:03d}_pre.png"
            cv2.imwrite(str(pre_frame_path), curr)

            # Execute operation
            op_msg = execute_operation(drv, op_name, cx, cy, W, H,
                                      drag_ms=(300, 900), rotate_steps=8)

            # Wait and capture post-action frame
            time.sleep(args.verify_wait_ms / 1000.0)
            post = capture_bgr(drv)
            post_frame_path = frames_dir / f"op_{i:03d}_post.png"
            cv2.imwrite(str(post_frame_path), post)

            # Check ground truth
            time.sleep(0.1)
            gt_ok = check_ground_truth(args.serial, op_name, time_window_sec=2.0)

            # Save operation metadata
            op_data = {
                'index': i,
                'operation': op_name,
                'is_supported': is_supported,
                'is_negative': is_negative,
                'detected': detected,
                'bbox': bbox,
                'center': [cx, cy],
                'pre_frame': str(pre_frame_path.name),
                'post_frame': str(post_frame_path.name),
                'gt_verified': gt_ok,
                'message': op_msg,
            }

            operations_file.write(json.dumps(op_data) + '\n')
            operations_file.flush()

            print(f"{op_name:15s} GT={'✓' if gt_ok else '✗'}  detected={detected}  {op_msg}")

            # Wait before next operation
            time.sleep(random.uniform(args.sleep_min, args.sleep_max))
            prev = curr

        print(f"\n{'='*70}")
        print(f"Collection complete!")
        print(f"Saved {args.rounds} operations to {output_dir}")
        print(f"{'='*70}\n")

    finally:
        operations_file.close()
        ver.stop()
        try:
            drv.quit()
        except:
            pass


# ==================== Phase 2: Eval ====================

def phase2_eval(args):
    """
    Phase 2: Load saved frames and run CV verification.
    """
    input_dir = Path(args.input_dir)

    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        return

    print(f"\n{'='*70}")
    print(f"PHASE 2: EVAL")
    print(f"{'='*70}")
    print(f"Input directory: {input_dir}")
    print(f"CV method: {args.cv_method}")
    print(f"Thresholds: max_ssim={args.max_ssim}, min_change_ratio={args.min_change_ratio}")

    # Load metadata
    metadata_path = input_dir / 'metadata.json'
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    print(f"\nLoaded metadata:")
    print(f"  Rounds: {metadata['rounds']}")
    print(f"  Seed: {metadata['seed']}")
    print(f"  Package: {metadata['package']}")

    # Load operations
    operations = []
    operations_path = input_dir / 'operations.jsonl'
    with open(operations_path, 'r') as f:
        for line in f:
            operations.append(json.loads(line.strip()))

    print(f"  Loaded {len(operations)} operations")

    # Statistics
    tp_count = 0
    tn_count = 0
    fp_count = 0
    fn_count = 0

    from collections import defaultdict
    tp_by_op = defaultdict(int)
    tn_by_op = defaultdict(int)
    fp_by_op = defaultdict(int)
    fn_by_op = defaultdict(int)

    frames_dir = input_dir / 'frames'

    print(f"\n{'='*70}")
    print(f"Running CV verification...")
    print(f"{'='*70}\n")

    # Process each operation
    for op_data in operations:
        i = op_data['index']
        op_name = op_data['operation']
        gt_verified = op_data['gt_verified']

        # Load frames
        pre_frame_path = frames_dir / op_data['pre_frame']
        post_frame_path = frames_dir / op_data['post_frame']

        pre_bgr = cv2.imread(str(pre_frame_path))
        post_bgr = cv2.imread(str(post_frame_path))

        if pre_bgr is None or post_bgr is None:
            print(f"[{i:03d}] Error loading frames, skipping")
            continue

        # Run CV verification
        cv_ok = verify_action(
            op_name,
            pre_bgr,
            post_bgr,
            extra={
                'method': args.cv_method,
                'max_ssim': args.max_ssim,
                'min_change_ratio': args.min_change_ratio,
                'diff_threshold': args.diff_threshold,
            }
        )

        # Also compute detailed metrics
        ssim_score = compute_ssim(pre_bgr, post_bgr)
        pixel_ratio = compute_pixel_diff_ratio(pre_bgr, post_bgr, args.diff_threshold)

        cv_verified = 1 if cv_ok else 0
        gt = 1 if gt_verified else 0

        # Update confusion matrix
        if cv_verified == 1 and gt == 1:
            tp_count += 1
            tp_by_op[op_name] += 1
            result = "TP ✓"
        elif cv_verified == 0 and gt == 0:
            tn_count += 1
            tn_by_op[op_name] += 1
            result = "TN ✓"
        elif cv_verified == 1 and gt == 0:
            fp_count += 1
            fp_by_op[op_name] += 1
            result = "FP ✗"
        elif cv_verified == 0 and gt == 1:
            fn_count += 1
            fn_by_op[op_name] += 1
            result = "FN ✗"

        print(f"[{i:03d}] {op_name:15s} CV={cv_verified} GT={gt} {result}  "
              f"SSIM={ssim_score:.3f} PixelDiff={pixel_ratio:.3f}")

    # Print results
    total_ops = tp_count + tn_count + fp_count + fn_count
    accuracy = (tp_count + tn_count) / max(1, total_ops)
    precision = tp_count / max(1, tp_count + fp_count)
    recall = tp_count / max(1, tp_count + fn_count)
    f1_score = 2 * precision * recall / max(0.001, precision + recall)

    print(f"\n{'='*70}")
    print(f"[EVALUATION RESULTS]")
    print(f"{'='*70}")
    print(f"Total operations: {total_ops}")
    print(f"CV method: {args.cv_method}")
    print(f"Parameters: max_ssim={args.max_ssim}, min_change_ratio={args.min_change_ratio}")

    print(f"\n{'-'*70}")
    print(f"[Confusion Matrix]")
    print(f"{'-'*70}")
    print(f"True Positive (TP):   {tp_count:4d}  (CV=1, GT=1) ✓")
    print(f"True Negative (TN):   {tn_count:4d}  (CV=0, GT=0) ✓")
    print(f"False Positive (FP):  {fp_count:4d}  (CV=1, GT=0) ✗")
    print(f"False Negative (FN):  {fn_count:4d}  (CV=0, GT=1) ✗")

    # Detailed breakdown
    if fn_count > 0 or fp_count > 0:
        print(f"\n{'-'*70}")
        print(f"[Detailed Breakdown by Operation Type]")
        print(f"{'-'*70}")

        if fn_count > 0:
            print(f"\n❌ False Negative ({fn_count} total):")
            for op, count in sorted(fn_by_op.items(), key=lambda x: -x[1]):
                percentage = 100.0 * count / fn_count
                print(f"  {op:20s}: {count:3d} ({percentage:5.1f}%)")

        if fp_count > 0:
            print(f"\n⚠️  False Positive ({fp_count} total):")
            for op, count in sorted(fp_by_op.items(), key=lambda x: -x[1]):
                percentage = 100.0 * count / fp_count
                print(f"  {op:20s}: {count:3d} ({percentage:5.1f}%)")

        if tp_count > 0:
            print(f"\n✓ True Positive ({tp_count} total):")
            for op, count in sorted(tp_by_op.items(), key=lambda x: -x[1]):
                percentage = 100.0 * count / tp_count
                print(f"  {op:20s}: {count:3d} ({percentage:5.1f}%)")

    print(f"\n{'-'*70}")
    print(f"[Overall Metrics]")
    print(f"{'-'*70}")
    print(f"Accuracy:  {accuracy:.4f} ({100.0*accuracy:.2f}%)")
    print(f"Precision: {precision:.4f} ({100.0*precision:.2f}%)")
    print(f"Recall:    {recall:.4f} ({100.0*recall:.2f}%)")
    print(f"F1-Score:  {f1_score:.4f} ({100.0*f1_score:.2f}%)")
    print(f"{'='*70}\n")


# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser(description='Two-Phase CV Evaluation')

    # Mode selection
    parser.add_argument('--mode', choices=['collect', 'eval'], required=True,
                       help='collect: execute operations and save frames | eval: load frames and run CV')

    # Common arguments
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')

    # Phase 1 (collect) arguments
    parser.add_argument('--output-dir', type=str, default='./eval_data/run001',
                       help='Output directory for collected data')
    parser.add_argument('--package', type=str, default='com.rooom.app',
                       help='App package name')
    parser.add_argument('--activity', type=str, default='auto',
                       help='Activity name')
    parser.add_argument('--serial', type=str, default=None,
                       help='Device serial number')
    parser.add_argument('--rounds', type=int, default=60,
                       help='Number of operations to execute')
    parser.add_argument('--negative-ratio', type=float, default=0.5,
                       help='Ratio of negative samples (operations outside AR objects)')
    parser.add_argument('--warmup-wait', type=float, default=3.0,
                       help='Wait time after app launch (seconds)')
    parser.add_argument('--verify-wait-ms', type=int, default=200,
                       help='Wait time after operation before capturing post frame (ms)')
    parser.add_argument('--sleep-min', type=float, default=0.3,
                       help='Minimum sleep between operations (seconds)')
    parser.add_argument('--sleep-max', type=float, default=0.8,
                       help='Maximum sleep between operations (seconds)')

    # Phase 2 (eval) arguments
    parser.add_argument('--input-dir', type=str, default='./eval_data/run001',
                       help='Input directory with collected data')
    parser.add_argument('--cv-method', choices=['ssim', 'pixel_diff', 'hybrid'], default='hybrid',
                       help='CV verification method')
    parser.add_argument('--max-ssim', type=float, default=0.95,
                       help='SSIM threshold (lower = more sensitive)')
    parser.add_argument('--min-change-ratio', type=float, default=0.01,
                       help='Minimum pixel change ratio (higher = less sensitive)')
    parser.add_argument('--diff-threshold', type=int, default=15,
                       help='Pixel difference threshold')

    args = parser.parse_args()

    if args.mode == 'collect':
        phase1_collect(args)
    else:
        phase2_eval(args)


if __name__ == '__main__':
    main()
