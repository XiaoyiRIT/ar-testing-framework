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
    """
    Run one round of Action Discovery.

    Args:
        drv:   原始 Appium driver（目前主要用于未来 UI 导航；当前版本不直接使用）
        detector: YOLODetector 实例
        sampler:  DefaultSampler 实例
        executor: AppiumExecutor 实例
        cfg:      全局配置字典

    Returns:
        bool: 是否至少完成了一组 target × op 的尝试。
    """
    dc = _build_cfg(cfg)
    verifier = Verifier.from_cfg(cfg)
    policy = NMPolicy.from_cfg(cfg)

    print(f"[AD] ops={dc.ops}, N={dc.N}, M={dc.M}")
    print(f"[AD] out_jsonl={dc.out_jsonl}")
    print(f"[AD] support_jsonl={dc.support_jsonl}")

    # 1) 初始截图 + 检测
    frame0 = executor.snapshot_screen()
    det0 = detector.detect(frame0)
    targets = _select_targets(det0, dc)

    if not targets:
        print("[AD] No detection targets found; aborting run_once.")
        return False

    print(f"[AD] Selected {len(targets)} targets.")

    # 2) 遍历 target × op，执行 N 次操作 & 记录 trial
    trial_fp = dc.out_jsonl.open("w", encoding="utf-8")
    support_records: List[Dict[str, Any]] = []

    # 用于 policy 决策
    per_target_op_results: Dict[str, List[bool]] = {}

    try:
        for tgt in targets:
            tgt_id = tgt["target_id"]
            for op in dc.ops:
                key = f"{tgt_id}:{op}"
                per_target_op_results[key] = []

                print(f"[AD] >> target={tgt_id}, op={op}")

                for k in range(dc.N):
                    t_trial_start = time.time()

                    # (1) 操作前截图
                    pre_img = executor.snapshot_screen()

                    # (2) 采样参数 + 执行操作
                    params = sampler.sample(op, tgt)
                    executor.perform(op, tgt, params)

                    # (3) 等待 AR 反应
                    time.sleep(dc.post_wait_s)

                    # (4) 操作后截图
                    post_img = executor.snapshot_screen()

                    # (5) 验证
                    ok = verifier.verify(op, pre_img, post_img, tgt, params)
                    per_target_op_results[key].append(bool(ok))

                    # (6) 记录一条 trial
                    trial_rec: Dict[str, Any] = {
                        "ts": time.time(),
                        "elapsed_ms": int((time.time() - t_trial_start) * 1000.0),
                        "target_id": tgt_id,
                        "op": op,
                        "k": k,
                        "ok": bool(ok),
                        "params": params,
                        "region": tgt,
                    }
                    trial_fp.write(json.dumps(trial_rec, ensure_ascii=False) + "\n")
                    trial_fp.flush()

    finally:
        trial_fp.close()

    # 3) 生成支持矩阵 & 写入 support.jsonl
    with dc.support_jsonl.open("w", encoding="utf-8") as sup_fp:
        for tgt in targets:
            tgt_id = tgt["target_id"]
            for op in dc.ops:
                key = f"{tgt_id}:{op}"
                results = per_target_op_results.get(key, [])
                if not results:
                    continue
                support = policy.decide_support(results)
                rec = {
                    "target_id": tgt_id,
                    "op": op,
                    "N": len(results),
                    "M": policy.M,
                    "ok_count": int(sum(1 for r in results if r)),
                    "support": bool(support),
                    "region": tgt,
                }
                sup_fp.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("[AD] run_once finished.")
    return True
