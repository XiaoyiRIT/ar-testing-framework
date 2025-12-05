# src/__main__.py
# -*- coding: utf-8 -*-

import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import yaml  # 需安装 pyyaml

# === 你自己的模块实现（按你的目录结构替换/补齐） ===
from src.discovery.run_discovery import run_once
from src.verifier.verifier import Verifier
from src.policy.policy import NMPolicy

from src.detector import YOLODetector
from src.executor import AppiumExecutor
from src.sampler import DefaultSampler


def load_cfg(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 组装输出目录与 JSONL 文件路径
    out_dir = cfg.get("runtime", {}).get("out_dir", None)
    if not out_dir:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        out_dir = f"runs/exp-{stamp}"
        cfg.setdefault("runtime", {})["out_dir"] = out_dir

    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    cfg["runtime"]["out_jsonl"] = str(out_dir_path / "trials.jsonl")
    cfg["runtime"]["support_jsonl"] = str(out_dir_path / "support.jsonl")

    # 默认项补齐
    cfg.setdefault("policy", {}).setdefault("N", 10)
    cfg["policy"].setdefault("M", 2)
    cfg.setdefault("thresholds", {}).setdefault("tau_move", 0.03)
    cfg["thresholds"].setdefault("tau_rot_deg", 8.0)
    cfg["thresholds"].setdefault("tau_ssim", 0.08)
    cfg.setdefault("runtime", {}).setdefault("post_wait_s", 0.4)

    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    # 如果你使用了 torch 或其他框架，可以在这里补充它们的随机种子


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="ActionDiscovery",
        description="Run Action Discovery pipeline (detector → sampler → executor → verifier → policy).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--cfg", type=str, required=True, help="Path to YAML config (e.g., configs/ad.yaml)")
    p.add_argument("--seed", type=int, default=42, help="Global random seed")
    p.add_argument(
        "--ops",
        type=str,
        default="drag,rotate,tap",
        help="Comma-separated ops to run (subset of: tap,drag,rotate,pinch_in,pinch_out)",
    )
    p.add_argument("--out-dir", type=str, default=None, help="Override runtime.out_dir")
    p.add_argument("--device-id", type=str, default=None, help="Override device id if needed")
    p.add_argument("--dry-run", action="store_true", help="Load everything but skip executing gestures")
    return p.parse_args(argv)


def make_components(cfg: dict, device_id_override: str | None = None):
    detector = YOLODetector.from_cfg(cfg)
    sampler = DefaultSampler.from_cfg(cfg)
    dev_id = device_id_override or cfg.get("device", None)
    executor = AppiumExecutor(cfg=cfg, device_id=dev_id)

    driver = executor.driver
    return driver, detector, sampler, executor


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    cfg = load_cfg(args.cfg)

    if args.out_dir:
        cfg["runtime"]["out_dir"] = args.out_dir
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)
        cfg["runtime"]["out_jsonl"] = str(Path(args.out_dir) / "trials.jsonl")
        cfg["runtime"]["support_jsonl"] = str(Path(args.out_dir) / "support.jsonl")

    if args.device_id:
        cfg["device"] = args.device_id

    cfg["seed"] = args.seed
    set_seed(args.seed)

    # 仅在需要时将 ops 覆盖到 cfg（给 sampler/executor 可见）
    ops = [s.strip() for s in args.ops.split(",") if s.strip()]
    cfg["ops"] = ops

    # 打印一次关键配置（便于追踪）
    print(json.dumps(
        {
            "out_dir": cfg["runtime"]["out_dir"],
            "out_jsonl": cfg["runtime"]["out_jsonl"],
            "support_jsonl": cfg["runtime"]["support_jsonl"],
            "ops": ops,
            "seed": cfg["seed"],
        },
        indent=2,
        ensure_ascii=False,
    ))

    # 构建组件
    driver, detector, sampler, executor = make_components(cfg, args.device_id)

    if args.dry_run:
        print("[DRY RUN] Components instantiated. Skipping execution.")
        return 0

    # Verifier/Policy 在 run_once 内部会实例化；也可以在这里先构建再传入（保持与之前骨架一致）
    # 这里沿用 run_once 内部创建 Verifier/NMPolicy 的设计
    ok = run_once(
        drv=driver,
        detector=detector,
        sampler=sampler,
        executor=executor,
        cfg=cfg,
    )

    print(f"[DONE] run_once -> {ok}. Outputs at: {cfg['runtime']['out_dir']}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
