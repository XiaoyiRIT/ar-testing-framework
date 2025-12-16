#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
将旧格式的 YOLO leave-one-out 目录名称统一转换成新格式。

旧格式示例：
    app-ood_idx0_1
    scene-ood_idx3_5
    object-ood_idx2_cube

新格式示例：
    app-ood_idx0_1_s0_yolo12n
    scene-ood_idx3_5_s0_yolo12n

使用方法：
    python rename_legacy_runs.py \
        --project runs/leave1out \
        --seed 0 \
        --model yolo12n.pt

加 --dryrun 可以只打印不执行：
    python rename_legacy_runs.py --project runs/app_ood_leave1out --seed 0 --model yolo12n.pt --dryrun
"""

import argparse
import re
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, required=True,
                        help="YOLO leave-one-out 项目根目录，例如 runs/leave1out")
    parser.add_argument("--seed", type=int, required=True,
                        help="用于新格式名中的 s{seed}")
    parser.add_argument("--model", type=str, required=True,
                        help="模型名，例如 yolo12n.pt（会自动取 stem = yolo12n）")
    parser.add_argument("--dryrun", action="store_true",
                        help="只打印重命名计划，不真正执行")

    args = parser.parse_args()

    project = Path(args.project)
    if not project.exists():
        print(f"[ERROR] project path not found: {project}")
        return

    model_stem = Path(args.model).stem
    seed = args.seed

    # 旧目录名正则：  oodtype_idx<idx>_<group>
    # 支持 app-ood / scene-ood / object-ood
    old_pattern = re.compile(
        r"^(app-ood|scene-ood|object-ood)_idx(\d+)_(.+)$"
    )

    print(f"[INFO] Scanning: {project}")
    print(f"[INFO] Model stem = {model_stem}, seed = {seed}")
    print("--------------------------------------------------")

    for d in project.iterdir():
        if not d.is_dir():
            continue

        m = old_pattern.match(d.name)
        if not m:
            continue

        ood_type, idx, group = m.groups()

        # 新格式
        new_name = f"{ood_type}_idx{idx}_{group}_s{seed}_{model_stem}"
        new_path = project / new_name

        # 已经是新格式的跳过
        if new_path.exists():
            print(f"[SKIP] target exists → {new_name}")
            continue

        print(f"[RENAME] {d.name}  →  {new_name}")

        if not args.dryrun:
            d.rename(new_path)

    print("--------------------------------------------------")
    print("[DONE]" if not args.dryrun else "[DRYRUN DONE]")


if __name__ == "__main__":
    main()
