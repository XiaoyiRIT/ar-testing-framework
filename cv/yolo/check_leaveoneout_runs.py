#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
检查 leave-one-group-out 的 YOLO 训练结果是否完整。

用法示例：
  python check_leaveoneout_runs.py \
    --project runs/app_ood_leave1out \
    --ood-type app-ood \
    --k 22
"""

import argparse
from pathlib import Path
from typing import Dict, List


def find_run_dir_for_idx(project: Path, ood_type: str, idx: int) -> List[Path]:
    """
    在 project 目录下查找名称以 "{ood_type}_idx{idx}_" 开头的子目录。
    例如：app-ood_idx2_11
    """
    prefix = f"{ood_type}_idx{idx}_"
    matches: List[Path] = []
    for d in project.iterdir():
        if d.is_dir() and d.name.startswith(prefix):
            matches.append(d)
    return matches


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check incomplete leave-one-group-out YOLO runs."
    )
    parser.add_argument(
        "--project",
        type=str,
        required=True,
        help="YOLO 训练的 project 目录（与 yolo_train_leave_one_group.py 中的 --project 一致）。",
    )
    parser.add_argument(
        "--ood-type",
        type=str,
        required=True,
        choices=["app-ood", "scene-ood"],
        help="当前检查的 OOD 类型。",
    )
    parser.add_argument(
        "--k",
        type=int,
        required=True,
        help="总的 group 数量（例如 app 一共有 22 个）。",
    )
    parser.add_argument(
        "--check-best",
        action="store_true",
        help="额外检查每个 run 的 weights/best.pt 是否存在（默认只检查目录存在）。",
    )

    args = parser.parse_args()

    project = Path(args.project)
    if not project.exists():
        print(f"[ERROR] project 目录不存在: {project}")
        return

    missing_run: List[int] = []
    multi_run: Dict[int, List[str]] = {}
    no_best: List[int] = []
    ok: List[int] = []

    for idx in range(args.k):
        runs = find_run_dir_for_idx(project, args.ood_type, idx)
        if not runs:
            missing_run.append(idx)
            continue
        if len(runs) > 1:
            multi_run[idx] = [str(d) for d in runs]

        # 默认认为只要目录存在，就算“有训练记录”
        run_dir = runs[0]

        if args.check-best:
            best_path = run_dir / "weights" / "best.pt"
            if not best_path.exists():
                no_best.append(idx)
            else:
                ok.append(idx)
        else:
            ok.append(idx)

    print("\n========== CHECK SUMMARY ==========")
    print(f"project   : {project}")
    print(f"ood_type  : {args.ood_type}")
    print(f"k (total) : {args.k}")

    print(f"\n[OK] 有训练目录的 index: {sorted(ok)}")

    if missing_run:
        print(f"\n[WARN] 缺少 run 目录的 index: {missing_run}")
    else:
        print("\n[OK] 所有 index 都有对应的 run 目录。")

    if multi_run:
        print("\n[WARN] 某些 index 对应多个 run 目录（可能重复运行）:")
        for idx, paths in multi_run.items():
            print(f"  idx={idx}:")
            for p in paths:
                print(f"    - {p}")

    if args.check-best:
        if no_best:
            print(f"\n[WARN] 没有 weights/best.pt 的 index: {no_best}")
        else:
            print("\n[OK] 所有 run 都包含 weights/best.pt。")

    print("\n========== END ==========")


if __name__ == "__main__":
    main()
