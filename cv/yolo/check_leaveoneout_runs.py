#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
检查 leave-one-group-out 的 YOLO 训练结果是否完整。

核心检查逻辑：
- 对于每个 index in [0, k-1]：
  - 在 project 目录下查找名称以 "{ood_type}_idx{idx}_" 开头的子目录
    例如：app-ood_idx0_1_s0_yolo12n
  - 若没有任何匹配目录 → 记为 missing
  - 若有目录，但所有目录下都缺少 weights/best.pt 和 weights/last.pt → 记为 no_model
  - 若至少一个目录下存在 weights/best.pt 或 weights/last.pt → 记为 ok

用法示例：
  python check_leaveoneout_runs.py \
    --project runs/app_ood_leave1out_olddata \
    --ood-type app-ood \
    --k 24

  python check_leaveoneout_runs.py \
    --project runs/scene_ood_leave1out_olddata \
    --ood-type scene-ood \
    --k 37
"""

import argparse
from pathlib import Path
from typing import Dict, List


def find_run_dirs_for_idx(project: Path, ood_type: str, idx: int) -> List[Path]:
    """
    在 project 目录下查找名称以 "{ood_type}_idx{idx}_" 开头的子目录。
    例如：app-ood_idx2_11, app-ood_idx2_11_s0_yolo12n 等。
    """
    prefix = f"{ood_type}_idx{idx}_"
    matches: List[Path] = []
    for d in project.iterdir():
        if d.is_dir() and d.name.startswith(prefix):
            matches.append(d)
    return matches


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check leave-one-group-out YOLO runs (whether models are trained)."
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
        choices=["app-ood", "scene-ood", "object-ood"],
        help="当前检查的 OOD 类型。",
    )
    parser.add_argument(
        "--k",
        type=int,
        required=True,
        help="总的 group 数量（例如 app 一共有 24 个）。",
    )

    args = parser.parse_args()

    project = Path(args.project)
    if not project.exists():
        print(f"[ERROR] project 目录不存在: {project}")
        return

    missing_idx: List[int] = []   # 没有任何 run 目录
    no_model_idx: List[int] = []  # 有目录但没有 best.pt / last.pt
    ok_idx: List[int] = []        # 至少有一个目录有 best.pt / last.pt
    multi_run: Dict[int, List[str]] = {}  # 某些 index 对应多个 run 目录

    for idx in range(args.k):
        runs = find_run_dirs_for_idx(project, args.ood_type, idx)
        if not runs:
            missing_idx.append(idx)
            continue

        if len(runs) > 1:
            multi_run[idx] = [str(d) for d in runs]

        # 检查这些目录下是否有训练好的模型（best.pt 或 last.pt）
        has_model = False
        for d in runs:
            best_path = d / "weights" / "best.pt"
            last_path = d / "weights" / "last.pt"
            if best_path.exists() or last_path.exists():
                has_model = True
                break

        if has_model:
            ok_idx.append(idx)
        else:
            no_model_idx.append(idx)

    # -------- 打印总结 --------
    print("\n========== CHECK SUMMARY ==========")
    print(f"project   : {project}")
    print(f"ood_type  : {args.ood_type}")
    print(f"k (total) : {args.k}")

    print(f"\n[OK] 已找到训练好模型(best/last)的 index:")
    print(f"  {sorted(ok_idx)}" if ok_idx else "  (none)")

    if missing_idx:
        print(f"\n[WARN] 完全没有 run 目录的 index:")
        print(f"  {missing_idx}")
    else:
        print("\n[OK] 所有 index 都有对应的 run 目录。")

    if no_model_idx:
        print(f"\n[WARN] 有 run 目录但缺少 weights/best.pt 和 weights/last.pt 的 index:")
        print(f"  {no_model_idx}")
    else:
        print("\n[OK] 所有 run 目录都包含 best.pt 或 last.pt。")

    if multi_run:
        print("\n[INFO] 某些 index 对应多个 run 目录（可能重复运行）:")
        for idx, paths in multi_run.items():
            print(f"  idx={idx}:")
            for p in paths:
                print(f"    - {p}")

    print("\n========== END ==========")


if __name__ == "__main__":
    main()
