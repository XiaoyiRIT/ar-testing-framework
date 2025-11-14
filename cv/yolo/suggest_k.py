#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
python suggest_k.py --data-csv data_stat.csv --group-col app
python suggest_k.py --data-csv data_stat.csv --group-col scene
python suggest_k.py --data-csv data_stat.csv --group-col object

"""

import argparse
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


def load_groups(data_csv: Path, group_col: str) -> np.ndarray:
    df = pd.read_csv(data_csv)
    if group_col not in df.columns:
        # 尝试大小写容错
        lower_map = {c.lower(): c for c in df.columns}
        if group_col.lower() not in lower_map:
            raise ValueError(f"CSV 中找不到列 {group_col!r}，实际列有：{list(df.columns)}")
        group_col = lower_map[group_col.lower()]
    groups = df[group_col].astype(str).values
    return groups


def evaluate_k_for_groups(
    groups: np.ndarray,
    k: int,
    target_val_ratio: float,
    min_val_samples: int,
    min_val_groups: int,
    min_train_groups: int,
) -> Optional[Dict]:
    """对给定的 K 做一次 GroupKFold 评估，若不满足约束则返回 None。"""
    n = len(groups)
    gkf = GroupKFold(n_splits=k)

    val_sizes: List[int] = []
    val_group_counts: List[int] = []
    train_group_counts: List[int] = []

    dummy_X = np.zeros(n)

    for train_idx, val_idx in gkf.split(dummy_X, groups=groups):
        val_sizes.append(len(val_idx))
        val_group_counts.append(len(np.unique(groups[val_idx])))
        train_group_counts.append(len(np.unique(groups[train_idx])))

    val_sizes = np.array(val_sizes)
    val_ratios = val_sizes / float(n)
    val_group_counts = np.array(val_group_counts)
    train_group_counts = np.array(train_group_counts)

    # 硬约束过滤
    if val_sizes.min() < min_val_samples:
        return None
    if val_group_counts.min() < min_val_groups:
        return None
    if train_group_counts.min() < min_train_groups:
        return None

    val_ratio_mean = float(val_ratios.mean())
    val_ratio_std = float(val_ratios.std())
    min_val_size = int(val_sizes.min())
    min_val_groups = int(val_group_counts.min())
    min_train_groups = int(train_group_counts.min())

    # 评分：越小越好
    score = abs(val_ratio_mean - target_val_ratio) / max(target_val_ratio, 1e-6) + val_ratio_std

    return {
        "k": k,
        "score": score,
        "val_ratio_mean": val_ratio_mean,
        "val_ratio_std": val_ratio_std,
        "min_val_size": min_val_size,
        "min_val_groups": min_val_groups,
        "min_train_groups": min_train_groups,
    }


def suggest_k(
    groups: np.ndarray,
    max_k: int = 10,
    target_val_ratio: float = 0.2,
    min_val_samples: int = 100,
    min_val_groups: int = 1,
    min_train_groups: int = 2,
) -> Dict:
    """遍历 K=2..max_k，自动选择最合适的 K。"""
    unique_groups = np.unique(groups)
    num_groups = len(unique_groups)
    n = len(groups)

    print(f"[INFO] 总样本数: {n}")
    print(f"[INFO] 不同 group 数量: {num_groups}")

    # K 不能超过 group 数
    max_k_feasible = min(max_k, num_groups)
    if max_k_feasible < 2:
        raise ValueError("group 数量太少，无法做 KFold（至少需要 2 个不同 group）。")

    candidates: List[Dict] = []

    for k in range(2, max_k_feasible + 1):
        print(f"\n[CHECK] 评估 K = {k} ...")
        stats = evaluate_k_for_groups(
            groups=groups,
            k=k,
            target_val_ratio=target_val_ratio,
            min_val_samples=min_val_samples,
            min_val_groups=min_val_groups,
            min_train_groups=min_train_groups,
        )
        if stats is None:
            print("  -> 不满足约束（val 太小 / group 太少），丢弃")
            continue

        print(
            f"  -> val_ratio_mean={stats['val_ratio_mean']:.3f}, "
            f"val_ratio_std={stats['val_ratio_std']:.3f}, "
            f"min_val_size={stats['min_val_size']}, "
            f"min_val_groups={stats['min_val_groups']}, "
            f"min_train_groups={stats['min_train_groups']}"
        )

        candidates.append(stats)

    if not candidates:
        raise RuntimeError("没有任何 K 满足约束，请调低 min_val_samples 或放宽约束。")

    # 以 score 最小为准，若 score 相同则倾向较小 K（训练代价更低）
    candidates_sorted = sorted(candidates, key=lambda d: (d["score"], d["k"]))
    best = candidates_sorted[0]

    print("\n========== 推荐结果 ==========")
    print(f"最佳 K: {best['k']}")
    print(f"  平均 val 比例: {best['val_ratio_mean']:.3f}")
    print(f"  val 比例标准差: {best['val_ratio_std']:.3f}")
    print(f"  最小 val 样本数: {best['min_val_size']}")
    print(f"  最少 val group 数: {best['min_val_groups']}")
    print(f"  最少 train group 数: {best['min_train_groups']}")

    print("\n所有候选 K 统计：")
    for s in candidates_sorted:
        print(
            f"  K={s['k']}: score={s['score']:.4f}, "
            f"val_mean={s['val_ratio_mean']:.3f}, "
            f"val_std={s['val_ratio_std']:.3f}, "
            f"min_val_size={s['min_val_size']}, "
            f"min_val_groups={s['min_val_groups']}, "
            f"min_train_groups={s['min_train_groups']}"
        )

    return best


def main() -> None:
    parser = argparse.ArgumentParser(
        description="根据 data_stat.csv 和 group 列，自动推荐最合适的 K 值（GroupKFold）。"
    )
    parser.add_argument("--data-csv", type=str, required=True, help="例如 data_stat.csv，包含 group 列")
    parser.add_argument(
        "--group-col",
        type=str,
        required=True,
        help="group 列名，比如 app / object / scene 等",
    )
    parser.add_argument(
        "--max-k",
        type=int,
        default=10,
        help="最大的候选 K 值（默认 10）",
    )
    parser.add_argument(
        "--target-val-ratio",
        type=float,
        default=0.2,
        help="期望的验证集比例（默认 0.2 ≈ 20%）",
    )
    parser.add_argument(
        "--min-val-samples",
        type=int,
        default=100,
        help="每折最少需要多少验证样本（默认 100）",
    )
    parser.add_argument(
        "--min-val-groups",
        type=int,
        default=1,
        help="每折验证集中最少 group 数（默认 1）",
    )
    parser.add_argument(
        "--min-train-groups",
        type=int,
        default=2,
        help="每折训练集中最少 group 数（默认 2）",
    )

    args = parser.parse_args()

    groups = load_groups(Path(args.data_csv), args.group_col)
    suggest_k(
        groups=groups,
        max_k=args.max_k,
        target_val_ratio=args.target_val_ratio,
        min_val_samples=args.min_val_samples,
        min_val_groups=args.min_val_groups,
        min_train_groups=args.min_train_groups,
    )


if __name__ == "__main__":
    main()
