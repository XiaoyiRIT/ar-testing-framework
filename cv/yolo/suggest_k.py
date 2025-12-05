#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根据 data_stat.csv 和某个 group 列（App/Object/Scene 等），
自动推荐一个较合适的 K，用于 GroupKFold。

示例用法：
python suggest_k.py --data-csv data_stat.csv --group-col App
python suggest_k.py --data-csv data_stat.csv --group-col Scene
python suggest_k.py --data-csv data_stat.csv --group-col Object

注意：
- 本脚本的评估逻辑和训练脚本中的 KFold 保持一致：
  * 在 “unique groups” 上做 GroupKFold（每个 group 权重相同）
  * 再用 group -> 样本数 来计算每折的验证集样本比例
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


# ----------------- 数据加载：group+样本数 -----------------


def load_group_counts(
    data_csv: Path,
    group_col: str,
) -> Tuple[np.ndarray, Dict[str, int], int]:
    """
    从 CSV 中读取 group 列，返回：

    - unique_groups: np.ndarray[str]，所有不重复 group id
    - group_counts: dict[group -> 样本数量]
    - total_samples: int，总样本数（行数）

    会做一层列名的大小写容错。
    """
    df = pd.read_csv(data_csv)
    if df.empty:
        raise RuntimeError(f"CSV 为空：{data_csv}")

    # 列名大小写容错
    lower_map = {c.lower(): c for c in df.columns}
    if group_col not in df.columns:
        if group_col.lower() not in lower_map:
            raise ValueError(f"CSV 中找不到列 {group_col!r}，实际列有：{list(df.columns)}")
        group_col = lower_map[group_col.lower()]

    groups_series = df[group_col].astype(str)
    total_samples = int(len(groups_series))

    # group -> 样本数量
    group_counts_series = groups_series.value_counts()
    unique_groups = group_counts_series.index.astype(str).to_numpy()
    group_counts = {g: int(group_counts_series[g]) for g in group_counts_series.index}

    return unique_groups, group_counts, total_samples


# ----------------- 评估某个 K -----------------


def evaluate_k_for_groups(
    unique_groups: np.ndarray,
    group_counts: Dict[str, int],
    total_samples: int,
    k: int,
    target_val_ratio: float,
    min_val_samples: int,
    min_val_groups: int,
    min_train_groups: int,
) -> Optional[Dict]:
    """
    在 unique_groups 上做一次 GroupKFold 评估，并基于“样本数”统计每折的验证集比例。

    参数：
    - unique_groups: 所有 group id（每个只出现一次）
    - group_counts: group -> 样本数量
    - total_samples: 数据集中总样本数
    - k: 候选的 K
    - 其它参数为各种约束/目标

    返回：
    - 若不满足约束（val 过小 / group 太少等），返回 None
    - 否则返回统计信息 dict：
      {
        "k", "score",
        "val_ratio_mean", "val_ratio_std",
        "min_val_size", "min_val_groups", "min_train_groups"
      }
    """
    num_groups = len(unique_groups)
    if k > num_groups:
        # K 不能大于 group 数
        return None

    # 在 “group 级别” 上做 GroupKFold
    gkf = GroupKFold(n_splits=k)
    dummy_X = np.zeros(num_groups)
    folds = gkf.split(dummy_X, y=None, groups=unique_groups)

    val_sample_sizes: List[int] = []
    val_group_counts: List[int] = []
    train_group_counts: List[int] = []

    for train_idx, val_idx in folds:
        # 当前折的 group id
        train_groups = unique_groups[train_idx]
        val_groups = unique_groups[val_idx]

        # 样本数：把这些 group 对应的样本数加起来
        val_size = int(sum(group_counts[g] for g in val_groups))
        train_size = int(sum(group_counts[g] for g in train_groups))

        val_sample_sizes.append(val_size)
        val_group_counts.append(len(val_groups))
        train_group_counts.append(len(train_groups))

    val_sizes = np.array(val_sample_sizes, dtype=float)
    val_ratios = val_sizes / float(total_samples)  # 每折验证集样本比例
    val_group_counts = np.array(val_group_counts, dtype=int)
    train_group_counts = np.array(train_group_counts, dtype=int)

    # -------- 硬约束过滤 --------
    if val_sizes.min() < min_val_samples:
        return None
    if val_group_counts.min() < min_val_groups:
        return None
    if train_group_counts.min() < min_train_groups:
        return None

    # 统计指标
    val_ratio_mean = float(val_ratios.mean())
    val_ratio_std = float(val_ratios.std())
    min_val_size = int(val_sizes.min())
    min_val_groups_achieved = int(val_group_counts.min())
    min_train_groups_achieved = int(train_group_counts.min())

    # 评分函数：越小越好
    #   - 第一项：平均验证集比例偏离 target_val_ratio 的相对误差
    #   - 第二项：各折验证比例的标准差（越均匀越好）
    score = (
        abs(val_ratio_mean - target_val_ratio) / max(target_val_ratio, 1e-6)
        + val_ratio_std
    )

    return {
        "k": k,
        "score": score,
        "val_ratio_mean": val_ratio_mean,
        "val_ratio_std": val_ratio_std,
        "min_val_size": min_val_size,
        "min_val_groups": min_val_groups_achieved,
        "min_train_groups": min_train_groups_achieved,
    }


# ----------------- 遍历 K，选择最优 -----------------


def suggest_k(
    unique_groups: np.ndarray,
    group_counts: Dict[str, int],
    total_samples: int,
    max_k: int = 10,
    target_val_ratio: float = 0.2,
    min_val_samples: int = 100,
    min_val_groups: int = 1,
    min_train_groups: int = 2,
) -> Dict:
    """
    遍历 K=2..max_k，自动选择最合适的 K。

    与训练脚本中的 KFold 设计一致：
    - GroupKFold 在 group 级进行划分；
    - 验证集比例以“样本数”而非 group 数计算。
    """
    num_groups = len(unique_groups)

    print(f"[INFO] 总样本数: {total_samples}")
    print(f"[INFO] 不同 group 数量: {num_groups}")

    # K 不能超过 group 数
    max_k_feasible = min(max_k, num_groups)
    if max_k_feasible < 2:
        raise ValueError("group 数量太少，无法做 KFold（至少需要 2 个不同 group）。")

    candidates: List[Dict] = []

    for k in range(2, max_k_feasible + 1):
        print(f"\n[CHECK] 评估 K = {k} ...")
        stats = evaluate_k_for_groups(
            unique_groups=unique_groups,
            group_counts=group_counts,
            total_samples=total_samples,
            k=k,
            target_val_ratio=target_val_ratio,
            min_val_samples=min_val_samples,
            min_val_groups=min_val_groups,
            min_train_groups=min_train_groups,
        )
        if stats is None:
            print("  -> 不满足约束（val 样本太少 / group 太少），丢弃")
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
        raise RuntimeError(
            "没有任何 K 满足约束，请尝试：\n"
            "  - 调低 --min-val-samples\n"
            "  - 或者减小 --max-k\n"
            "  - 或者检查 group 是否过于不平衡"
        )

    # 以 score 最小为准，若 score 相同则倾向较小 K（训练代价更低）
    candidates_sorted = sorted(candidates, key=lambda d: (d["score"], d["k"]))
    best = candidates_sorted[0]

    print("\n========== 推荐结果 ==========")
    print(f"最佳 K: {best['k']}")
    print(f"  平均 val 样本比例: {best['val_ratio_mean']:.3f}")
    print(f"  val 比例标准差:    {best['val_ratio_std']:.3f}")
    print(f"  最小 val 样本数:   {best['min_val_size']}")
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


# ----------------- CLI -----------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="根据 data_stat.csv 和 group 列，自动推荐最合适的 K 值（GroupKFold）。"
    )
    parser.add_argument(
        "--data-csv",
        type=str,
        required=True,
        help="例如 data_stat.csv，包含 group 列",
    )
    parser.add_argument(
        "--group-col",
        type=str,
        required=True,
        help="group 列名，比如 App / Object / Scene 等",
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
        help="期望的验证集“样本比例”（默认 0.2 ≈ 20%）。\n"
             "如果你的 train 脚本 ratio 是 0.7,0.15,0.15，"
             "可以把这里改成 0.15 来匹配。",
    )
    parser.add_argument(
        "--min-val-samples",
        type=int,
        default=100,
        help="每折最少需要多少验证样本（默认 100）。",
    )
    parser.add_argument(
        "--min-val-groups",
        type=int,
        default=1,
        help="每折验证集中最少 group 数（默认 1）。",
    )
    parser.add_argument(
        "--min-train-groups",
        type=int,
        default=2,
        help="每折训练集中最少 group 数（默认 2）。",
    )

    args = parser.parse_args()

    unique_groups, group_counts, total_samples = load_group_counts(
        Path(args.data_csv), args.group_col
    )
    suggest_k(
        unique_groups=unique_groups,
        group_counts=group_counts,
        total_samples=total_samples,
        max_k=args.max_k,
        target_val_ratio=args.target_val_ratio,
        min_val_samples=args.min_val_samples,
        min_val_groups=args.min_val_groups,
        min_train_groups=args.min_train_groups,
    )


if __name__ == "__main__":
    main()
