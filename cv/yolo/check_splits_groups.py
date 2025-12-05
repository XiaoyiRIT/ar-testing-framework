#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
检查 .split（或指定 splits 目录）下每个 split 子目录中：
1. train/val/test 是否在“文件级别”互斥；
2. 指定的 group 列（如 App/Object/Scene）在“group 级别”是否互斥。

- 文件级别：同一图片路径是否同时出现在 train/val/test；
- group 级别：例如 App 维度上，train 与 test 是否共享同一个 App。

使用示例：

python check_splits_groups.py \
  --split-outdir .split \
  --data-csv data_stat.csv \
  --group-cols App,Object,Scene

如只想检查 App：

python check_splits_groups.py \
  --split-outdir .split \
  --data-csv data_stat.csv \
  --group-cols App
"""

import argparse
from pathlib import Path
from typing import Dict, Set, Tuple, List

import pandas as pd


# ---------- 通用小工具 ----------

def load_list(path: Path) -> Set[str]:
    """读取一个 txt 文件，每行一个路径，返回去重后的 set。"""
    if not path.exists():
        return set()
    lines = path.read_text(encoding="utf-8").splitlines()
    return {ln.strip() for ln in lines if ln.strip()}


def _find_filename_column(df: pd.DataFrame) -> str:
    """根据列名自动推测 filename 列。"""
    lower_map = {c.lower(): c for c in df.columns}
    for key in ["filename", "file", "image", "img"]:
        if key in lower_map:
            return lower_map[key]
    # 没有匹配到时，用第一列，但给出提醒
    print(f"[WARN] 未找到典型 filename 列，默认使用第一列: {df.columns[0]!r}")
    return df.columns[0]


# ---------- group 映射构建 ----------

def build_name_to_groups(
    data_csv: Path,
    group_cols: List[str],
) -> Tuple[Dict[str, Dict[str, str]], List[str]]:
    """
    从 data_csv 读取 filename 和指定 group 列，构建映射：

    name_to_groups[basename][group_col] = group_id

    返回:
      - name_to_groups: { "xxx.jpg": { "App": "xxxApp", "Object": "obj1", ... }, ... }
      - valid_group_cols: 实际在 CSV 中存在的 group 列（自动过滤不存在的）
    """
    df = pd.read_csv(data_csv)
    if df.empty:
        raise RuntimeError(f"CSV 为空: {data_csv}")

    filename_col = _find_filename_column(df)
    lower_map = {c.lower(): c for c in df.columns}

    # 过滤出真正存在的 group 列
    valid_group_cols: List[str] = []
    for g in group_cols:
        if g in df.columns:
            valid_group_cols.append(g)
        elif g.lower() in lower_map:
            valid_group_cols.append(lower_map[g.lower()])
        else:
            print(f"[WARN] group 列 {g!r} 不在 CSV 中，跳过该列。")

    if not valid_group_cols:
        print("[WARN] 未找到任何有效的 group 列，不会进行 group 级别检查。")
        return {}, []

    print(f"[INFO] 使用 filename 列: {filename_col!r}")
    print(f"[INFO] 使用 group 列: {valid_group_cols}")

    name_to_groups: Dict[str, Dict[str, str]] = {}

    for _, row in df.iterrows():
        raw_name = str(row[filename_col])
        basename = Path(raw_name).name  # 仅使用文件名部分
        grp_dict = name_to_groups.setdefault(basename, {})
        for col in valid_group_cols:
            grp_dict[col] = str(row[col])

    return name_to_groups, valid_group_cols


# ---------- 单个 split 目录检查 ----------

def check_one_split_dir_files(split_dir: Path) -> Tuple[bool, str]:
    """
    检查单个 split 子目录中 train/val/test 在“文件级别”是否互斥。

    返回:
      ok, msg
        ok = True  : 互斥
        ok = False : 存在重复文件
    """
    train_txt = split_dir / "train.txt"
    val_txt = split_dir / "val.txt"
    test_txt = split_dir / "test.txt"

    if not train_txt.exists():
        return True, "no train.txt, skipped"

    train_set = load_list(train_txt)
    val_set = load_list(val_txt)
    test_set = load_list(test_txt)

    inter_tr_va = train_set & val_set
    inter_tr_te = train_set & test_set
    inter_va_te = val_set & test_set

    leak = False
    parts = []

    if inter_tr_va:
        leak = True
        parts.append(f"train ∩ val = {len(inter_tr_va)}")
    if inter_tr_te:
        leak = True
        parts.append(f"train ∩ test = {len(inter_tr_te)}")
    if inter_va_te:
        leak = True
        parts.append(f"val ∩ test = {len(inter_va_te)}")

    if leak:
        detail = "; ".join(parts)
        return False, detail

    return True, "OK"


def check_one_split_dir_groups(
    split_dir: Path,
    name_to_groups: Dict[str, Dict[str, str]],
    group_cols: List[str],
) -> Tuple[bool, str]:
    """
    检查单个 split 子目录中，在指定 group 列维度上 train/val/test 是否互斥。

    这里的“互斥”指的是：同一个 group id（例如某个 App 名）不能同时出现在
    train 与 val、train 与 test、val 与 test 中。

    返回:
      ok, msg
        ok = True  : 所有 group 列均互斥
        ok = False : 某些 group 列存在 group 级别的数据泄露
    """
    if not group_cols or not name_to_groups:
        # 没有 group 配置或映射，视为 OK
        return True, "no group check"

    train_txt = split_dir / "train.txt"
    val_txt = split_dir / "val.txt"
    test_txt = split_dir / "test.txt"

    if not train_txt.exists():
        return True, "no train.txt, skipped"

    train_paths = load_list(train_txt)
    val_paths = load_list(val_txt)
    test_paths = load_list(test_txt)

    # 基于 basename 做 filename -> group_id 映射
    def to_basenames(paths: Set[str]) -> Set[str]:
        return {Path(p).name for p in paths}

    train_names = to_basenames(train_paths)
    val_names = to_basenames(val_paths)
    test_names = to_basenames(test_paths)

    # 统计有多少文件在 CSV 映射中缺失（仅做提示，不作为硬错误）
    def count_missing(names: Set[str]) -> int:
        return sum(1 for n in names if n not in name_to_groups)

    miss_train = count_missing(train_names)
    miss_val = count_missing(val_names)
    miss_test = count_missing(test_names)
    if miss_train or miss_val or miss_test:
        print(
            f"[WARN] {split_dir.name}: 在 CSV 映射中有文件缺失 "
            f"(train={miss_train}, val={miss_val}, test={miss_test})"
        )

    leak = False
    details: List[str] = []

    for col in group_cols:
        # 将每个集合中的文件名映射为 group_id 集合
        def collect_groups(names: Set[str]) -> Set[str]:
            groups = set()
            for n in names:
                info = name_to_groups.get(n)
                if info is None:
                    continue
                if col in info:
                    groups.add(info[col])
            return groups

        g_train = collect_groups(train_names)
        g_val = collect_groups(val_names)
        g_test = collect_groups(test_names)

        inter_tr_va = g_train & g_val
        inter_tr_te = g_train & g_test
        inter_va_te = g_val & g_test

        parts = []
        if inter_tr_va:
            leak = True
            parts.append(f"train ∩ val ({col}) = {len(inter_tr_va)}")
        if inter_tr_te:
            leak = True
            parts.append(f"train ∩ test ({col}) = {len(inter_tr_te)}")
        if inter_va_te:
            leak = True
            parts.append(f"val ∩ test ({col}) = {len(inter_va_te)}")

        if parts:
            details.append("; ".join(parts))

    if leak:
        return False, " | ".join(details) if details else "group leak"
    else:
        return True, "OK"


# ---------- 主逻辑 ----------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="检查 .split 下 train/val/test 的文件级与 group 级互斥性。"
    )
    parser.add_argument(
        "--split-outdir",
        type=str,
        default=".split",
        help="保存各个 split 子目录的根目录（默认 .split）",
    )
    parser.add_argument(
        "--data-csv",
        type=str,
        required=True,
        help="如 data_stat.csv，包含 filename 和 App/Object/Scene 等列。",
    )
    parser.add_argument(
        "--group-cols",
        type=str,
        default="",
        help="逗号分隔的 group 列名，例如 App,Object,Scene。留空则不做 group 级检查。",
    )

    args = parser.parse_args()
    split_root = Path(args.split_outdir)
    data_csv = Path(args.data_csv)

    if not split_root.exists():
        raise FileNotFoundError(f"split-outdir 不存在: {split_root}")
    if not data_csv.exists():
        raise FileNotFoundError(f"data-csv 不存在: {data_csv}")

    group_cols = [g.strip() for g in args.group_cols.split(",") if g.strip()]
    name_to_groups, valid_group_cols = build_name_to_groups(data_csv, group_cols)

    leak_file_dirs: List[str] = []
    leak_group_dirs: List[str] = []

    checked = 0
    skipped = 0

    print(f"[INFO] 检查 splits 根目录: {split_root}")

    for d in sorted(split_root.iterdir(), key=lambda p: p.name):
        if not d.is_dir():
            continue

        ok_files, msg_files = check_one_split_dir_files(d)
        if msg_files == "no train.txt, skipped":
            skipped += 1
            continue

        checked += 1

        ok_groups, msg_groups = check_one_split_dir_groups(
            d, name_to_groups, valid_group_cols
        )

        # 输出单个目录结果
        if ok_files and ok_groups:
            print(f"[OK]    {d.name}: files=OK, groups=OK")
        else:
            if not ok_files:
                print(f"[LEAK]  {d.name}: 文件级泄露: {msg_files}")
                leak_file_dirs.append(d.name)
            if not ok_groups:
                print(f"[LEAK]  {d.name}: group级泄露: {msg_groups}")
                leak_group_dirs.append(d.name)

    print("\n========== SUMMARY ==========")
    print(f"已检查 split 子目录数量: {checked}")
    print(f"跳过（无 train.txt）数量: {skipped}")

    if not leak_file_dirs and not leak_group_dirs:
        print("✅ 文件级 + group 级均未发现数据泄露。")
    else:
        if leak_file_dirs:
            print("\n⚠ 文件级泄露目录（train/val/test 有重复图片）：")
            for name in sorted(set(leak_file_dirs)):
                print(f"  - {name}")
        if leak_group_dirs:
            print("\n⚠ group 级泄露目录（train/val/test 共享同一 group，例如同一 App）：")
            for name in sorted(set(leak_group_dirs)):
                print(f"  - {name}")


if __name__ == "__main__":
    main()
