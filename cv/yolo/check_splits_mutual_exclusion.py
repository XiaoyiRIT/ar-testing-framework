#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
检查 .split（或指定 splits 目录）下每个 split 子目录中
train/val/test 的数据是否互斥。

- 若 train/val/test 之间有图片路径重叠，则视为“数据泄露”，打印该目录名。
- 默认只检查文件级别（同一图片路径是否被重复使用）。
- 使用方式示例：

  python check_splits_mutual_exclusion.py --split-outdir .split

"""

import argparse
from pathlib import Path
from typing import Set, Tuple


def load_list(path: Path) -> Set[str]:
    """读取一个 txt 文件，每行一个路径，返回去重后的 set。"""
    if not path.exists():
        return set()
    lines = path.read_text(encoding="utf-8").splitlines()
    return {ln.strip() for ln in lines if ln.strip()}


def check_one_split_dir(split_dir: Path) -> Tuple[bool, str]:
    """
    检查单个 split 子目录中 train/val/test 是否互斥。

    返回值:
      (ok, msg)
        ok = True  : 互斥（没有重叠）
        ok = False : 存在重叠（数据泄露），msg 中包含细节
    """
    train_txt = split_dir / "train.txt"
    val_txt = split_dir / "val.txt"
    test_txt = split_dir / "test.txt"

    # 没有 train.txt 的目录直接跳过（可能是 last.yaml 等）
    if not train_txt.exists():
        return True, "no train.txt, skipped"

    train_set = load_list(train_txt)
    val_set = load_list(val_txt)
    test_set = load_list(test_txt)

    # 至少有一个集合为空时，也照样检查，但一般视为“没有泄露”
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="检查 splits 目录下 train/val/test 是否互斥（文件级别）。"
    )
    parser.add_argument(
        "--split-outdir",
        type=str,
        default=".split",
        help="保存各个 split 子目录的根目录（默认 .split）",
    )

    args = parser.parse_args()
    split_root = Path(args.split_outdir)

    if not split_root.exists():
        raise FileNotFoundError(f"split-outdir 不存在: {split_root}")

    leak_dirs = []
    checked = 0
    skipped = 0

    print(f"[INFO] 检查 splits 根目录: {split_root}")

    for d in sorted(split_root.iterdir(), key=lambda p: p.name):
        if not d.is_dir():
            continue

        ok, msg = check_one_split_dir(d)
        if msg == "no train.txt, skipped":
            skipped += 1
            continue

        checked += 1
        if ok:
            print(f"[OK]    {d.name}: {msg}")
        else:
            print(f"[LEAK]  {d.name}: {msg}")
            leak_dirs.append(d.name)

    print("\n========== SUMMARY ==========")
    print(f"已检查 split 子目录数量: {checked}")
    print(f"跳过（无 train.txt）数量: {skipped}")
    if not leak_dirs:
        print("✅ 未发现 train/val/test 之间的文件级数据泄露。")
    else:
        print("⚠ 发现以下目录存在数据泄露（train/val/test 有重复图片）：")
        for name in leak_dirs:
            print(f"  - {name}")


if __name__ == "__main__":
    main()
