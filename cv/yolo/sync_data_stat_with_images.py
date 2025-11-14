#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
同步 data_stat.csv 与实际 images 目录。

功能：
- 从 data_stat.csv 中删除那些在 images_root 下找不到对应图片文件的行。
- 仅根据“文件名”匹配（不看路径），即：
  - data_stat.csv 里的 filename 列如果是 "foo.jpg" 或 "subdir/foo.jpg"，
  - 都会被当成 "foo.jpg" 来和 images_root 下的实际文件名匹配。

用法示例：

python sync_data_stat_with_images.py \
  --csv data_stat.csv \
  --images-root /Users/yangxiaoyi/datasets/myar/images \
  --out data_stat_updated.csv

如果你想直接覆盖原文件（记得自己先备份）：
python sync_data_stat_with_images.py \
  --csv data_stat.csv \
  --images-root /Users/yangxiaoyi/datasets/myar/images \
  --inplace
"""

import argparse
from pathlib import Path
from typing import Set

import pandas as pd

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def collect_existing_filenames(images_root: Path) -> Set[str]:
    """收集 images_root 下所有图片的“文件名”（不含路径），用于匹配 CSV 的 filename 列。"""
    existing = set()
    for p in images_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            existing.add(p.name)  # 只保留文件名本身
    return existing


def find_filename_column(df: pd.DataFrame) -> str:
    """在 CSV 中找到 filename 列（大小写不敏感）。"""
    lower_map = {c.lower(): c for c in df.columns}
    for key in ["filename", "file", "image", "img"]:
        if key in lower_map:
            return lower_map[key]
    raise ValueError(
        f"data_stat.csv 中找不到 filename 列，实际列为：{list(df.columns)}。\n"
        f"请确保有一列名为 filename / Filename 等类似名称。"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="根据实际 images 目录，清理 data_stat.csv 中已经不存在的图片行。"
    )
    parser.add_argument("--csv", type=str, required=True, help="data_stat.csv 路径")
    parser.add_argument(
        "--images-root",
        type=str,
        required=True,
        help="图片根目录（会递归扫描其中所有图片文件）",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="输出 CSV 路径（默认不填则需要使用 --inplace 覆盖原文件）",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="直接覆盖原 CSV（请自行做好备份）。与 --out 互斥。",
    )

    args = parser.parse_args()

    csv_path = Path(args.csv)
    images_root = Path(args.images_root)

    if not csv_path.exists():
        raise FileNotFoundError(f"找不到 CSV 文件：{csv_path}")
    if not images_root.exists():
        raise FileNotFoundError(f"找不到图片根目录：{images_root}")

    if args.inplace and args.out is not None:
        raise ValueError("--inplace 与 --out 不能同时使用")

    # 1) 读取 CSV
    df = pd.read_csv(csv_path)
    fn_col = find_filename_column(df)

    print(f"[INFO] 读取 CSV：{csv_path}")
    print(f"[INFO] 使用列作为 filename：{fn_col!r}")
    n_before = len(df)

    # 2) 收集现有图片文件名
    existing_fns = collect_existing_filenames(images_root)
    print(f"[INFO] images_root 下找到图片数量：{len(existing_fns)}")

    # 3) 过滤：仅保留在 existing_fns 中的行
    def row_has_image(x: str) -> bool:
        if not isinstance(x, str):
            return False
        # 如果 CSV 里 filename 含有子路径，只取最后一段
        name = Path(x).name
        return name in existing_fns

    mask = df[fn_col].apply(row_has_image)
    df_kept = df[mask].copy()
    n_after = len(df_kept)
    n_removed = n_before - n_after

    print(f"[INFO] 原始行数：{n_before}")
    print(f"[INFO] 保留行数：{n_after}")
    print(f"[INFO] 删除行数：{n_removed}")

    # 4) 输出
    if args.inplace:
        out_path = csv_path
    else:
        if args.out is None:
            # 默认输出到同目录下一个带后缀的文件
            out_path = csv_path.with_name(csv_path.stem + "_updated.csv")
        else:
            out_path = Path(args.out)

    df_kept.to_csv(out_path, index=False)
    print(f"[INFO] 已写出更新后的 CSV：{out_path}")


if __name__ == "__main__":
    main()
