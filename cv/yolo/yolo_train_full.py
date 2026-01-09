#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
YOLO Full-Data Training Script (Final)
--------------------------------------
目标：
- 不再做 OOD / kfold / leave-one-out。
- 直接使用 CSV 中的全部样本作为训练集。
- 默认跳过验证（val=False），完全不做 evaluation/validation。
- 仍然保留：
  - 从 CSV 推断 filename 列
  - images_root + filename 解析到绝对路径
  - 写 train.txt 和 data.yaml
  - dryrun（只生成文件不训练）
  - batch/workers/device/seed 等训练参数

用法示例：
python yolo_train_full.py \
  --data-csv data_stat.csv \
  --images-root /path/to/images \
  --labels-root /path/to/labels \
  --model yolov12n.pt \
  --project runs/train_full \
  --name full_yolo12n_s0 \
  --epochs 100 \
  --imgsz 640 \
  --batch 16 \
  --workers 8

只生成 train.txt/data.yaml 不训练：
python yolo_train_full.py ... --dryrun
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from ultralytics import YOLO


# -------------------- Utils --------------------


def pick_device(prefer: str = "auto") -> str:
    import platform
    import torch

    if prefer != "auto":
        return prefer
    if platform.system() == "Darwin" and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def set_global_seed(seed: int) -> None:
    """仅影响训练随机性，不影响数据划分（这里也没有划分）。"""
    import random
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_csv(data_csv: Path) -> pd.DataFrame:
    if not data_csv.exists():
        raise FileNotFoundError(f"CSV not found: {data_csv}")
    df = pd.read_csv(data_csv)
    if df.empty:
        raise RuntimeError(f"CSV is empty: {data_csv}")
    return df


def _find_filename_column(df: pd.DataFrame) -> str:
    """自动推断存放文件名/路径的列名"""
    lower_map = {c.lower(): c for c in df.columns}
    for key in ["filename", "file", "image", "img", "image_path", "img_path", "path"]:
        if key in lower_map:
            return lower_map[key]
    print(f"[WARN] 未找到常见 filename/file/image/img/path 列，默认使用第一列: {df.columns[0]}")
    return df.columns[0]


def _resolve_image_path(images_root: Path, name: str) -> Optional[str]:
    """
    根据 CSV 中的文件名解析真实图片路径（绝对路径）。
    支持：
    - CSV 中本身就是绝对路径
    - 相对于 images_root 的相对路径
    - 只给 basename（在 images_root 下查找）
    """
    p = Path(str(name))

    # 绝对路径且存在
    if p.is_absolute() and p.exists():
        return str(p.resolve())

    # 相对于 images_root
    cand = images_root / p
    if cand.exists():
        return str(cand.resolve())

    # 只用 basename 再试一次
    cand2 = images_root / p.name
    if cand2.exists():
        return str(cand2.resolve())

    return None


# -------------------- Main --------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO full-data training (no evaluation)")

    # Data
    parser.add_argument("--data-csv", type=str, required=True, help="包含图片文件名/路径的 CSV。")
    parser.add_argument(
        "--image-col",
        type=str,
        default=None,
        help="可选：显式指定 CSV 中存放文件名/路径的列名；不指定则自动推断。",
    )
    parser.add_argument("--images-root", type=str, required=True, help="图片根目录。")
    parser.add_argument("--labels-root", type=str, required=True, help="YOLO label txt 根目录。")
    parser.add_argument("--model", type=str, required=True, help="YOLO 模型或权重，如 yolov12n.pt。")

    # Train
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--seed", type=int, default=0)

    # Output
    parser.add_argument("--project", type=str, default="runs/train_full")
    parser.add_argument("--name", type=str, default=None, help="实验名；不指定则自动生成。")
    parser.add_argument("--splits-dir", type=str, default="splits_full", help="保存 train.txt/data.yaml 的目录。")

    # No evaluation
    parser.add_argument(
        "--val",
        action="store_true",
        help="如果开启，则让 YOLO 在训练过程中跑验证（val=True）。默认不跑（val=False）。",
    )

    # Dryrun
    parser.add_argument("--dryrun", action="store_true", help="只生成 train.txt/data.yaml，不启动训练。")

    args = parser.parse_args()

    data_csv = Path(args.data_csv)
    images_root = Path(args.images_root)
    labels_root = Path(args.labels_root)

    if not images_root.exists():
        print(f"[ERROR] images-root not found: {images_root}")
        sys.exit(1)
    if not labels_root.exists():
        print(f"[ERROR] labels-root not found: {labels_root}")
        sys.exit(1)

    df = _load_csv(data_csv)
    print(f"[INFO] Loaded CSV: {data_csv} (rows={len(df)})")

    if args.image_col is not None:
        if args.image_col not in df.columns:
            print(f"[ERROR] image-col {args.image_col!r} not found in CSV. Columns={list(df.columns)}")
            sys.exit(1)
        filename_col = args.image_col
    else:
        filename_col = _find_filename_column(df)

    print(f"[INFO] Using filename column: {filename_col}")

    # Resolve all image paths
    train_list: List[str] = []
    missing = 0
    for _, row in df.iterrows():
        name = row[filename_col]
        resolved = _resolve_image_path(images_root, str(name))
        if resolved is None:
            missing += 1
            continue
        train_list.append(resolved)

    # 去重（可选但推荐，避免 CSV 有重复行）
    train_list = sorted(set(train_list))

    print(f"[INFO] Resolved images: train={len(train_list)}, missing={missing}")
    if not train_list:
        print("[ERROR] No images resolved. Please check --images-root and filename column.")
        sys.exit(1)

    # Write split files
    splits_root = Path(args.splits_dir)
    splits_root.mkdir(parents=True, exist_ok=True)

    exp_name = args.name
    if exp_name is None:
        model_stem = Path(args.model).stem
        exp_name = f"full_s{args.seed}_{model_stem}"
    out_dir = splits_root / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)

    train_txt = out_dir / "train.txt"
    with open(train_txt, "w", encoding="utf-8") as f:
        for p in train_list:
            f.write(p + "\n")
    print(f"[INFO] train.txt written: {train_txt}")

    # data.yaml：train/val 都指向 train（但我们默认 val=False，不跑验证）
    data_yaml = out_dir / "data.yaml"
    yaml_lines = [
        f"train: {train_txt.resolve()}",
        f"val:   {train_txt.resolve()}  # (optional) same as train; training uses --val flag",
        "",
        "nc: 2",
        "names:",
        "  0: AR_Object",
        "  1: UI_Element",
    ]
    data_yaml.write_text("\n".join(yaml_lines), encoding="utf-8")
    print(f"[INFO] data.yaml written: {data_yaml}")

    # 保存 labels_root 方便你以后如果还要调试/可视化
    (out_dir / "labels_root.txt").write_text(str(labels_root.resolve()), encoding="utf-8")
    print(f"[INFO] labels_root.txt written: {out_dir / 'labels_root.txt'}")

    if args.dryrun:
        print("\n[DRYRUN] 已生成 train.txt/data.yaml，但不会启动训练。")
        print(f"[DRYRUN] out_dir: {out_dir}")
        sys.exit(0)

    # Train
    set_global_seed(args.seed)
    device = pick_device(args.device)
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Loading YOLO model: {args.model}")

    model = YOLO(args.model)

    train_kwargs = dict(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        project=args.project,
        name=exp_name,
        device=device,
        val=bool(args.val),  # 默认 False：不跑验证/评估
    )
    if args.batch is not None:
        train_kwargs["batch"] = args.batch
    if args.workers is not None:
        train_kwargs["workers"] = args.workers

    print(f"[INFO] Training started → project={args.project}, name={exp_name}, val={bool(args.val)}")
    model.train(**train_kwargs)
    print("[INFO] Training finished.")


if __name__ == "__main__":
    main()
