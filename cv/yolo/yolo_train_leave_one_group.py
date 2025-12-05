#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
YOLO Leave-One-Group-Out Training Script
----------------------------------------

功能摘要：
- 只保留 train + test，不再有单独 val set。
- k = group（App 或 Scene）的去重个数。
- Leave-One-Group-Out：每次选一个 group 做 test，其余全部做 train。
- 两种运行模式：
  1) 不给 --kfold-index：只打印 k 和所有 group 的 index + 样本行数，直接退出；
  2) 给 --kfold-index：构建对应的 train/test/data.yaml，并（可选）启动 YOLO 训练。
- 支持 --dryrun：生成所有 split 文件和 data.yaml，但不真正启动训练。
- group 列统一使用 str.strip() 清洗，避免因空格等字符导致计数不一致。

python yolo_train_leave_one_group.py \
    --data-csv data_stat.csv \
    --ood-type scene-ood \
    --kfold-index 2 \
    --images-root /Users/yangxiaoyi/datasets/myar/images \
    --labels-root /Users/yangxiaoyi/datasets/myar/labels \
    --model yolov12n.pt \
    --dryrun

"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from ultralytics import YOLO


# -------------------- Utility functions --------------------


def pick_device(prefer: str = "auto") -> str:
    import torch
    import platform

    if prefer != "auto":
        return prefer
    if platform.system() == "Darwin" and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def set_global_seed(seed: int) -> None:
    """仅影响训练随机性，不影响数据划分。"""
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
    """自动推断存放文件名的列（参考 dynsplit 的风格）"""
    lower_map = {c.lower(): c for c in df.columns}
    for key in ["filename", "file", "image", "img"]:
        if key in lower_map:
            return lower_map[key]
    print(f"[WARN] 未找到 filename/file/image/img 列，默认使用第一列: {df.columns[0]}")
    return df.columns[0]


def _guess_group_col(df: pd.DataFrame, ood_type: str, group_col_arg: Optional[str]) -> str:
    """根据 ood_type 自动推断 group 列（App / Scene），或使用显式指定的列名。"""
    if group_col_arg is not None:
        if group_col_arg not in df.columns:
            raise ValueError(
                f"指定的 group 列 {group_col_arg!r} 不在 CSV 中。\n可用列: {list(df.columns)}"
            )
        return group_col_arg

    lower_map = {c.lower(): c for c in df.columns}

    if ood_type == "app-ood":
        for key in ["app", "app_id", "appname", "package", "package_name"]:
            if key in lower_map:
                return lower_map[key]
        raise ValueError("无法自动找到 App 列，请使用 --group-col 显式指定。")

    if ood_type == "scene-ood":
        for key in ["scene", "scene_id"]:
            if key in lower_map:
                return lower_map[key]
        raise ValueError("无法自动找到 Scene 列，请使用 --group-col 显式指定。")

    raise ValueError(f"不支持的 ood-type={ood_type}")


def _resolve_image_path(images_root: Path, name: str) -> Optional[str]:
    """根据 CSV 中的文件名解析真实图片路径（绝对路径）"""
    p = Path(name)

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


# -------------------- Main Logic --------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO Leave-One-Group-Out training")

    # Data
    parser.add_argument("--data-csv", type=str, required=True,
                        help="例如 data_stat.csv，包含图片文件名和 App/Scene 等信息。")
    parser.add_argument("--ood-type", type=str, required=True,
                        choices=["app-ood", "scene-ood"],
                        help="当前要构建的 OOD 类型，只支持 app-ood / scene-ood。")
    parser.add_argument("--group-col", type=str, default=None,
                        help="可选，显式指定 group 列名；不指定则按 ood-type 自动推断。")
    parser.add_argument("--image-col", type=str, default=None,
                        help="可选，显式指定文件名列名；不指定则自动从 filename/file/image/img 推断。")

    # kfold index
    parser.add_argument("--kfold-index", type=int, default=None,
                        help="Leave-One-Group-Out 中第几个 group 作为 test (0-based)。"
                             "若不提供，仅打印 k 和 group 列表，不训练。")

    # Training params（只影响训练，不影响数据划分）
    parser.add_argument("--images-root", type=str, default=None,
                        help="图片根目录，用于与 CSV 文件名列组合解析最终路径。")
    parser.add_argument("--labels-root", type=str, default=None,
                        help="YOLO label txt 根目录，会写入 labels_root.txt 方便 eval。")
    parser.add_argument("--model", type=str, default=None,
                        help="YOLO 模型名称或权重路径，如 yolov12n.pt 或 runs/.../best.pt。")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    # batch / workers 改为“可选”：默认 None，不传给 YOLO，使用其默认设置
    parser.add_argument("--batch", type=int, default=None,
                        help="可选：batch size；若不提供则使用 YOLO 默认。")
    parser.add_argument("--workers", type=int, default=None,
                        help="可选：dataloader workers 数；若不提供则使用 YOLO 默认。")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--seed", type=int, default=0,
                        help="仅控制训练随机性，不影响数据划分。")

    # Output
    parser.add_argument("--project", type=str, default="runs/leave1out",
                        help="YOLO 训练的 project 目录。")
    parser.add_argument("--name", type=str, default=None,
                        help="实验名称，若不指定则自动根据 ood-type 和 group 生成。")
    parser.add_argument("--splits-dir", type=str, default="splits_leave1out",
                        help="保存 train/test/data.yaml 的目录根。")

    # Dryrun
    parser.add_argument(
        "--dryrun",
        action="store_true",
        help="若启用，则只构建 train/test/data.yaml，打印信息但不启动训练。",
    )

    args = parser.parse_args()

    # -------- 读取 CSV --------
    df = _load_csv(Path(args.data_csv))
    print(f"[INFO] Loaded CSV: {args.data_csv}")

    # -------- 推断 group 列并做清洗（strip）--------
    group_col = _guess_group_col(df, args.ood_type, args.group_col)
    print(f"[INFO] Using group column: {group_col}")

    # 统一清洗 group 列：转 str + 去掉首尾空格，避免 "21" vs "21 " 这种分裂
    df[group_col] = df[group_col].astype(str).str.strip()
    group_series = df[group_col]

    group_counts: Dict[str, int] = {}
    for g in group_series:
        group_counts[g] = group_counts.get(g, 0) + 1

    unique_groups = sorted(group_counts.keys())
    k = len(unique_groups)

    print(f"[INFO] Total groups (k) = {k}")
    for i, g in enumerate(unique_groups):
        print(f"  index={i:3d}  group={g}  rows={group_counts[g]}")

    # -------- 未提供 --kfold-index：只报告 k，不训练 --------
    if args.kfold_index is None:
        print(
            "\n[INFO] 未提供 --kfold-index，本次不会启动训练。\n"
            "      如果要对第 i 个 group 做 Leave-One-Out 训练，请加参数："
            " --kfold-index i  （0 <= i < k）"
        )
        sys.exit(0)

    # -------- 检查 kfold-index 合法性 --------
    idx = args.kfold_index
    if not (0 <= idx < k):
        print(f"[ERROR] --kfold-index {idx} out of range [0, {k-1}]")
        sys.exit(1)

    test_group = unique_groups[idx]
    print(f"\n[INFO] Selected test group = {test_group} (index={idx})")

    # -------- 启动训练 / dryrun 前，检查必要参数 --------
    if args.images_root is None or args.labels_root is None or args.model is None:
        print(
            "\n[ERROR] 启动训练（或 dryrun）时必须提供以下参数：\n"
            "  --images-root   --labels-root   --model\n"
        )
        sys.exit(1)

    images_root = Path(args.images_root)
    labels_root = Path(args.labels_root)

    if not images_root.exists():
        print(f"[ERROR] images-root not found: {images_root}")
        sys.exit(1)
    if not labels_root.exists():
        print(f"[ERROR] labels-root not found: {labels_root}")
        sys.exit(1)

    # -------- 确定文件名列 --------
    if args.image_col is not None:
        if args.image_col not in df.columns:
            print(
                f"[ERROR] image-col {args.image_col!r} not found in CSV.\n"
                f"Columns = {list(df.columns)}"
            )
            sys.exit(1)
        filename_col = args.image_col
    else:
        filename_col = _find_filename_column(df)

    print(f"[INFO] Using filename column: {filename_col}")

    # -------- 根据 group 构建 train/test 路径列表 --------
    train_list: List[str] = []
    test_list: List[str] = []
    missing_count = 0

    for _, row in df.iterrows():
        # 再保险一次 strip（虽然上面已经清洗过 df[group_col]）
        g = str(row[group_col]).strip()
        name = str(row[filename_col])
        resolved = _resolve_image_path(images_root, name)
        if resolved is None:
            missing_count += 1
            continue
        if g == test_group:
            test_list.append(resolved)
        else:
            train_list.append(resolved)

    print(
        f"[INFO] Paths resolved: train={len(train_list)}, test={len(test_list)}, "
        f"missing={missing_count}"
    )

    if not train_list or not test_list:
        print("[WARN] train 或 test 为空，训练结果可能不可靠，请检查数据划分。")

    # -------- 写出 train/test/data.yaml --------
    splits_root = Path(args.splits_dir)
    splits_root.mkdir(parents=True, exist_ok=True)

    safe_group = str(test_group).replace("/", "_").replace(" ", "_")
    subdir = splits_root / f"{args.ood_type}_{safe_group}"
    subdir.mkdir(parents=True, exist_ok=True)

    train_txt = subdir / "train.txt"
    test_txt = subdir / "test.txt"

    with open(train_txt, "w", encoding="utf-8") as f:
        for p in train_list:
            f.write(p + "\n")
    with open(test_txt, "w", encoding="utf-8") as f:
        for p in test_list:
            f.write(p + "\n")

    print(f"[INFO] train.txt written: {train_txt}")
    print(f"[INFO] test.txt  written: {test_txt}")

    # data.yaml：val 复用 train（没有单独 val set）
    data_yaml = subdir / "data.yaml"
    yaml_lines = [
        f"train: {train_txt.resolve()}",
        f"val:   {train_txt.resolve()}  # no val set, use train",
        f"test:  {test_txt.resolve()}",
        "",
        "nc: 2",
        "names:",
        "  0: AR_Object",
        "  1: UI_Element",
    ]
    data_yaml.write_text("\n".join(yaml_lines), encoding="utf-8")
    print(f"[INFO] data.yaml written: {data_yaml}")

    # 写 labels_root.txt，方便 eval 脚本自动读取
    (subdir / "labels_root.txt").write_text(str(labels_root.resolve()), encoding="utf-8")

    # -------- 如果是 dryrun：这里直接退出，不训练 --------
    if args.dryrun:
        print("\n[DRYRUN] 已生成 train/test/data.yaml，但不会启动 YOLO 训练。")
        print(f"[DRYRUN] train.txt: {train_txt}")
        print(f"[DRYRUN] test.txt : {test_txt}")
        print(f"[DRYRUN] data.yaml: {data_yaml}")
        print(f"[DRYRUN] labels_root.txt: {subdir / 'labels_root.txt'}")
        print("[DRYRUN] 退出程序。")
        sys.exit(0)

    # -------- 真正开始训练 YOLO --------
    set_global_seed(args.seed)
    device = pick_device(args.device)
    print(f"[INFO] Using device: {device}")

    print(f"[INFO] Loading YOLO model: {args.model}")
    model = YOLO(args.model)

    if args.name is None:
        args.name = f"{args.ood_type}_idx{idx}_{safe_group}"

    print(f"[INFO] Training started → project={args.project}, name={args.name}")

    # 构建 train() 的参数字典，batch / workers 仅在显式提供时才传给 YOLO
    train_kwargs = dict(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        project=args.project,
        name=args.name,
        device=device,
    )
    if args.batch is not None:
        train_kwargs["batch"] = args.batch
    if args.workers is not None:
        train_kwargs["workers"] = args.workers

    model.train(**train_kwargs)

    print("[INFO] Training finished.")


if __name__ == "__main__":
    main()
