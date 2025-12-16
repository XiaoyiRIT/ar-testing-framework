#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
YOLO Leave-One-Group-Out Training Script
----------------------------------------

功能摘要：
- 只保留 train + test，不再有单独 val set。
- app-ood / scene-ood：
    * k = App / Scene 去重个数；
    * Leave-One-Group-Out：每次选 1 个 App/Scene 做 test，其余做 train。
- object-ood（当前版本）：
    * 先将所有 Object 随机均分成 N 个 meta-group（默认 N=10，可通过 --object-folds 指定）；
    * k = N；
    * 第 i 折：把第 i 个 meta-group 中的所有 Object 作为 test，其余 Object 作为 train。
- 两种运行模式：
  1) 不给 --kfold-index：只打印 group / object 分布（以及 object-ood 下的 folds 信息），直接退出；
  2) 给 --kfold-index：构建对应的 train/test/data.yaml，并（可选）启动 YOLO 训练。
- 支持 --dryrun：生成所有 split 文件和 data.yaml，但不真正启动训练。
- group 列统一使用 str.strip() 清洗，避免因空格等字符导致计数不一致。
- 支持 app-ood / scene-ood / object-ood。
- 自动跳过机制：
    * 对于某个 fold（idx, group）：
        1) 在 project 目录下查找所有以 "{ood_type}_idx{idx}_{group}" 开头的子目录；
        2) 若其中任一目录存在 weights/best.pt 或 weights/last.pt，则认为该 fold 已经训完 → 直接跳过；
        3) 若没有任何目录有模型：
              * 删除所有这些旧目录（视为未完成 / 废弃）；
              * 新建一个目录名："{ood_type}_idx{idx}_{group}_s{seed}_{model_stem}"，用于本次训练。
"""

import argparse
import sys
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
    """自动推断存放文件名的列"""
    lower_map = {c.lower(): c for c in df.columns}
    for key in ["filename", "file", "image", "img"]:
        if key in lower_map:
            return lower_map[key]
    print(f"[WARN] 未找到 filename/file/image/img 列，默认使用第一列: {df.columns[0]}")
    return df.columns[0]


def _guess_group_col(df: pd.DataFrame, ood_type: str, group_col_arg: Optional[str]) -> str:
    """
    根据 ood_type 自动推断 group 列（App / Scene / Object），
    或使用显式指定的列名。
    """
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

    if ood_type == "object-ood":
        for key in ["object", "object_id", "obj", "category", "class"]:
            if key in lower_map:
                return lower_map[key]
        raise ValueError(
            "无法自动找到 Object 列，请使用 --group-col 显式指定。\n"
            "推荐列名示例：Object / object / object_id / obj / category / class"
        )

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


def _find_existing_runs(project: Path, base_prefix: str) -> List[Path]:
    """
    在 project 下查找所有以 base_prefix 开头的子目录。
    例如 base_prefix = 'scene-ood_idx1_Scene_2'：
        - scene-ood_idx1_Scene_2
        - scene-ood_idx1_Scene_2_s0_yolo12n
        - scene-ood_idx1_Scene_2_s0_yolo12n_s0_yolo12n
      都会被视为同一个 fold 的候选目录。
    """
    result: List[Path] = []
    if not project.exists():
        return result
    for d in project.iterdir():
        if d.is_dir() and d.name.startswith(base_prefix):
            result.append(d)
    return sorted(result)


def _build_object_folds(
    objects: List[str],
    group_counts: Dict[str, int],
    num_folds: int,
    seed: int,
) -> Tuple[Dict[str, int], List[List[str]], List[int]]:
    """
    将所有 object 随机均分成 num_folds 个 fold。
    返回：
      - obj_to_fold: object -> fold_id
      - folds:      每个 fold 的 object 列表
      - fold_rows:  每个 fold 的样本总数（按 group_counts 汇总）
    """
    if num_folds <= 0:
        num_folds = 1
    num_folds = min(num_folds, len(objects))  # 不要超过 object 总数

    rng = np.random.RandomState(seed)
    objs = list(objects)
    rng.shuffle(objs)

    folds: List[List[str]] = [[] for _ in range(num_folds)]
    for i, obj in enumerate(objs):
        folds[i % num_folds].append(obj)

    obj_to_fold: Dict[str, int] = {}
    fold_rows: List[int] = []
    for fid, obj_list in enumerate(folds):
        total_rows = sum(group_counts[o] for o in obj_list)
        fold_rows.append(total_rows)
        for o in obj_list:
            obj_to_fold[o] = fid

    return obj_to_fold, folds, fold_rows


# -------------------- Main Logic --------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO Leave-One-Group-Out training")

    # Data
    parser.add_argument(
        "--data-csv",
        type=str,
        required=True,
        help="例如 data_stat.csv，包含图片文件名和 App/Scene/Object 等信息。",
    )
    parser.add_argument(
        "--ood-type",
        type=str,
        required=True,
        choices=["app-ood", "scene-ood", "object-ood"],
        help="当前要构建的 OOD 类型：app-ood / scene-ood / object-ood。",
    )
    parser.add_argument(
        "--group-col",
        type=str,
        default=None,
        help="可选，显式指定 group 列名；不指定则按 ood-type 自动推断。",
    )
    parser.add_argument(
        "--image-col",
        type=str,
        default=None,
        help="可选，显式指定文件名列名；不指定则自动从 filename/file/image/img 推断。",
    )

    # kfold index
    parser.add_argument(
        "--kfold-index",
        type=int,
        default=None,
        help=(
            "Leave-One-Group-Out 中第几个 group 作为 test (0-based)。"
            "若不提供，仅打印 group / fold 信息，不训练。"
        ),
    )

    # object-ood 额外参数：折数
    parser.add_argument(
        "--object-folds",
        type=int,
        default=10,
        help="仅在 --ood-type object-ood 时生效：将所有 Object 随机均分成 N 个 meta-group（默认 10）。",
    )

    # Training params（只影响训练，不影响数据划分）
    parser.add_argument(
        "--images-root",
        type=str,
        default=None,
        help="图片根目录，用于与 CSV 文件名列组合解析最终路径。",
    )
    parser.add_argument(
        "--labels-root",
        type=str,
        default=None,
        help="YOLO label txt 根目录，会写入 labels_root.txt 方便 eval。",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="YOLO 模型名称或权重路径，如 yolo12n.pt 或 runs/.../best.pt。",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--batch",
        type=int,
        default=None,
        help="可选：batch size；若不提供则使用 YOLO 默认。",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="可选：dataloader workers 数；若不提供则使用 YOLO 默认。",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="仅控制训练随机性，不影响数据划分。",
    )

    # Output
    parser.add_argument(
        "--project",
        type=str,
        default="runs/leave1out",
        help="YOLO 训练的 project 目录。",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="实验名称，若不指定则自动根据 ood-type 和 group 生成。",
    )
    parser.add_argument(
        "--splits-dir",
        type=str,
        default="splits_leave1out",
        help="保存 train/test/data.yaml 的目录根。",
    )

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

    # 统一清洗 group 列：转 str + 去掉首尾空格
    df[group_col] = df[group_col].astype(str).str.strip()
    group_series = df[group_col]

    group_counts: Dict[str, int] = {}
    for g in group_series:
        group_counts[g] = group_counts.get(g, 0) + 1

    unique_groups = sorted(group_counts.keys())

    print(f"[INFO] Raw groups (unique values in {group_col}) = {len(unique_groups)}")
    for i, g in enumerate(unique_groups):
        print(f"  raw_index={i:3d}  group={g}  rows={group_counts[g]}")

    obj_to_fold: Optional[Dict[str, int]] = None
    folds_objects: Optional[List[List[str]]] = None
    fold_rows: Optional[List[int]] = None

    # -------- 针对 object-ood：把 Object 随机均分成 N 个 fold --------
    if args.ood_type == "object-ood":
        num_folds = args.object_folds
        if num_folds <= 0:
            num_folds = 1
        if num_folds > len(unique_groups):
            num_folds = len(unique_groups)
        print(
            f"\n[INFO] object-ood: 将 {len(unique_groups)} 个 Object 随机均分成 "
            f"{num_folds} 个 fold（seed={args.seed}）"
        )
        obj_to_fold, folds_objects, fold_rows = _build_object_folds(
            unique_groups, group_counts, num_folds, args.seed
        )
        k = num_folds
        for fid in range(num_folds):
            objs = folds_objects[fid]
            print(
                f"  fold={fid:2d}  n_objects={len(objs):3d}  rows={fold_rows[fid]:5d}"
            )
    else:
        # app-ood / scene-ood：一个 group 一个 fold
        k = len(unique_groups)

    # -------- 未提供 --kfold-index：只报告信息，不训练 --------
    if args.kfold_index is None:
        print(
            "\n[INFO] 未提供 --kfold-index，本次不会启动训练。\n"
            f"      当前 ood-type = {args.ood_type}, group 列 = {group_col}\n"
            f"      有效 fold 数量 k = {k}\n"
            "      如果要对第 i 个 fold 做训练，请加参数： --kfold-index i  （0 <= i < k）"
        )
        sys.exit(0)

    # -------- 检查 kfold-index 合法性 --------
    idx = args.kfold_index
    if not (0 <= idx < k):
        print(f"[ERROR] --kfold-index {idx} out of range [0, {k-1}]")
        sys.exit(1)

    # -------- 根据类型确定 “test fold” 的描述 --------
    if args.ood_type == "object-ood":
        test_fold = idx
        print(f"\n[INFO] Selected object fold = {test_fold} (0-based)")
        safe_group = f"objFold{test_fold}"
    else:
        test_group = unique_groups[idx]
        print(f"\n[INFO] Selected test group = {test_group} (index={idx})")
        safe_group = str(test_group).replace("/", "_").replace(" ", "_")

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

    # -------- 根据 fold 构建 train/test 路径列表 --------
    train_list: List[str] = []
    test_list: List[str] = []
    missing_count = 0

    for _, row in df.iterrows():
        g = str(row[group_col]).strip()
        name = str(row[filename_col])
        resolved = _resolve_image_path(images_root, name)
        if resolved is None:
            missing_count += 1
            continue

        if args.ood_type == "object-ood":
            assert obj_to_fold is not None
            fold_id = obj_to_fold[g]
            if fold_id == idx:
                test_list.append(resolved)
            else:
                train_list.append(resolved)
        else:
            # app-ood / scene-ood：直接按 group 分
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

    # 如果是 object-ood，可以把 object→fold 分配也保存一份，方便复现
    if args.ood_type == "object-ood" and obj_to_fold is not None:
        mapping_txt = subdir / "object_folds.txt"
        with open(mapping_txt, "w", encoding="utf-8") as f:
            f.write(f"# seed={args.seed}, object_folds={args.object_folds}\n")
            for oid in sorted(obj_to_fold.keys()):
                f.write(f"{oid}\t{obj_to_fold[oid]}\n")
        print(f"[INFO] object fold mapping written: {mapping_txt}")

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

    # -------- 真正开始训练 YOLO 之前：自动跳过 + 删除未完成 run --------
    project = Path(args.project)
    project.mkdir(parents=True, exist_ok=True)

    base_prefix = f"{args.ood_type}_idx{idx}_{safe_group}"
    existing_runs = _find_existing_runs(project, base_prefix)

    # 1) 先看这些目录里有没有已经训练好的模型
    trained_dir: Optional[Path] = None
    for d in existing_runs:
        best_pt = d / "weights" / "best.pt"
        last_pt = d / "weights" / "last.pt"
        if best_pt.exists() or last_pt.exists():
            trained_dir = d
            break

    if trained_dir is not None:
        print(f"[SKIP] fold={idx} 已存在训练好的模型: {trained_dir}")
        sys.exit(0)

    # 2) 若没有任何模型：删除所有旧目录，然后新建一个规范目录名
    if existing_runs:
        print(f"[CLEAN] fold={idx} 发现 {len(existing_runs)} 个无模型目录，将删除：")
        for d in existing_runs:
            print(f"        - {d}")
            shutil.rmtree(d, ignore_errors=True)

    model_stem = Path(args.model).stem
    if args.name is not None:
        exp_name = args.name
    else:
        exp_name = f"{base_prefix}_s{args.seed}_{model_stem}"

    args.name = exp_name
    exp_dir = project / exp_name
    print(f"[RUN] fold={idx} 使用目录: {exp_dir}")

    # -------- 真正开始训练 YOLO --------
    set_global_seed(args.seed)
    device = pick_device(args.device)
    print(f"[INFO] Using device: {device}")

    print(f"[INFO] Loading YOLO model: {args.model}")
    model = YOLO(args.model)

    print(f"[INFO] Training started → project={args.project}, name={args.name}")

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
