#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dynamic split + YOLO training/testing script.

功能概述
--------
1. 从 data_stat.csv 动态划分 train/val/test：
   - random              : 随机划分样本（不看 group）
   - app-ood             : 按 App 列分组做 OOD
   - object-ood          : 按 Object 列分组做 OOD
   - scene-ood           : 按 Scene 列分组做 OOD
   - group-ood           : 按任意指定列分组做 OOD (--group-col)

2. 支持 group 级 K 折 (GroupKFold)，用于 OOD 稳定评估：
   - kfold=0 或 1 : 不使用 K 折
   - kfold>1      : 按 group 做 K 折，在当前折中再根据样本数切出 train/test

3. 生成的 data.yaml 固定为 2 类：
   - 0: AR_Object
   - 1: UI_Element

4. 训练输出目录自动命名为：
   <split-mode>_<weights-stem>_s<seed>_k<kfold>_f<fold>_<timestamp>
   例如：
   app-ood_yolo11n_s42_k0_f0_1763082385
   避免多任务/slurm 并发写同一 exp 目录。

5. 提供一个简单的 test 子命令做无标签批量推理。

用法示例：

# 1) 随机分割（不复制文件；从 CSV 读取文件名并写入三份 list.txt）
python test_yolo_train_dynsplit.py train \
  --images-root /Users/yangxiaoyi/datasets/myar/images \
  --labels-root /Users/yangxiaoyi/datasets/myar/labels \
  --data-csv data_stat.csv \
  --split-mode random --ratios 0.7,0.15,0.15 \
  --weights yolo11n.pt --epochs 100 --imgsz 640

# 2) App-OOD（按 app 组互斥）+ 5 折交叉验证第0折（train/val 来自 K 折；test 额外保留一批 app 组）
python test_yolo_train_dynsplit.py train \
  --images-root /Users/yangxiaoyi/datasets/myar/images \
  --labels-root /Users/yangxiaoyi/datasets/myar/labels \
  --data-csv data_stat.csv \
  --split-mode app-ood --group-col App \
  --ratios 0.7,0.15,0.15 --seed 10\
  --weights yolo11n.pt --epochs 100 --imgsz 640
  
  --kfold 5 --fold-index 0 

# 3) 仅评测（中心点命中 GT 算成功），从 list 文件读入
python test_yolo_train_dynsplit.py eval \
  --image-list split/random_s42_k0_f0_1763082385/test.txt \
  --labels-root /Users/yangxiaoyi/datasets/myar/labels \
  --weights runs/train/exp4/weights/best.pt \
  --class-id 0 --imgsz 640 --conf 0.05 --save-vis
"""

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from ultralytics import YOLO


# ----------------- 基本工具 -----------------


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


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


@dataclass
class SplitLists:
    train: List[str]
    val: List[str]
    test: List[str]


# ----------------- 路径解析与 CSV 读取 -----------------

def _find_existing_split_yaml(args) -> Optional[Path]:
    """
    对于 KFold（kfold > 1），尝试在 split-outdir 下复用已有的 split 目录。
    返回其 data.yaml 路径；若不存在则返回 None。
    """
    # 只在 kfold>1 时尝试复用；random/单次划分不复用，保持原语义
    if not args.kfold or args.kfold <= 1:
        return None

    split_outdir = Path(args.split_outdir)
    if not split_outdir.exists():
        return None

    mode = args.split_mode
    seed = args.seed
    kfold = args.kfold
    fold_index = args.fold_index

    prefix = f"{mode}_s{seed}_k{kfold}_f{fold_index}_"
    candidates = [
        d for d in split_outdir.iterdir()
        if d.is_dir() and d.name.startswith(prefix)
    ]
    if not candidates:
        return None

    # 如有多个，按目录名排序取最后一个（时间戳更大）
    candidates.sort(key=lambda p: p.name)
    chosen = candidates[-1]
    yaml_path = chosen / "data.yaml"
    if not yaml_path.exists():
        return None

    print(f"[SPLIT] 复用已有 split 目录: {chosen}")
    return yaml_path


def _load_csv(data_csv: Path) -> pd.DataFrame:
    if not data_csv.exists():
        raise FileNotFoundError(f"data_csv 不存在: {data_csv}")
    df = pd.read_csv(data_csv)
    if df.empty:
        raise RuntimeError(f"data_csv 为空: {data_csv}")
    return df


def _find_filename_column(df: pd.DataFrame) -> str:
    lower_map = {c.lower(): c for c in df.columns}
    for key in ["filename", "file", "image", "img"]:
        if key in lower_map:
            return lower_map[key]
    # 如果没有匹配到，默认用第一列（但给出警告）
    print(f"[WARN] 未找到典型 filename 列，默认使用第一列: {df.columns[0]!r}")
    return df.columns[0]


def _guess_group_col(df: pd.DataFrame, split_mode: str, group_col_arg: Optional[str]) -> Optional[str]:
    if split_mode == "random":
        return None
    if group_col_arg:
        if group_col_arg not in df.columns:
            raise ValueError(f"指定的 group 列 {group_col_arg!r} 不在 CSV 列中: {list(df.columns)}")
        return group_col_arg

    # 自动推断：根据 split_mode 试着找 App/Object/Scene 等列
    lower_map = {c.lower(): c for c in df.columns}
    if split_mode == "app-ood":
        for key in ["app", "app_id", "appname"]:
            if key in lower_map:
                return lower_map[key]
        raise ValueError("app-ood 模式下无法自动找到 App 列，请使用 --group-col 显式指定。")
    if split_mode == "object-ood":
        for key in ["object", "obj", "object_id"]:
            if key in lower_map:
                return lower_map[key]
        raise ValueError("object-ood 模式下无法自动找到 Object 列，请使用 --group-col 显式指定。")
    if split_mode == "scene-ood":
        for key in ["scene", "scene_id"]:
            if key in lower_map:
                return lower_map[key]
        raise ValueError("scene-ood 模式下无法自动找到 Scene 列，请使用 --group-col 显式指定。")

    if split_mode == "group-ood":
        raise ValueError("group-ood 模式必须提供 --group-col。")

    return None


def _resolve_image_path(images_root: Path, name: str) -> Optional[str]:
    """根据 CSV 中的 filename 字段解析真实图片路径（绝对路径）。

    支持：
    - 直接写文件名: foo.jpg
    - 带子路径: terrace/foo.jpg
    """
    p = Path(name)
    # 已经是绝对路径且存在
    if p.is_absolute() and p.exists():
        return str(p.resolve())

    # 相对于 images_root 的相对路径
    cand = images_root / p
    if cand.exists():
        return str(cand.resolve())

    # 只用 basename 再试一次（应对 CSV 中写了子路径而实际 flatten 的情况）
    cand2 = images_root / p.name
    if cand2.exists():
        return str(cand2.resolve())

    return None


# ----------------- 分割逻辑 -----------------


def _parse_ratios(ratios_str: str) -> Tuple[float, float, float]:
    parts = [float(x) for x in ratios_str.split(",")]
    if len(parts) != 3:
        raise ValueError(f"--ratios 需要 3 个值，例如 0.7,0.15,0.15，当前为: {ratios_str!r}")
    s = sum(parts)
    if s <= 0:
        raise ValueError("--ratios 的和必须大于 0")
    parts = [x / s for x in parts]
    return parts[0], parts[1], parts[2]


def _split_groups_by_size(
    unique_groups: np.ndarray,
    groups_map: Dict[str, List[str]],
    ratios: Tuple[float, float, float],
    rng: np.random.RandomState,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """在保证 group 互斥的前提下，让**样本数量**尽量接近 ratios。

    策略：
    - 统计每个 group 的样本数；
    - 随机打乱 group 顺序；
    - 依次把当前 group 分配给使 (curr+size)/target 最接近 1 且未明显溢出的 split；
    - 极端情况下某个 split 没有 group，则从 group 最多的 split 借 1 个。
    """
    total_samples = sum(len(groups_map[g]) for g in unique_groups)
    if total_samples == 0:
        raise RuntimeError("总样本数为 0，无法分割。")

    targets = [ratios[i] * total_samples for i in range(3)]  # train/val/test 目标样本数

    order = list(unique_groups)
    rng.shuffle(order)

    curr_samples = [0.0, 0.0, 0.0]
    split_ids: List[List[str]] = [[], [], []]

    for gid in order:
        size = float(len(groups_map[gid]))
        best_j = None
        best_score = None

        for j in range(3):
            if targets[j] <= 0:
                continue

            new_sum = curr_samples[j] + size
            ratio = new_sum / targets[j]

            # score: (是否超过1, |ratio-1|, j)
            over = 1 if ratio > 1.0 else 0
            score = (over, abs(ratio - 1.0), j)

            if best_score is None or score < best_score:
                best_score = score
                best_j = j

        if best_j is None:
            best_j = 0  # 防御性兜底

        split_ids[best_j].append(gid)
        curr_samples[best_j] += size

    # 若极端情况下某个 split 为空，从 group 最多的 split 借 1 个过来
    for j in range(3):
        if len(split_ids[j]) == 0:
            donor = max(range(3), key=lambda x: len(split_ids[x]))
            if len(split_ids[donor]) > 1:
                moved = split_ids[donor].pop()
                split_ids[j].append(moved)

    curr_samples = [sum(len(groups_map[g]) for g in split_ids[i]) for i in range(3)]
    achieved = [s / total_samples for s in curr_samples]
    print(
        f"[SPLIT-GROUP] target ratios={ratios}, "
        f"achieved (by samples)={[round(x, 3) for x in achieved]}"
    )

    return (
        np.array(split_ids[0], dtype=object),
        np.array(split_ids[1], dtype=object),
        np.array(split_ids[2], dtype=object),
    )


def _to_lists_by_group(
    df: pd.DataFrame,
    images_root: Path,
    ratios: Tuple[float, float, float],
    seed: int,
    split_mode: str,
    group_col: Optional[str],
    kfold: int,
    fold_index: int,
) -> SplitLists:
    """核心分割逻辑：根据 split_mode/group_col/kfold 生成 train/val/test 的图片路径列表。"""
    rng = np.random.RandomState(seed)

    filename_col = _find_filename_column(df)

    # 收集 (img_path, group_id)
    all_items: List[Tuple[str, Optional[str]]] = []
    for _, row in df.iterrows():
        raw_name = str(row[filename_col])
        im = _resolve_image_path(images_root, raw_name)
        if im is None:
            continue
        gid = None
        if group_col is not None and group_col in df.columns:
            gid = str(row[group_col])
        all_items.append((im, gid))

    if not all_items:
        raise RuntimeError("没有从 CSV 中解析到任何图像路径，请检查 images_root 和 filename 列。")

    # 情况 1：random 模式（不使用 group）
    if split_mode == "random" or group_col is None:
        paths = [im for im, _ in all_items]
        rng.shuffle(paths)
        n = len(paths)
        r_train, r_val, r_test = ratios
        n_test = int(round(n * r_test))
        n_val = int(round(n * r_val))
        n_train = max(0, n - n_val - n_test)
        train_list = paths[:n_train]
        val_list = paths[n_train:n_train + n_val]
        test_list = paths[n_train + n_val:]
        print(
            f"[SPLIT-RANDOM] total={n}, "
            f"train/val/test={len(train_list)}/{len(val_list)}/{len(test_list)}"
        )
        return SplitLists(train=train_list, val=val_list, test=test_list)

    # 情况 2：按 group 做 OOD / KFold
    groups_map: Dict[str, List[str]] = {}
    for im, gid in all_items:
        gid = gid if gid is not None else "__NA__"
        groups_map.setdefault(gid, []).append(im)

    unique_groups = np.array(list(groups_map.keys()))

    # KFold 情况：先在 group 上做 K 折，再在训练侧按样本数切 train/test
    if kfold and kfold > 1:
        if kfold > len(unique_groups):
            raise ValueError(
                f"kfold={kfold} 大于 group 数量={len(unique_groups)}，无法进行 GroupKFold。"
            )
        gkf = GroupKFold(n_splits=kfold)
        X_dummy = np.arange(len(unique_groups))
        y_dummy = np.zeros_like(X_dummy)
        folds = list(gkf.split(X_dummy, y_dummy, groups=unique_groups))
        if not (0 <= fold_index < kfold):
            raise ValueError(f"fold-index 越界: 0 <= {fold_index} < {kfold} 不成立")
        train_val_idx, val_idx = folds[fold_index]

        train_val_groups = unique_groups[train_val_idx]
        val_groups = unique_groups[val_idx]

        # 在 train_val_groups 内切出 train/test
        tv_sum = ratios[0] + ratios[2]
        inner_ratios = (ratios[0] / tv_sum, 0.0, ratios[2] / tv_sum)
        train_groups, _, test_groups = _split_groups_by_size(
            train_val_groups,
            groups_map,
            inner_ratios,
            rng,
        )

        train_list, val_list, test_list = [], [], []
        for g in train_groups:
            train_list.extend(groups_map[g])
        for g in val_groups:
            val_list.extend(groups_map[g])
        for g in test_groups:
            test_list.extend(groups_map[g])

        print(
            f"[SPLIT-KFOLD] groups total={len(unique_groups)}, "
            f"fold={fold_index}/{kfold}, "
            f"train/val/test={len(train_list)}/{len(val_list)}/{len(test_list)}"
        )
        return SplitLists(train=train_list, val=val_list, test=test_list)

    # 非 KFold 情况：直接在全部 group 上按样本数近似 ratios 切 train/val/test
    train_groups, val_groups, test_groups = _split_groups_by_size(
        unique_groups, groups_map, ratios, rng
    )

    train_list, val_list, test_list = [], [], []
    for g in train_groups:
        train_list.extend(groups_map[g])
    for g in val_groups:
        val_list.extend(groups_map[g])
    for g in test_groups:
        test_list.extend(groups_map[g])

    print(
        f"[SPLIT-GROUP-FINAL] groups={len(unique_groups)}, "
        f"train/val/test={len(train_list)}/{len(val_list)}/{len(test_list)}"
    )
    return SplitLists(train=train_list, val=val_list, test=test_list)


# ----------------- 写 list + data.yaml + last -----------------


def _write_lists_and_yaml(out_dir: Path, lists: SplitLists, labels_root: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    train_txt = out_dir / "train.txt"
    val_txt = out_dir / "val.txt"
    test_txt = out_dir / "test.txt"

    # 写 train/val/test 的 list 文件
    for p, arr in [(train_txt, lists.train), (val_txt, lists.val), (test_txt, lists.test)]:
        with open(p, "w", encoding="utf-8") as f:
            for x in arr:
                f.write(str(x) + "\n")

    # 生成 data.yaml，固定为 2 个类别
    yaml_path = out_dir / "data.yaml"
    yaml_str = (
        "path: .\n"
        f"train: {train_txt.resolve()}\n"
        f"val: {val_txt.resolve()}\n"
        f"test: {test_txt.resolve()}\n"
        "nc: 2\n"
        "names:\n"
        "  0: AR_Object\n"
        "  1: UI_Element\n"
    )
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_str)

    # 记录 labels_root 供其它脚本复用
    with open(out_dir / "labels_root.txt", "w", encoding="utf-8") as f:
        f.write(str(labels_root.resolve()))

    return yaml_path


def _record_last(split_outdir: Path, yaml_path: Path, lists: SplitLists) -> None:
    """在 split_outdir 下写 last.yaml 和 last/train/val/test.txt 作为当前 split 的快照。"""
    splits_dir = split_outdir
    splits_dir.mkdir(parents=True, exist_ok=True)

    last_dir = splits_dir / "last"
    last_dir.mkdir(parents=True, exist_ok=True)

    with open(splits_dir / "last.yaml", "w", encoding="utf-8") as f:
        f.write(str(yaml_path.resolve()))

    for name, arr in {"train": lists.train, "val": lists.val, "test": lists.test}.items():
        list_path = last_dir / f"{name}.txt"
        with open(list_path, "w", encoding="utf-8") as f:
            for x in arr:
                f.write(str(x) + "\n")


def _ensure_ultra_yaml(args) -> Path:
    """生成 data.yaml + train/val/test list，并返回 data.yaml 的路径。"""
    images_root = Path(args.images_root)
    labels_root = Path(args.labels_root)
    data_csv = Path(args.data_csv)

    if not images_root.exists():
        raise FileNotFoundError(f"images_root 不存在: {images_root}")
    if not labels_root.exists():
        raise FileNotFoundError(f"labels_root 不存在: {labels_root}")
    if not data_csv.exists():
        raise FileNotFoundError(f"data_csv 不存在: {data_csv}")

    df = _load_csv(data_csv)
    ratios = _parse_ratios(args.ratios)
    group_col = _guess_group_col(df, args.split_mode, args.group_col)

    lists = _to_lists_by_group(
        df=df,
        images_root=images_root,
        ratios=ratios,
        seed=args.seed,
        split_mode=args.split_mode,
        group_col=group_col,
        kfold=args.kfold,
        fold_index=args.fold_index,
    )

    ts = int(time.time())
    mode = args.split_mode
    out_dir = Path(args.split_outdir) / f"{mode}_s{args.seed}_k{args.kfold}_f{args.fold_index}_{ts}"
    yaml_path = _write_lists_and_yaml(out_dir, lists, labels_root)

    _record_last(Path(args.split_outdir), yaml_path, lists)

    print(f"[SPLIT] yaml = {yaml_path}")
    print(
        f"[SPLIT] train/val/test 数量 = "
        f"{len(lists.train)}/{len(lists.val)}/{len(lists.test)}"
    )
    return yaml_path


# ----------------- 子命令：train -----------------


def cmd_train(args) -> None:
    device = pick_device(args.device)

    # ① 优先在 KFold 情况下尝试复用已有 split 目录
    yaml_path = _find_existing_split_yaml(args)
    if yaml_path is None:
        # 若不存在旧 split，则正常生成新的 split + data.yaml
        yaml_path = _ensure_ultra_yaml(args)

    # ② 构造唯一的实验名称，避免 slurm 并发写同一目录
    mode = args.split_mode
    weights_stem = Path(args.weights).stem  # yolo11n.pt -> yolo11n
    parent_name = yaml_path.parent.name     # 例如 "object-ood_s42_k5_f0_1763..."

    if parent_name.startswith(f"{mode}_"):
        suffix = parent_name[len(mode) + 1 :]  # 去掉 "object-ood_" 前缀
        auto_name = f"{mode}_{weights_stem}_{suffix}"
    else:
        ts = int(time.time())
        auto_name = f"{mode}_{weights_stem}_s{args.seed}_k{args.kfold}_f{args.fold_index}_{ts}"

    # 用户显式给了 --name 则优先使用，否则采用 auto_name
    if args.name and args.name != "exp":
        exp_name = args.name
    else:
        exp_name = auto_name

    exp_dir = Path(args.project) / exp_name
    best_weights = exp_dir / "weights" / "best.pt"

    # ③ 如果已经有训练好的 best.pt，则跳过本次训练
    if best_weights.exists():
        print(f"[SKIP] 检测到 fold 已有训练好的模型: {best_weights}")
        print(f"[SKIP] 跳过本次训练（split-mode={mode}, kfold={args.kfold}, fold={args.fold_index}）")
        return

    print(f"[TRAIN] data.yaml = {yaml_path}")
    print(f"[TRAIN] project   = {args.project}")
    print(f"[TRAIN] exp name  = {exp_name}")
    print(f"[TRAIN] device    = {device}")

    model = YOLO(args.weights)
    t0 = time.perf_counter()
    results = model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        lr0=args.lr0,
        patience=args.patience,
        seed=args.seed,
        workers=args.workers,
        project=args.project,
        name=exp_name,
        pretrained=True,
        verbose=False,  # 如果你已经按我们前面讨论关掉详细日志，可以保留这一行
    )
    dt = time.perf_counter() - t0
    print(results)
    print(f"[TIME] training wall time = {dt:.2f}s")



# ----------------- 子命令：test（无标签批量推理） -----------------


def _load_list(list_path: Path) -> List[str]:
    if not list_path.exists():
        raise FileNotFoundError(f"list 文件不存在: {list_path}")
    lines = [ln.strip() for ln in list_path.read_text(encoding="utf-8").splitlines()]
    return [ln for ln in lines if ln]


def cmd_test(args) -> None:
    device = pick_device(args.device)
    model = YOLO(args.weights)

    images: List[str] = []

    if args.image_list:
        list_path = Path(args.image_list)
        paths = _load_list(list_path)
        images.extend(paths)
        print(f"[TEST] 从 image_list 读取 {len(paths)} 张图片。")
    elif args.test_dir:
        test_dir = Path(args.test_dir)
        if not test_dir.exists():
            raise FileNotFoundError(f"test-dir 不存在: {test_dir}")
        for p in test_dir.rglob("*"):
            if p.suffix.lower() in IMAGE_EXTS and p.is_file():
                images.append(str(p.resolve()))
        print(f"[TEST] 从 test-dir 扫描到 {len(images)} 张图片。")
    else:
        raise ValueError("test 需要 --image_list/--image-list 或 --test-dir 其中之一。")

    if not images:
        print("[TEST] 没有图片可供推理。")
        return

    print(f"[TEST] device = {device}")
    out_dir = Path(args.project) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Ultralytics YOLO 的 predict 可以直接传 list
    t0 = time.perf_counter()
    _ = model.predict(
        source=images,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        device=device,
        project=args.project,
        name=args.name,
        save=args.save,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
        verbose=True,
    )
    dt = time.perf_counter() - t0
    print(f"[TIME] test wall time = {dt:.2f}s")


# ----------------- CLI -----------------


def add_split_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--images-root",
        type=str,
        required=True,
        help="图片根目录（与 data_stat.csv 中 filename 列对应）。",
    )
    p.add_argument(
        "--labels-root",
        type=str,
        required=True,
        help="YOLO label txt 根目录。",
    )
    p.add_argument(
        "--data-csv",
        type=str,
        required=True,
        help="例如 data_stat.csv，包含 filename、App/Object/Scene 等列。",
    )
    p.add_argument(
        "--split-mode",
        type=str,
        default="random",
        choices=["random", "app-ood", "object-ood", "scene-ood", "group-ood"],
        help="数据划分模式。",
    )
    p.add_argument(
        "--group-col",
        type=str,
        default=None,
        help="group-ood 或自动 OOD 失败时的列名，例如 App/Object/Scene。",
    )
    p.add_argument(
        "--ratios",
        type=str,
        default="0.7,0.15,0.15",
        help="train,val,test 的比例，例如 0.7,0.15,0.15。",
    )
    p.add_argument(
        "--split-outdir",
        type=str,
        default=".split",
        help="生成 list 与 data.yaml 的输出目录（默认 .split）。",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="用于随机划分的随机种子。",
    )
    p.add_argument(
        "--kfold",
        type=int,
        default=0,
        help="group 级 KFold，0 或 1 表示不启用。",
    )
    p.add_argument(
        "--fold-index",
        type=int,
        default=0,
        help="KFold 中当前折编号，从 0 开始。",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Dynamic split + YOLO train/test"
    )
    sub = parser.add_subparsers(dest="subcommand", required=True)

    # train 子命令
    p_train = sub.add_parser("train", help="动态划分 + 训练 YOLO 模型")
    add_split_args(p_train)
    p_train.add_argument(
        "--weights",
        type=str,
        required=True,
        help="YOLO 预训练权重路径，例如 yolo11n.pt",
    )
    p_train.add_argument("--epochs", type=int, default=100)
    p_train.add_argument("--batch", type=int, default=16)
    p_train.add_argument("--imgsz", type=int, default=640)
    p_train.add_argument("--lr0", type=float, default=0.01)
    p_train.add_argument("--patience", type=int, default=50)
    p_train.add_argument("--workers", type=int, default=8)
    p_train.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
    )
    p_train.add_argument(
        "--project",
        type=str,
        default="runs/train",
        help="Ultralytics project 目录。",
    )
    p_train.add_argument(
        "--name",
        type=str,
        default="exp",
        help="实验名称（默认自动生成）。",
    )
    p_train.set_defaults(func=cmd_train)

    # test 子命令（无标签推理）
    p_test = sub.add_parser("test", help="无标签批量推理")
    p_test.add_argument(
        "--image_list",
        "--image-list",
        dest="image_list",
        type=str,
        default=None,
        help="list 文件路径，每行一个图像路径。",
    )
    p_test.add_argument(
        "--test-dir",
        type=str,
        default=None,
        help="如果未提供 image_list，则从此目录递归扫描图片。",
    )
    p_test.add_argument(
        "--weights",
        type=str,
        required=True,
        help="YOLO 权重路径，例如 runs/train/xxx/weights/best.pt",
    )
    p_test.add_argument("--imgsz", type=int, default=640)
    p_test.add_argument("--conf", type=float, default=0.25)
    p_test.add_argument("--iou", type=float, default=0.6)
    p_test.add_argument("--max-det", type=int, default=100)
    p_test.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
    )
    p_test.add_argument(
        "--project",
        type=str,
        default="runs/predict",
        help="predict 输出 project 目录。",
    )
    p_test.add_argument(
        "--name",
        type=str,
        default="exp",
        help="predict 输出子目录名。",
    )
    p_test.add_argument(
        "--save",
        action="store_true",
        help="是否保存带框可视化图片。",
    )
    p_test.add_argument(
        "--save-txt",
        action="store_true",
        help="是否保存 YOLO txt 预测结果。",
    )
    p_test.add_argument(
        "--save-conf",
        action="store_true",
        help="是否在 txt 中保存置信度。",
    )
    p_test.set_defaults(func=cmd_test)

    args = parser.parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
