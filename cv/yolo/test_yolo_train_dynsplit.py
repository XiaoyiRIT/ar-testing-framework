#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
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
  --ratios 0.7,0.15,0.15 \
  --weights yolo11n.pt --epochs 100 --imgsz 640
  
  --kfold 5 --fold-index 0 

# 3) 仅评测（中心点命中 GT 算成功），从 list 文件读入
python test_yolo_train_dynsplit.py eval \
  --image-list .splits/last/test.txt \
  --labels-root /path/dataset/labels \
  --weights runs/train/exp/weights/best.pt \
  --class-id 0 --imgsz 640 --conf 0.05 --save-vis

说明：
- 本脚本不会复制或移动任何图像/标签，只会在 .splits/ 目录下生成 train.txt/val.txt/test.txt
- 自动生成临时 data.yaml（指向三个 list 文件），传给 ultralytics.YOLO 使用
- 支持 random、app-ood、object-ood、scene-ood；也可用 --group-col=自定义列 实现任意组互斥
"""

import argparse
import csv
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO

try:
    import pandas as pd
except Exception as e:
    print("[ERROR] 需要 pandas：pip install pandas")
    raise

try:
    from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold
except Exception as e:
    print("[ERROR] 需要 scikit-learn：pip install scikit-learn")
    raise

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# =========================
# 设备选择（与原脚本一致风格）
# =========================

def pick_device(prefer: str = "auto") -> str:
    if prefer != "auto":
        return prefer
    try:
        import torch
        import platform
        if platform.system() == "Darwin" and torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"

# =========================
# 分割与 list/yaml 生成
# =========================

@dataclass
class SplitLists:
    train: List[str]
    val: List[str]
    test: List[str]


def _read_csv(data_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(data_csv)
    # 规范列名
    cols = {c.lower(): c for c in df.columns}
    # 兼容常见列名
    for need in ["filename", "app", "object", "scene"]:
        if need not in cols and need not in df.columns:
            # 尝试大小写容错
            for c in df.columns:
                if c.lower() == need:
                    cols[need] = c
    # 至少需要 filename
    fn_col = cols.get("filename", "filename")
    if fn_col not in df.columns:
        raise ValueError("CSV 必须包含 filename 列")
    return df


def _resolve_image_path(images_root: Path, name: str) -> Optional[str]:
    p = images_root / name
    if p.suffix == "":
        # 允许不带后缀的 filename；尝试匹配
        for ext in IMAGE_EXTS:
            cand = (images_root / f"{name}{ext}")
            if cand.exists():
                return str(cand.resolve())
        return None
    return str(p.resolve())


def _to_lists_by_group(
    df: pd.DataFrame,
    images_root: Path,
    ratios: Tuple[float, float, float],
    seed: int,
    group_col: Optional[str] = None,
    kfold: int = 0,
    fold_index: int = 0,
) -> SplitLists:
    """按组（可选）进行随机/组互斥切分；不复制文件，仅返回图像绝对路径 list。

    注意：
    - random 模式下不使用 group，直接在样本级别按 ratios 切分；
    - *-ood 模式下使用 group，并尽量让最终 "样本数量" 接近给定 ratios，
      而不是简单按 group 数量来切，避免某些 group 过大导致比例严重失衡。
    """
    rng = np.random.RandomState(seed)
    all_items: List[Tuple[str, Optional[str]]] = []  # (img_path, group_id)

    # 解析路径并可选读取 group
    for _, row in df.iterrows():
        name = str(row.get("filename") if "filename" in row else row.get("Filename"))
        if not isinstance(name, str):
            continue
        im = _resolve_image_path(images_root, name)
        if im is None:
            continue
        gid = None
        if group_col and group_col in row:
            gid = str(row[group_col])
        all_items.append((im, gid))

    if not all_items:
        raise RuntimeError("没有从 CSV 解析到任何图像路径，请检查 images_root/filename 列")

    # ------------------ helper：按 group 尺寸做贪心近似比例切分 ------------------
    def split_groups_by_size(
        unique_groups: np.ndarray,
        groups_map: Dict[str, List[str]],
        ratios_: Tuple[float, float, float],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """在保证 group 互斥的前提下，让**样本数量**尽量接近 ratios 指定的比例。

        策略：
        - 先统计每个 group 的样本量；
        - 按随机顺序依次遍历每个 group；
        - 每次把当前 group 分配给 “(curr + size) / target 最靠近 1 且未明显溢出的”那个 split；
        - 这样 target 大的 train 桶会优先被填满，val/test 较小的桶会先饱和。
        """
        total_samples = sum(len(groups_map[g]) for g in unique_groups)
        if total_samples == 0:
            raise RuntimeError("总样本数为 0，无法分割")

        targets = [ratios_[i] * total_samples for i in range(3)]  # train/val/test 目标样本数

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

                # 希望 ratio 尽量接近 1；同时优先考虑还没“过满”的桶
                # score 的含义：
                #   (是否已超过 1, |ratio-1|, j)
                # 这样：
                #   - 先选“未超过 1”的桶
                #   - 再选 |ratio-1| 更小的
                #   - 再用 j 做一个稳定的 tie-break（保证 deterministic）
                over = 1 if ratio > 1.0 else 0
                score = (over, abs(ratio - 1.0), j)

                if best_score is None or score < best_score:
                    best_score = score
                    best_j = j

            if best_j is None:
                # 理论上不会出现，保险起见
                best_j = 0

            split_ids[best_j].append(gid)
            curr_samples[best_j] += size

        # 若极端情况下某个 split 没有 group，则从 group 数最多的 split 借 1 个过来
        for j in range(3):
            if len(split_ids[j]) == 0:
                donor = max(range(3), key=lambda x: len(split_ids[x]))
                if len(split_ids[donor]) > 1:
                    moved = split_ids[donor].pop()
                    split_ids[j].append(moved)

        # 重新统计样本数并打印实际比例方便调试
        curr_samples = [sum(len(groups_map[g]) for g in split_ids[i]) for i in range(3)]
        achieved = [s / total_samples if total_samples > 0 else 0.0 for s in curr_samples]
        print(
            f"[SPLIT-GROUP] target ratios={ratios_}, "
            f"achieved (by samples)={[round(x, 3) for x in achieved]}"
        )
        return (
            np.array(split_ids[0], dtype=object),
            np.array(split_ids[1], dtype=object),
            np.array(split_ids[2], dtype=object),
        )


    # ------------------ 主体逻辑 ------------------

    # 情况 1：不按 group（random 模式），直接按样本切分
    if not group_col:
        paths = [im for im, _ in all_items]
        rng.shuffle(paths)
        n = len(paths)
        n_test = int(round(n * ratios[2]))
        n_val = int(round(n * ratios[1]))
        n_train = max(0, n - n_val - n_test)
        train_list = paths[:n_train]
        val_list = paths[n_train:n_train + n_val]
        test_list = paths[n_train + n_val:]
        return SplitLists(train=train_list, val=val_list, test=test_list)

    # 情况 2：按 group 做切分（*-ood 模式）
    # 收集 group -> paths
    groups: Dict[str, List[str]] = {}
    for im, gid in all_items:
        gid = gid if gid is not None else "__NA__"
        groups.setdefault(gid, []).append(im)
    unique_groups = np.array(list(groups.keys()))

    if kfold and kfold > 1:
        # K 折的目标主要是评估稳定性，这里仍沿用 GroupKFold 按 group 数量平衡，
        # 不强求每折严格匹配样本比例，避免逻辑过重。
        gkf = GroupKFold(n_splits=kfold)
        X = unique_groups
        y = np.zeros(len(unique_groups))
        folds = list(gkf.split(X, y, groups=unique_groups))
        if not (0 <= fold_index < kfold):
            raise ValueError("fold-index 越界")
        train_val_idx, val_idx = folds[fold_index]
        train_groups = unique_groups[train_val_idx]
        val_groups = unique_groups[val_idx]

        # 在 train_groups 内按样本数近似 ratios[0]:ratios[2] 再切出 test
        tv_groups = train_groups
        # train:test 的目标比例（只在 train+test 内重新归一化）
        tv_sum = ratios[0] + ratios[2]
        inner_ratios = (ratios[0] / tv_sum, 0.0, ratios[2] / tv_sum)
        inner_train_groups, _, test_groups = split_groups_by_size(tv_groups, groups, inner_ratios)

        train_list, val_list, test_list = [], [], []
        for g in inner_train_groups:
            train_list.extend(groups[g])
        for g in val_groups:
            val_list.extend(groups[g])
        for g in test_groups:
            test_list.extend(groups[g])

    else:
        # 非 K 折：直接在所有 group 上按样本数近似 ratios 切 train/val/test
        train_groups, val_groups, test_groups = split_groups_by_size(unique_groups, groups, ratios)

        train_list, val_list, test_list = [], [], []
        for g in train_groups:
            train_list.extend(groups[g])
        for g in val_groups:
            val_list.extend(groups[g])
        for g in test_groups:
            test_list.extend(groups[g])

    return SplitLists(train=train_list, val=val_list, test=test_list)


def _write_lists_and_yaml(out_dir: Path, lists: SplitLists, labels_root: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    train_txt = out_dir / "train.txt"
    val_txt = out_dir / "val.txt"
    test_txt = out_dir / "test.txt"

    # 1) 写 train/val/test 的 list 文件
    for p, arr in [(train_txt, lists.train), (val_txt, lists.val), (test_txt, lists.test)]:
        with open(p, "w", encoding="utf-8") as f:
            for x in arr:
                f.write(str(x) + "\n")

    # 2) 生成 data.yaml，固定为 2 个类别：0=AR_Object, 1=UI_Element
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

    # 3) 记录 labels_root 供 eval 默认推断
    with open(out_dir / "labels_root.txt", "w", encoding="utf-8") as f:
        f.write(str(labels_root.resolve()))

    return yaml_path


# =========================
# 训练 / 验证 / 推理 / 评测
# =========================

def _ensure_ultra_yaml(args) -> Optional[Path]:
    """基于 CSV/模式生成 txt 列表与 yaml；若未提供 data-csv 则返回 None 表示使用用户的 --data。"""
    if not args.data_csv:
        return None

    images_root = Path(args.images_root)
    labels_root = Path(args.labels_root)
    df = _read_csv(Path(args.data_csv))

    # 选择 group 列
    mode = args.split_mode.lower()
    group_col = None
    if mode in {"app-ood", "object-ood", "scene-ood"}:
        if args.group_col:
            group_col = args.group_col
        else:
            group_col = {"app-ood": "app", "object-ood": "object", "scene-ood": "scene"}[mode]
    elif mode == "group-ood":
        if not args.group_col:
            raise ValueError("group-ood 模式必须指定 --group-col")
        group_col = args.group_col

    ratios = tuple(float(x) for x in args.ratios.split(","))  # e.g., "0.7,0.15,0.15"
    if len(ratios) != 3:
        raise ValueError("--ratios 需要三个用逗号分隔的小数，例如 0.7,0.15,0.15")

    lists = _to_lists_by_group(
        df=df,
        images_root=images_root,
        ratios=ratios, seed=args.seed,
        group_col=group_col,
        kfold=args.kfold, fold_index=args.fold_index,
    )

    ts = int(time.time())
    out_dir = Path(args.split_outdir) / f"{mode}_s{args.seed}_k{args.kfold}_f{args.fold_index}_{ts}"
    yaml_path = _write_lists_and_yaml(out_dir, lists, labels_root)

    # ========= 修复部分：先创建 .split / .splits 以及 last 目录 =========
    splits_dir = Path(args.split_outdir)
    splits_dir.mkdir(parents=True, exist_ok=True)

    last_dir = splits_dir / "last"
    last_dir.mkdir(parents=True, exist_ok=True)

    # 记录这次生成的 yaml 路径
    with open(splits_dir / "last.yaml", "w", encoding="utf-8") as f:
        f.write(str(yaml_path.resolve()))

    # 同时把这次的 train/val/test 列表也保存一份，方便 eval/复现实验
    for name, arr in {"train": lists.train, "val": lists.val, "test": lists.test}.items():
        list_path = last_dir / f"{name}.txt"
        with open(list_path, "w", encoding="utf-8") as f:
            for x in arr:
                f.write(str(x) + "\n")

    print(f"[SPLIT] yaml = {yaml_path}")
    print(f"[SPLIT] train/val/test 数量 = {len(lists.train)}/{len(lists.val)}/{len(lists.test)}")
    return yaml_path



def _load_list(list_file: Path) -> List[Path]:
    with open(list_file, "r", encoding="utf-8") as f:
        return [Path(x.strip()) for x in f if x.strip()]


def _yolo_label_for_image(img_path: Path, labels_root: Path) -> Path:
    return labels_root / (img_path.stem + ".txt")


def _draw_vis(img_path: Path, save_path: Path, gt_xyxy: Optional[Tuple[float,float,float,float]], pred_center: Optional[Tuple[float,float]]):
    im = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(im)
    if gt_xyxy is not None:
        x1,y1,x2,y2 = gt_xyxy
        draw.rectangle([x1,y1,x2,y2], outline=(0,255,0), width=3)
    if pred_center is not None:
        cx, cy = pred_center
        r = 4
        draw.ellipse([cx-r,cy-r,cx+r,cy+r], fill=(255,0,0))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(save_path)


def _load_gt_one(label_path: Path, img_wh: Tuple[int,int], want_class: int) -> Optional[Tuple[float,float,float,float]]:
    if not label_path.exists():
        return None
    W, H = img_wh
    with open(label_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    for ln in lines:
        ss = ln.split()
        if len(ss) < 5:
            continue
        cls = int(float(ss[0]))
        if cls != want_class:
            continue
        cx, cy, w, h = map(float, ss[1:5])
        x_c, y_c = cx * W, cy * H
        bw, bh = w * W, h * H
        x1, y1 = x_c - bw/2, y_c - bh/2
        x2, y2 = x_c + bw/2, y_c + bh/2
        return (x1, y1, x2, y2)
    return None

# ----------------- commands -----------------

def cmd_train(args):
    device = pick_device(args.device)
    yaml_path = _ensure_ultra_yaml(args) or Path(args.data)

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
        name=args.name,
        pretrained=True,
    )
    dt = time.perf_counter() - t0
    print(results)
    print(f"[TIME] training wall time = {dt:.2f}s")


def cmd_val(args):
    device = pick_device(args.device)
    yaml_path = _ensure_ultra_yaml(args) or Path(args.data)

    model = YOLO(args.weights)
    metrics = model.val(
        data=str(yaml_path),
        split="val",
        imgsz=args.imgsz,
        device=device,
        iou=0.7,
        conf=0.001,
        project=args.project,
        name=args.name,
    )
    print(metrics)


def cmd_test(args):
    device = pick_device(args.device)
    model = YOLO(args.weights)

    images = []
    if args.image_list:
        images = _load_list(Path(args.image_list))
    elif args.test_dir:
        images = [p for p in Path(args.test_dir).rglob("*") if p.suffix.lower() in IMAGE_EXTS]
    else:
        raise ValueError("test 需要 --image-list 或 --test-dir")

    for r in model.predict(
        source=[str(p) for p in images],
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        device=device,
        project=args.project,
        name=args.name,
        stream=False,
        save=True,
        save_txt=True,
        save_conf=True,
        verbose=False,
    ):
        pass
    print("[INFO] 推理完成，查看 runs 目录输出。")


def cmd_eval(args):
    device = pick_device(args.device)
    model = YOLO(args.weights)

    if args.image_list:
        images = _load_list(Path(args.image_list))
        labels_root = Path(args.labels_root)
    elif args.test_dir:
        images = [p for p in Path(args.test_dir).rglob("*") if p.suffix.lower() in IMAGE_EXTS]
        labels_root = Path(args.labels_root)
    else:
        # 回落到 last 列表（便于与 train/val 同步）
        last = Path(args.split_outdir) / "last" / "test.txt"
        labels_hint = Path(args.split_outdir) / "labels_root.txt"
        if not last.exists() or not labels_hint.exists():
            raise RuntimeError("未提供 --image-list/--test-dir，且找不到 .splits/last/test.txt")
        images = _load_list(last)
        labels_root = Path(labels_hint.read_text().strip())

    out_root = Path(args.project) / args.name
    out_root.mkdir(parents=True, exist_ok=True)
    vis_dir = out_root / "vis" if args.save_vis else None
    report_csv = out_root / "report.csv"

    want_class = int(args.class_id)

    n_total = 0
    n_with_gt = 0
    n_success = 0
    n_no_label = 0
    n_no_pred = 0

    rows: List[Dict[str, str]] = []

    for img_path in images:
        img_path = Path(img_path)
        if not img_path.exists():
            continue
        n_total += 1
        with Image.open(img_path) as im:
            W, H = im.size
        label_path = _yolo_label_for_image(img_path, labels_root)
        gt_xyxy = _load_gt_one(label_path, (W, H), want_class)

        if gt_xyxy is None:
            n_no_label += 1
            status = "no_label"
            pred_center = None
            r = model.predict(
                source=str(img_path), imgsz=args.imgsz, conf=args.conf, iou=args.iou,
                max_det=args.max_det, device=device, verbose=False
            )[0]
            if r and r.boxes is not None and r.boxes.data.numel() > 0 and r.boxes.cls is not None:
                b = r.boxes
                import numpy as np
                mask = (b.cls.int().cpu().numpy() == want_class)
                if mask.any():
                    confs = b.conf.cpu().numpy()[mask]
                    idx_local = int(np.argmax(confs))
                    xyxy = b.xyxy.cpu().numpy()[mask][idx_local]
                    cx = (xyxy[0] + xyxy[2]) / 2.0
                    cy = (xyxy[1] + xyxy[3]) / 2.0
                    pred_center = (float(cx), float(cy))
            if args.save_vis and vis_dir is not None:
                _draw_vis(img_path, vis_dir / f"{img_path.stem}.jpg", None, pred_center)
            rows.append({
                "image": str(img_path),
                "status": status,
                "success": "NA",
                "gt_xyxy": "NA",
                "pred_center": "NA" if pred_center is None else f"{pred_center[0]:.1f},{pred_center[1]:.1f}",
            })
            continue

        n_with_gt += 1
        r = model.predict(
            source=str(img_path), imgsz=args.imgsz, conf=args.conf, iou=args.iou,
            max_det=args.max_det, device=device, verbose=False
        )[0]

        pred_center = None
        status = "pred_ok"
        if r is None or r.boxes is None or r.boxes.data.numel() == 0 or r.boxes.cls is None:
            n_no_pred += 1
            status = "no_pred"
        else:
            b = r.boxes
            import numpy as np
            cls_np = b.cls.int().cpu().numpy()
            conf_np = b.conf.cpu().numpy()
            xyxy_np = b.xyxy.cpu().numpy()
            mask = (cls_np == want_class)
            if not mask.any():
                n_no_pred += 1
                status = "no_pred"
            else:
                confs = conf_np[mask]
                idx_local = int(np.argmax(confs))
                xyxy = xyxy_np[mask][idx_local]
                cx = (xyxy[0] + xyxy[2]) / 2.0
                cy = (xyxy[1] + xyxy[3]) / 2.0
                pred_center = (float(cx), float(cy))

        success = False
        if pred_center is not None:
            x1, y1, x2, y2 = gt_xyxy
            cx, cy = pred_center
            success = (x1 <= cx <= x2) and (y1 <= cy <= y2)
            if success:
                n_success += 1

        if args.save_vis and vis_dir is not None:
            _draw_vis(img_path, vis_dir / f"{img_path.stem}.jpg", gt_xyxy, pred_center)

        rows.append({
            "image": str(img_path),
            "status": status,
            "success": "1" if success else "0",
            "gt_xyxy": f"{gt_xyxy[0]:.1f},{gt_xyxy[1]:.1f},{gt_xyxy[2]:.1f},{gt_xyxy[3]:.1f}",
            "pred_center": "NA" if pred_center is None else f"{pred_center[0]:.1f},{pred_center[1]:.1f}",
        })

    success_rate = (n_success / max(1, n_with_gt)) * 100.0

    with open(report_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image","status","success","gt_xyxy","pred_center"])
        writer.writeheader()
        writer.writerows(rows)

    print("\n========== EVAL (Center-in-GT) ==========")
    print(f"Total images           : {n_total}")
    print(f"With GT (denominator)  : {n_with_gt}")
    print(f"Success (hit center)   : {n_success}")
    print(f"No GT label            : {n_no_label}")
    print(f"No prediction (class)  : {n_no_pred}")
    print(f"Success Rate           : {success_rate:.2f}%")
    print(f"Report CSV             : {report_csv}")
    if args.save_vis:
        print(f"Visualizations         : {vis_dir}")

# ----------------- main & args -----------------

def main():
    parser = argparse.ArgumentParser(description="YOLO 动态分割 + OOD/CV 支持版（不复制数据，仅写 list 与 yaml）")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # 通用动态分割参数（可选）
    def add_split_args(p):
        p.add_argument("--data-csv", type=str, default=None, help="data_stat.csv（含 filename, app, object, scene 等列）")
        p.add_argument("--images-root", type=str, default=None, help="图片根目录（与 CSV 的 filename 拼接）")
        p.add_argument("--labels-root", type=str, default=None, help="YOLO labels 根目录（txt 与图片同名）")
        p.add_argument("--split-mode", type=str, default="random", choices=[
            "random", "app-ood", "object-ood", "scene-ood", "group-ood"
        ])
        p.add_argument("--group-col", type=str, default=None, help="自定义组列名（配合 group-ood 使用或覆盖默认列名）")
        p.add_argument("--ratios", type=str, default="0.7,0.15,0.15", help="train,val,test 比例")
        p.add_argument("--kfold", type=int, default=0, help=">1 启用按组 K 折（train/val）；test 由 train 侧再切出")
        p.add_argument("--fold-index", type=int, default=0, help="K 折中的折号（0..k-1）")
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--split-outdir", type=str, default=".splits", help="生成 list 与 yaml 的输出目录")

    # train
    p_train = sub.add_parser("train", help="训练：支持 CSV 动态分割或直接用现有 data.yaml")
    add_split_args(p_train)
    p_train.add_argument("--data", type=str, default="myar2.yaml", help="若未提供 CSV，则使用该 data.yaml")
    p_train.add_argument("--weights", type=str, default="yolo11n.pt")
    p_train.add_argument("--epochs", type=int, default=100)
    p_train.add_argument("--imgsz", type=int, default=640)
    p_train.add_argument("--batch", type=int, default=16)
    p_train.add_argument("--lr0", type=float, default=0.01)
    p_train.add_argument("--patience", type=int, default=30)
    p_train.add_argument("--workers", type=int, default=4)
    p_train.add_argument("--device", type=str, default="auto", choices=["auto","cpu","cuda","mps"])
    p_train.add_argument("--project", type=str, default="runs/train")
    p_train.add_argument("--name", type=str, default="exp")
    p_train.set_defaults(func=cmd_train)

    # val
    p_val = sub.add_parser("val", help="验证集评测：支持 CSV 动态分割或 data.yaml")
    add_split_args(p_val)
    p_val.add_argument("--data", type=str, default="myar2.yaml")
    p_val.add_argument("--weights", type=str, default="runs/train/exp/weights/best.pt")
    p_val.add_argument("--imgsz", type=int, default=640)
    p_val.add_argument("--device", type=str, default="auto", choices=["auto","cpu","cuda","mps"])
    p_val.add_argument("--project", type=str, default="runs/val")
    p_val.add_argument("--name", type=str, default="exp")
    p_val.set_defaults(func=cmd_val)

    # test（无标签推理）
    p_test = sub.add_parser("test", help="批量推理：支持 --image-list 或 --test-dir")
    p_test.add_argument("--image-list", type=str, default=None, help="list 文件（每行一个图像绝对路径）")
    p_test.add_argument("--test-dir", type=str, default=None)
    p_test.add_argument("--weights", type=str, default="runs/train/exp/weights/best.pt")
    p_test.add_argument("--imgsz", type=int, default=640)
    p_test.add_argument("--conf", type=float, default=0.05)
    p_test.add_argument("--iou", type=float, default=0.6)
    p_test.add_argument("--max-det", type=int, default=100)
    p_test.add_argument("--device", type=str, default="auto", choices=["auto","cpu","cuda","mps"])
    p_test.add_argument("--project", type=str, default="runs/predict")
    p_test.add_argument("--name", type=str, default="exp")
    p_test.set_defaults(func=cmd_test)

    # eval（中心点命中 GT）
    p_eval = sub.add_parser("eval", help="有标签评测：从 --image-list/--test-dir/.splits/last 读取图像")
    p_eval.add_argument("--image-list", type=str, default=None)
    p_eval.add_argument("--test-dir", type=str, default=None)
    p_eval.add_argument("--labels-root", type=str, default=None, help="若使用 --image-list，需指定 labels 根目录")
    p_eval.add_argument("--weights", type=str, default="runs/train/exp/weights/best.pt")
    p_eval.add_argument("--class-id", type=int, default=0)
    p_eval.add_argument("--imgsz", type=int, default=640)
    p_eval.add_argument("--conf", type=float, default=0.05)
    p_eval.add_argument("--iou", type=float, default=0.6)
    p_eval.add_argument("--max-det", type=int, default=100)
    p_eval.add_argument("--device", type=str, default="auto", choices=["auto","cpu","cuda","mps"])
    p_eval.add_argument("--project", type=str, default="runs/eval-ctr")
    p_eval.add_argument("--name", type=str, default="exp")
    p_eval.add_argument("--save-vis", action="store_true")
    p_eval.add_argument("--split-outdir", type=str, default=".splits")
    p_eval.set_defaults(func=cmd_eval)

    args = parser.parse_args()

    # 约束：若使用 CSV 动态分割，要求提供 images_root/labels_root
    if getattr(args, "data_csv", None):
        if not args.images_root or not args.labels_root:
            raise ValueError("使用 --data-csv 时必须提供 --images-root 与 --labels-root")

    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
