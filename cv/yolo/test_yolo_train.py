#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
python test_yolo_train.py train --data myar_scene.yaml --weights yolo11n.pt --epochs 200 --imgsz 640
python test_yolo_train.py val   --data myar2.yaml --weights runs/train/exp/weights/best.pt
python test_yolo_train.py test  --test-dir /Users/yangxiaoyi/datasets/myar_scene/images/test --weights runs/train/exp5/weights/best.pt --conf 0.05

新增：有标签评测（中心点命中 GT 判成功）
python test_yolo_train.py eval --data myar2.yaml --weights runs/train/exp/weights/best.pt \
    --split test --class-id 0 --conf 0.05 --report-latex --save-vis
    
python test_yolo_train.py eval \
  --test-dir /Users/yangxiaoyi/datasets/myar_scene/images/test \
  --labels-dir /Users/yangxiaoyi/datasets/myar_scene/labels/test \
  --weights runs/train/exp5/weights/best.pt \
  --class-id 0 --conf 0.05 --report-latex --save-vis


也支持显式目录：
python test_yolo_train.py eval --test-dir path/to/images --labels-dir path/to/labels --weights ... --class-id 0
"""

import argparse
import sys
import platform
from tqdm import tqdm
from pathlib import Path
from ultralytics import YOLO
import time
from datetime import timedelta
import csv
import math
from typing import List, Tuple, Optional, Dict

from PIL import Image  # 用于读取图像宽高
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

def pick_device(prefer="auto"):
    # 简单设备选择：优先 MPS（Apple Silicon），其次 CUDA，最后 CPU
    if prefer != "auto":
        return prefer
    try:
        import torch
        if platform.system() == "Darwin" and torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"

def ensure_yaml(data_yaml: str):
    p = Path(data_yaml)
    if not p.exists():
        print(f"[WARN] data yaml 不存在：{p.resolve()}，请确认路径。")
    return str(p)

def cmd_train(args):
    device = pick_device(args.device)
    print(f"[INFO] device = {device}")

    model = YOLO(args.weights)

    # 计时开始
    _t0 = time.perf_counter()

    results = model.train(
        data=ensure_yaml(args.data),
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
        pretrained=True
    )

    # 计时结束并输出
    elapsed_s = time.perf_counter() - _t0
    avg_per_epoch = elapsed_s / max(1, args.epochs)
    print(results)
    print(f"[TIME] training wall time = {timedelta(seconds=int(elapsed_s))} "
          f"({elapsed_s:.2f}s), avg/epoch = {avg_per_epoch:.2f}s")

def cmd_val(args):
    device = pick_device(args.device)
    print(f"[INFO] device = {device}")

    model = YOLO(args.weights)
    metrics = model.val(
        data=ensure_yaml(args.data),
        split="val",              # 依赖 data.yaml 的 val
        imgsz=args.imgsz,
        device=device,
        iou=0.7,                  # 可按需调整
        conf=0.001,               # 评测用低阈值，做 PR 曲线更稳
        project=args.project,
        name=args.name
    )
    print(metrics)                # dict-like，含 mAP50、mAP50-95 等

def cmd_test(args):
    device = pick_device(args.device)
    print(f"[INFO] device = {device}")

    model = YOLO(args.weights)

    test_dir = Path(args.test_dir)
    # 统计测试集图片数量（递归）
    total_imgs = sum(1 for p in test_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS)

    if total_imgs == 0:
        print(f"[WARN] 在 {test_dir} 未找到图片（支持扩展名：{sorted(IMAGE_EXTS)}）")
        return

    pbar = tqdm(total=total_imgs, desc="Inferencing (stream)", unit="img")

    # 关键：stream=True，并通过 for 循环消费生成器来触发推理与保存
    for r in model.predict(
            source=str(test_dir),
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            device=device,
            project=args.project,
            name=args.name,
            stream=True,        # ✅ 流式
            save=True,          # 保存渲染图到 runs/predict-*
            save_txt=True,      # 保存 YOLO txt
            save_conf=True,
            verbose=False       # 由 tqdm 负责展示进度
        ):
        # r.path 是当前图片路径字符串
        pbar.set_postfix_str(Path(r.path).name[:40])
        pbar.update(1)

    pbar.close()
    print("[INFO] 推理完成，查看 runs 目录输出。")

# ===========================
# 新增：有标签评测（中心点命中 GT 算成功）
# ===========================

def _find_images(root: Path) -> List[Path]:
    return sorted([p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS])

def _yolo_label_path_for_image(img_path: Path, labels_dir: Optional[Path]) -> Path:
    if labels_dir is not None:
        return labels_dir / (img_path.stem + ".txt")
    # 尝试从 images/... 推断到 labels/...
    parts = list(img_path.parts)
    try:
        idx = parts.index("images")
        parts[idx] = "labels"
        label_path = Path(*parts).with_suffix(".txt")
        return label_path
    except ValueError:
        # 没有 images 这一层，就放同级 txt
        return img_path.with_suffix(".txt")

def _load_gt_one(img_path: Path, label_path: Path, want_class: int, img_wh: Tuple[int,int]) -> Optional[Tuple[float,float,float,float]]:
    """
    读取 YOLO txt 格式（class cx cy w h，归一化），返回该图里“目标类”的一个 GT 框（默认唯一）。
    返回为像素级 xyxy（float）。若 label 不存在或目标类不存在，返回 None。
    """
    if not label_path.exists():
        return None
    W, H = img_wh
    with open(label_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    # 先找目标类的第一行（默认每图一个 AR_Object）
    for ln in lines:
        ss = ln.split()
        if len(ss) < 5:
            continue
        cls = int(float(ss[0]))
        if cls != want_class:
            continue
        cx, cy, w, h = map(float, ss[1:5])
        # 归一化 -> 像素
        x_c = cx * W
        y_c = cy * H
        bw  = w  * W
        bh  = h  * H
        x1 = x_c - bw/2
        y1 = y_c - bh/2
        x2 = x_c + bw/2
        y2 = y_c + bh/2
        return (x1, y1, x2, y2)
    # 没有该类
    return None

def _point_in_box(cx: float, cy: float, box_xyxy: Tuple[float,float,float,float]) -> bool:
    x1,y1,x2,y2 = box_xyxy
    return (cx >= x1) and (cx <= x2) and (cy >= y1) and (cy <= y2)

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _draw_vis(img_path: Path, save_path: Path, gt_xyxy: Optional[Tuple[float,float,float,float]], pred_center: Optional[Tuple[float,float]]):
    """
    简单可视化：GT bbox(绿)，Pred 中心(红点)。使用 PIL 画。
    """
    from PIL import ImageDraw
    im = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(im)
    if gt_xyxy is not None:
        x1,y1,x2,y2 = gt_xyxy
        draw.rectangle([x1,y1,x2,y2], outline=(0,255,0), width=3)
    if pred_center is not None:
        cx, cy = pred_center
        r = 4
        draw.ellipse([cx-r,cy-r,cx+r,cy+r], fill=(255,0,0))
    _ensure_dir(save_path.parent)
    im.save(save_path)

def cmd_eval(args):
    """
    评测口径：每图默认只有一个 AR_Object（可通过 --class-id 指定类别）。
    取预测该类的最高置信度框，中心点落入 GT bbox 算 success。
    """
    device = pick_device(args.device)
    print(f"[INFO] device = {device}")

    model = YOLO(args.weights)

    # 决定图像与标签来源
    images: List[Path] = []
    if args.test_dir:
        images = _find_images(Path(args.test_dir))
        labels_dir = Path(args.labels_dir) if args.labels_dir else None
    else:
        # 使用 data.yaml + split，解析出路径
        # ultralytics 的 Dataset 解析较重，这里直接按常见结构约定：
        # <dataset_root>/images/<split> 与 <dataset_root>/labels/<split>
        # 通过 data.yaml 的 'path' 或 'test' 字段推断；为了简单稳妥，要求用户提供 --test-dir 更直接。
        print("[WARN] 未显式提供 --test-dir，尝试使用 data.yaml + --split 推断。建议显式传入 --test-dir/--labels-dir")
        data_yaml = Path(args.data) if args.data else Path("data.yaml")
        if not data_yaml.exists():
            print(f"[ERROR] data yaml 不存在：{data_yaml.resolve()}")
            return
        import yaml
        with open(data_yaml, "r", encoding="utf-8") as f:
            yd = yaml.safe_load(f)
        split_key = args.split or "test"
        # 可能直接给出 images/test 的路径
        if split_key in yd:
            img_root = Path(yd[split_key])
        else:
            # 尝试 path + images/split
            root = Path(yd.get("path", "."))
            img_root = root / "images" / split_key
        images = _find_images(img_root)
        labels_dir = img_root.as_posix().replace("/images/", "/labels/")
        labels_dir = Path(labels_dir)

    if not images:
        print("[WARN] 未找到任何测试图像。")
        return

    out_root = Path(args.project) / args.name
    _ensure_dir(out_root)
    vis_dir = out_root / "vis" if args.save_vis else None
    report_csv = out_root / "report.csv"

    want_class = int(args.class_id)

    n_total = 0
    n_with_gt = 0
    n_success = 0
    n_no_label = 0
    n_no_pred = 0

    rows: List[Dict[str, str]] = []

    pbar = tqdm(images, desc="Evaluating (center-in-GT)", unit="img")

    for img_path in pbar:
        n_total += 1

        # 加载 GT
        with Image.open(img_path) as im:
            W, H = im.size
        label_path = _yolo_label_path_for_image(img_path, labels_dir)
        gt_xyxy = _load_gt_one(img_path, label_path, want_class, (W,H))

        if gt_xyxy is None:
            n_no_label += 1
            status = "no_label"
            pred_center = None
            # 仍可做推理，但不计入命中率分母
            r = model.predict(
                source=str(img_path), imgsz=args.imgsz, conf=args.conf, iou=args.iou,
                max_det=args.max_det, device=device, verbose=False
            )[0]
            # 取该类最高置信度（可选）
            pred_center = None
            if r and r.boxes is not None and r.boxes.data.numel() > 0:
                b = r.boxes
                if b.cls is not None:
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

        # 有 GT：进入评测分母
        n_with_gt += 1

        # 推理
        r = model.predict(
            source=str(img_path), imgsz=args.imgsz, conf=args.conf, iou=args.iou,
            max_det=args.max_det, device=device, verbose=False
        )[0]

        pred_center = None
        status = "pred_ok"

        if r is None or r.boxes is None or r.boxes.data.numel() == 0:
            n_no_pred += 1
            status = "no_pred"
        else:
            b = r.boxes
            if b.cls is None:
                n_no_pred += 1
                status = "no_pred"
            else:
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
            success = _point_in_box(pred_center[0], pred_center[1], gt_xyxy)
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

    # 汇总
    success_rate = (n_success / max(1, n_with_gt)) * 100.0

    # 输出 CSV
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

    if args.report_latex:
        # 简单 LaTeX 表格（适合论文直接粘贴）
        latex = rf"""
\begin{table}[h]
\centering
\begin{tabular}{{lrrrrr}}
\toprule
Method & \#Imgs & \#GT & Success & No-Label & No-Pred & Success(\%) \\
\midrule
YOLO(ctr\_hit) & {n_total} & {n_with_gt} & {n_success} & {n_no_label} & {n_no_pred} & {success_rate:.2f} \\
\bottomrule
\end{tabular}
\caption{{Center-in-GT evaluation on {args.split or 'test'} set (class={want_class}).}}
\end{table}
""".strip()
        print("\n----- LaTeX (copy & paste) -----\n")
        print(latex)
        print("\n--------------------------------\n")

def main():
    parser = argparse.ArgumentParser(description="YOLO AR 训练/评测/测试一体脚本")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # 通用默认
    default_yaml = "myar2.yaml"
    default_weights = "yolo11n.pt"

    # train
    p_train = sub.add_parser("train", help="训练：使用 data.yaml 的 train/val")
    p_train.add_argument("--data", type=str, default=default_yaml)
    p_train.add_argument("--weights", type=str, default=default_weights)
    p_train.add_argument("--epochs", type=int, default=100)
    p_train.add_argument("--imgsz", type=int, default=640)
    p_train.add_argument("--batch", type=int, default=16)
    p_train.add_argument("--lr0", type=float, default=0.01)
    p_train.add_argument("--patience", type=int, default=30)
    p_train.add_argument("--seed", type=int, default=42)
    p_train.add_argument("--workers", type=int, default=4)
    p_train.add_argument("--device", type=str, default="auto", choices=["auto","cpu","cuda","mps"])
    p_train.add_argument("--project", type=str, default="runs/train")
    p_train.add_argument("--name", type=str, default="exp")
    p_train.set_defaults(func=cmd_train)

    # val
    p_val = sub.add_parser("val", help="验证集评测：需要 data.yaml 的 val 有标签")
    p_val.add_argument("--data", type=str, default=default_yaml)
    p_val.add_argument("--weights", type=str, default="runs/train/exp/weights/best.pt")
    p_val.add_argument("--imgsz", type=int, default=640)
    p_val.add_argument("--device", type=str, default="auto", choices=["auto","cpu","cuda","mps"])
    p_val.add_argument("--project", type=str, default="runs/val")
    p_val.add_argument("--name", type=str, default="exp")
    p_val.set_defaults(func=cmd_val)

    # test（无标签推理）
    p_test = sub.add_parser("test", help="对 images/test 批量推理（通常无标签）")
    p_test.add_argument("--test-dir", type=str, default="myar2/images/test")
    p_test.add_argument("--weights", type=str, default="runs/train/exp/weights/best.pt")
    p_test.add_argument("--imgsz", type=int, default=640)
    p_test.add_argument("--conf", type=float, default=0.05)  # 低阈值，提升召回
    p_test.add_argument("--iou", type=float, default=0.6)
    p_test.add_argument("--max-det", type=int, default=100)
    p_test.add_argument("--device", type=str, default="auto", choices=["auto","cpu","cuda","mps"])
    p_test.add_argument("--project", type=str, default="runs/predict")
    p_test.add_argument("--name", type=str, default="exp")
    p_test.set_defaults(func=cmd_test)

    # eval（有标签评测：中心点命中 GT）
    p_eval = sub.add_parser("eval", help="有标签评测：取目标类最高置信度框中心点落入 GT 框即成功")
    p_eval.add_argument("--data", type=str, default=default_yaml, help="可选，当未提供 --test-dir 时用于推断数据根目录")
    p_eval.add_argument("--split", type=str, default="test", help="配合 --data 使用的分割名（默认 test）")
    p_eval.add_argument("--test-dir", type=str, default=None, help="显式指定测试图片目录（优先）")
    p_eval.add_argument("--labels-dir", type=str, default=None, help="显式指定 labels 目录（YOLO txt），不传则按 images->labels 推断")
    p_eval.add_argument("--weights", type=str, default="runs/train/exp/weights/best.pt")
    p_eval.add_argument("--class-id", type=int, default=0, help="目标类 id（默认为 0）")
    p_eval.add_argument("--imgsz", type=int, default=640)
    p_eval.add_argument("--conf", type=float, default=0.05)
    p_eval.add_argument("--iou", type=float, default=0.6)
    p_eval.add_argument("--max-det", type=int, default=100)
    p_eval.add_argument("--device", type=str, default="auto", choices=["auto","cpu","cuda","mps"])
    p_eval.add_argument("--project", type=str, default="runs/eval-ctr")
    p_eval.add_argument("--name", type=str, default="exp")
    p_eval.add_argument("--report-latex", action="store_true", help="打印 LaTeX 表格到控制台")
    p_eval.add_argument("--save-vis", action="store_true", help="保存可视化（GT 框+预测中心点）")
    p_eval.set_defaults(func=cmd_eval)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    sys.exit(main())
