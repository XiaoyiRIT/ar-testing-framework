#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
YOLO 评测脚本（中心点命中 GT）+ 调试信息 + 可视化错误样本 + Top-K 试错机制

在原脚本基础上新增：
- EXIF Orientation 自动转正（逐图，默认开启）
- 兜底逆时针旋转 --rotate-ccw（默认 0，可设 90/180/270）
- 推理与可视化统一使用“转正后的图”，避免坐标系不一致

用法示例（你当前场景）：
python eval_yolo_center_hit_debug.py \
  --test-dir datasets/temp \
  --labels-root datasets/myar/labels_2 \
  --weights runs/app_ood_leave1out/app-ood_idx11_2_s0_yolo12n/weights/best.pt \
  --class-id 0 \
  --imgsz 640 --conf 0.05 --topk 1 \
  --project runs/temp --name temp_debug \
  --save-vis

若个别图片仍不对（你观察多为逆时针 90 度），再加：
  --rotate-ccw 90
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from ultralytics import YOLO

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


# ----------------- 工具函数 -----------------


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


def debug_load_list(list_file: Path) -> List[Path]:
    print(f"[DEBUG] 准备读取 image_list: {list_file}")
    if not list_file.exists():
        print(f"[ERROR] image_list 文件不存在：{list_file}")
        return []

    raw_lines: List[str] = list_file.read_text(encoding="utf-8").splitlines()
    print(f"[DEBUG] list 文件原始行数（包含空行）: {len(raw_lines)}")

    stripped = [ln.strip() for ln in raw_lines]
    non_empty = [ln for ln in stripped if ln]
    print(f"[DEBUG] list 文件非空行数: {len(non_empty)}")

    images: List[Path] = []
    missing: List[str] = []

    for ln in non_empty:
        p = Path(ln)
        if not p.is_absolute():
            p = p.resolve()
        if p.exists():
            images.append(p)
        else:
            missing.append(str(p))

    print(f"[DEBUG] 实际存在的图片数量: {len(images)}")
    print(f"[DEBUG] 丢失的图片数量: {len(missing)}")

    if images:
        print("[DEBUG] 前 5 个有效图片路径示例：")
        for p in images[:5]:
            print(f"         {p}")
    if missing:
        print("[WARN] 前 5 个不存在的图片路径示例：")
        for s in missing[:5]:
            print(f"         {s}")

    return images


def _fix_orientation(im: Image.Image, exif_transpose: bool, rotate_ccw: int) -> Image.Image:
    """逐图处理方向：先 EXIF 转正，再额外逆时针旋转。"""
    if exif_transpose:
        im = ImageOps.exif_transpose(im)
    if rotate_ccw not in (0, 90, 180, 270):
        raise ValueError("--rotate-ccw 必须是 0/90/180/270")
    if rotate_ccw != 0:
        im = im.rotate(rotate_ccw, expand=True)  # PIL rotate 是逆时针
    return im


def draw_vis_on_image(
    im_rgb: Image.Image,
    save_path: Path,
    gt_xyxy: Optional[Tuple[float, float, float, float]],
    pred_xyxy: Optional[Tuple[float, float, float, float]],
    pred_center: Optional[Tuple[float, float]],
) -> None:
    """
    在给定图像对象上画：
    - 绿色框：GT bbox
    - 红色框：预测 bbox（top-1）
    - 红点：用于评估的预测中心（top-1 或命中的那个）
    """
    draw = ImageDraw.Draw(im_rgb)

    if gt_xyxy is not None:
        x1, y1, x2, y2 = gt_xyxy
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)

    if pred_xyxy is not None:
        px1, py1, px2, py2 = pred_xyxy
        draw.rectangle([px1, py1, px2, py2], outline=(255, 0, 0), width=3)

    if pred_center is not None:
        cx, cy = pred_center
        r = 4
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(255, 0, 0))

    save_path.parent.mkdir(parents=True, exist_ok=True)
    im_rgb.save(save_path)


def load_gt_one(
    label_path: Path,
    img_wh: Tuple[int, int],
    want_class: int,
) -> Optional[Tuple[float, float, float, float]]:
    """从单个 txt 中取指定 class 的第一个目标，返回 xyxy 像素坐标。"""
    if not label_path.exists():
        return None
    W, H = img_wh
    with open(label_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    for ln in lines:
        ss = ln.split()
        if len(ss) < 5:
            continue
        try:
            cls = int(float(ss[0]))
        except ValueError:
            continue
        if cls != want_class:
            continue
        cx, cy, w, h = map(float, ss[1:5])
        x_c, y_c = cx * W, cy * H
        bw, bh = w * W, h * H
        x1, y1 = x_c - bw / 2, y_c - bh / 2
        x2, y2 = x_c + bw / 2, y_c + bh / 2
        return (x1, y1, x2, y2)
    return None


# ----------------- 主评测逻辑 -----------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="YOLO 评测（中心点命中 GT，支持 Top-K）+ list/label 调试输出 + 可视化错误样本"
    )
    parser.add_argument("--image_list", type=str, default=None, help="每行一个图片路径（建议绝对路径）。")
    parser.add_argument("--test-dir", type=str, default=None, help="递归扫描目录中的图片。")
    parser.add_argument("--labels-root", type=str, required=True, help="label txt 根目录，文件名需与图片同名。")
    parser.add_argument("--weights", type=str, required=True, help="YOLO 权重路径（best.pt）。")
    parser.add_argument("--class-id", type=int, default=0, help="评估的目标类别 ID（默认 0）")
    parser.add_argument("--topk", type=int, default=1, help="Top-K 试错：任一中心命中 GT 即算成功。")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--max-det", type=int, default=100)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--project", type=str, default="runs/eval-ctr-debug")
    parser.add_argument("--name", type=str, default="exp")
    parser.add_argument("--save-vis", action="store_true")

    # ✅ 新增：逐图 EXIF 转正（默认开启）
    parser.add_argument("--no-exif-transpose", action="store_true", help="关闭 EXIF 转正（一般不建议）")
    # ✅ 新增：兜底逆时针旋转
    parser.add_argument("--rotate-ccw", type=int, default=0, help="额外逆时针旋转：0/90/180/270")

    args = parser.parse_args()

    if args.topk <= 0:
        print("[WARN] --topk 必须为正数，已自动设置为 1")
        args.topk = 1

    exif_transpose = not args.no_exif_transpose

    labels_root = Path(args.labels_root)
    print(f"[DEBUG] labels_root = {labels_root}")
    if not labels_root.exists():
        print(f"[ERROR] labels_root 不存在：{labels_root}")
        return
    if not labels_root.is_dir():
        print(f"[ERROR] labels_root 不是一个目录：{labels_root}")
        return

    if args.image_list:
        images = debug_load_list(Path(args.image_list))
    elif args.test_dir:
        test_dir = Path(args.test_dir)
        print(f"[DEBUG] 使用 test-dir 扫描图片: {test_dir}")
        if not test_dir.exists():
            print(f"[ERROR] test-dir 不存在：{test_dir}")
            return
        images = [p for p in test_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS]
        images.sort(key=lambda p: str(p))  # 稳定顺序
        print(f"[DEBUG] test-dir 中找到图片数量: {len(images)}")
    else:
        print("[ERROR] 必须提供 --image_list 或 --test-dir 其中之一。")
        return

    if not images:
        print("[ERROR] 没有任何图片可供评测。")
        return

    # 统计：有多少图片有对应 label
    num_have_label = 0
    for img_path in images:
        if (labels_root / (img_path.stem + ".txt")).exists():
            num_have_label += 1
    print(f"[DEBUG] 共 {len(images)} 张图片，其中 {num_have_label} 张有 label 文件（按 stem 匹配）。")

    device = pick_device(args.device)
    print(f"[DEBUG] 选择设备: {device}")
    print(f"[DEBUG] 尝试加载权重: {args.weights}")
    model = YOLO(args.weights)
    print("[DEBUG] 模型加载完成。")

    out_root = Path(args.project) / args.name
    out_root.mkdir(parents=True, exist_ok=True)

    vis_root: Optional[Path] = None
    if args.save_vis:
        vis_root = out_root / "vis"
        for sub in ["success", "fail", "no_label", "no_pred"]:
            (vis_root / sub).mkdir(parents=True, exist_ok=True)

    report_csv = out_root / "report.csv"
    want_class = int(args.class_id)

    n_total = 0
    n_with_gt = 0
    n_success = 0
    n_no_label = 0
    n_no_pred = 0

    rows: List[Dict[str, str]] = []

    print("[DEBUG] 开始逐张图片评测...")
    for idx, img_path in enumerate(images):
        img_path = Path(img_path)
        if not img_path.exists():
            continue

        n_total += 1
        if idx < 5:
            print(f"[DEBUG] 第 {idx} 张图片: {img_path}")

        # ✅ 关键：用转正后的图像参与 W/H、推理、可视化
        im0 = Image.open(img_path).convert("RGB")
        im = _fix_orientation(im0, exif_transpose=exif_transpose, rotate_ccw=args.rotate_ccw)
        W, H = im.size

        label_path = labels_root / (img_path.stem + ".txt")
        gt_xyxy = load_gt_one(label_path, (W, H), want_class)

        pred_center: Optional[Tuple[float, float]] = None
        pred_xyxy: Optional[Tuple[float, float, float, float]] = None
        pred_conf_val: Optional[float] = None
        hit_center: Optional[Tuple[float, float]] = None
        status = "pred_ok"
        success = False

        # ✅ 用转正后的图像推理（避免坐标系不一致）
        im_np = np.array(im)
        r = model.predict(
            source=im_np,
            imgsz=args.imgsz,
            conf=args.conf,
            max_det=args.max_det,
            device=device,
            verbose=False,
        )[0]

        # ---- 没有 label 的情况 ----
        if gt_xyxy is None:
            n_no_label += 1
            status = "no_label"

            if (
                r is not None
                and r.boxes is not None
                and r.boxes.data.numel() > 0
                and r.boxes.cls is not None
            ):
                b = r.boxes
                cls_np = b.cls.int().cpu().numpy()
                conf_np = b.conf.cpu().numpy()
                xyxy_np = b.xyxy.cpu().numpy()
                mask = (cls_np == want_class)
                if mask.any():
                    confs = conf_np[mask]
                    xyxy_cls = xyxy_np[mask]
                    idx_sorted = np.argsort(-confs)
                    idx_local = int(idx_sorted[0])
                    xyxy = xyxy_cls[idx_local]
                    pred_xyxy = (float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3]))
                    pred_conf_val = float(confs[idx_local])
                    cx = (xyxy[0] + xyxy[2]) / 2.0
                    cy = (xyxy[1] + xyxy[3]) / 2.0
                    pred_center = (float(cx), float(cy))

            if args.save_vis and vis_root is not None:
                save_path = (vis_root / "no_label") / f"{img_path.stem}.jpg"
                draw_vis_on_image(im.copy(), save_path, None, pred_xyxy, pred_center)

            rows.append(
                {
                    "image": str(img_path),
                    "status": status,
                    "success": "NA",
                    "gt_xyxy": "NA",
                    "pred_center": "NA" if pred_center is None else f"{pred_center[0]:.1f},{pred_center[1]:.1f}",
                    "pred_xyxy": "NA" if pred_xyxy is None else f"{pred_xyxy[0]:.1f},{pred_xyxy[1]:.1f},{pred_xyxy[2]:.1f},{pred_xyxy[3]:.1f}",
                    "pred_conf": "NA" if pred_conf_val is None else f"{pred_conf_val:.4f}",
                    "topk": str(args.topk),
                    "hit_center": "NA",
                }
            )
            continue

        # ---- 有 GT 的情况 ----
        n_with_gt += 1

        if (
            r is None
            or r.boxes is None
            or r.boxes.data.numel() == 0
            or r.boxes.cls is None
        ):
            n_no_pred += 1
            status = "no_pred"
        else:
            b = r.boxes
            cls_np = b.cls.int().cpu().numpy()
            conf_np = b.conf.cpu().numpy()
            xyxy_np = b.xyxy.cpu().numpy()
            mask = (cls_np == want_class)
            if not mask.any():
                n_no_pred += 1
                status = "no_pred"
            else:
                confs_cls = conf_np[mask]
                xyxy_cls = xyxy_np[mask]

                idx_sorted = np.argsort(-confs_cls)

                idx_top1 = int(idx_sorted[0])
                xyxy_top1 = xyxy_cls[idx_top1]
                pred_xyxy = (float(xyxy_top1[0]), float(xyxy_top1[1]), float(xyxy_top1[2]), float(xyxy_top1[3]))
                pred_conf_val = float(confs_cls[idx_top1])
                cx1 = (xyxy_top1[0] + xyxy_top1[2]) / 2.0
                cy1 = (xyxy_top1[1] + xyxy_top1[3]) / 2.0
                pred_center = (float(cx1), float(cy1))

                x1, y1, x2, y2 = gt_xyxy
                k = min(args.topk, len(idx_sorted))
                for j in idx_sorted[:k]:
                    xyxy_j = xyxy_cls[int(j)]
                    cx = (xyxy_j[0] + xyxy_j[2]) / 2.0
                    cy = (xyxy_j[1] + xyxy_j[3]) / 2.0
                    cx_f, cy_f = float(cx), float(cy)
                    if x1 <= cx_f <= x2 and y1 <= cy_f <= y2:
                        success = True
                        hit_center = (cx_f, cy_f)
                        break

        if success:
            n_success += 1

        if args.save_vis and vis_root is not None:
            if status == "no_pred":
                subdir = "no_pred"
            else:
                subdir = "success" if success else "fail"
            save_path = (vis_root / subdir) / f"{img_path.stem}.jpg"
            vis_center = hit_center if hit_center is not None else pred_center
            draw_vis_on_image(im.copy(), save_path, gt_xyxy, pred_xyxy, vis_center)

        rows.append(
            {
                "image": str(img_path),
                "status": status,
                "success": "1" if success else "0",
                "gt_xyxy": f"{gt_xyxy[0]:.1f},{gt_xyxy[1]:.1f},{gt_xyxy[2]:.1f},{gt_xyxy[3]:.1f}",
                "pred_center": "NA" if pred_center is None else f"{pred_center[0]:.1f},{pred_center[1]:.1f}",
                "pred_xyxy": "NA" if pred_xyxy is None else f"{pred_xyxy[0]:.1f},{pred_xyxy[1]:.1f},{pred_xyxy[2]:.1f},{pred_xyxy[3]:.1f}",
                "pred_conf": "NA" if pred_conf_val is None else f"{pred_conf_val:.4f}",
                "topk": str(args.topk),
                "hit_center": "NA" if hit_center is None else f"{hit_center[0]:.1f},{hit_center[1]:.1f}",
            }
        )

    success_rate = (n_success / n_with_gt) * 100.0 if n_with_gt > 0 else 0.0

    with open(report_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image",
                "status",
                "success",
                "gt_xyxy",
                "pred_center",
                "pred_xyxy",
                "pred_conf",
                "topk",
                "hit_center",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\n========== EVAL (Center-in-GT, DEBUG, Top-K) ==========")
    print(f"Total images           : {n_total}")
    print(f"With GT (denominator)  : {n_with_gt}")
    print(f"Success (hit center)   : {n_success}")
    print(f"No GT label            : {n_no_label}")
    print(f"No prediction (class)  : {n_no_pred}")
    print(f"Top-K (K)              : {args.topk}")
    print(f"EXIF transpose         : {exif_transpose}")
    print(f"Rotate CCW             : {args.rotate_ccw}")
    print(f"Success Rate           : {success_rate:.2f}%")
    print(f"Report CSV             : {report_csv}")
    if args.save_vis:
        print(f"Visualizations root    : {vis_root}")


if __name__ == "__main__":
    main()
