#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
python test_yolo_train.py train --data myar2.yaml --epochs 100 --imgsz 640

python test_yolo_train.py val --data myar2.yaml --weights runs/train/exp/weights/best.pt

python test_yolo_train.py test --test-dir testimages --weights runs/train/exp4/weights/best.pt --conf 0.05
"""

import argparse
import sys
import platform
from tqdm import tqdm
from pathlib import Path
from ultralytics import YOLO

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

    # 1) 选择基模型：默认 yolo11s.pt（更稳），可用 --weights 指定
    model = YOLO(args.weights)

    # 2) 开始训练：只使用 data yaml 的 train/val
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
    print(results)

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

def main():
    parser = argparse.ArgumentParser(description="YOLO AR 训练/评测/测试一体脚本")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # 通用默认
    #default_yaml = "myar2.yaml"   # 如你仍使用 myar_simp.yaml，请改成它
    default_yaml = "myar2.yaml"
    #default_weights = "yolo11n.yaml"  # 更稳；若想更快可换回 yolo11n.pt
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

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    sys.exit(main())
