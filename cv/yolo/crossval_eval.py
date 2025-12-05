#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
对 K 折训练好的 YOLO 模型进行一键式 cross validation 评测。

逻辑：
- 根据 split-mode / seed / kfold / fold-index 规则，自动匹配：
  - .split 下对应的 test.txt
  - runs/train 下对应的 best.pt
- 对每个 fold 调用 eval_yolo_center_hit_debug.py 做评测
- 解析其中的 Success Rate，最后输出每折结果 + mean/std

使用示例（app-ood, k=5）：

python crossval_eval.py \
  --split-mode app-ood \
  --kfold 5 \
  --seed 10 \
  --weights-stem yolo12n \
  --split-outdir .split \
  --train-project runs/train \
  --eval-script eval_yolo_center_hit_debug.py \
  --labels-root /home/xy3371/Yolo/datasets/myar/labels \
  --class-id 0 --imgsz 640 --conf 0.05

"""

import argparse
import math
import re
import statistics
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple


def find_split_dir(
    split_outdir: Path,
    mode: str,
    seed: int,
    kfold: int,
    fold_index: int,
) -> Path:
    """在 split_outdir 下找到当前 fold 对应的目录."""
    prefix = f"{mode}_s{seed}_k{kfold}_f{fold_index}_"
    candidates = [
        d for d in split_outdir.iterdir()
        if d.is_dir() and d.name.startswith(prefix)
    ]
    if not candidates:
        raise FileNotFoundError(
            f"在 {split_outdir} 下找不到前缀为 {prefix!r} 的 split 目录，"
            f"请确认已经跑过对应 fold 的训练。"
        )
    # 如有多个，按名字排序选最后一个（时间戳更大）
    candidates.sort(key=lambda p: p.name)
    chosen = candidates[-1]
    print(f"[INFO] fold={fold_index}: 使用 split 目录: {chosen}")
    return chosen


def build_exp_name(
    mode: str,
    weights_stem: str,
    split_dir_name: str,
) -> str:
    """根据 split 目录名推导训练 exp 目录名."""
    # split_dir_name 例如: "app-ood_s42_k5_f0_1763082385"
    if not split_dir_name.startswith(f"{mode}_"):
        raise ValueError(
            f"split 目录名 {split_dir_name!r} 不以 {mode + '_'} 开头，无法推导 exp name"
        )
    suffix = split_dir_name[len(mode) + 1 :]  # 去掉 "app-ood_"
    exp_name = f"{mode}_{weights_stem}_{suffix}"
    return exp_name


def run_single_eval(
    eval_script: Path,
    image_list: Path,
    labels_root: Path,
    weights: Path,
    class_id: int,
    imgsz: int,
    conf: float,
    iou: float,
    project: Path,
    name: str,
    save_vis: bool = False,
) -> float:
    """调用 eval_yolo_center_hit_debug.py 做一次评测，并解析 Success Rate."""
    cmd = [
        "python",
        str(eval_script),
        "--image_list",
        str(image_list),
        "--labels-root",
        str(labels_root),
        "--weights",
        str(weights),
        "--class-id",
        str(class_id),
        "--imgsz",
        str(imgsz),
        "--conf",
        str(conf),
        "--iou",
        str(iou),
        "--project",
        str(project),
        "--name",
        name,
    ]
    if save_vis:
        cmd.append("--save-vis")

    print(f"[CMD] {' '.join(cmd)}")

    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,  # 我们自己检查 returncode
    )

    if proc.returncode != 0:
        print("[ERROR] eval 脚本返回非零退出码")
        print("-------- STDOUT --------")
        print(proc.stdout)
        print("-------- STDERR --------")
        print(proc.stderr)
        raise RuntimeError("eval 运行失败，请检查上方输出。")

    # 解析 Success Rate           : XX.XX%
    m = re.search(r"Success Rate\s*:\s*([0-9.]+)%", proc.stdout)
    if not m:
        print("-------- STDOUT --------")
        print(proc.stdout)
        raise RuntimeError("未在输出中找到 'Success Rate : xx.xx%' 行，无法解析准确率。")

    acc = float(m.group(1))
    print(f"[RESULT] Success Rate = {acc:.2f}%")
    return acc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="对 K 折训练好的 YOLO 模型进行一键 cross validation 评估"
    )
    parser.add_argument(
        "--split-mode",
        type=str,
        required=True,
        choices=["random", "app-ood", "object-ood", "scene-ood", "group-ood"],
        help="与训练时 --split-mode 对齐，用于推导目录名。",
    )
    parser.add_argument(
        "--kfold",
        type=int,
        required=True,
        help="K 折数量（例如 5）。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="与训练时 --seed 对齐（默认为 42）。",
    )
    parser.add_argument(
        "--weights-stem",
        type=str,
        required=True,
        help="训练时预训练权重文件名去掉 .pt 之后的名字，例如 yolo11n",
    )
    parser.add_argument(
        "--split-outdir",
        type=str,
        default=".split",
        help="训练时用于存放 split 的目录（默认 .split）。",
    )
    parser.add_argument(
        "--train-project",
        type=str,
        default="runs/train",
        help="训练输出的 project 目录（默认 runs/train）。",
    )
    parser.add_argument(
        "--eval-script",
        type=str,
        default="eval_yolo_center_hit_debug.py",
        help="eval 脚本路径（默认当前目录下 eval_yolo_center_hit_debug.py）。",
    )
    parser.add_argument(
        "--labels-root",
        type=str,
        required=True,
        help="label txt 根目录，与 eval 脚本的 --labels-root 一致。",
    )
    parser.add_argument(
        "--class-id",
        type=int,
        default=0,
        help="评估目标类别（默认 0）。",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="输入图像尺寸（默认 640）。",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.05,
        help="预测的置信度阈值（默认 0.05）。",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.6,
        help="NMS 的 IoU 阈值（默认 0.6）。",
    )
    parser.add_argument(
        "--eval-project",
        type=str,
        default="runs/eval-ctr-kfold",
        help="eval 输出的 project 目录（默认 runs/eval-ctr-kfold）。",
    )
    parser.add_argument(
        "--save-vis",
        action="store_true",
        help="是否保存可视化结果（会比较占空间，调试时可开启）。",
    )

    args = parser.parse_args()

    split_outdir = Path(args.split_outdir)
    train_project = Path(args.train_project)
    eval_script = Path(args.eval_script)
    labels_root = Path(args.labels_root)
    eval_project = Path(args.eval_project)

    if not eval_script.exists():
        raise FileNotFoundError(f"eval 脚本不存在：{eval_script}")
    if not labels_root.exists():
        raise FileNotFoundError(f"labels_root 不存在：{labels_root}")
    if not split_outdir.exists():
        raise FileNotFoundError(f"split_outdir 不存在：{split_outdir}")
    if not train_project.exists():
        raise FileNotFoundError(f"train_project 不存在：{train_project}")

    kfold = args.kfold
    mode = args.split_mode
    seed = args.seed
    weights_stem = args.weights_stem

    accuracies: List[float] = []

    print(f"[INFO] 开始对 split-mode={mode}, kfold={kfold} 做 cross validation 评估")

    for fold in range(kfold):
        print("\n" + "=" * 60)
        print(f"[INFO] 处理 fold {fold}/{kfold - 1}")

        # 1. 找到对应的 split 目录
        split_dir = find_split_dir(split_outdir, mode, seed, kfold, fold)
        test_list = split_dir / "test.txt"
        if not test_list.exists():
            raise FileNotFoundError(f"fold={fold} 对应的 test.txt 不存在: {test_list}")

        # 2. 推导训练 exp 目录名 + best.pt
        exp_name = build_exp_name(mode, weights_stem, split_dir.name)
        exp_dir = train_project / exp_name
        weights = exp_dir / "weights" / "best.pt"
        if not weights.exists():
            raise FileNotFoundError(
                f"fold={fold} 对应的权重文件不存在: {weights}，"
                f"请确认已经完成该 fold 的训练。"
            )
        print(f"[INFO] fold={fold}: 使用模型: {weights}")

        # 3. 执行单折 eval
        fold_eval_name = f"{mode}_k{kfold}_f{fold}"
        acc = run_single_eval(
            eval_script=eval_script,
            image_list=test_list,
            labels_root=labels_root,
            weights=weights,
            class_id=args.class_id,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            project=eval_project,
            name=fold_eval_name,
            save_vis=args.save_vis,
        )
        accuracies.append(acc)

    print("\n" + "#" * 60)
    print(f"[SUMMARY] split-mode={mode}, kfold={kfold}")
    for i, acc in enumerate(accuracies):
        print(f"  Fold {i}: {acc:.2f}%")

    if len(accuracies) > 1:
        mean = statistics.mean(accuracies)
        std = statistics.pstdev(accuracies)  # population std
        print(f"\n  Mean accuracy : {mean:.2f}%")
        print(f"  Std  accuracy : {std:.2f}%")
    else:
        print("\n  仅有一个 fold，未计算 std。")


if __name__ == "__main__":
    main()
